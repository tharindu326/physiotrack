import cv2
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Any, Tuple
from tqdm import tqdm
from physiotrack.modules.Yolo.classes_and_palettes import COLORS
from physiotrack.core.overlay import draw_label
from physiotrack.core.radar_view import RadarView
from physiotrack.core.depth_view import DepthView
from physiotrack.core.ego_view import EgoVideoView
from physiotrack.core.rom_skeleton_view import ROMSkeletonView
from physiotrack.signals.plotting.keypoint_plotter import KeypointMotionPlotter
from physiotrack.signals.plotting.angle_plotter import JointAnglePlotter
from physiotrack.capture.orientation import resolve_rotation, apply_rotation
from physiotrack.signals.motion.features import DEFAULT_ROM_MOVEMENTS, ROM_DEFINITIONS
from physiotrack.utils import get_screen_size, resize_frame_for_display

from ..core.boxes import assign_ids
from ..core.panel import attach_stack
from .._logging import get_logger
from .writer import open_video_writer
from ..results import (FrameResult, Instance, Result, ResultMeta,
                       VideoResults)

logger = get_logger(__name__)


class Video:
    """High-level orchestrator that runs the full inference pipeline over a video.

    ``Video`` ties together the individual predictors -- detection, pose
    estimation, tracking, segmentation, face detection and analysis, and depth --
    and drives them frame-by-frame (optionally in batches) across a clip, camera
    device or RTSP stream. It composites every enabled model's output onto each
    frame, builds the requested side panels (keypoint-motion plot, joint-angle /
    ROM grids, ROM skeleton canvas, top-down radar view, depth view, ego-video
    view), writes an annotated output video, and returns / saves the per-frame
    results as structured data.

    You attach models by passing already-constructed predictor instances to the
    constructor (``detector``, ``pose``, ``tracker`` ...); any left as ``None`` are
    simply skipped. Nothing runs until you call [`run`][physiotrack.Video.run].
    Every stage is optional, so the same class covers a bare pose-only pass as well
    as the complete multi-model pipeline.

    The pipeline order per batch is: detection -> tracking -> pose -> segmentation
    -> faces (detection, subject ids, face stages) -> depth -> overlay/compositing. When both a custom
    ``detector`` and a ``pose`` estimator are supplied, the pose estimator must be
    a ``Pose.Custom`` instance (so it consumes external boxes); otherwise a
    ``ValueError`` is raised at run time.

    Attributes:
        video_path (str | Path | int): The original ``source`` argument.
        source_identifier (str): Human-readable id derived from the source (file
            stem, RTSP host, or ``"CAM_device_<n>"``); used to name outputs/logs.
        total_frames (int | None): Frame count for seekable files, else ``None``
            (streams / cameras).
        video_fps (float): Native frames-per-second of the source, e.g. ``29.97``
            (falls back to ``30.0`` when it cannot be read). Timestamps, the output
            video and the vitals all use this exact rate.
        width (int): Effective output frame width in pixels (already accounts for
            ``orient`` 90/270 dimension swap).
        height (int): Effective output frame height in pixels.
        batch_size (int): Number of frames processed per batch (>= 1).
        detectors (list): Detector instances (empty if none supplied).
        segmentators (list): Segmentation instances (empty if none supplied).
        pose_estimator: The attached [`Pose`][physiotrack.Pose] predictor or ``None``.
        tracker: The attached [`Tracker`][physiotrack.Tracker] or ``None``.
        output_path (Path): Directory where outputs are written.
        rom_movements (list[str]): Resolved list of enabled ROM movement names.

    Example:
        ```python
        import physiotrack as pt

        # Pose-only pass over a clip (see examples/pose_video.py)
        pose = pt.Pose.VRStudent(verbose=False, device=0)
        video = pt.Video(source="clip.mp4", pose=pose, output_dir="output", verbose=True)
        results = video.run("output/clip_poses.mp4", "output/clip_result.json")
        logger.info(f"Processed {len(results)} frames")
        ```

    See Also:
        [`Detection`][physiotrack.Detection]: person / object detectors to pass as ``detector``.
        [`Pose`][physiotrack.Pose]: pose estimators to pass as ``pose``.
        [`Tracker`][physiotrack.Tracker]: multi-object tracker to pass as ``tracker``.
        [`Result`][physiotrack.Result]: per-frame result object produced by the predictors.
    """

    def __init__(self,
                 source: Union[str, Path, int],
                 *,
                 detector=None,
                 pose=None,
                 segmenter=None,
                 tracker=None,
                 face=None,
                 face_stages=None,
                 depth=None,
                 ego_video: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 fps: Optional[int] = None,
                 resize: Optional[Tuple[int, int]] = None,
                 rotate: bool = False,
                 orient=0,
                 floor_map: Optional[List[Tuple[int, int]]] = None,
                 floor_map_background: Optional[Union[str, np.ndarray]] = None,
                 floor_map_rotation: int = 90,
                 depth_colormap: str = 'inferno',
                 plot_keypoint: Optional[int] = None,
                 plot_keypoint_name: Optional[str] = None,
                 plot_angles: bool = False,
                 angle_joints: Optional[List[str]] = None,
                 rom=None,
                 rom_render: bool = True,
                 rppg: bool = False,
                 hrv: bool = False,
                 respiration: bool = False,
                 respiration_source: str = "pulse",
                 rppg_method: str = "POS",
                 rppg_roi=None,
                 rppg_window_sec: Optional[float] = None,
                 device: Union[str, int] = 'cpu',
                 verbose: bool = False,
                 show_fps: bool = False,
                 show: bool = False,
                 batch_size: int = 1):
        """Configure the pipeline: attach models, panels and I/O options.

        Opens the ``source`` immediately (to read fps / resolution / frame count)
        and builds any side-panel views that were requested, but does not process
        any frames -- call [`run`][physiotrack.Video.run] to start. All model
        arguments accept an already-constructed predictor instance (or ``None`` to
        skip that stage). ``detector`` and ``segmenter`` additionally accept a
        ``list`` of instances to run several of the same kind.

        Args:
            source (str | Path | int): Video source. A file path, an ``rtsp://``
                URL, or an integer camera-device index (e.g. ``0``). RTSP streams
                and camera devices have no known frame count.
            detector (optional): Detection predictor, or a ``list`` of them, whose
                boxes feed pose/segmentation/tracking. Defaults to ``None`` (no
                detection). See [`Detection`][physiotrack.Detection]. When a
                detector is combined with a pose estimator, the pose estimator must
                be a ``Pose.Custom``.
            pose (optional): Pose estimator producing keypoints per frame. Defaults
                to ``None``. See [`Pose`][physiotrack.Pose].
            segmenter (optional): Segmentation predictor, or a ``list`` of them,
                whose masks are colorized and alpha-blended onto the frame.
                Defaults to ``None``.
            tracker (optional): Multi-object tracker assigning stable IDs to
                detections; runs frame-by-frame (no batching). Defaults to
                ``None``. See [`Tracker`][physiotrack.Tracker].
            face (Face | VRFace, optional): Face detector run on every frame. With a
                ``tracker``, each face is given the track id of the subject whose box
                contains it, so face signals can follow a person over time. Its faces
                are exported as [`FrameResult.faces`][physiotrack.FrameResult].
                Defaults to ``None``. Leave it out when ``detector`` is itself a face
                detector: the (tracked) detections are then the faces.
            face_stages (list[FaceStage], optional): Face stages applied, in order, to
                every face -- e.g. ``[FaceOrientation(), FaceLandmarks(),
                FaceExpression()]``. A stage that needs another's output (e.g.
                [`GazeEstimator`][physiotrack.GazeEstimator] needs
                [`FaceLandmarks`][physiotrack.FaceLandmarks]) must come after it.
                Requires faces from ``face`` or a face ``detector``. Defaults to
                ``None``.
            depth (optional): Monocular depth estimator; enables the depth side
                view. Defaults to ``None``.
            ego_video (str | Path, optional): Path to an ego-centric video to
                overlay as a synchronized side view. Defaults to ``None``.
            output_dir (str | Path, optional): Directory for outputs; created if
                missing. Defaults to ``None`` (uses the current working directory).
            fps (int, optional): Target processing frame rate. Frames are subsampled
                so roughly this many frames per source-second are processed and the
                output video is written at this rate. Defaults to ``None`` (process
                every frame at the source fps).
            resize (tuple[int, int], optional): ``(width, height)`` to resize every
                frame to before inference and output. Defaults to ``None`` (keep
                native resolution).
            rotate (bool): If ``True``, rotate every frame 90 degrees clockwise
                (via ``cv2.ROTATE_90_CLOCKWISE``) during preprocessing, swapping the
                output dimensions. Applied after ``orient``. Defaults to ``False``.
            orient (int | str, optional): Explicit orientation fix applied to every
                frame, one of ``0`` / ``90`` / ``180`` / ``270`` degrees (also
                accepts ``None`` / ``"none"`` for no rotation); unknown values fall
                back to ``0``. Use this for phone clips whose display rotation lives
                in container metadata rather than the pixels. ``90``/``270`` swap the
                effective ``width``/``height``. Defaults to ``0`` (no rotation).
            floor_map (list[tuple[int, int]], optional): Four ``(x, y)`` image
                corner points delimiting the floor region; enables the top-down
                radar view (requires both ``tracker`` and ``pose`` at run time).
                Defaults to ``None`` (radar view disabled).
            floor_map_background (str | np.ndarray, optional): Background for the
                radar canvas. ``None`` / ``"default"`` uses a plain canvas;
                ``"auto"`` / ``"extract"`` warps the floor region out of the first
                frame via homography; a path string or image array supplies a
                pre-made floor plan. Defaults to ``None``.
            floor_map_rotation (int): Rotation in degrees (``0`` / ``90`` / ``180``
                / ``270``) applied to orient the radar view. Defaults to ``90``.
            depth_colormap (str): Matplotlib colormap name for the depth view, e.g.
                ``"inferno"``, ``"viridis"``, ``"magma"``, ``"plasma"`` or ``"jet"``.
                Only used when ``depth`` is provided. Defaults to ``"inferno"``.
            plot_keypoint (int, optional): COCO keypoint id to plot as a live motion
                signal in a top-right panel (e.g. ``9`` = left wrist, ``10`` = right
                wrist). Requires a pose estimator at run time. Defaults to ``None``
                (no motion plot).
            plot_keypoint_name (str, optional): Display label for the motion plot.
                Defaults to ``None`` (falls back to ``"keypoint_<id>"``).
            plot_angles (bool): If ``True``, overlay a live joint-angle panel
                (interior joint angles) on the left. Ignored with a warning if no
                pose estimator is provided. Defaults to ``False``.
            angle_joints (list[str], optional): Subset of joint names to show in the
                angle panel, e.g. ``["leftElbow", "rightElbow", "leftKnee",
                "rightKnee"]``. Only used when ``plot_angles`` is ``True``. Defaults
                to ``None`` (shows all available joints).
            rom (bool | list[str], optional): Clinical range-of-motion overlays.
                ``True`` enables the default ROM movement set; a ``list`` selects
                specific movement names (filtered against the known ROM definitions);
                ``None`` / ``False`` disables ROM. Requires a pose estimator.
                Defaults to ``None``.
            rom_render (bool): If ``True`` (and ``rom`` is enabled with a pose
                estimator), draw the white full-room skeleton canvas with
                color-coded ROM arcs. Defaults to ``True``.
            rppg (bool): If ``True``, overlay the contactless rPPG panels (the
                blood-volume-pulse trace and the heart rate in bpm) in the top-right
                stack, and add ``vitals`` (``hr``, ``snr``) to the JSON output. The pulse
                skin ROI is a SegFace segmentation by default (see ``rppg_roi``) -- no
                separate face detector is required. Defaults to ``False``.
            hrv (bool): If ``True``, overlay a heart-rate-variability panel (RMSSD,
                SDNN, pNN50, SD1/SD2, LF/HF) and add ``vitals["hrv"]`` to the JSON.
                Uses a longer (60 s) rPPG window for stable indices. Same skin ROI as
                ``rppg`` (``rppg_roi``). Defaults to ``False``.
            respiration (bool): If ``True``, overlay a respiration-rate panel
                (breaths/min) and add ``vitals["respiration"]`` to the JSON. The signal
                source is chosen by ``respiration_source``. Defaults to ``False``.
            respiration_source (str): Where respiration is derived from when
                ``respiration`` is enabled. ``"pulse"`` (default) uses the rPPG amplitude
                modulation (RIAV/RSA), sharing the rPPG estimator and its ``rppg_roi``
                skin segmentation (SegFace by default). ``"motion"`` uses shoulder/torso
                motion via
                [`respiration_from_motion`][physiotrack.signals.respiration_from_motion],
                reusing the per-frame pose keypoints (no extra inference) and requiring a
                ``pose`` estimator instead of a face. Ignored with a warning if the
                required model is missing.
            rppg_method (str): rPPG extraction method for the ``rppg`` / ``hrv`` panels,
                one of ``"POS"``, ``"CHROM"``, ``"LGI"``, ``"OMIT"``. Defaults to
                ``"POS"`` (the most motion-robust; see
                [`benchmark_rppg_methods`][physiotrack.signals.benchmark_rppg_methods]).
            rppg_roi (callable | object, optional): The skin-**segmentation** provider
                that the rPPG / HR / HRV (and pulse respiration) panels sample -- always
                a segmentation mask, never a raw face box. Defaults to ``None``, which
                builds a SegFace
                [`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor] (it finds
                faces itself, so no separate face detector is needed). Pass a custom
                provider to override it -- a callable ``roi(frame_bgr) -> mask`` or an
                object exposing ``skin_mask(frame_bgr, boxes=None) -> mask`` (e.g. a
                face-neck segmentation model for VR/occluded-face cases) -- returning a
                boolean ``(H, W)`` mask (or ``None`` to skip that frame). ``boxes`` are
                the frame's face boxes when the pipeline finds faces (``face=`` or a
                face ``detector``), else ``None``. Provide a
                ``FaceSkinExtractor(device=...)`` here to override the device the
                default extractor uses (which is ``device``).
            rppg_window_sec (float, optional): Sliding-window length (seconds) of the
                rPPG estimator. Longer windows give more stable HR/HRV but a later first
                reading (the first value appears after ~60% of the window fills).
                Defaults to ``None``, which auto-selects ``60`` s when ``hrv`` is enabled
                (needed for stable HRV) and ``15`` s otherwise. Set a smaller value
                (e.g. ``10``) for short clips so panels populate sooner.
            verbose (bool): If ``True``, print setup/progress info and show a tqdm
                progress bar. Defaults to ``False``.
            show_fps (bool): If ``True``, print real-time and end-of-run
                component-wise FPS statistics. Defaults to ``False``.
            show (bool): If ``True``, display the annotated frames in a live OpenCV
                window during processing (press ``q`` to quit). Defaults to ``False``.
            batch_size (int): Number of frames processed together per pipeline step;
                values below ``1`` are clamped to ``1``. Tracking always runs
                frame-by-frame regardless. Defaults to ``1``.

        Raises:
            ValueError: If ``source`` cannot be opened by OpenCV; if ``face_stages`` are
                given without a face source or out of order; or if faces would come
                from two places (a face ``detector`` and ``face``).

        Example:
            ```python
            import physiotrack as pt

            # Pose + explicit detector + tracker (see examples/tracker_aided_pose_video.py)
            pose = pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.b_wholebody, device=0)
            detector = pt.Detection.VRStudent(device=0)
            cfg = pt.TrackerConfig(); cfg.tracker_type = "ocsort"; cfg.classes = [0]
            tracker = pt.Tracker(config=cfg)

            video = pt.Video(
                source="clip.mp4",
                pose=pose,
                detector=detector,
                tracker=tracker,
                output_dir="output",
                batch_size=4,
                verbose=True,
            )
            video.run("output/clip_poses.mp4", "output/clip_result.json")

            # Faces of tracked people, with head pose and face mesh; blinks per person
            video = pt.Video(
                source="clip.mp4",
                detector=pt.Detection.Person(), tracker=pt.Tracker(),
                face=pt.Face(),
                face_stages=[pt.FaceOrientation(), pt.FaceLandmarks()],
            )
            results = video.run()
            blinks = pt.signals.detect_blinks(results, detection_id=1)
            ```

        Note:
            Constructing the object opens the video capture and (when
            ``plot_keypoint`` is set) briefly opens a second capture to read the fps.
        """

        self.video_path = source
        # Support both single instance and list of instances for detector and segmenter
        self.detectors = detector if isinstance(detector, list) else ([detector] if detector is not None else [])
        self.segmentators = segmenter if isinstance(segmenter, list) else ([segmenter] if segmenter is not None else [])
        self.tracker = tracker
        self.pose_estimator = pose
        self.face_detector = face
        self.face_stages = list(face_stages or [])
        # A face detector passed as ``detector`` makes the (tracked) subjects the faces.
        face_detectors = [getattr(d, "task", "detect") == "face" for d in self.detectors]
        self._faces_from_detector = bool(face_detectors) and all(face_detectors)
        if any(face_detectors) and not self._faces_from_detector:
            raise ValueError(
                "A face detector was mixed with other detectors in `detector`. Pass the "
                "face detector as `face=` so its faces are matched to the tracked subjects.")
        if self._faces_from_detector and face is not None:
            raise ValueError(
                "`detector` is already a face detector, so its detections are the faces; "
                "drop `face=` (or use a person detector with `face=`).")
        if self.face_stages:
            from ..face.base import check_stage_order
            check_stage_order(self.face_stages)
            if face is None and not self._faces_from_detector:
                raise ValueError(
                    "face_stages need faces: pass face=pt.Face() (or VRFace), or use a "
                    "face detector as `detector`.")
        self.depth_estimator = depth
        self.ego_video_path = ego_video
        self.verbose = verbose
        self.show_fps = show_fps
        self.show_output = show
        self.required_fps = fps
        self.frame_resize = resize
        self.frame_rotate = rotate
        self.floor_map = floor_map
        self.batch_size = max(1, batch_size)  # Ensure batch size is at least 1

        # Get screen size for display if show_output is enabled
        self.screen_width = None
        self.screen_height = None
        if self.show_output:
            self.screen_width, self.screen_height = get_screen_size(verbose=self.verbose)

        # Initialize radar view with background mode and rotation
        self.radar_view = RadarView(
            floor_map=floor_map,
            background=floor_map_background,
            rotation=floor_map_rotation
        ) if floor_map else None

        # Initialize depth view if depth estimator is provided
        # Match width to radar view if available, otherwise use default
        self.depth_view = None
        if depth is not None:
            depth_max_width = 320  # default
            if self.radar_view is not None:
                depth_max_width = self.radar_view.canvas_size[0]
            self.depth_view = DepthView(
                max_width=int(depth_max_width * 1.2),  # ~20% larger (matches ROM skeleton)
                max_height=720,  # Allow taller to preserve aspect ratio
                colormap=depth_colormap,
                show_title=True
            )

        # Initialize ego video view if ego video path is provided
        # Match width to radar view if available, otherwise use default
        self.ego_view = None
        if ego_video is not None:
            ego_max_width = 320  # default
            if self.radar_view is not None:
                ego_max_width = self.radar_view.canvas_size[0]
            self.ego_view = EgoVideoView(
                ego_video_path=str(ego_video),
                max_width=ego_max_width,
                max_height=600,  # Allow taller to preserve aspect ratio
                show_title=True
            )

        # Initialize keypoint motion plotter
        self.motion_plotter = None
        if plot_keypoint is not None:
            # Get video FPS for the plotter
            cap_temp = cv2.VideoCapture(source)
            video_fps = float(cap_temp.get(cv2.CAP_PROP_FPS))
            if not video_fps > 0:
                video_fps = 30.0
            cap_temp.release()
            
            if plot_keypoint_name is None:
                plot_keypoint_name = f"keypoint_{plot_keypoint}"
            
            self.motion_plotter = KeypointMotionPlotter(
                keypoint_id=plot_keypoint,
                keypoint_name=plot_keypoint_name,
                window_size=300,  # 10 seconds at 30fps
                canvas_width=450,  # Reduced width for better performance
                canvas_height=180,  # Reduced height for better performance
                filter_signal=True,
                filter_bandpass=(0.5, 5.0),
                fps=float(video_fps)
            )

        if output_dir:
            self.output_path = Path(output_dir)
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = Path.cwd()

        # Orientation fix: opt-in, default 0 (none). Phones store display rotation as
        # container metadata rather than baking it into the pixels, so frames can decode
        # sideways/upside down; pass orient=90/180/270 to rotate when a clip needs it.
        # There is no auto/metadata mode (it is unreliable across builds). Applied to
        # every frame in preprocess_frame; 90/270 swaps the effective dimensions.
        self.device = device
        self._rotation = resolve_rotation(orient)
        if self._rotation:
            logger.info(f"[orientation] rotating frames {self._rotation} deg (orient).")
        self._open_source(source)

        rom_enabled = bool(rom) and rom is not False
        self.rom_movements = []
        if rom_enabled:
            self.rom_movements = (list(DEFAULT_ROM_MOVEMENTS) if rom is True
                                  else [m for m in rom if m in ROM_DEFINITIONS])

        # Joint-angle panel (top-left): interior joint angles (plot_angles) and/or
        # the clinical ROM angle *values* as color-coded rows (rom). The same colors
        # mark the arcs on the skeleton canvas below.
        self.angle_plotter = None
        if plot_angles or rom_enabled:
            if self.pose_estimator is None:
                if self.verbose:
                    warnings.warn(
                        "plot_angles/rom were requested but no pose estimator was given, so "
                        "the angle panel is disabled. Pass pose=... to enable it.",
                        RuntimeWarning, stacklevel=2)
            else:
                self.angle_plotter = JointAnglePlotter(
                    joints=angle_joints if plot_angles else [],
                    rom=self.rom_movements if rom_enabled else None,
                    fps=float(self.video_fps),
                )

        # Skeleton canvas (left, under the angle panel): the person's skeleton on a
        # white full-room canvas with color-coded ROM arcs. Shown when rom is on and
        # rom_render is True.
        self.rom_skeleton_view = None
        if rom_enabled and rom_render and self.rom_movements and self.pose_estimator is not None:
            rom_base_width = self.radar_view.canvas_size[0] if self.radar_view is not None else 320
            self.rom_skeleton_view = ROMSkeletonView(
                max_width=int(rom_base_width * 1.2), max_height=720, show_title=True  # ~20% larger
            )

        if respiration and respiration_source not in ("pulse", "motion"):
            raise ValueError(
                f"respiration_source must be 'pulse' or 'motion', got {respiration_source!r}")

        # rPPG vitals (top-right stack): a single shared HeartRateEstimator drives the
        # rPPG / HR / HRV panels -- and the respiration panel too when
        # ``respiration_source == 'pulse'`` -- so the pulse is computed once per frame.
        # The pulse skin ROI is ALWAYS a SEGMENTATION mask, never a raw face box: by
        # default a SegFace face-skin segmentation (FaceSkinExtractor, which finds faces
        # itself -- no face detector needed), or a custom ``rppg_roi`` mask provider
        # (e.g. a face-neck segmentation model for VR/occluded-face cases).
        if rppg_roi is not None and not (hasattr(rppg_roi, "skin_mask") or callable(rppg_roi)):
            raise TypeError(
                "rppg_roi must be a callable roi(frame) -> mask or an object with "
                "skin_mask(frame, boxes=None) -> mask.")
        self.rppg_roi = rppg_roi
        self.rppg_estimator = None
        self.rppg_panels = []
        self._hrv_panel = None
        self._resp_panel = None       # pulse-derived respiration panel (source='pulse')
        self._skin_extractor = None   # default SegFace skin segmenter (built on demand)
        pulse_resp = respiration and respiration_source == "pulse"
        if rppg or hrv or pulse_resp:
            from physiotrack.signals import (HeartRateEstimator, RPPGPlotter,
                                              HeartRatePlotter, HRVPlotter,
                                              RespirationPlotter, FaceSkinExtractor)
            # Default skin ROI = SegFace segmentation; a custom provider overrides it.
            if self.rppg_roi is None:
                self._skin_extractor = FaceSkinExtractor(device=self.device)
                self.rppg_roi = self._skin_extractor
            window_sec = (float(rppg_window_sec) if rppg_window_sec is not None
                          else (60.0 if hrv else 15.0))
            self.rppg_estimator = HeartRateEstimator(
                rppg_method, float(self.video_fps), window_sec=window_sec)
            if rppg:
                self.rppg_panels.append(RPPGPlotter(estimator=self.rppg_estimator))
                self.rppg_panels.append(HeartRatePlotter(estimator=self.rppg_estimator))
            if hrv:
                self._hrv_panel = HRVPlotter(estimator=self.rppg_estimator)
                self.rppg_panels.append(self._hrv_panel)
            if pulse_resp:
                self._resp_panel = RespirationPlotter(estimator=self.rppg_estimator)
                self.rppg_panels.append(self._resp_panel)

        # Motion-derived respiration (``respiration_source == 'motion'``): reuse the pose
        # keypoints the pipeline already computes each frame -- no extra inference and no
        # face needed. The shoulder-y signal is recomputed every ``_resp_recompute_every``
        # frames over a rolling buffer of pose records (see ``respiration_from_motion``).
        self._resp_motion_panel = None
        self._resp_records = None
        self._resp_recompute_every = 30
        if respiration and respiration_source == "motion":
            if self.pose_estimator is None:
                if self.verbose:
                    warnings.warn(
                        "respiration_source='motion' was requested but no pose estimator was "
                        "given, so motion-based respiration is disabled. Pass pose=... to enable it.",
                        RuntimeWarning, stacklevel=2)
            else:
                from collections import deque
                from physiotrack.signals import RespirationPlotter
                self._resp_motion_panel = RespirationPlotter(
                    fps=float(self.video_fps), source="motion")
                # ~45 s of frames comfortably covers the low respiration frequency band.
                self._resp_records = deque(maxlen=max(64, int(self.video_fps * 45)))

        if self.verbose:
            logger.info(f"Video properties: {self.width}x{self.height}, {self.video_fps} FPS")
            logger.info(f"Source: {self.source_identifier}")
            logger.info(f"Batch size: {self.batch_size}")
    def _collect_performance(self, frame_count, total_time, avg_fps, frames_read=None):
        """Gather end-to-end and per-stage throughput for the run just finished.

        Args:
            frame_count (int): Frames that went through the models.
            total_time (float): Wall-clock seconds for the whole run.
            avg_fps (float): Overall pipeline frames per second, over processed frames.
            frames_read (int, optional): Frames decoded from the source, including any
                skipped by ``fps=`` subsampling. Defaults to ``None`` (same as
                ``frame_count``).

        Returns:
            dict: ``{"batch_size", "frames", "seconds", "fps", "stages"}`` where
                ``stages`` maps a stage name (e.g. ``"detector[0]"``, ``"pose"``,
                ``"tracker"``) to ``{"fps": float, "ms_per_frame": float}``. Stages that
                were not configured are absent.

        Note:
            Also stored as :attr:`performance` after each
            [`run`][physiotrack.Video.run], so throughput can be reported or compared
            programmatically rather than read off the console.
        """
        stages = {}

        def record(name, component):
            if component is not None and hasattr(component, 'get_avg_fps'):
                stages[name] = {
                    "fps": component.get_avg_fps(),
                    "ms_per_frame": component.get_avg_inference_time(),
                }

        for idx, detector in enumerate(self.detectors):
            record(f"detector[{idx}]", detector)
        record("pose", getattr(self.pose_estimator, 'pose_estimator', None))
        record("tracker", self.tracker)
        for idx, segmentor in enumerate(self.segmentators):
            record(f"segmentor[{idx}]", segmentor)
        record("face_detector", self.face_detector)
        for stage in self.face_stages:
            record(type(stage).__name__, stage)
        record("depth", self.depth_estimator)

        return {
            "batch_size": self.batch_size,
            "frames": frame_count,
            "frames_read": frame_count if frames_read is None else frames_read,
            "seconds": total_time,
            "fps": avg_fps,
            "stages": stages,
        }

    @staticmethod
    def _format_performance(metrics):
        """Render :meth:`_collect_performance` output as a readable report.

        Args:
            metrics (dict): The mapping returned by :meth:`_collect_performance`.

        Returns:
            str: A multi-line report, emitted as a single log record so it cannot be
                interleaved with other output.
        """
        rule = "=" * 60
        lines = [
            rule,
            "PERFORMANCE METRICS",
            rule,
            f"Batch size:              {metrics['batch_size']}",
            f"Overall pipeline FPS:    {metrics['fps']:.2f}",
            f"Total frames processed:  {metrics['frames']}",
            f"Frames decoded:          {metrics.get('frames_read', metrics['frames'])}",
            f"Total time:              {metrics['seconds']:.2f}s",
            "-" * 60,
        ]
        for name, stage in metrics["stages"].items():
            lines.append(f"{name:<24} {stage['fps']:8.2f} FPS "
                         f"({stage['ms_per_frame']:.2f} ms/frame)")
        lines.append(rule)
        return "\n".join(lines)

    @staticmethod
    def _attach_track_ids(pose_results_batch, track_ids_batch):
        """Stamp persistent track ids onto pose detections, in place.

        The pose backends label their detections with the positional index of the box
        they came from, which is only meaningful within a single frame. Tracking runs
        first and supplies one id per box, in the same order, so this replaces each
        positional index with the subject's persistent id.

        That identity is what makes any cross-frame analysis valid: without it, the
        per-frame tables built by
        [`extract_keypoints_sequence`][physiotrack.signals.extract_keypoints_sequence]
        interleave subjects, and the finite differences behind velocity and
        acceleration end up subtracting one person's position from another's.

        Args:
            pose_results_batch (list): One per-frame ``detections`` list (as produced by
                [`process_batch_pose`][physiotrack.Video.process_batch_pose]). Modified
                in place.
            track_ids_batch (list): Per-frame list of track ids aligned with the boxes
                pose was given, or ``None`` for frames that were not tracked.
        """
        for detections, track_ids in zip(pose_results_batch, track_ids_batch):
            if not track_ids or not detections:
                continue
            for position, detection in enumerate(detections):
                if isinstance(detection, dict) and position < len(track_ids):
                    detection['id'] = track_ids[position]

    def _open_source(self, source):
        """Point the pipeline at a source and read its stream properties.

        Opens ``source`` as the active capture and refreshes everything derived from
        it: the source identifier and frame count, the frame rate, and the frame
        dimensions (swapped when ``orient`` requests a 90/270 degree rotation). Any
        capture already open is released first.

        This is called by ``__init__`` and again by
        [`batch_run`][physiotrack.Video.batch_run] for each input, so a single
        configured pipeline can be pointed at successive clips without the two paths
        drifting apart.

        Args:
            source (str | int): Video file path, ``rtsp://`` URL, or camera index.

        Raises:
            ValueError: If the source cannot be opened.
        """
        previous = getattr(self, "cap", None)
        if previous is not None:
            previous.release()

        self.video_path = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {source}")

        self._setup_source_info()

        # Keep the exact rate (29.97, not 29): timestamps and every per-second measure
        # derive from it. Only the subsampling cycle in run() rounds it.
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.video_fps > 0:
            self.video_fps = 30.0  # Default FPS
            if self.verbose:
                logger.info(f'Using default FPS: {self.video_fps}')
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._rotation in (90, 270):
            self.width, self.height = self.height, self.width

    def _setup_source_info(self):
        """Setup source identifier and total frames based on video path type."""
        if isinstance(self.video_path, (str, Path)):
            source = str(self.video_path)
            path_prefix = ''.join(letter for letter in source.split(':')[0] if letter.isalnum())
            if path_prefix == 'rtsp':
                if self.verbose:
                    logger.info(f'Start processing RTSP stream {source}')
                source_name = ".".join(source.split('@')[-1].split('.')[:-1]).replace(':', '-').replace('/', '_')
                self.source_identifier = f'{source_name}'
                self.total_frames = None
            else:
                if self.verbose:
                    logger.info(f'Start processing video {self.video_path}')
                source_name = Path(self.video_path).stem
                self.source_identifier = f'{source_name}'
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames = frame_count if frame_count > 0 else None
        elif isinstance(self.video_path, int):
            if self.verbose:
                logger.info(f'Start processing camera device {self.video_path}')
            self.source_identifier = f'CAM_device_{self.video_path}'
            self.total_frames = None
        else:
            self.source_identifier = 'unknown_source'
            self.total_frames = None

    @staticmethod
    def select_frames(camera_fps: int, required_fps: Optional[int]) -> List[int]:
        """Choose which 1-based frame positions (within each source-second) to process.

        Given the source frame rate and a desired processing rate, returns the
        indices (in the range ``1..camera_fps``) that should be kept so that
        roughly ``required_fps`` frames are processed per second of video. Used by
        [`run`][physiotrack.Video.run] to subsample frames.

        Args:
            camera_fps (int): Native frames-per-second of the source.
            required_fps (int, optional): Desired processing rate. If ``None`` or
                equal to ``camera_fps``, every frame in the second is selected.

        Returns:
            list[int]: The 1-based frame positions to process within each second.

        Example:
            ```python
            import physiotrack as pt
            pt.Video.select_frames(30, 5)  # -> 5 evenly spaced indices in 1..30
            ```
        """
        if required_fps == camera_fps or required_fps is None:
            return [int(i) for i in np.arange(1, camera_fps+1, dtype=float)]

        delta = camera_fps - 1
        step = delta / required_fps
        y = np.arange(0, required_fps, dtype=float) * step + 1
        return [int(i) for i in y]

    def preprocess_frame(self, frame):
        """Apply orientation, resize and 90-degree rotation to a raw decoded frame.

        Applies, in order: the explicit ``orient`` rotation, the ``resize`` target
        size, and the ``rotate`` 90-degree clockwise rotation -- each only if that
        option was enabled in the constructor. Called on every frame read from the
        source before inference.

        Args:
            frame (np.ndarray): A decoded BGR frame of shape ``(H, W, 3)``.

        Returns:
            np.ndarray: The preprocessed BGR frame. Its dimensions may differ from
                the input when ``orient`` is 90/270, ``resize`` is set, or
                ``rotate`` is ``True``.
        """
        if self._rotation:
            frame = apply_rotation(frame, self._rotation)
        if self.frame_resize:
            frame = cv2.resize(frame, self.frame_resize)
        if self.frame_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    
    def process_batch_detections(self, frames_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Run all configured detectors over a batch of frames.

        Each detector runs once over the whole batch (``detect_batch``) when it
        supports it, else frame by frame, and draws its boxes in its own palette
        colour -- labelled with the detector index when several are configured. With no
        detectors configured, returns copies of the frames with empty detection lists.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``.

        Returns:
            list[tuple[np.ndarray, list[np.ndarray]]]: One ``(combined_frame,
                all_detections)`` tuple per input frame. ``combined_frame`` is the
                frame with detection boxes drawn; ``all_detections`` is a list with
                one ``(N, 6)`` array per detector, each row
                ``(x1, y1, x2, y2, conf, cls)``.
        """
        per_detector = []
        for detector in self.detectors:
            if hasattr(detector, 'detect_batch'):
                outputs = [results[0] for results, _ in detector.detect_batch(frames_batch)]
            else:
                outputs = [detector.detect(frame)[0][0] for frame in frames_batch]
            per_detector.append([r.boxes.data.cpu().numpy() for r in outputs])

        color_names = list(COLORS.keys())
        batch_results = []
        for frame_index, frame in enumerate(frames_batch):
            combined_frame = frame.copy()
            all_detections = []
            for det_index, detections_per_frame in enumerate(per_detector):
                detections = detections_per_frame[frame_index]
                color = tuple(COLORS[color_names[det_index % len(color_names)]])
                for x1, y1, x2, y2, conf, cls in detections:
                    cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if len(per_detector) > 1:
                        draw_label(combined_frame, (int(x1), int(y1) - 20),
                                   f"D{det_index}-C{int(cls)}: {conf:.2f}",
                                   size=18, color=color, bold=True)
                all_detections.append(detections)
            batch_results.append((combined_frame, all_detections))
        return batch_results

    def process_batch_pose(self, frames_batch: List[np.ndarray], boxes_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, Any]]:
        """Run the pose estimator over a batch of frames and draw keypoints.

        Passes the batch (and per-frame detection boxes, if any) to the pose
        estimator's ``predict``, which returns one [`Result`][physiotrack.Result]
        per frame. If no pose estimator is configured, frames pass through with
        ``None`` results.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``,
                typically already annotated with detection boxes.
            boxes_batch (list[np.ndarray]): Per-frame person boxes to pose on, each
                an ``(N, 4)`` int array, or ``None`` for a frame with no boxes.

        Returns:
            list[tuple[np.ndarray, Any]]: One ``(result_frame, pose_results)`` tuple
                per frame. ``result_frame`` has keypoints drawn on it;
                ``pose_results`` is the ``detections`` list from the result's
                ``to_dict()`` (per-person keypoints and metadata).
        """
        batch_results = []
        
        if self.pose_estimator is None:
            for frame in frames_batch:
                batch_results.append((frame, None))
            return batch_results
        
        # Pose predict() accepts a list and returns one Result per frame.
        results = self.pose_estimator.predict(frames_batch, boxes_batch)
        for frame, result in zip(frames_batch, results):
            pose_results = result.to_dict()['instances']
            # Draw keypoints onto the (detection-annotated) frame.
            result_frame = result.plot(boxes=False, labels=False, keypoints=True)
            batch_results.append((result_frame, pose_results))

        return batch_results
    
    def process_batch_segmentation(self, frames_batch: List[np.ndarray], 
                                  all_detections_batch: List[List[np.ndarray]],
                                  overlay_frames: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Run all segmentators over a batch and alpha-blend their masks onto frames.

        Gathers each segmentator's segmentation map (using its batched API when
        available), optionally restricts masks to detector boxes when the
        segmentator has ``bbox_filter`` enabled, colorizes and merges the maps
        across segmentators, then blends the result over the base frame (70/30).
        With no segmentators configured, returns ``frames_batch`` unchanged.

        Args:
            frames_batch (list[np.ndarray]): Clean BGR frames used for segmentation
                inference, shape ``(H, W, 3)``.
            all_detections_batch (list[list[np.ndarray]]): Per-frame detection
                arrays (one ``(N, 6)`` array per detector) used for optional
                bbox-based mask filtering.
            overlay_frames (list[np.ndarray], optional): Frames the colorized masks
                are composited onto (e.g. frames already carrying pose/tracking
                drawings). Defaults to ``None`` (composite onto ``frames_batch``).

        Returns:
            list[np.ndarray]: One BGR frame per input with the merged, colorized
                segmentation overlay blended in.
        """
        batch_results = []

        if len(self.segmentators) == 0:
            return frames_batch

        # Prepare colors for merging
        color_names = list(COLORS.keys())
        color_list = [COLORS[name] for name in color_names]

        # First, gather segmentation maps per segmentator, using segment_batch if available
        base_frames_for_overlay = overlay_frames if overlay_frames is not None else frames_batch
        seg_maps_per_segmentator = []  # List[List[np.ndarray]] indexed [seg_idx][frame_idx]
        for seg_idx, segmentator in enumerate(self.segmentators):
            # Try wrapper's batch API, then inner segmentor batch API, else per-frame
            batched_outputs = None

            if hasattr(segmentator, 'segmentor') and hasattr(segmentator.segmentor, 'segment_batch'):
                batched_outputs = segmentator.segmentor.segment_batch(frames_batch)

            seg_maps_for_this_segmentator = []
            if batched_outputs is not None:
                # Expect list of (seg_img, seg_map)
                for (_, seg_map) in batched_outputs:
                    seg_maps_for_this_segmentator.append(seg_map)
            else:
                # Fallback: per-frame segmentation without filtering; we'll filter below
                for frame in frames_batch:
                    seg_map = segmentator.predict(frame).seg_map
                    seg_maps_for_this_segmentator.append(seg_map)

            # Apply optional bbox filtering per frame, matching single-frame logic
            for frame_idx, seg_map in enumerate(seg_maps_for_this_segmentator):
                filter_bboxes = None
                all_detections = all_detections_batch[frame_idx]
                if getattr(segmentator, 'bbox_filter', False) and len(all_detections) > 0:
                    if segmentator.detector_index is not None and segmentator.detector_index < len(all_detections):
                        detections = all_detections[segmentator.detector_index]
                        if segmentator.detector_class_filter is not None:
                            class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                            class_mask = np.isin(detections[:, 5], class_filter)
                            filter_bboxes = detections[class_mask][:, :4] if np.any(class_mask) else None
                        else:
                            filter_bboxes = detections[:, :4]
                    else:
                        all_dets = np.vstack(all_detections) if len(all_detections) > 0 else None
                        if all_dets is not None:
                            if segmentator.detector_class_filter is not None:
                                class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                                class_mask = np.isin(all_dets[:, 5], class_filter)
                                filter_bboxes = all_dets[class_mask][:, :4] if np.any(class_mask) else None
                            else:
                                filter_bboxes = all_dets[:, :4]

                if filter_bboxes is not None and len(filter_bboxes) > 0:
                    h, w = seg_map.shape[:2]
                    filtered_map = np.zeros((h, w), dtype=seg_map.dtype)
                    for bbox in filter_bboxes:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        filtered_map[y1:y2, x1:x2] = seg_map[y1:y2, x1:x2]
                    seg_maps_for_this_segmentator[frame_idx] = filtered_map

            seg_maps_per_segmentator.append(seg_maps_for_this_segmentator)

        # Now merge per-frame across segmentators and overlay
        for frame_idx, frame in enumerate(frames_batch):
            h, w = frame.shape[:2]
            combined_segmentation_map = np.zeros((h, w), dtype=np.uint8)
            combined_segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

            for seg_idx, seg_maps in enumerate(seg_maps_per_segmentator):
                seg_map = seg_maps[frame_idx]
                seg_mask = seg_map > 0
                if not np.any(seg_mask):
                    continue

                # Remap class IDs and colorize
                remapped_map = seg_map.copy()
                remapped_map[seg_mask] = (seg_idx * 100) + seg_map[seg_mask]

                unique_classes = np.unique(seg_map[seg_mask])
                seg_colored = np.zeros_like(combined_segmentation_img)
                for cls_id in unique_classes:
                    if cls_id > 0:
                        cls_mask = seg_map == cls_id
                        color_idx = (seg_idx * 10 + int(cls_id)) % len(color_list)
                        seg_colored[cls_mask] = color_list[color_idx]

                combined_segmentation_map[seg_mask] = remapped_map[seg_mask]
                combined_segmentation_img[seg_mask] = seg_colored[seg_mask]

            base_frame = base_frames_for_overlay[frame_idx]
            result_frame = cv2.addWeighted(base_frame, 0.7, combined_segmentation_img, 0.3, 0)
            batch_results.append(result_frame)

        return batch_results
    
    def process_batch_faces(self, frames_batch: List[np.ndarray],
                            subjects_batch: List[List[Instance]]) -> Optional[List[Result]]:
        """Find the faces of a batch, give them subject ids, and run the face stages.

        Faces come from the ``face`` detector (run on the clean frames) or, when
        ``detector`` is a face detector, are the frame's subjects themselves. With a
        tracker, a detected face takes the track id of the subject box that contains it
        (one-to-one, see [`assign_ids`][physiotrack.core.boxes.assign_ids]); without
        one its ``id`` is ``None``. The ``face_stages`` then run in order.

        Args:
            frames_batch (list[np.ndarray]): Clean BGR frames.
            subjects_batch (list[list[Instance]]): Each frame's tracked (or detected)
                subjects.

        Returns:
            list[Result] | None: One ``task="face"`` result per frame, or ``None`` when
                the pipeline has no face source.
        """
        if self._faces_from_detector:
            faces = [Result(orig_img=frame, instances=list(subjects), task="face")
                     for frame, subjects in zip(frames_batch, subjects_batch)]
        elif self.face_detector is not None:
            faces = self.face_detector.predict(list(frames_batch))
            if self.tracker is not None:
                for face_result, subjects in zip(faces, subjects_batch):
                    tracked = [s for s in subjects if s.box is not None and s.id is not None]
                    boxed = [f for f in face_result.instances if f.box is not None]
                    ids = assign_ids([f.box for f in boxed], [s.box for s in tracked],
                                     [s.id for s in tracked])
                    id_of = {id(f): i for f, i in zip(boxed, ids)}
                    face_result.instances = [f.replace(id=id_of.get(id(f)))
                                             for f in face_result.instances]
        else:
            return None
        for stage in self.face_stages:
            faces = stage.predict(list(frames_batch), faces)
        return faces

    def process_batch_depth(self, frames_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate a depth map for each frame in a batch.

        Delegates to the depth estimator's ``predict`` (which accepts a list and
        returns one ``DepthResult`` per frame). Returns all ``None`` when no depth
        estimator is configured.

        Args:
            frames_batch (list[np.ndarray]): BGR frames of shape ``(H, W, 3)``.

        Returns:
            list[np.ndarray | None]: One ``(H, W)`` depth map per frame, or ``None``
                entries when depth estimation is disabled.
        """
        if self.depth_estimator is None:
            return [None for _ in frames_batch]

        # predict() accepts a list and returns one DepthResult per frame.
        results = self.depth_estimator.predict(frames_batch)
        return [r.depth for r in results]

    def _rppg_roi_mask(self, clean_frame: np.ndarray, face_boxes=None):
        """Boolean skin ROI mask for rPPG from the configured ``rppg_roi`` segmenter.

        ``rppg_roi`` is a segmentation provider -- either the default SegFace
        [`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor], or a custom mask
        provider: a callable ``roi(frame) -> mask`` or an object exposing
        ``skin_mask(frame, boxes=None) -> mask`` (e.g. a face-neck segmentation model).
        Providers with ``skin_mask`` receive this frame's face boxes when the pipeline
        found faces, so the default extractor segments them instead of detecting the
        faces a second time. An empty or missing mask skips the frame.
        """
        roi = self.rppg_roi
        if hasattr(roi, "skin_mask"):
            mask = roi.skin_mask(clean_frame, boxes=face_boxes)
        else:
            mask = roi(clean_frame)
        return mask if mask is not None and np.asarray(mask).any() else None

    def _update_rppg(self, clean_frame: np.ndarray, face_boxes=None) -> None:
        """Feed one clean frame to the shared rPPG estimator and refresh derived panels.

        Skin RGB is sampled from ``clean_frame`` (before any overlay is drawn) over the
        ``rppg_roi`` **segmentation** mask -- the SegFace face-skin segmentation by
        default, or a custom provider's mask. The estimator is updated exactly once; the
        HRV and respiration panels only recompute their cached values (they do not
        re-update the estimator window).

        Args:
            clean_frame (np.ndarray): The frame before any overlay.
            face_boxes (np.ndarray, optional): This frame's face boxes when the
                pipeline found faces, reused as the skin-segmentation input. Defaults
                to ``None``.
        """
        if self.rppg_estimator is None:
            return
        self.rppg_estimator.update(clean_frame,
                                   roi_mask=self._rppg_roi_mask(clean_frame, face_boxes))
        for panel in self.rppg_panels:
            if hasattr(panel, "refresh"):
                panel.refresh()

    def run(self,
            output_video: Optional[Union[str, Path]] = None,
            output_json: Optional[Union[str, Path]] = None,
            progress_callback: Optional[callable] = None) -> VideoResults:
        """Process the whole video and return per-frame results.

        Reads the source frame-by-frame (subsampling to ``fps`` if set), runs every
        configured stage in batches -- detection, pose, tracking, segmentation, face
        orientation, depth -- composites the enabled overlays and side panels, and
        optionally writes an annotated video (H.264 when available, otherwise
        MPEG-4 with a warning) and a JSON dump.
        Blocks until the source is exhausted (or the user presses ``q`` when
        ``show=True``).

        Args:
            output_video (str | Path, optional): Path to write the annotated MP4.
                Defaults to ``None`` (no video written).
            output_json (str | Path, optional): Path to write the per-frame results
                as JSON. Defaults to ``None`` (no JSON written).
            progress_callback (callable, optional): Called after each processed
                frame as ``progress_callback(frame_id, total_frames, pose_results)``.
                Defaults to ``None``.

        Returns:
            VideoResults: One [`FrameResult`][physiotrack.FrameResult] per processed
                frame. Each exposes its subjects as
                [`Instance`][physiotrack.Instance] objects — so the object model
                survives video processing — plus ``meta`` (frame index, timestamp,
                source fps) and, when the relevant stage is enabled, ``vitals``,
                ``faces`` (the analysed faces) and ``track_box``. The subjects come from the
                richest attached stage: pose instances with named keypoints
                (``task="pose"``), else tracked instances with persistent ``id``s
                (``task="track"``), else bare detections (``task="detect"``). The
                sequence serializes with
                [`to_json`][physiotrack.VideoResults.to_json] and is accepted directly by
                the ``physiotrack.signals`` sequence functions.

        Raises:
            ValueError: If a custom ``detector`` is configured alongside a pose
                estimator that is not a ``Pose.Custom`` instance.

        Example:
            ```python
            import physiotrack as pt
            pose = pt.Pose.VRStudent(device=0)
            video = pt.Video(source="clip.mp4", pose=pose, output_dir="output")
            results = video.run("output/clip.mp4", "output/clip.json")

            for frame in results:
                print(frame.meta.timestamp, len(frame))
                for person in frame:
                    print(person.id, person.keypoints.by_name("nose"))
            ```

        Note:
            The writer tries H.264 (``avc1``) first and falls back to MPEG-4
            (``mp4v``) with a warning. Its frame rate is ``fps`` when set,
            otherwise the source fps.
        """
        
        pbar = None
        if self.total_frames and self.verbose:
            pbar = tqdm(total=self.total_frames, desc=f'Processing {self.source_identifier}')
        
        frames_per_cycle = int(round(self.video_fps))  # one source-second, in frames
        selected_frame_ids = self.select_frames(frames_per_cycle, self.required_fps)
        out_writer = None
        if output_video:
            if self.frame_resize:
                output_width, output_height = self.frame_resize
            else:
                output_width, output_height = self.width, self.height
            if self.frame_rotate:
                output_width, output_height = output_height, output_width

            effective_fps = self.required_fps if self.required_fps else self.video_fps

            # Shared with the examples so there is one implementation of "open a writer
            # that actually works": OpenCV does not raise when an encoder is missing, it
            # just silently discards every frame.
            out_writer = open_video_writer(output_video, effective_fps,
                                           (output_width, output_height))
        
        all_detection_data = VideoResults()
        frame_count = 0
        frame_filter_count = 1
        start_time = time.time()
        last_fps_print_time = start_time
        fps_print_interval = 2.0  # Print FPS every 2 seconds for real-time monitoring
        
        # Batch collection variables
        frame_batch = []
        frame_batch_metadata = []
        
        # Flag to track if we need to extract floor background from first frame
        extract_floor_from_first_frame = (
            self.radar_view is not None and 
            self.radar_view.background_mode in ["auto", "extract"]
        )

        while True:
            # Collect frames into batch
            batch_ready = False
            
            while len(frame_batch) < self.batch_size:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video, process remaining batch if any
                    if len(frame_batch) > 0:
                        batch_ready = True
                    break
                
                if pbar:
                    pbar.update(1)
                
                video_timestamp = round(frame_count / self.video_fps, 3)
                frame = self.preprocess_frame(frame)
                
                # Extract floor area from first frame if needed
                if extract_floor_from_first_frame and frame_count == 0:
                    if self.verbose:
                        logger.info("Extracting floor area from first frame and transforming to top-down view...")
                    self.radar_view.set_background_from_frame(frame)
                    extract_floor_from_first_frame = False  # Only do this once
                
                # Check if this frame should be processed
                if frame_filter_count in selected_frame_ids:
                    frame_batch.append(frame)
                    frame_batch_metadata.append({
                        'frame_id': frame_count,
                        'timestamp': video_timestamp,
                        'frame_filter_count': frame_filter_count
                    })
                
                frame_count += 1
                frame_filter_count = frame_filter_count + 1 if frame_filter_count < frames_per_cycle else 1
                
                # Check if batch is full
                if len(frame_batch) == self.batch_size:
                    batch_ready = True
                    break
            
            # Process batch if ready
            if batch_ready and len(frame_batch) > 0:
                # Step 1: Batch detection
                detection_results = self.process_batch_detections(frame_batch)
                
                # Extract frames and detections from results
                frames_with_detections = [r[0] for r in detection_results]
                all_detections_batch = [r[1] for r in detection_results]
                
                # Step 2: Tracking, frame by frame (the tracker has no batch path).
                #
                # This runs *before* pose so that the boxes pose receives are the
                # tracked ones and each pose instance can inherit its subject's
                # persistent track id. With pose first, its instance ids were merely
                # per-frame enumeration indices — meaningless across frames — and the
                # subject-lock filter below mutated the box list after pose had already
                # consumed it, so the filter did nothing at all.
                frames_with_tracking = []
                online_targets_batch = []
                boxes_batch = []
                track_ids_batch = []
                track_instances_batch = []

                for frame, all_detections in zip(frames_with_detections, all_detections_batch):
                    boxes, track_ids, online_targets = None, None, []
                    track_instances = []

                    if len(all_detections) > 0:
                        detections = np.vstack(all_detections)

                        if self.tracker is not None:
                            track_result = self.tracker.track(frame, detections)
                            frame = track_result.rendered
                            online_targets = track_result.raw
                            track_instances = track_result.instances

                            tracked = [(inst.box, inst.id) for inst in track_result.instances
                                       if inst.box is not None]
                            if (self.tracker.locked_subject_id is not None
                                    and self.tracker.locked_subject_box is not None):
                                # Subject lock: pose only the locked subject.
                                locked_box = np.asarray(
                                    self.tracker.locked_subject_box, dtype=float).reshape(-1)[:4]
                                tracked = [(locked_box, self.tracker.locked_subject_id)]

                            if tracked:
                                boxes = np.array([t[0] for t in tracked], dtype=int)
                                track_ids = [t[1] for t in tracked]
                        else:
                            # No tracker: pose every detection, but there are no
                            # persistent ids to attach.
                            boxes = detections[:, :-2].astype(int)

                    frames_with_tracking.append(frame)
                    online_targets_batch.append(online_targets)
                    boxes_batch.append(boxes)
                    track_ids_batch.append(track_ids)
                    track_instances_batch.append(track_instances)

                # The frame's subjects before pose: the tracker's instances (persistent
                # ids) or, without a tracker, the bare detections.
                if self.tracker is not None:
                    subjects_batch = track_instances_batch
                else:
                    subjects_batch = [
                        [Instance(box=np.array(row[:4], dtype=np.float32),
                                  confidence=float(row[4]), cls=int(row[5]))
                         for det_rows in frame_detections for row in det_rows]
                        for frame_detections in all_detections_batch
                    ]

                # Step 3: Batch pose estimation (if applicable)
                pose_results_batch = []
                if self.pose_estimator:
                    if self.pose_estimator.__class__.__name__ != "Custom" and len(self.detectors) > 0:
                        raise ValueError("Please use Pose.Custom class if you want to use a custom detector with the Video class.")

                    pose_batch_results = self.process_batch_pose(frames_with_tracking, boxes_batch)
                    frames_with_pose = [r[0] for r in pose_batch_results]
                    pose_results_batch = [r[1] for r in pose_batch_results]
                else:
                    frames_with_pose = frames_with_tracking
                    pose_results_batch = [[] for _ in frame_batch]

                # Step 4: Replace each pose instance's positional index with its
                # subject's persistent track id. The pose backends return detections in
                # the same order as the boxes they were given, so position maps to
                # track id.
                if self.tracker is not None:
                    self._attach_track_ids(pose_results_batch, track_ids_batch)

                frames_with_tracking = frames_with_pose

                # Step 5: Batch segmentation
                # Inference uses clean frames; overlay uses frames_with_tracking to keep pose drawings
                if len(self.segmentators) > 0:
                    frames_after_seg = self.process_batch_segmentation(frame_batch, all_detections_batch, overlay_frames=frames_with_tracking)
                else:
                    frames_after_seg = frames_with_tracking
                
                # Step 6: Faces -- found on the clean frames, tied to their subject's track
                # id, analysed by the face stages, and drawn onto the annotated frames.
                faces_batch = self.process_batch_faces(frame_batch, subjects_batch)
                if faces_batch is not None:
                    frames_after_seg = [
                        faces.plot(image=frame, boxes=not self._faces_from_detector)
                        for frame, faces in zip(frames_after_seg, faces_batch)
                    ]

                # Step 7: Batch depth estimation
                depth_maps_batch = []
                if self.depth_estimator is not None:
                    depth_maps_batch = self.process_batch_depth(frame_batch)
                else:
                    depth_maps_batch = [None for _ in frame_batch]

                # Step 8: Process overlays (radar view, depth view) and save results
                for idx, (result_frame, metadata, pose_results, online_targets, depth_map) in enumerate(
                        zip(frames_after_seg, frame_batch_metadata, pose_results_batch,
                            online_targets_batch, depth_maps_batch)):
                    faces = faces_batch[idx] if faces_batch is not None else None
                    
                    # Top-right stack (top -> bottom): motion plot, then the rPPG vitals
                    # panels, then motion-derived respiration. Update each panel's data
                    # first; attach_stack renders them once and handles the offsets.
                    if self.motion_plotter and self.pose_estimator is not None:
                        self.motion_plotter.update(pose_results, metadata['timestamp'])

                    # Skin is sampled from the CLEAN frame (frame_batch[idx]), never the
                    # overlaid result_frame, so the pulse signal is not corrupted.
                    if self.rppg_estimator is not None:
                        self._update_rppg(frame_batch[idx],
                                          faces.boxes if faces is not None else None)

                    # Motion-based respiration: reuse the pose keypoints already computed
                    # this frame (no extra inference), buffer them, and recompute the
                    # shoulder-motion respiration rate on the cadence.
                    if self._resp_motion_panel is not None:
                        self._resp_records.append(
                            {'timestamp': metadata['timestamp'],
                             'instances': pose_results or []})
                        panel = self._resp_motion_panel
                        panel._frames += 1
                        if panel._frames % self._resp_recompute_every == 0:
                            from physiotrack.signals import respiration_from_motion
                            panel.resp_wave, panel.rate = respiration_from_motion(
                                list(self._resp_records), float(self.video_fps))

                    top_right = []
                    if self.motion_plotter and self.pose_estimator is not None:
                        top_right.append(self.motion_plotter)
                    if self.rppg_estimator is not None:
                        top_right.extend(self.rppg_panels)
                    top_right.append(self._resp_motion_panel)   # None is skipped
                    result_frame = attach_stack(result_frame, top_right, 'top_right', 10)

                    # (Left-side kinematics stack -- joint-angle grid, ROM grid,
                    #  skeleton -- is composited together after the right-side views.)

                    # Bottom-right stack (bottom -> up): radar, depth, ego video.
                    bottom_right = []
                    if self.radar_view and self.tracker is not None and self.pose_estimator is not None:
                        self.radar_view.update(online_targets, pose_results)
                        bottom_right.append(self.radar_view)
                    if self.depth_view and depth_map is not None:
                        self.depth_view.update(depth_map)
                        bottom_right.append(self.depth_view)
                    if self.ego_view is not None:
                        self.ego_view.read_frame(metadata['frame_id'])
                        bottom_right.append(self.ego_view)
                    result_frame = attach_stack(result_frame, bottom_right,
                                                'bottom_right', 10)

                    # Left-side kinematics stack (top -> bottom): joint-angle grid,
                    # ROM grid (both transparent 2-column L|R panels), then the white
                    # full-room skeleton canvas with color-coded ROM arcs. All share
                    # the skeleton's width so they line up.
                    if self.pose_estimator is not None and (self.angle_plotter is not None
                                                            or self.rom_skeleton_view is not None):
                        first_kps = pose_results[0].get('keypoints') if pose_results else None
                        if self.angle_plotter is not None:
                            self.angle_plotter.update(pose_results, metadata['timestamp'])
                        if self.rom_skeleton_view is not None:
                            self.rom_skeleton_view.update(first_kps, self.rom_movements, result_frame.shape)
                            grid_w = self.rom_skeleton_view.canvas_size[0]
                        else:
                            grid_w = self.angle_plotter.canvas_width if self.angle_plotter else 320

                        # The angle plotter fuses its joint and ROM grids into one
                        # canvas, so the stack is just [grids, skeleton].
                        if self.angle_plotter is not None:
                            self.angle_plotter.canvas_width = grid_w
                        result_frame = attach_stack(
                            result_frame,
                            [self.angle_plotter, self.rom_skeleton_view],
                            'top_left', 10)

                    # Store frame data
                    track_box = None
                    if (self.tracker is not None
                            and getattr(self.tracker, 'locked_subject_box', None) is not None):
                        track_box = self.tracker.locked_subject_box.tolist()

                    # Instances come back from the pose backend as serialized dicts, so
                    # rebuild them into the object model: the frame result must expose
                    # Instance/Keypoints objects, not raw dicts. Detector/tracker-only
                    # pipelines (no pose) export their subjects too — the tracker's
                    # instances already carry persistent ids, and bare detections carry
                    # box/confidence/class — so a face-tracking or detection-only run
                    # produces per-frame results instead of empty frames.
                    architecture = getattr(self.pose_estimator, 'architecture', None)
                    if self.pose_estimator is not None:
                        frame_task = 'pose'
                        frame_instances = [
                            Instance.from_dict(det, architecture or "WHOLEBODY")
                            for det in (pose_results or [])
                        ]
                    else:
                        frame_task = 'track' if self.tracker is not None else 'detect'
                        frame_instances = list(subjects_batch[idx])

                    frame_meta = ResultMeta(
                        frame_index=metadata['frame_id'],
                        timestamp=metadata['timestamp'],
                        fps=float(self.video_fps),
                    )

                    # Add rPPG-derived vitals if enabled (nan -> None for clean JSON).
                    vitals = None
                    if self.rppg_estimator is not None or self._resp_motion_panel is not None:
                        def _clean(v):
                            return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
                        vitals = {}
                        if self.rppg_estimator is not None:
                            vitals['hr'] = _clean(self.rppg_estimator.hr)
                            vitals['snr'] = _clean(self.rppg_estimator.snr)
                            if self._hrv_panel is not None and self._hrv_panel.hrv:
                                vitals['hrv'] = {k: _clean(v) for k, v in self._hrv_panel.hrv.items()}
                        # Respiration comes from whichever source is configured.
                        if self._resp_panel is not None:
                            vitals['respiration'] = _clean(self._resp_panel.rate)
                        elif self._resp_motion_panel is not None:
                            vitals['respiration'] = _clean(self._resp_motion_panel.rate)

                    all_detection_data.append(FrameResult(
                        result=Result(
                            orig_img=result_frame,
                            instances=frame_instances,
                            task=frame_task,
                            architecture=architecture,
                            meta=frame_meta,
                        ),
                        meta=frame_meta,
                        vitals=vitals,
                        faces=faces,
                        track_box=track_box,
                    ))
                    
                    if out_writer:
                        out_writer.write(result_frame)

                    # Show output in real-time if enabled
                    if self.show_output:
                        display_frame = resize_frame_for_display(result_frame, self.screen_width, self.screen_height)
                        cv2.imshow('PhysioTrack - Full Inference', display_frame)
                        # Wait 1ms and check if user pressed 'q' to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("\nUser interrupted processing (pressed 'q')")
                            break

                    if progress_callback:
                        progress_callback(metadata['frame_id'], self.total_frames, pose_results)
                
                # Clear batch
                frame_batch = []
                frame_batch_metadata = []
                
                # Real-time FPS monitoring
                if self.show_fps:
                    current_time = time.time()
                    if current_time - last_fps_print_time >= fps_print_interval:
                        elapsed = current_time - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        # Build component-wise FPS dict
                        fps_dict = {"Pipeline": f"{current_fps:.2f}"}
                        
                        if len(self.detectors) > 0:
                            for idx, detector in enumerate(self.detectors):
                                if hasattr(detector, 'get_avg_fps'):
                                    det_fps = detector.get_avg_fps()
                                    fps_dict[f"Det[{idx}]"] = f"{det_fps:.2f}"
                        
                        if self.pose_estimator and hasattr(self.pose_estimator, 'pose_estimator'):
                            if hasattr(self.pose_estimator.pose_estimator, 'get_avg_fps'):
                                pose_fps = self.pose_estimator.pose_estimator.get_avg_fps()
                                fps_dict["Pose"] = f"{pose_fps:.2f}"
                        
                        if self.tracker and hasattr(self.tracker, 'get_avg_fps'):
                            tracker_fps = self.tracker.get_avg_fps()
                            fps_dict["Track"] = f"{tracker_fps:.2f}"
                        
                        if len(self.segmentators) > 0:
                            for idx, segmentor in enumerate(self.segmentators):
                                if hasattr(segmentor, 'get_avg_fps'):
                                    seg_fps = segmentor.get_avg_fps()
                                    fps_dict[f"Seg[{idx}]"] = f"{seg_fps:.2f}"
                        
                        if self.face_detector is not None:
                            fps_dict["Face"] = f"{self.face_detector.get_avg_fps():.2f}"
                        for stage in self.face_stages:
                            fps_dict[type(stage).__name__] = f"{stage.get_avg_fps():.2f}"

                        if self.depth_estimator and hasattr(self.depth_estimator, 'get_avg_fps'):
                            depth_fps = self.depth_estimator.get_avg_fps()
                            fps_dict["Depth"] = f"{depth_fps:.2f}"

                        fps_dict["Batch"] = f"{self.batch_size}"
                        
                        if pbar:
                            # Update progress bar with FPS info
                            pbar.set_postfix(fps_dict)
                        else:
                            # A self-overwriting terminal status line, deliberately not
                            # logged: it relies on a carriage return, which a log handler
                            # would turn into one new line per refresh.
                            fps_parts = [f"{k}: {v}" for k, v in fps_dict.items()]
                            fps_display = " | ".join(fps_parts)
                            sys.stdout.write("\r" + fps_display)
                            sys.stdout.flush()
                        
                        last_fps_print_time = current_time
            
            # Check if we're done
            if not ret and len(frame_batch) == 0:
                break
            
        if pbar:
            pbar.close()
            
        self.cap.release()
        if out_writer:
            out_writer.release()

        # Release ego video view if used
        if self.ego_view is not None:
            self.ego_view.release()

        # Close display window if show_output was enabled
        if self.show_output:
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        frames_processed = len(all_detection_data)
        # Throughput must be measured over the frames that actually went through the
        # models; `frame_count` includes the ones the fps= subsampler skipped, which
        # would report the decode rate instead.
        avg_fps = frames_processed / total_time if total_time > 0 else 0

        # Timings are collected unconditionally and kept on the instance, so a caller can
        # read them (e.g. to tabulate throughput) instead of scraping stdout.
        self.performance = self._collect_performance(frames_processed, total_time, avg_fps,
                                                    frames_read=frame_count)
        if self.show_fps:
            logger.info("\n%s", self._format_performance(self.performance))

        if output_json:
            self._save_json_data(all_detection_data, output_json)
        
        return all_detection_data
    
    def batch_run(self, 
                  input_paths: List[Union[str, Path]],
                  output_dir: Union[str, Path],
                  save_videos: bool = True,
                  save_json: bool = True):
        """Process several videos with this pipeline's configuration.

        Iterates the given paths, calling [`run`][physiotrack.Video.run] for each
        and writing ``<name>_processed.mp4`` / ``<name>_result.json`` into
        ``output_dir``.

        Args:
            input_paths (list[str | Path]): Video files to process.
            output_dir (str | Path): Directory for all outputs; created if missing.
            save_videos (bool): If ``True``, write an annotated MP4 per input.
                Defaults to ``True``.
            save_json (bool): If ``True``, write a JSON result file per input.
                Defaults to ``True``.

        Returns:
            dict[str, VideoResults]: Maps each input's file stem to its per-frame
                results (the return value of [`run`][physiotrack.Video.run]).

        Raises:
            ValueError: If one of ``input_paths`` cannot be opened.

        Note:
            The loaded models and all overlay settings are shared across the batch;
            only the video source changes. Each clip's own frame rate and dimensions
            are re-read, so inputs may differ in resolution and frame rate.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for input_path in input_paths:
            input_path = Path(input_path)
            video_name = input_path.stem

            if self.verbose:
                logger.info(f"Processing video: {input_path}")
            # Re-point the pipeline at this clip. `run` releases the capture when it
            # finishes, so without this every input after the first would read from an
            # exhausted, already-released capture and return no frames.
            self._open_source(str(input_path))

            video_output_path = output_dir / f"{video_name}_processed.mp4" if save_videos else None
            json_output_path = output_dir / f"{video_name}_result.json" if save_json else None

            detection_data = self.run(video_output_path, json_output_path)
            results[video_name] = detection_data

        return results
    
    def _save_json_data(self, results: "VideoResults", output_path: Union[str, Path]):
        """Write the per-frame results to a JSON file.

        Args:
            results (VideoResults): The frames to serialize.
            output_path (str | Path): Destination ``.json`` file.

        Note:
            Delegates to [`VideoResults.to_json`][physiotrack.VideoResults.to_json], so
            the file format is defined in one place and can be read back with
            ``VideoResults.from_dict_list``.
        """
        results.to_json(output_path)
        if self.verbose:
            logger.info("JSON data saved to: %s", output_path)
