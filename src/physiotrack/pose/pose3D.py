import json
import logging
import os
from datetime import datetime

import numpy as np

from .._logging import get_logger
from ..models import Models
from ..modules.DDHPose.inference import DDHPoseInference
from ..modules.MotionBERT.inference import MotionBERTInference
from ..results import Pose3DResult, ResultMeta
from ..signals.keypoints import as_frame_records, as_keypoint_dicts
from .canonicalizer import CanonicalView, canonicalize_pose
from .utils import add_3d_keypoints, coco17_to_h36m, coco17_to_halpe26

logger = get_logger(__name__)

# ``render_and_save`` (3D-video rendering) pulls in heavy/optional deps (smplx, ipdb).
# Import it lazily so Pose3D can be imported/used without them unless rendering is requested.

_COCO17_JOINTS = 17


def _video_properties(vid_path):
    """Read ``(fps, (width, height))`` from a video without decoding frames.

    Args:
        vid_path (str | os.PathLike): Path to the video.

    Returns:
        tuple[float, tuple[int, int]]: Frame rate and ``(width, height)``.
    """
    import imageio

    meta = imageio.get_reader(str(vid_path), 'ffmpeg').get_meta_data()
    return meta['fps'], meta['size']


def _as_coco17_sequence(source):
    """Normalise any accepted 2D-pose input into a ``(N, 17, 3)`` COCO array.

    Accepts a raw array, a [`VideoResults`][physiotrack.VideoResults], or a list of
    per-frame records. Lifting is single-subject, so the first subject of each frame is
    taken; frames with no subject contribute a zero-confidence pose, which keeps the
    sequence temporally aligned with the source video instead of silently shortening it.

    Args:
        source (np.ndarray | VideoResults | list): The 2D pose sequence.

    Returns:
        np.ndarray: ``(N, 17, 3)`` keypoints as ``(x, y, confidence)``.

    Raises:
        ValueError: If an array has the wrong shape, or no frames could be read.
    """
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=float)
        if arr.ndim != 3 or arr.shape[1] < _COCO17_JOINTS or arr.shape[2] not in (2, 3):
            raise ValueError(
                f"2D keypoints must be (N, 17, 2) or (N, 17, 3) in COCO order, got "
                f"{tuple(arr.shape)}. Whole-body keypoints are accepted: only the first "
                f"17 (the COCO body joints) are used."
            )
        arr = arr[:, :_COCO17_JOINTS, :]
        if arr.shape[2] == 2:
            arr = np.concatenate([arr, np.ones((*arr.shape[:2], 1))], axis=2)
        return arr

    frames = []
    for record in as_frame_records(source):
        instances = record.get('instances') or []
        pose = np.zeros((_COCO17_JOINTS, 3))
        if instances:
            for kp in as_keypoint_dicts(instances[0]):
                if kp['id'] < _COCO17_JOINTS:
                    pose[kp['id']] = (kp['x'], kp['y'], kp['confidence'])
        frames.append(pose)

    if not frames:
        raise ValueError(
            "No frames found in the 2D pose input, so there is nothing to lift."
        )
    return np.stack(frames)


class Pose3D:
    """Lift 2D pose keypoints to 3D joint positions from a video.

    ``Pose3D`` wraps two temporal 3D-lifting backends, selected automatically from
    the model's metadata:

    - **MotionBERT** — transformer lifter driven by a config YAML and checkpoint.
    - **DDHPose** (``DDH``) — diffusion-based lifter with its own sampling params.

    Like every other predictor, [`predict`][physiotrack.Pose3D] takes keypoints in
    memory and returns a result object — here a
    [`Pose3DResult`][physiotrack.Pose3DResult] holding ``(N, 17, 3)`` joints in
    Human3.6M order. It differs in one respect that is intrinsic to the task rather
    than a design choice: lifting is **sequence-level**, because a temporal model needs
    a window of 2D frames (``clip_len``) to produce each 3D frame. A single frame
    therefore cannot be lifted in isolation, and lifting is single-subject.

    For the offline workflow where the 2D pass already wrote a JSON file, use
    [`predict_json`][physiotrack.pose.pose3D.Pose3D.predict_json], which additionally
    renders a 3D video and saves an ``.npy`` array.

    Attributes:
        model (Models.Pose3D): The loaded 3D model enum.
        pose3d_framework (str): Backend name, ``"MotionBERT"`` or ``"DDH"``.
        pose3d_estimator: The underlying backend inference object.
        device (str | int): Compute device the model runs on.
        pixel (bool): Whether outputs are in pixel space (MotionBERT).
        render_video (bool): Whether ``predict_json`` renders a 3D ``.mp4`` when an
            output path is given.
        save_npy (bool): Whether ``predict_json`` saves the raw 3D array as ``.npy``
            when an output path is given.
        clip_len (int): Temporal window length used by the lifter.

    Example:
        ```python
        import physiotrack as pt

        # 2D pass, then lift -- the keypoints never touch the filesystem.
        results = pt.Video(source="clip.mp4", pose=pt.Pose.Person()).run()

        lifter = pt.Pose3D(model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite)
        poses = lifter.predict(results, fps=30, canonical_view=pt.CanonicalView.FRONT)

        poses.poses.shape                  # (N, 17, 3)
        poses.by_name("left_wrist")        # (N, 3) trajectory of one joint
        ```

    See Also:
        [`Pose`][physiotrack.Pose]: produces the 2D keypoints this consumes.
        [`Pose3DResult`][physiotrack.Pose3DResult]: the returned sequence.
    """

    def __init__(self, model=None, device='cpu',
                 config=None,
                 clip_len=243,
                 pixel=False,
                 render_video=True,
                 save_npy=True,
                 testloader_params=None,
                 # DDHPose specific parameters
                 boneindex_h36m='0,1,1,2,2,3,0,4,4,5,5,6,0,7,7,8,8,9,9,10,8,11,11,12,12,13,8,14,14,15,15,16',
                 number_of_frames=243,
                 test_time_augmentation=True,
                 timestep=1000,
                 scale=1.0,
                 cs=512,
                 dep=8,
                 joints_left=[4, 5, 6, 11, 12, 13],
                 joints_right=[1, 2, 3, 14, 15, 16],
                 num_proposals=300,
                 sampling_timesteps=5,
                 verbose=False,
                 **kwargs):
        """Load a 3D-lifting model and configure its backend.

        Args:
            model (Models.Pose3D, optional): 3D model enum to load, e.g.
                ``Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite`` or
                ``Models.Pose3D.DDH.best``. Defaults to ``None``, which loads
                ``Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite``.
            device (str | int, optional): Compute device, e.g. ``'cpu'``,
                ``'cuda'`` or a CUDA index. Defaults to ``'cpu'``.
            config (str, optional): Path to a MotionBERT config YAML. Defaults to
                ``None``, which resolves to the packaged config named after the
                model (ignored by the DDH backend).
            clip_len (int, optional): MotionBERT temporal window (frames).
                Defaults to ``243``.
            pixel (bool, optional): MotionBERT flag to keep outputs in pixel
                space. Defaults to ``False``.
            render_video (bool, optional): Render a 3D visualization ``.mp4`` when
                ``predict`` is called with an ``out_path``. Defaults to ``True``.
            save_npy (bool, optional): Save the raw 3D pose array as ``.npy`` when
                ``predict`` is called with an ``out_path``. Defaults to ``True``.
            testloader_params (dict, optional): Extra MotionBERT test-loader
                parameters. Defaults to ``None``.
            boneindex_h36m (str, optional): DDHPose bone-index string (H3.6M
                parent-child pairs). Defaults to the standard 16-bone H3.6M
                skeleton.
            number_of_frames (int, optional): DDHPose temporal window (frames).
                Defaults to ``243``.
            test_time_augmentation (bool, optional): DDHPose flip test-time
                augmentation. Defaults to ``True``.
            timestep (int, optional): DDHPose diffusion timesteps. Defaults to
                ``1000``.
            scale (float, optional): DDHPose diffusion scale. Defaults to ``1.0``.
            cs (int, optional): DDHPose channel size. Defaults to ``512``.
            dep (int, optional): DDHPose network depth. Defaults to ``8``.
            joints_left (list[int], optional): Left-side joint indices for DDHPose
                flip augmentation. Defaults to ``[4, 5, 6, 11, 12, 13]``.
            joints_right (list[int], optional): Right-side joint indices for
                DDHPose flip augmentation. Defaults to ``[1, 2, 3, 14, 15, 16]``.
            num_proposals (int, optional): DDHPose diffusion proposal count.
                Defaults to ``300``.
            sampling_timesteps (int, optional): DDHPose DDIM sampling steps.
                Defaults to ``5``.
            verbose (bool, optional): Emit load/progress messages at ``INFO`` rather
                than ``DEBUG``. Defaults to ``False``.
            **kwargs (Any): Reserved for forward compatibility; unused by the current
                backends.

        Raises:
            ValueError: If the model's backend is neither ``MotionBERT`` nor
                ``DDH``.

        Note:
            The first time a validated model is loaded its weights are
            auto-downloaded to the per-user weight cache (see
            [`Models.resolve`][physiotrack.Models.resolve]).
        """

        if model is None:
            model = Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite

        model_path = Models.resolve(model)

        Models.validate_pose3d_model(model)
    
        self.verbose = verbose
        self.minfo = Models.info(model)
        self.pose3d_framework = self.minfo['backend']
        logger.log(logging.INFO if verbose else logging.DEBUG,
                   'Initiating %s %s for 3D pose estimation', self.pose3d_framework, model.name)
        
        # Initialize 3D pose estimator based on framework
        if self.pose3d_framework == 'MotionBERT':
            if config is None:
                config = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MotionBERT', 'configs', f'{model.name}.yaml')
            self.pose3d_estimator = MotionBERTInference(
                config_path=config,
                checkpoint_path=model_path,
                clip_len=clip_len,
                testloader_params=testloader_params,
                device=device
            )
        elif self.pose3d_framework == 'DDH':
            self.pose3d_estimator = DDHPoseInference(
                boneindex_h36m=boneindex_h36m,
                number_of_frames=number_of_frames,
                test_time_augmentation=test_time_augmentation,
                timestep=timestep,
                scale=scale,
                cs=cs,
                dep=dep,
                joints_left=joints_left,
                joints_right=joints_right,
                num_proposals=num_proposals,
                sampling_timesteps=sampling_timesteps,
                checkpoint_path=model_path,
                device=device
            )
        else:
            raise ValueError(f"Invalid 3D model type: {self.pose3d_framework}")
        
        # Store parameters
        self.model = model
        self.device = device
        self.pixel = pixel
        self.render_video = render_video
        self.save_npy = save_npy
        self.clip_len = clip_len
    
    def predict(self, keypoints_2d, fps=None, frame_size=None,
                canonical_view=None, canonical_model=None,
                focus=None, scale_range=None, no_conf=None, flip=None, rootrel=None,
                gt_2d=None, batch_size=64) -> "Pose3DResult":
        """Lift a 2D keypoint sequence to 3D.

        Takes keypoints in memory and returns a result object, like every other
        predictor in the library. The lifter is *temporal*: it consumes a window of
        frames (``clip_len``) to produce each 3D frame, so the whole sequence is passed
        at once rather than frame by frame.

        Args:
            keypoints_2d (np.ndarray | VideoResults | list): The 2D sequence, as either
                a ``(N, 17, 2)``/``(N, 17, 3)`` COCO-17 array, a
                [`VideoResults`][physiotrack.VideoResults] from
                [`Video.run`][physiotrack.Video.run], or a list of per-frame records.
                For the non-array forms the first subject of each frame is used, since
                lifting is single-subject.
            fps (float, optional): Source frame rate, recorded on the result and used
                when rendering. Defaults to ``None``.
            frame_size (tuple[int, int], optional): ``(width, height)`` of the source
                video in pixels. Required for DDHPose, which normalises pixel
                coordinates, and for ``pixel=True`` with MotionBERT.
            canonical_view (CanonicalView, optional): Rotate the lifted poses to a fixed
                viewpoint -- ``View.FRONT``, ``View.BACK``, ``View.LEFT_SIDE`` or
                ``View.RIGHT_SIDE``. Defaults to ``None`` (no canonicalization).
            canonical_model (Models.Pose3D.Canonicalizer.Models, optional):
                Canonicalization model, used only when ``canonical_view`` is set.
            focus (int, optional): MotionBERT subject index. Defaults to ``None``.
            scale_range (list | tuple, optional): MotionBERT 2D scale range.
            no_conf (bool, optional): MotionBERT flag to drop keypoint confidence.
            flip (bool, optional): MotionBERT flip test-time augmentation.
            rootrel (bool, optional): MotionBERT root-relative output flag.
            gt_2d (bool, optional): MotionBERT flag to treat the 2D input as ground truth.
            batch_size (int, optional): DDHPose sampling batch size. Defaults to ``64``.

        Returns:
            Pose3DResult: The lifted sequence, ``(N, 17, 3)`` in Human3.6M joint order.
                Coordinates are root-relative and unitless unless the estimator was
                constructed with ``pixel=True``.

        Raises:
            ValueError: If the keypoints cannot be interpreted as a COCO-17 sequence, or
                if a required ``frame_size`` is missing.

        Example:
            ```python
            import physiotrack as pt

            results = pt.Video(source="clip.mp4", pose=pt.Pose.Person()).run()
            lifter = pt.Pose3D(model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite)
            poses = lifter.predict(results, fps=30, canonical_view=pt.CanonicalView.FRONT)
            poses.by_name("left_wrist")     # (N, 3) trajectory
            ```

        See Also:
            [`predict_json`][physiotrack.pose.pose3D.Pose3D.predict_json]: the offline
            file-based workflow, which also writes rendered video and ``.npy`` output.
        """
        keypoints = _as_coco17_sequence(keypoints_2d)

        if self.pose3d_framework == 'MotionBERT':
            poses = self.pose3d_estimator.infer(
                coco17_to_halpe26(keypoints),
                pixel=self.pixel,
                focus=focus,
                scale_range=scale_range,
                no_conf=no_conf,
                flip=flip,
                rootrel=rootrel,
                gt_2d=gt_2d,
                fps=fps,
                frame_size=frame_size,
            )
        else:  # DDH -- the constructor rejects anything else
            if frame_size is None:
                raise ValueError(
                    "DDHPose normalises pixel coordinates, so it needs the source frame "
                    "size: pass frame_size=(width, height). MotionBERT does not require it."
                )
            poses = self.pose3d_estimator.infer(
                coco17_to_h36m(keypoints),
                batch_size=batch_size,
                fps=fps,
                frame_size=frame_size,
            )

        if canonical_view:
            poses = canonicalize_pose(poses, model=canonical_model, view=canonical_view)

        return Pose3DResult(
            poses=poses,
            fps=fps if fps is not None else getattr(self.pose3d_estimator, 'fps_in', None),
            view=canonical_view,
            meta=ResultMeta(
                fps=fps,
                model=self.minfo['path'],
                device=str(self.device),
                units={"poses": "pixels" if self.pixel else "relative"},
            ),
        )

    def predict_json(self, json_path, vid_path, out_path=None, canonical_view=None,
                     canonical_model=None, **kwargs):
        """Lift a 2D-pose JSON file, optionally writing rendered and array output.

        The offline counterpart to [`predict`][physiotrack.Pose3D]: it reads a 2D-pose
        JSON produced by [`Video.run`][physiotrack.Video.run], lifts it, and augments the
        per-frame records with 3D keypoints so the JSON can be re-saved. Use this when
        the 2D pass and the lifting pass are separate steps; use ``predict`` when the
        keypoints are already in memory.

        Args:
            json_path (str | os.PathLike): 2D-pose JSON path.
            vid_path (str | os.PathLike): Source video, used for frame rate and size.
            out_path (str | os.PathLike, optional): Directory for the rendered ``.mp4``
                (when ``render_video``), the ``.npy`` array (when ``save_npy``) and a
                ``*_with_3d_keypoints.json``. Defaults to ``None`` (nothing written).
            canonical_view (CanonicalView, optional): Canonical viewpoint to rotate to.
            canonical_model (Models.Pose3D.Canonicalizer.Models, optional):
                Canonicalization model.
            **kwargs (Any): Forwarded to [`predict`][physiotrack.Pose3D].

        Returns:
            tuple[list, Pose3DResult]: The per-frame records augmented with 3D keypoints,
                and the lifted sequence. The records are returned because this method's
                purpose is to enrich an existing JSON; ``predict`` is the interface that
                returns a result object alone.

        Example:
            ```python
            import physiotrack as pt

            lifter = pt.Pose3D(model=pt.Models.Pose3D.DDH.best, device="cuda")
            frames, poses = lifter.predict_json(
                "output/clip_result.json", "clip.mp4", out_path="output/",
                canonical_view=pt.CanonicalView.FRONT,
            )
            ```
        """
        with open(json_path, 'r') as f:
            frames_data = json.load(f)

        fps, frame_size = _video_properties(vid_path)
        result = self.predict(frames_data, fps=fps, frame_size=frame_size,
                             canonical_view=canonical_view,
                             canonical_model=canonical_model, **kwargs)
        poses = result.poses

        if out_path:
            os.makedirs(out_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.render_video:
                from ..modules.MotionBERT.utils.vismo import render_and_save
                render_and_save(poses, f'{out_path}/X3D_{timestamp}.mp4',
                                fps=result.fps or 30)
            if self.save_npy:
                np.save(f'{out_path}/X3D_{timestamp}.npy', poses)

        frames_data = add_3d_keypoints(frames_data, poses)

        if out_path:
            base_name = os.path.splitext(os.path.basename(str(json_path)))[0]
            output_json_path = os.path.join(out_path, f"{base_name}_with_3d_keypoints.json")
            with open(output_json_path, 'w') as f:
                json.dump(frames_data, f, indent=2)

        return frames_data, result

    def predict_batch(self, json_paths, vid_paths, out_paths=None, **kwargs):
        """Run [`predict_json`][physiotrack.pose.pose3D.Pose3D.predict_json] over many videos.

        Args:
            json_paths (list[str]): 2D-pose JSON paths, one per video.
            vid_paths (list[str]): Source video paths, aligned with ``json_paths``.
            out_paths (list[str], optional): Per-video output directories. Defaults to
                ``None`` (nothing written for any video).
            **kwargs (Any): Forwarded to
                [`predict_json`][physiotrack.pose.pose3D.Pose3D.predict_json].

        Returns:
            list[tuple[list, Pose3DResult]]: One ``(frames_data, result)`` pair per video,
                in input order.
        """
        if out_paths is None:
            out_paths = [None] * len(json_paths)

        return [self.predict_json(json_path=j, vid_path=v, out_path=o, **kwargs)
                for j, v, o in zip(json_paths, vid_paths, out_paths)]
