import logging

from ..detect import Detection
from ..models import Models
from ..modules import SapiensPoseEstimation, VitInference, YoloPose
from ..results import Result, Instance, Keypoints
import os
import numpy as np

from .._logging import get_logger
from ..core.predictor import PredictorMixin

logger = get_logger(__name__)


class PoseBase(PredictorMixin):
    """Shared implementation behind every 2D pose predictor preset.

    ``PoseBase`` wires a pose-estimation backend (ViTPose, YOLO-Pose or Sapiens,
    selected automatically from the model's metadata) to the unified
    [`Result`][physiotrack.Result] API. It is not used directly; instead use one
    of the [`Pose`][physiotrack.Pose] presets ([`Pose.Person`][physiotrack.Pose],
    [`Pose.VRStudent`][physiotrack.Pose] or [`Pose.Custom`][physiotrack.Pose]),
    which subclass it. Rendering is delegated to
    [`Result.plot`][physiotrack.Result.plot], so no drawing flags are configured
    on the estimator.

    Top-down backends (ViTPose, Sapiens) require person boxes; when none are
    passed to [`predict`][physiotrack.Pose] a person detector is created
    on demand and run first. YOLO-Pose is single-stage and needs no boxes.

    Attributes:
        architecture (str): Keypoint layout of the loaded model, upper-cased,
            e.g. ``"WHOLEBODY"`` (133+ keypoints) or ``"COCO"`` (17 body
            keypoints). Propagated to every returned ``Result`` and used to name
            keypoints (see [`config`][physiotrack.pose.pose3D.Pose3D]).
        pose_framework (str): Backend name — one of ``"ViTPose"``, ``"YOLO"`` or
            ``"Sapiens"``.
        pose_estimator: The underlying backend instance performing inference.
        detector: Lazily-created person detector used to supply boxes for
            top-down backends, or ``None`` until first needed.
        device (str | int): Compute device the model runs on.
        classes (list[int] | None): Detector class-id filter, if any.
        verbose (bool): Whether the backend logs verbosely.

    Example:
        ```python
        import cv2
        import physiotrack as pt

        pose = pt.Pose.Person()          # whole-body ViTPose + person detector
        frame = cv2.imread("frame_1.png")
        result = pose.predict(frame)     # auto-detects people
        wrist = result[0].keypoints.by_name("left_wrist")
        cv2.imwrite("out.png", result.plot())
        ```

    See Also:
        [`Pose3D`][physiotrack.pose.pose3D.Pose3D]: lift these 2D keypoints to 3D.
        [`Keypoints`][physiotrack.Keypoints]: the per-instance keypoint container.
    """

    detector = None
    default_model = None
    detector_class = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None, device='cpu',
                 detector_model=None, detector_conf=0.25, detector_iou=0.45,
                 verbose=False, **kwargs):
        """Load a pose model and (for top-down backends) prepare its detector.

        Args:
            model (Models.Pose, optional): Pose model enum to load (e.g.
                ``Models.Pose.ViTPose.WholeBody.b_wholebody``). Defaults to
                ``None``, which falls back to the subclass's ``default_model``;
                if the subclass sets no default a ``ValueError`` is raised.
            conf (float, optional): Confidence threshold in ``[0.0, 1.0]`` for the
                pose backend (used by YOLO-Pose). Defaults to ``0.25``.
            iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]`` for the
                pose backend (used by YOLO-Pose). Defaults to ``0.45``.
            classes (list[int], optional): Restrict detections to these class
                ids. Defaults to ``None`` (all classes).
            device (str | int, optional): Compute device, e.g. ``'cpu'``,
                ``'cuda'`` or a CUDA device index like ``0``. Defaults to
                ``'cpu'``.
            detector_model (Models.Detection, optional): Person-detector model
                for top-down backends. Defaults to ``None`` (the preset's default
                detector is used).
            detector_conf (float, optional): Confidence threshold in
                ``[0.0, 1.0]`` for the person detector. Defaults to ``0.25``.
            detector_iou (float, optional): IoU threshold in ``[0.0, 1.0]`` for
                the person detector. Defaults to ``0.45``.
            verbose (bool, optional): Enable verbose backend logging. Defaults to
                ``False``.
            **kwargs (Any): Extra keyword arguments forwarded to the YOLO-Pose backend.

        Raises:
            ValueError: If no ``model`` is given and the subclass defines no
                ``default_model``, or if the model's backend is unrecognized.

        Note:
            The first time a validated model is loaded its weights are
            auto-downloaded to the per-user weight cache (see
            [`Models.resolve`][physiotrack.Models.resolve]).
        """
        if model is None:
            if self.default_model is None:
                raise ValueError(
                    f"{type(self).__name__} needs a model: pass model=<a Models.Pose "
                    f"member>, or use a preset such as Pose.Person() that supplies one. "
                    f"Browse the options with Models.list(task='Pose')."
                )
            model = self.default_model
        Models.validate_pose_model(model)

        model_path = Models.resolve(model)

        self.minfo = Models.info(model)
        self.architecture = self.minfo['group'].upper()
        self.pose_framework = self.minfo['backend']
        # `verbose` selects the level, so a quiet predictor stays quiet without
        # muting the rest of the library.
        logger.log(logging.INFO if verbose else logging.DEBUG,
                   'Initiating %s %s for pose estimation', self.pose_framework, model.name)

        # Pose backend. Rendering is delegated to Result.plot(), so overlay/draw flags
        # are off here.
        if self.pose_framework == 'ViTPose':
            self.pose_estimator = VitInference(model, device, False)
        elif self.pose_framework == 'YOLO':
            self.pose_estimator = YoloPose(model, device, conf, iou, classes,
                                           False, False, False, verbose, **kwargs)
        elif self.pose_framework == 'Sapiens':
            self.pose_estimator = SapiensPoseEstimation(model, device)
        else:
            raise ValueError(
                f"No pose backend for {model.name!r} (backend {self.pose_framework!r}). "
                f"Supported backends are: ViTPose, YOLO, Sapiens."
            )

        # Optional person detector used when boxes aren't supplied to predict().
        if self.detector_class is not None:
            self.detector = self.detector_class(
                model=detector_model, conf=detector_conf, iou=detector_iou,
                classes=classes, device=device, verbose=verbose,
            )

        self.detector_model = detector_model
        self.device = device
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.classes = classes
        self.verbose = verbose

    def _ensure_detector(self):
        if self.detector is None:
            if self.detector_class is not None:
                self.detector = self.detector_class()
            else:
                self.detector = Detection.Person(
                    device=self.device, conf=self.detector_conf,
                    iou=self.detector_iou, classes=self.classes, verbose=self.verbose,
                )
        return self.detector

    def _to_result(self, frame, frame_data) -> Result:
        instances = []
        for det in (frame_data or {}).get('detections', []):
            kps = (Keypoints(det['keypoints'], self.architecture)
                   if det.get('keypoints') else None)
            box = (np.array(det['bbox'], dtype=np.float32)
                   if det.get('bbox') is not None else None)
            instances.append(Instance(id=det.get('id'), box=box, keypoints=kps))
        return Result(orig_img=frame, instances=instances, task='pose',
                      architecture=self.architecture)

    def predict(self, source, boxes=None):
        """Estimate 2D pose on an image or a list of images.

        For top-down backends (ViTPose, Sapiens), if ``boxes`` is omitted the
        people in each frame are auto-detected with a person detector before pose
        estimation. YOLO-Pose ignores ``boxes`` and detects people itself.

        Args:
            source (np.ndarray | list[np.ndarray]): A single BGR frame of shape
                ``(H, W, 3)``, or a list of such frames for batch inference.
            boxes (list | np.ndarray, optional): Person boxes as
                ``[[x1, y1, x2, y2], ...]`` (or, for batch input, one such list
                per frame). Defaults to ``None`` (people are auto-detected when
                the backend needs boxes).

        Returns:
            Result | list[Result]: A [`Result`][physiotrack.Result] with
                ``task="pose"`` for a single frame, or a ``list[Result]`` when
                ``source`` is a list. Iterate the result to get per-person
                [`Instance`][physiotrack.Instance] objects and read their
                [`Keypoints`][physiotrack.Keypoints], e.g.
                ``result[0].keypoints.by_name("left_wrist")``.

        Example:
            ```python
            import cv2
            import physiotrack as pt

            pose = pt.Pose.Person()
            frame = cv2.imread("frame_1.png")
            result = pose.predict(frame)          # or pose.predict(frame, boxes)
            for person in result:
                nose = person.keypoints.by_name("nose")
                print(nose.x, nose.y, nose.confidence)
            cv2.imwrite("out.png", result.plot())
            ```
        """
        frames, was_batch = self._as_frames(source)
        if was_batch:
            return self._predict_batch(frames, boxes)
        return self._unwrap([self._predict_one(frames[0], boxes)], was_batch)

    def _predict_one(self, frame, boxes=None) -> Result:
        if boxes is None and self.pose_framework in ("ViTPose", "Sapiens"):
            boxes = self._ensure_detector().predict(frame).boxes.astype(int)
        _, frame_data = self.pose_estimator.inference(frame, boxes)
        return self._to_result(frame, frame_data)

    def _predict_batch(self, frames, boxes_list=None):
        if not isinstance(boxes_list, list) or len(boxes_list) != len(frames):
            boxes_list = [None] * len(frames)
        if hasattr(self.pose_estimator, 'inference_batch_frames'):
            # Auto-detect boxes where missing.
            resolved = []
            for frame, b in zip(frames, boxes_list):
                if b is None and self.pose_framework in ("ViTPose", "Sapiens"):
                    b = self._ensure_detector().predict(frame).boxes.astype(int)
                resolved.append(b)
            outputs = self.pose_estimator.inference_batch_frames(frames, resolved)
            return [self._to_result(frame, frame_data)
                    for frame, (_, frame_data) in zip(frames, outputs)]
        return [self._predict_one(f, b) for f, b in zip(frames, boxes_list)]


class Pose:
    """Namespace of ready-to-use 2D pose-estimation presets.

    ``Pose`` is not instantiated directly; it groups predictor classes that share
    the [`PoseBase`][physiotrack.Pose] machinery. Pick a preset by use
    case:

    - [`Pose.Person`](#) â€” general whole-body pose with a generic person detector.
    - [`Pose.VRStudent`](#) â€” whole-body pose paired with the VR-student detector.
    - [`Pose.Custom`](#) â€” any supported pose model you name explicitly.

    Each preset returns a [`Result`][physiotrack.Result] with ``task="pose"`` whose
    instances carry [`Keypoints`][physiotrack.Keypoints]. See the individual preset
    docstrings for their default models.

    Example:
        ```python
        import cv2
        import physiotrack as pt

        pose = pt.Pose.Person()
        result = pose.predict(cv2.imread("frame_1.png"))
        print(f"{len(result)} people, architecture={result.architecture}")
        ```

    See Also:
        [`Pose3D`][physiotrack.pose.pose3D.Pose3D]: lift 2D keypoints to 3D.
    """

    def __new__(cls, *args, **kwargs):
        """Refuse direct instantiation of the preset namespace.

        Raises:
            TypeError: Always. ``Pose`` groups the presets; it is not itself a
                predictor, and instantiating it used to return an object with no
                model attached, which failed later with a confusing error.
        """
        presets = [n for n in vars(cls) if not n.startswith("_")
                   and isinstance(vars(cls)[n], type)]
        raise TypeError(
            f"Pose is a namespace of presets, not a predictor. Use one of: "
            f"{', '.join(f'Pose.{p}()' for p in presets)} "
            f"— for example Pose.Person()."
        )

    class Custom(PoseBase):
        """Pose predictor for an explicitly chosen pose model.

        Wraps any model validated by ``Models.validate_pose_model`` (ViTPose,
        YOLO-Pose or Sapiens); the backend is inferred from the model's metadata.
        Unlike the other presets, ``model`` is required. See
        [`PoseBase.__init__`][physiotrack.Pose] for the shared keyword
        arguments.

        Args:
            model (Models.Pose): Pose model enum to load (required).
            conf (float, optional): Backend confidence threshold in
                ``[0.0, 1.0]``. Defaults to ``0.25``.
            iou (float, optional): Backend IoU threshold in ``[0.0, 1.0]``.
                Defaults to ``0.45``.
            classes (list[int], optional): Class-id filter. Defaults to ``None``.
            device (str | int, optional): Compute device. Defaults to ``'cpu'``.
            detector_model (Models.Detection, optional): Person detector for
                top-down backends. Defaults to ``None``.
            detector_conf (float, optional): Detector confidence threshold.
                Defaults to ``0.25``.
            detector_iou (float, optional): Detector IoU threshold. Defaults to
                ``0.45``.
            verbose (bool, optional): Verbose logging. Defaults to ``False``.
            **kwargs (Any): Forwarded to the YOLO-Pose backend.

        Example:
            ```python
            import physiotrack as pt

            pose = pt.Pose.Custom(
                pt.Models.Pose.ViTPose.WholeBody.b_wholebody, device="cuda",
            )
            result = pose.predict(frame)
            ```
        """

        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None, device='cpu',
                     detector_model=None, detector_conf=0.25, detector_iou=0.45,
                     verbose=False, **kwargs):
            Models.validate_pose_model(model)
            super().__init__(model=model, conf=conf, iou=iou, classes=classes,
                             device=device, detector_model=detector_model,
                             detector_conf=detector_conf, detector_iou=detector_iou,
                             verbose=verbose, **kwargs)

    class VRStudent(PoseBase):
        """Whole-body pose preset paired with the VR-student person detector.

        Uses the ``Detection.VRStudent`` detector to supply person boxes and
        defaults to the ViTPose whole-body model
        (``Models.Pose.ViTPose.WholeBody.b_wholebody``, 133+ keypoints). All
        constructor arguments are those of
        [`PoseBase.__init__`][physiotrack.Pose]; ``model`` may be omitted
        to use the default.

        Example:
            ```python
            import physiotrack as pt

            pose = pt.Pose.VRStudent(device=0, verbose=False)
            result = pose.predict(frame)
            ```
        """

        detector_class = Detection.VRStudent
        default_model = Models.Pose.ViTPose.WholeBody.b_wholebody

    class Person(PoseBase):
        """General whole-body pose preset with a generic person detector.

        Uses the ``Detection.Person`` detector to supply person boxes and defaults
        to the ViTPose whole-body model
        (``Models.Pose.ViTPose.WholeBody.b_wholebody``, 133+ keypoints). All
        constructor arguments are those of
        [`PoseBase.__init__`][physiotrack.Pose]; ``model`` may be omitted
        to use the default.

        Example:
            ```python
            import cv2
            import physiotrack as pt

            pose = pt.Pose.Person()
            result = pose.predict(cv2.imread("frame_1.png"))
            cv2.imwrite("out.png", result.plot())
            ```
        """

        detector_class = Detection.Person
        default_model = Models.Pose.ViTPose.WholeBody.b_wholebody
