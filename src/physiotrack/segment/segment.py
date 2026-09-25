import logging

from ..models import Models
from ..modules import Segmentor, SapiensSegmentation
from ..results import Result, Instance
import numpy as np

from .._logging import get_logger
from ..core.boxes import clip_box
from ..core.predictor import PredictorMixin

logger = get_logger(__name__)


class SegmentationBase(PredictorMixin):
    """Shared implementation for the segmentation presets.

    Selects a backend (``YOLO``, ``Sapiens`` or ``SegFace``) from the model
    enum's registry metadata, wraps it, and exposes the unified
    [`predict`][physiotrack.Segmentation] interface returning a
    [`Result`][physiotrack.Result] with a ``.seg_map`` class-index array.

    Not used directly; instantiate a ``Segmentation.*`` preset.

    Attributes:
        model (Models.Segmentation.*): The resolved model enum in use.
        segmentation_framework (str): Backend name, one of ``"YOLO"``,
            ``"Sapiens"`` or ``"SegFace"``.
        device (int | str): Inference device.
        conf (float): Confidence threshold (YOLO backend).
        iou (float): NMS/IoU threshold (YOLO backend).
        classes (list[int] | None): Class filter (YOLO backend).
    """

    default_model = None

    def __init__(self, model=None, *, conf=0.25, iou=0.45, classes=None,
                 device='cpu', filter=None, verbose=False, **kwargs):
        """Configure a segmenter.

        Args:
            model (Models.Segmentation.*, optional): A validated segmentation
                model enum. Defaults to ``None`` (uses the preset's class-level
                ``default_model``).
            conf (float, optional): Confidence threshold in ``[0.0, 1.0]`` (YOLO
                backend). Defaults to ``0.25``.
            iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]`` (YOLO
                backend). Defaults to ``0.45``.
            classes (list[int], optional): Restrict segmentation to these class
                ids (YOLO backend). Defaults to ``None`` (all classes).
            device (int | str, optional): Inference device, e.g. ``'cpu'``,
                ``'cuda'``, ``'mps'`` or a device index. Defaults to ``'cpu'``.
            filter (dict, optional): Bounding-box filtering options with keys
                ``'bbox_filter'`` (bool), ``'detector_index'`` and
                ``'detector_class_filter'``. Defaults to ``None`` (no filtering).
            verbose (bool, optional): Print backend inference logs. Defaults to
                ``False``.
            **kwargs (Any): Additional keyword arguments forwarded to the underlying
                segmentation backend.

        Raises:
            ValueError: If no model can be resolved, or the model maps to an
                unsupported backend.

        Note:
            On first use the model weights are auto-downloaded from Hugging Face
            and cached.
        """
        if model is None:
            if self.default_model is None:
                raise ValueError(
                    f"{type(self).__name__} needs a model: pass model=<a "
                    f"Models.Segmentation member>, or use a preset such as "
                    f"Segmentation.Person() that supplies one. Browse the options with "
                    f"Models.list(task='Segmentation')."
                )
            model = self.default_model

        Models.validate_seg_model(model)

        model_path = Models.resolve(model)

        self.minfo = Models._get_model_info(model)
        self.segmentation_framework = self.minfo['backend']
        logger.log(logging.INFO if verbose else logging.DEBUG,
                   'Initiating %s %s for segmentation', self.segmentation_framework, model.name)

        if self.segmentation_framework == 'YOLO':
            self.segmentor = Segmentor(model, device, conf, iou, classes,
                                       False, verbose, **kwargs)
        elif self.segmentation_framework == 'Sapiens':
            self.segmentor = SapiensSegmentation(model, device)
        elif self.segmentation_framework == 'SegFace':
            from ..modules import SegFaceInference
            self.segmentor = SegFaceInference(model_path, input_resolution=512, device=device)
        else:
            raise ValueError(
                f"No segmentation backend for {model.name!r} "
                f"(backend {self.segmentation_framework!r}). Supported backends are: "
                f"YOLO, Sapiens, SegFace."
            )

        self.model = model
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.verbose = verbose

        # Optional bbox-based filtering of the segmentation map.
        filter = filter or {}
        self.bbox_filter = filter.get('bbox_filter', False)
        self.detector_index = filter.get('detector_index', None)
        self.detector_class_filter = filter.get('detector_class_filter', None)

    # -- inference ----------------------------------------------------------- #
    def _segment_map(self, frame) -> np.ndarray:
        if self.segmentation_framework == 'YOLO':
            _, segmentation_map = self.segmentor.segment(frame)
        else:  # Sapiens
            segmentation_map = self.segmentor.inference(frame)
        return segmentation_map

    @staticmethod
    def _filter_by_boxes(segmentation_map, bboxes) -> np.ndarray:
        h, w = segmentation_map.shape[:2]
        filtered = np.zeros((h, w), dtype=segmentation_map.dtype)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            filtered[y1:y2, x1:x2] = segmentation_map[y1:y2, x1:x2]
        return filtered

    def predict(self, source, boxes=None):
        """Run segmentation on one image or a batch of images.

        Args:
            source (np.ndarray | list[np.ndarray] | tuple[np.ndarray]): A single
                BGR image ``(H, W, 3)`` or a list/tuple of such frames for batch
                inference.
            boxes (list | np.ndarray, optional): Bounding boxes
                ``[[x1, y1, x2, y2], ...]``; only the segmentation inside these
                boxes is kept (the rest is zeroed). Defaults to ``None`` (keep the
                full-frame map).

        Returns:
            Result | list[Result]: A [`Result`][physiotrack.Result] with
                ``task="segment"`` whose ``.seg_map`` is the ``(H, W)`` class-index
                array, or a ``list[Result]`` when ``source`` is a list/tuple.
                ``result.plot()`` overlays the colorized segmentation.

        Example:
            ```python
            import physiotrack as pt

            seg = pt.Segmentation.Person()
            result = seg.predict(frame)          # or: seg(frame)
            seg_map = result.seg_map             # (H, W) class-index map
            overlay = result.plot()
            ```

        See Also:
            [`Result`][physiotrack.Result]: the returned segmentation container.
        """
        frames, was_batch = self._as_frames(source)
        source = frames if was_batch else frames[0]
        if was_batch:
            return [self._predict_one(frame, boxes) for frame in source]
        return self._predict_one(source, boxes)

    def _predict_one(self, frame, boxes=None) -> Result:
        segmentation_map = self._segment_map(frame)
        if boxes is not None and len(boxes) > 0:
            segmentation_map = self._filter_by_boxes(segmentation_map, boxes)
        return Result(orig_img=frame, instances=[], task="segment",
                      seg_map=segmentation_map)

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        if hasattr(self.segmentor, 'get_avg_inference_time'):
            return self.segmentor.get_avg_inference_time()
        return 0.0

    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        if hasattr(self.segmentor, 'get_avg_fps'):
            return self.segmentor.get_avg_fps()
        return 0.0


class Segmentation:
    """Segmentation predictors, grouped as ready-to-use presets.

    ``Segmentation`` is a namespace of nested predictor classes covering three
    backends (YOLO, Sapiens, SegFace). Instantiate a preset, then call
    [`predict`][physiotrack.Segmentation] (or the instance directly) to
    get a [`Result`][physiotrack.Result] with a ``.seg_map`` class-index array;
    ``result.plot()`` overlays the colorized mask.

    Presets:
        - [`Person`][physiotrack.Segmentation.Person]: whole-person instance
          segmentation (YOLO).
        - [`VRHead`][physiotrack.Segmentation.VRHead]: VR-head segmentation (YOLO).
        - [`BodyPart`][physiotrack.Segmentation.BodyPart]: body-part parsing
          (Sapiens/Goliath).
        - [`Face`][physiotrack.Segmentation.Face]: 19-class face-part parsing
          (SegFace/CelebAMask-HQ).
        - [`Custom`][physiotrack.Segmentation.Custom]: any validated segmentation
          model.

    Example:
        ```python
        import physiotrack as pt

        seg = pt.Segmentation.Person()
        seg_map = seg.predict(frame).seg_map     # (H, W) class map
        ```

    Note:
        Weights are auto-downloaded from Hugging Face on first use and cached.

    See Also:
        [`Detection`][physiotrack.Detection]: box-based prediction.
    """

    def __new__(cls, *args, **kwargs):
        """Refuse direct instantiation of the preset namespace.

        Raises:
            TypeError: Always. ``Segmentation`` groups the presets; it is not itself a
                predictor, and instantiating it used to return an object with no
                model attached, which failed later with a confusing error.
        """
        presets = [n for n in vars(cls) if not n.startswith("_")
                   and isinstance(vars(cls)[n], type)]
        raise TypeError(
            f"Segmentation is a namespace of presets, not a predictor. Use one of: "
            f"{', '.join(f'Segmentation.{p}()' for p in presets)} "
            f"— for example Segmentation.Person()."
        )

    class Custom(SegmentationBase):
        """Segmenter backed by any user-specified validated segmentation model.

        Example:
            ```python
            import physiotrack as pt
            from physiotrack import Models

            seg = pt.Segmentation.Custom(model=Models.Segmentation.YOLO.PERSON.l_person)
            result = seg.predict(frame)
            ```
        """

        def __init__(self, model, *, conf=0.25, iou=0.45, classes=None,
                     device='cpu', filter=None, verbose=False, **kwargs):
            """Configure a custom segmenter.

            Args:
                model (Models.Segmentation.*): A validated segmentation model
                    enum, e.g. ``Models.Segmentation.YOLO.PERSON.m_person``.
                conf (float, optional): Confidence threshold in ``[0.0, 1.0]``
                    (YOLO backend). Defaults to ``0.25``.
                iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]`` (YOLO
                    backend). Defaults to ``0.45``.
                classes (list[int], optional): Restrict to these class ids (YOLO
                    backend). Defaults to ``None`` (all classes).
                device (int | str, optional): Inference device. Defaults to
                    ``'cpu'``.
                filter (dict, optional): Bounding-box filtering options. See
                    [`SegmentationBase`][physiotrack.Segmentation].
                    Defaults to ``None``.
                verbose (bool, optional): Print backend inference logs. Defaults
                    to ``False``.
                **kwargs (Any): Forwarded to the underlying segmentation backend.
            """
            Models.validate_seg_model(model)
            super().__init__(model=model, conf=conf, iou=iou, classes=classes,
                             device=device, filter=filter, verbose=verbose, **kwargs)

    class VRHead(SegmentationBase):
        """VR-head segmenter.

        Wraps ``Models.Segmentation.YOLO.VRHEAD.M11`` (YOLO backend). See
        [`SegmentationBase`][physiotrack.Segmentation] for
        constructor arguments.
        """
        default_model = Models.Segmentation.YOLO.VRHEAD.M11

    class Person(SegmentationBase):
        """Whole-person instance segmenter.

        Wraps ``Models.Segmentation.YOLO.PERSON.m_person`` (YOLO backend). See
        [`SegmentationBase`][physiotrack.Segmentation] for
        constructor arguments.
        """
        default_model = Models.Segmentation.YOLO.PERSON.m_person

    class BodyPart(SegmentationBase):
        """Body-part parser (Sapiens / Goliath).

        Wraps ``Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG`` (Sapiens
        backend). See
        [`SegmentationBase`][physiotrack.Segmentation] for
        constructor arguments.
        """
        default_model = Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG

    class Face(SegmentationBase):
        """SegFace face-part parser (CelebAMask-HQ, 19 classes).

        Wraps ``Models.Segmentation.SegFace.Face.swinb_celeba_512`` (SegFace
        backend). Unlike the whole-frame segmenters, SegFace runs on face crops.
        If [`predict`][physiotrack.Segmentation] is called without
        ``boxes``, faces are auto-detected with a YOLO face detector (mirroring
        how ``Pose`` auto-detects people). The returned
        [`Result`][physiotrack.Result] carries a full-frame ``seg_map`` of
        face-part class indices and a 19-class ``palette``; ``result.plot()``
        overlays the parsing with that palette.

        Example:
            ```python
            import physiotrack as pt

            parse = pt.Segmentation.Face()
            result = parse.predict(frame)        # auto-detects faces if no boxes
            overlay = result.plot()
            ```

        See Also:
            [`Detection.Face`][physiotrack.Detection.Face]: the auto-detector used
                when ``boxes`` is omitted.
        """
        default_model = Models.Segmentation.SegFace.Face.swinb_celeba_512

        def __init__(self, model=None, *, device='cpu', face_detector=None,
                     face_conf=0.25, face_iou=0.45, verbose=False, **kwargs):
            """Configure the SegFace face-part parser.

            Args:
                model (Models.Segmentation.SegFace.*, optional): Override the
                    default SegFace model. Defaults to ``None`` (uses
                    ``swinb_celeba_512``).
                device (int | str, optional): Inference device, e.g. ``'cpu'``,
                    ``'cuda'`` or a device index. Defaults to ``'cpu'``.
                face_detector (optional): A pre-built face detector to reuse for
                    auto-detection when ``predict`` is called without ``boxes``.
                    Defaults to ``None`` (lazily builds a
                    [`Detection.Face`][physiotrack.Detection.Face]).
                face_conf (float, optional): Confidence threshold in ``[0.0, 1.0]``
                    for the auto-built face detector. Defaults to ``0.25``.
                face_iou (float, optional): NMS/IoU threshold in ``[0.0, 1.0]``
                    for the auto-built face detector. Defaults to ``0.45``.
                verbose (bool, optional): Print backend inference logs. Defaults
                    to ``False``.
                **kwargs (Any): Forwarded to
                    [`SegmentationBase`][physiotrack.Segmentation].
            """
            super().__init__(model=model, device=device, verbose=verbose, **kwargs)
            self._face_detector = face_detector
            self._face_conf = face_conf
            self._face_iou = face_iou

        def _ensure_detector(self):
            if self._face_detector is None:
                from ..detect import Detection
                self._face_detector = Detection.Face(
                    conf=self._face_conf, iou=self._face_iou, device=self.device)
            return self._face_detector

        def _predict_one(self, frame, boxes=None) -> Result:
            from ..modules.SegFace import CELEBA_CLASSES, CELEBA_PALETTE
            if boxes is None:
                boxes = self._ensure_detector().predict(frame).boxes

            h, w = frame.shape[:2]
            seg_map = np.zeros((h, w), dtype=np.int32)
            instances = []
            for box in (boxes if boxes is not None else []):
                clipped = clip_box(box, frame.shape)
                if clipped is None:
                    continue
                x1, y1, x2, y2 = clipped
                parsing = self.segmentor.infer(frame[y1:y2, x1:x2])
                fg = parsing > 0
                seg_map[y1:y2, x1:x2][fg] = parsing[fg]
                instances.append(Instance(box=np.array([x1, y1, x2, y2], dtype=float)))

            names = {i: n for i, n in enumerate(CELEBA_CLASSES)}
            return Result(orig_img=frame, instances=instances, task="segment",
                          seg_map=seg_map, names=names, palette=CELEBA_PALETTE)
