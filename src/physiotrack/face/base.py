"""The shared contract of the per-face analysis stages.

A face stage takes faces that were already found -- a face detector's
[`Result`][physiotrack.Result], a tracker's [`TrackResult`][physiotrack.TrackResult], or
plain boxes -- and adds one field to every face. Stages therefore chain::

    faces = pt.Face()(frame)
    faces = pt.FaceOrientation()(frame, faces)
    faces = pt.FaceLandmarks()(frame, faces)

and the same objects plug into [`Video`][physiotrack.Video] as ``face_stages=[...]``.
"""
import importlib
import time
from collections import deque
from typing import Any, List, Optional, Sequence

import numpy as np

from ..core.boxes import clip_box
from ..core.predictor import PredictorMixin
from ..results import Instance, Result, TrackResult

__all__ = ["FaceStage", "check_stage_order"]


def require_extra(module: str, stage: str):
    """Import an optional dependency of a face stage, or explain how to install it.

    Args:
        module (str): The module to import, e.g. ``"mediapipe"``.
        stage (str): The stage that needs it, for the error message.

    Returns:
        module: The imported module.

    Raises:
        ImportError: If the module is missing, naming the ``face`` extra.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{stage} needs the optional '{module.split('.')[0]}' package. Install the "
            f"face extra: pip install 'physiotrack[face]'"
        ) from exc


class FaceStage(PredictorMixin):
    """Base class of the per-face analysis stages.

    Subclasses set :attr:`provides` (the capability they add) and implement
    :meth:`_infer_batch`; the base class handles the accepted inputs, keeps ``N`` output
    faces aligned with the ``N`` input faces (a face the model could not analyse keeps
    its instance, with the stage's field left ``None``), preserves every field already
    set -- ``id``, ``box``, results of earlier stages -- and records timings.

    Attributes:
        provides (str): The capability this stage adds, e.g. ``"orientation"`` or
            ``"landmarks"``.
        requires (tuple[str, ...]): Capabilities an earlier stage must provide.
        device (str | int): Compute device the stage was built for.
    """

    provides: str = ""
    requires: tuple = ()

    def __init__(self):
        """Initialise the timing buffer shared by every stage."""
        self._inference_times = deque(maxlen=100)

    # -- to be implemented by each stage --------------------------------------- #
    def _infer_batch(self, frames: List[np.ndarray],
                     faces: List[List[Instance]]) -> List[List[Any]]:
        """Analyse every face of every frame.

        Args:
            frames (list[np.ndarray]): BGR frames.
            faces (list[list[Instance]]): The faces of each frame; each has a ``box``.

        Returns:
            list[list[Any]]: Per frame, one value per face (``None`` when that face
                could not be analysed), aligned with ``faces``.
        """
        raise NotImplementedError

    def _update(self, instance: Instance, value: Any) -> Instance:
        """Return ``instance`` with this stage's ``value`` applied."""
        return instance.replace(**{self.provides: value})

    def _output_architecture(self, faces: Result) -> Optional[str]:
        """Keypoint layout of the output result (stages that add keypoints override)."""
        return faces.architecture

    def _check_input(self, faces: Result) -> None:
        """Raise if ``faces`` lacks what this stage needs (overridden per stage)."""

    # -- the public contract ---------------------------------------------------- #
    def predict(self, source, faces=None):
        """Run the stage on the faces of one image or a batch of images.

        Args:
            source (str | os.PathLike | np.ndarray | Sequence): A single BGR image
                ``(H, W, 3)``, a path to an image file, or a sequence of either.
            faces (Result | TrackResult | np.ndarray | list, optional): The faces to
                analyse. For a single image: a face ``Result`` (e.g. from
                [`Face`][physiotrack.Face] or an earlier stage), a ``TrackResult``, or
                ``(N, 4)`` boxes ``[x1, y1, x2, y2]``. For a batch: a list with one such
                entry per image. Defaults to ``None``, which treats each whole image as
                one face.

        Returns:
            Result | list[Result]: A ``task="face"`` [`Result`][physiotrack.Result] per
                image whose instances are the input faces, in order, with this stage's
                field filled in.

        Raises:
            ValueError: If a batch's ``faces`` does not have one entry per image, or the
                faces are missing something this stage requires.
        """
        frames, was_batch = self._as_frames(source)
        if was_batch:
            if faces is None:
                faces = [None] * len(frames)
            if not isinstance(faces, (list, tuple)) or len(faces) != len(frames):
                raise ValueError(
                    f"For a batch of {len(frames)} images, pass `faces` as a list with "
                    f"one entry per image."
                )
            per_frame = [self._as_face_result(f, frame) for f, frame in zip(faces, frames)]
        else:
            per_frame = [self._as_face_result(faces, frames[0])]
        return self._unwrap(self._run(frames, per_frame), was_batch)

    def _run(self, frames: List[np.ndarray], faces: List[Result]) -> List[Result]:
        """Apply the stage to already-normalised face results."""
        for face_result in faces:
            self._check_input(face_result)
        valid = [[inst for inst in r.instances if _in_frame(inst.box, frame.shape)]
                 for frame, r in zip(frames, faces)]

        started = time.perf_counter()
        values = self._infer_batch(frames, valid) if any(valid) else [[] for _ in frames]
        elapsed = time.perf_counter() - started
        for _ in frames:
            self._inference_times.append(elapsed / len(frames))

        out = []
        for frame, face_result, frame_valid, frame_values in zip(frames, faces, valid, values):
            by_identity = {id(inst): value for inst, value in zip(frame_valid, frame_values)}
            instances = [self._update(inst, by_identity.get(id(inst)))
                         for inst in face_result.instances]
            out.append(Result(orig_img=frame, instances=instances, task="face",
                              architecture=self._output_architecture(face_result),
                              meta=face_result.meta))
        return out

    @staticmethod
    def _as_face_result(faces, frame: np.ndarray) -> Result:
        """Normalise one image's ``faces`` argument into a face ``Result``."""
        if faces is None:
            height, width = frame.shape[:2]
            instances = [Instance(box=np.array([0, 0, width, height], dtype=np.float32))]
            return Result(orig_img=frame, instances=instances, task="face")
        if isinstance(faces, Result):
            return Result(orig_img=frame, instances=list(faces.instances), task="face",
                          architecture=faces.architecture, meta=faces.meta)
        if isinstance(faces, TrackResult):
            return Result(orig_img=frame, instances=list(faces.instances), task="face")
        boxes = np.asarray(faces, dtype=np.float32)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError(f"Face boxes must have shape (N, 4), got {boxes.shape}.")
        return Result(orig_img=frame, task="face",
                      instances=[Instance(box=box[:4].copy()) for box in boxes])

    # -- timings (read by Video's performance summary) ------------------------ #
    def get_avg_inference_time(self) -> float:
        """Return the mean per-frame inference time.

        Returns:
            float: Milliseconds per frame over the last 100 frames, ``0.0`` before the
                first call.
        """
        if not self._inference_times:
            return 0.0
        return float(np.mean(self._inference_times)) * 1000.0

    def get_avg_fps(self) -> float:
        """Return the mean throughput.

        Returns:
            float: Frames per second over the last 100 frames, ``0.0`` before the first
                call.
        """
        ms = self.get_avg_inference_time()
        return 1000.0 / ms if ms > 0 else 0.0


def _in_frame(box, shape) -> bool:
    """Whether ``box`` exists, is finite, and covers some pixels of the frame."""
    return (box is not None and bool(np.isfinite(box[:4]).all())
            and clip_box(box, shape) is not None)


def check_stage_order(stages: Sequence[FaceStage]) -> None:
    """Check that every stage's requirements are provided by an earlier stage.

    Args:
        stages (Sequence[FaceStage]): The stages in the order they will run.

    Raises:
        TypeError: If an element is not a ``FaceStage``.
        ValueError: If a stage needs a capability no earlier stage provides, naming the
            stage to add in front of it.

    Example:
        ```python
        check_stage_order([pt.FaceLandmarks(), pt.GazeEstimator()])   # ok
        check_stage_order([pt.GazeEstimator()])                       # ValueError
        ```
    """
    provided = set()
    for stage in stages:
        if not isinstance(stage, FaceStage):
            raise TypeError(
                f"{type(stage).__name__} is not a face stage; face_stages accepts "
                f"FaceOrientation, FaceLandmarks, FaceExpression, GazeEstimator, "
                f"FaceQuality and FaceRegions instances."
            )
        missing = [need for need in stage.requires if need not in provided]
        if missing:
            providers = {cls.provides: cls.__name__ for cls in _stage_classes()}
            names = ", ".join(providers.get(need, need) for need in missing)
            raise ValueError(
                f"{type(stage).__name__} needs {', '.join(missing)} from an earlier "
                f"stage; put {names} before it."
            )
        provided.add(stage.provides)


def _stage_classes():
    """Every concrete face-stage class, for naming the stage that provides a need."""
    found, pending = [], list(FaceStage.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return [cls for cls in found if cls.provides]
