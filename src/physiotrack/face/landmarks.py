"""478-point face mesh of detected faces with the MediaPipe Face Landmarker."""
import weakref

import cv2
import numpy as np

from ..core.boxes import crop_square
from ..models import Models
from ..results import Keypoints
from .base import FaceStage, require_extra

__all__ = ["FaceLandmarks"]


class FaceLandmarks(FaceStage):
    """Dense face mesh of each face: 468 surface points plus 10 iris points.

    A face stage built on the MediaPipe Face Landmarker (Kartynnik et al., "Real-time
    Facial Surface Geometry from Monocular Video on Mobile GPUs", CVPR-W 2019). Each face
    is cropped square around its box, the landmarker runs on the crop, and the points
    are mapped back to frame pixels. The result's faces carry ``keypoints`` with the
    ``"FACEMESH"`` layout, which the face signals
    ([`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio],
    [`mouth_aspect_ratio`][physiotrack.signals.mouth_aspect_ratio],
    [`iris_position`][physiotrack.signals.iris_position]) and
    [`GazeEstimator`][physiotrack.GazeEstimator] consume.

    The landmarker reports no per-point confidence, so the keypoints carry
    ``confidence=None``. A face in which it finds no face keeps ``keypoints=None``.

    The square crop can also contain a neighbouring face, which the landmarker may
    pick instead. A mesh is therefore kept only if its centre lies inside the face's
    own box and no other face's box both contains it and is centred closer to it;
    otherwise the face keeps ``keypoints=None`` rather than a neighbour's mesh.

    Each face is cropped to a square 1.25x its longer box side: the landmarker's
    internal face detector needs some context around the face, and this matches the
    20 %-padded face boxes of the 300-W validation (see the face validation guide).

    Attributes:
        model (Models.Face.Landmarks): The checkpoint in use.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.FaceLandmarks()(frame, pt.Face()(frame))
        mesh = faces[0].keypoints                 # Keypoints, architecture "FACEMESH"
        iris = mesh.by_name("left_iris_center")
        ear = pt.signals.eye_aspect_ratio(faces[0])
        ```

    Note:
        Requires the ``face`` extra: ``pip install 'physiotrack[face]'``.

    See Also:
        [`GazeEstimator`][physiotrack.GazeEstimator]: 3D gaze from this mesh.
    """

    provides = "landmarks"
    crop_scale = 1.25

    def __init__(self, model=None):
        """Load the MediaPipe Face Landmarker.

        Args:
            model (Models.Face.Landmarks, optional): Checkpoint. Defaults to ``None``,
                meaning ``Models.Face.Landmarks.face_landmarker``.

        Raises:
            ValueError: If ``model`` is not a ``Models.Face.Landmarks`` member.
            ImportError: If ``mediapipe`` is not installed.

        Note:
            MediaPipe runs on the CPU. The landmarker is released automatically when
            the stage is garbage-collected or the interpreter exits; call
            [`close`][physiotrack.FaceLandmarks.close] to release it earlier.
        """
        super().__init__()
        require_extra("mediapipe", "FaceLandmarks")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = Models.Face.Landmarks.face_landmarker if model is None else model
        Models.validate_face_model(model, "Landmarks")
        self.model = model
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=Models.resolve(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        # MediaPipe prints a spurious error if its landmarker is left to interpreter
        # teardown; weakref.finalize releases it on collection or at exit instead.
        self._finalizer = weakref.finalize(self, self._landmarker.close)

    def _update(self, instance, value):
        return instance.replace(keypoints=value)

    def _output_architecture(self, faces):
        return "FACEMESH"

    def _infer_batch(self, frames, faces):
        values = []
        for frame, frame_faces in zip(frames, faces):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = [face.box for face in frame_faces]
            values.append([self._mesh(rgb, box, boxes[:i] + boxes[i + 1:])
                           for i, box in enumerate(boxes)])
        return values

    def _mesh(self, rgb, box, other_boxes):
        crop, (left, top), side = crop_square(rgb, box, self.crop_scale)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(crop))
        found = self._landmarker.detect(image).face_landmarks
        if not found:
            return None
        points = np.array([[left + p.x * side, top + p.y * side] for p in found[0]])
        if not _mesh_belongs(np.median(points, axis=0), box, other_boxes):
            return None
        return Keypoints([{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(points)],
                         "FACEMESH")

    def close(self) -> None:
        """Release the MediaPipe landmarker now rather than at collection or exit.

        The stage cannot be used afterwards; calling ``close`` again does nothing.
        """
        self._finalizer()


def _mesh_belongs(center, box, other_boxes):
    """Whether a mesh centred at ``center`` is the face in ``box`` rather than a neighbour.

    Args:
        center (array-like): The mesh centre ``(x, y)`` in frame pixels.
        box (array-like): The face's box ``[x1, y1, x2, y2]``.
        other_boxes (Sequence[array-like]): The boxes of the other faces in the frame.

    Returns:
        bool: ``True`` if ``center`` lies inside ``box`` and no other box that also
            contains it has its centre closer to ``center``.
    """
    center = np.asarray(center, dtype=float)

    def contains(b):
        return b[0] <= center[0] <= b[2] and b[1] <= center[1] <= b[3]

    def distance(b):
        return np.hypot((b[0] + b[2]) / 2 - center[0], (b[1] + b[3]) / 2 - center[1])

    if not contains(box):
        return False
    own = distance(box)
    return not any(contains(b) and distance(b) < own for b in other_boxes)
