"""Facial expression of detected faces with an EmotiEffLib model."""
from typing import Union

import cv2

from ..core.boxes import clip_box
from ..core.device import torch_device
from ..models import Models
from .base import FaceStage, require_extra

__all__ = ["FaceExpression"]


class FaceExpression(FaceStage):
    """Facial expression of each face: an AffectNet category, optionally valence/arousal.

    A face stage built on EmotiEffLib's EfficientNets (Savchenko, IEEE SISY 2021),
    trained on AffectNet. It classifies the face box crop into ``Anger``, ``Contempt``,
    ``Disgust``, ``Fear``, ``Happiness``, ``Neutral``, ``Sadness`` and ``Surprise``
    (the ``*_7`` model has no ``Contempt``) and sets ``expression = {"label",
    "confidence", "scores"}``: the most probable category, its probability, and the
    probability of every category. The multi-task ``enet_b0_8_va_mtl`` model also adds
    ``"valence"`` and ``"arousal"``, EmotiEffLib's continuous affect estimates (roughly
    in ``[-1, 1]``).

    The categories describe the visible facial configuration. They are not a measure of
    what a person feels.

    Attributes:
        model (Models.Face.Expression): The checkpoint in use.
        labels (tuple[str, ...]): The categories, in score order.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.FaceExpression()(frame, pt.Face()(frame))
        print(faces[0].expression["label"], faces[0].expression["confidence"])
        ```

    Note:
        Requires the ``face`` extra: ``pip install 'physiotrack[face]'``. The weights
        are released for non-commercial research (AffectNet terms); see
        ``THIRD_PARTY_LICENSES.md``.
    """

    provides = "expression"

    def __init__(self, model=None, device: Union[str, int] = "cpu"):
        """Load the expression classifier.

        Args:
            model (Models.Face.Expression, optional): Checkpoint. Defaults to ``None``,
                meaning ``Models.Face.Expression.enet_b0_8_best_afew`` (EfficientNet-B0
                fine-tuned on AFEW). Others: ``enet_b0_8_best_vgaf`` (fine-tuned on
                VGAF), ``enet_b0_8_va_mtl`` (adds valence/arousal), and the larger
                EfficientNet-B2 ``enet_b2_8`` / ``enet_b2_7``.
            device (str | int, optional): ``"cpu"``, ``"cuda"``, ``"cuda:<i>"`` or a
                CUDA index. CUDA needs the ``onnxruntime-gpu`` build. Defaults to
                ``"cpu"``.

        Raises:
            ValueError: If ``model`` is not a ``Models.Face.Expression`` member.
            ImportError: If ``onnxruntime`` is not installed.
            RuntimeError: If CUDA is requested but unavailable to ONNX Runtime.
        """
        super().__init__()
        require_extra("onnxruntime", "FaceExpression")
        from ..modules.EmotiEffLib import ExpressionRecognizer

        model = Models.Face.Expression.enet_b0_8_best_afew if model is None else model
        Models.validate_face_model(model, "Expression")
        self.model = model
        self.device = device
        self._recognizer = ExpressionRecognizer(Models.resolve(model), torch_device(device))
        self.labels = self._recognizer.labels

    def _infer_batch(self, frames, faces):
        crops, owners = [], []
        for frame_index, (frame, frame_faces) in enumerate(zip(frames, faces)):
            for face_index, face in enumerate(frame_faces):
                box = clip_box(face.box, frame.shape)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                owners.append((frame_index, face_index))

        values = [[None] * len(frame_faces) for frame_faces in faces]
        if crops:
            probabilities, affect = self._recognizer.predict(crops)
            for row, ((frame_index, face_index), probs) in enumerate(zip(owners, probabilities)):
                best = int(probs.argmax())
                value = {
                    "label": self.labels[best],
                    "confidence": float(probs[best]),
                    "scores": {label: float(p) for label, p in zip(self.labels, probs)},
                }
                if affect is not None:
                    value["valence"], value["arousal"] = (float(v) for v in affect[row])
                values[frame_index][face_index] = value
        return values
