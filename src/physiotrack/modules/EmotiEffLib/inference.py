"""Facial-expression recognition with EmotiEffLib ONNX models.

Savchenko, "Facial expression and attributes recognition based on multi-task learning of
lightweight neural networks", IEEE SISY 2021; code and weights from
https://github.com/sb-ai-lab/EmotiEffLib (Apache-2.0). The preprocessing below is the
library's own ONNX path (``EmotiEffLibRecognizerOnnx``): RGB, resize to the model's
input size, scale to [0, 1], ImageNet mean/std, NCHW. The class sets and the multi-task
layout (8 expression logits followed by valence and arousal) are EmotiEffLib's.
"""
from typing import List

import cv2
import numpy as np

#: Class order of the 8-class AffectNet models.
AFFECTNET8 = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral",
              "Sadness", "Surprise")
#: Class order of the 7-class AffectNet models (no Contempt).
AFFECTNET7 = ("Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ExpressionRecognizer:
    """Runs an EmotiEffLib ONNX classifier on face crops.

    The input size and the output layout are read from the model: 8 or 7 expression
    logits, or 10 for the multi-task models (8 logits, then valence and arousal).

    Args:
        weights_path (str): Path to the ``.onnx`` model.
        device (str): ``"cpu"`` or ``"cuda[:<i>]"``.

    Raises:
        RuntimeError: If CUDA is requested but ONNX Runtime has no CUDA provider.
        ValueError: If the model's output layout is not one of EmotiEffLib's.
    """

    def __init__(self, weights_path: str, device: str):
        import onnxruntime as ort

        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "FaceExpression was asked to run on CUDA, but this ONNX Runtime "
                    "build has no CUDA provider. Install onnxruntime-gpu or use "
                    "device='cpu'."
                )
            index = int(device.split(":")[1]) if ":" in device else 0
            providers = [("CUDAExecutionProvider", {"device_id": index})]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights_path, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_size = int(model_input.shape[-1])
        outputs = int(self.session.get_outputs()[0].shape[-1])
        layouts = {8: (AFFECTNET8, False), 7: (AFFECTNET7, False), 10: (AFFECTNET8, True)}
        if outputs not in layouts:
            raise ValueError(f"Unexpected expression-model output width {outputs}.")
        self.labels, self.multitask = layouts[outputs]

    def _preprocess(self, crop_rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop_rgb, (self.input_size, self.input_size)).astype(np.float32) / 255.0
        return ((x - _MEAN) / _STD).transpose(2, 0, 1)

    def predict(self, crops_rgb: List[np.ndarray]):
        """Class probabilities (and valence / arousal) for a batch of RGB face crops.

        Args:
            crops_rgb (list[np.ndarray]): ``(H, W, 3)`` uint8 RGB face crops.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: ``(N, C)`` softmax probabilities in
                :attr:`labels` order, and ``(N, 2)`` valence / arousal for the
                multi-task models (``None`` otherwise).
        """
        batch = np.stack([self._preprocess(c) for c in crops_rgb]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: batch})[0]
        logits = outputs[:, :len(self.labels)]
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs, (outputs[:, -2:] if self.multitask else None)
