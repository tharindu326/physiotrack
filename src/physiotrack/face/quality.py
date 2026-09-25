"""Image-quality indicators of detected faces."""
import cv2

from ..core.boxes import clip_box
from .base import FaceStage

__all__ = ["FaceQuality"]


class FaceQuality(FaceStage):
    """Image quality of each face crop: brightness, sharpness and relative size.

    A model-free face stage for screening faces before trusting downstream measures
    (landmarks, expression, rPPG): dark, blurred or tiny faces are the usual cause of
    unreliable results. Each face gets ``quality = {"brightness", "sharpness",
    "area_ratio"}``:

    - ``brightness`` -- mean grey level of the crop in ``[0, 1]``.
    - ``sharpness`` -- variance of the Laplacian of the grey crop (Pech-Pacheco et al.,
      ICPR 2000; ranked among the most reliable focus measures by Pertuz et al.,
      Pattern Recognition 2013). The crop is first resized to 112 x 112 px, so the value
      compares focus across faces of different sizes rather than their resolution.
    - ``area_ratio`` -- face box area over frame area.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.FaceQuality()(frame, pt.Face()(frame))
        sharp = [f for f in faces if f.quality["sharpness"] > 100]
        ```
    """

    provides = "quality"
    #: Side of the square the crop is resized to before measuring sharpness.
    SHARPNESS_SIZE = 112

    def __init__(self):
        """Create the stage (it has no model or options)."""
        super().__init__()

    def _infer_batch(self, frames, faces):
        values = []
        for frame, frame_faces in zip(frames, faces):
            frame_area = float(frame.shape[0] * frame.shape[1])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_values = []
            for face in frame_faces:
                box = clip_box(face.box, frame.shape)
                if box is None:
                    frame_values.append(None)
                    continue
                x1, y1, x2, y2 = box
                crop = gray[y1:y2, x1:x2]
                fixed = cv2.resize(crop, (self.SHARPNESS_SIZE, self.SHARPNESS_SIZE),
                                   interpolation=cv2.INTER_AREA)
                frame_values.append({
                    "brightness": float(crop.mean() / 255.0),
                    "sharpness": float(cv2.Laplacian(fixed, cv2.CV_64F).var()),
                    "area_ratio": float((x2 - x1) * (y2 - y1) / frame_area),
                })
            values.append(frame_values)
        return values
