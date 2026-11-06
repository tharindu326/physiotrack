"""Facial regions of detected faces from SegFace face parsing."""
from typing import Union

import numpy as np

from ..core.boxes import clip_box
from .base import FaceStage

__all__ = ["FaceRegions"]


class FaceRegions(FaceStage):
    """Facial regions of each face: which face parts its box contains, and how much.

    A face stage built on SegFace face parsing
    ([`Segmentation.Face`][physiotrack.Segmentation], 19 CelebAMask-HQ classes). Each
    face is parsed on its own box, so overlapping faces do not bleed into each other,
    and gets ``regions = {"pixel_counts", "fractions"}``: per visible class (``skin``,
    ``hair``, ``l_eye``, ``mouth``, ``eye_g``, ``hat``, ...), its pixel count and its
    share of the face box. ``regions["fractions"].get("skin", 0.0)`` is the share of
    visible facial skin -- the signal source of rPPG -- which drops when the face is
    occluded (glasses, a VR headset, a hand, hair).

    Attributes:
        segmenter (Segmentation.Face): The face parser in use.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.FaceRegions()(frame, pt.Face()(frame))
        skin = faces[0].regions["fractions"].get("skin", 0.0)
        ```

    See Also:
        [`Segmentation.Face`][physiotrack.Segmentation]: the frame-level face parsing
            with its colour overlay.
    """

    provides = "regions"

    def __init__(self, model=None, device: Union[str, int] = "cpu"):
        """Load the SegFace face parser.

        Args:
            model (Models.Segmentation.SegFace.Face, optional): Checkpoint. Defaults to
                ``None``, meaning ``swinb_celeba_512``.
            device (str | int, optional): ``"cpu"``, ``"cuda"``, ``"cuda:<i>"`` or a
                CUDA index. Defaults to ``"cpu"``.
        """
        super().__init__()
        from ..segment import Segmentation

        self.device = device
        self.segmenter = Segmentation.Face(model=model, device=device)

    def _infer_batch(self, frames, faces):
        values = []
        for frame, frame_faces in zip(frames, faces):
            frame_values = []
            for face in frame_faces:
                x1, y1, x2, y2 = clip_box(face.box, frame.shape)
                parsing = self.segmenter.predict(frame, boxes=[face.box])
                region = parsing.seg_map[y1:y2, x1:x2]
                classes, counts = np.unique(region[region > 0], return_counts=True)
                pixel_counts = {parsing.names[int(c)]: int(n) for c, n in zip(classes, counts)}
                frame_values.append({
                    "pixel_counts": pixel_counts,
                    "fractions": {name: n / region.size for name, n in pixel_counts.items()},
                })
            values.append(frame_values)
        return values
