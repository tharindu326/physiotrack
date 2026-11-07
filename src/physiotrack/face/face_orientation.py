"""Head orientation (yaw / pitch / roll) of detected faces with 6DRepNet360."""
from typing import Union

import cv2

from ..core.boxes import crop_square
from ..core.device import torch_device
from ..models import Models
from .base import FaceStage

__all__ = ["FaceOrientation"]


class FaceOrientation(FaceStage):
    """Head orientation (yaw / pitch / roll) of each face, using 6DRepNet360.

    A face stage: give it faces from [`Face`][physiotrack.Face] /
    [`VRFace`][physiotrack.VRFace] (or boxes, or an earlier stage's result) and it
    returns the same faces with ``orientation = {"yaw", "pitch", "roll"}`` in degrees.
    ``result.plot()`` draws the head axes.

    Each face is analysed on a square crop 1.2x its longer box side, centred on the box
    -- the crop size validated on AFLW (see the face validation guide). At the frame
    border the crop is padded rather than clipped, so it stays square and centred.

    Attributes:
        model (Models.Face.Orientation): The checkpoint in use.
        device (str | int): Compute device.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.VRFace(device=0)(frame)
        faces = pt.FaceOrientation(model=pt.Models.Face.Orientation.VR, device=0)(frame, faces)
        for face in faces:
            print(face.orientation)          # {"yaw": .., "pitch": .., "roll": ..}
        annotated = faces.plot()
        ```

    See Also:
        [`Face`][physiotrack.Face]: the face detector that supplies the faces.
        [`FaceLandmarks`][physiotrack.FaceLandmarks]: the face mesh of the same faces.
    """

    provides = "orientation"
    crop_scale = 1.2

    def __init__(self, model=None, device: Union[str, int] = "cpu"):
        """Load the head-orientation model.

        Args:
            model (Models.Face.Orientation, optional): Checkpoint. Defaults to ``None``,
                meaning ``Models.Face.Orientation.default`` (6DRepNet360 trained on
                300W-LP + Panoptic); ``Models.Face.Orientation.VR`` is tuned for faces
                wearing a VR headset.
            device (str | int, optional): ``"cpu"``, ``"cuda"``, ``"cuda:<i>"`` or a
                CUDA index. Defaults to ``"cpu"``.

        Raises:
            ValueError: If ``model`` is not a ``Models.Face.Orientation`` member.

        Note:
            The weights are downloaded into the model cache on first use.
        """
        super().__init__()
        from ..modules._6DRepNet360 import HeadPoseEstimator

        model = Models.Face.Orientation.default if model is None else model
        Models.validate_face_model(model, "Orientation")
        self.model = model
        self.device = device
        self._estimator = HeadPoseEstimator(Models.resolve(model), torch_device(device))

    def _infer_batch(self, frames, faces):
        crops, owners = [], []
        for frame_index, (frame, frame_faces) in enumerate(zip(frames, faces)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for face in frame_faces:
                crops.append(crop_square(rgb, face.box, self.crop_scale)[0])
                owners.append(frame_index)
        angles = self._estimator.estimate(crops)

        values = [[] for _ in frames]
        for frame_index, (yaw, pitch, roll) in zip(owners, angles):
            values[frame_index].append(
                {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)})
        return values
