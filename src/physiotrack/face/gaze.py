"""3D gaze direction of detected faces with the ptgaze models."""
from typing import Optional, Union

import cv2
import numpy as np

from ..core.device import torch_device
from ..models import Models
from .base import FaceStage, require_extra

__all__ = ["GazeEstimator"]

# Registry member -> normalisation / network family (modules.Gaze.MODES).
_MODE_OF = {
    "eth_xgaze_resnet18": "eth_xgaze",
    "mpiifacegaze_resnet_simple": "mpiifacegaze",
    "mpiigaze_resnet_preact": "mpiigaze",
}


class GazeEstimator(FaceStage):
    """3D gaze direction of each face, from the face image and its face mesh.

    A face stage following the ptgaze reference implementation, with three models:

    - ``eth_xgaze_resnet18`` (default) — full face, ETH-XGaze (Zhang et al., ECCV 2020);
      robust to large head rotations.
    - ``mpiifacegaze_resnet_simple`` — full face, MPIIFaceGaze (Zhang et al., CVPR-W
      2017).
    - ``mpiigaze_resnet_preact`` — each eye separately, MPIIGaze (Zhang et al., CVPR
      2015); the face's gaze is the normalised mean of its two eyes.

    All three use the data normalisation of Zhang, Sugano & Bulling (ETRA 2018), which
    needs a head pose; it is fitted to the ``"FACEMESH"`` keypoints of
    [`FaceLandmarks`][physiotrack.FaceLandmarks], so run this stage after that one.

    Each face gets ``gaze = {"pitch", "yaw", "vector"}``. ``vector`` is the unit gaze
    direction in camera coordinates (x right, y down, z forward, so looking straight at
    the camera is ``[0, 0, -1]``). ``yaw = atan2(x, -z)`` is positive towards the image
    right and ``pitch = atan2(y, sqrt(x^2 + z^2))`` is positive downwards, both in
    degrees. These are camera-frame gaze angles; they are unrelated to the head
    ``orientation`` angles of [`FaceOrientation`][physiotrack.FaceOrientation]. A face
    without a mesh, or whose head fit fails, keeps ``gaze=None``.

    Attributes:
        model (Models.Face.Gaze): The checkpoint in use.
        camera_matrix (np.ndarray | None): Fixed intrinsics, or ``None`` for the
            per-frame default.
        dist_coeffs (np.ndarray | None): Lens distortion coefficients, or ``None``.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.Face()(frame)
        faces = pt.FaceLandmarks()(frame, faces)
        faces = pt.GazeEstimator()(frame, faces)
        print(faces[0].gaze["yaw"], faces[0].gaze["pitch"])
        ```

    Note:
        Requires the ``face`` extra: ``pip install 'physiotrack[face]'``. The weights
        are trained on ETH-XGaze (CC BY-NC-SA 4.0) and MPIIGaze / MPIIFaceGaze, whose
        terms are non-commercial; see ``THIRD_PARTY_LICENSES.md``.
    """

    provides = "gaze"
    requires = ("landmarks",)

    def __init__(self, model=None, device: Union[str, int] = "cpu",
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """Load the gaze model.

        Args:
            model (Models.Face.Gaze, optional): Checkpoint. Defaults to ``None``,
                meaning ``Models.Face.Gaze.eth_xgaze_resnet18``.
            device (str | int, optional): ``"cpu"``, ``"cuda"``, ``"cuda:<i>"`` or a
                CUDA index. Defaults to ``"cpu"``.
            camera_matrix (np.ndarray, optional): ``(3, 3)`` intrinsics of the camera
                that recorded the frames. Defaults to ``None``: an uncalibrated pinhole
                camera per frame, with focal length equal to the image width and the
                principal point at the image centre. Calibrated intrinsics make the
                fitted head distance, and so the gaze, more accurate.
            dist_coeffs (np.ndarray, optional): OpenCV lens-distortion coefficients of
                that camera (4, 5, 8, 12 or 14 values). When given, the frame and the
                face mesh are undistorted before the head fit, as ptgaze does with a
                calibration file; requires ``camera_matrix``. Defaults to ``None`` (no
                distortion).

        Raises:
            ValueError: If ``model`` is not a ``Models.Face.Gaze`` member,
                ``camera_matrix`` is not ``(3, 3)``, or ``dist_coeffs`` is given without
                ``camera_matrix``.
            ImportError: If ``safetensors`` is not installed.
        """
        super().__init__()
        require_extra("safetensors", "GazeEstimator")
        from ..modules.Gaze import GazeModel

        model = Models.Face.Gaze.eth_xgaze_resnet18 if model is None else model
        Models.validate_face_model(model, "Gaze")
        if camera_matrix is not None:
            camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix must be (3, 3), got {camera_matrix.shape}.")
        if dist_coeffs is not None:
            if camera_matrix is None:
                raise ValueError("dist_coeffs needs the camera_matrix they belong to.")
            dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self.model = model
        self.device = device
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self._model = GazeModel(Models.resolve(model), _MODE_OF[model.name],
                                torch_device(device))

    def _check_input(self, faces):
        if faces.architecture != "FACEMESH":
            raise ValueError(
                "GazeEstimator needs faces carrying FACEMESH keypoints; pass the result "
                "of FaceLandmarks (e.g. faces = pt.FaceLandmarks()(frame, faces))."
            )

    def _infer_batch(self, frames, faces):
        from ..modules.Gaze import default_camera_matrix

        values = []
        for frame, frame_faces in zip(frames, faces):
            meshed = [i for i, face in enumerate(frame_faces) if face.keypoints is not None]
            frame_values = [None] * len(frame_faces)
            if meshed:
                camera = (self.camera_matrix if self.camera_matrix is not None
                          else default_camera_matrix(frame.shape[1], frame.shape[0]))
                points = [frame_faces[i].keypoints.xy[:468].astype(np.float64) for i in meshed]
                if self.dist_coeffs is not None:
                    frame = cv2.undistort(frame, camera, self.dist_coeffs)
                    points = [cv2.undistortPoints(p.reshape(-1, 1, 2), camera,
                                                  self.dist_coeffs, P=camera).reshape(-1, 2)
                              for p in points]
                for i, vector in zip(meshed, self._model.estimate(frame, points, camera)):
                    if vector is None:
                        continue
                    x, y, z = vector
                    frame_values[i] = {
                        "pitch": float(np.degrees(np.arctan2(y, np.hypot(x, z)))),
                        "yaw": float(np.degrees(np.arctan2(x, -z))),
                        "vector": [float(x), float(y), float(z)],
                    }
            values.append(frame_values)
        return values
