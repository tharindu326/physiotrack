"""Appearance-based 3D gaze estimation with the ptgaze models.

Three models, following the ptgaze reference implementation
(https://github.com/hysts/pytorch_mpiigaze_demo, MIT): ETH-XGaze (Zhang et al., ECCV
2020; full face, ResNet-18), MPIIFaceGaze (Zhang et al., CVPR-W 2017; full face) and
MPIIGaze (Zhang et al., CVPR 2015; one patch per eye plus head pose). Each fits a 3D
face template to the face-mesh landmarks (PnP), warps the face or eyes into a
normalised camera (Zhang, Sugano & Bulling, "Revisiting Data Normalization for
Appearance-Based Gaze Estimation", ETRA 2018), regresses normalised pitch/yaw, and
rotates the gaze back into the camera frame.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import face_model

_IMAGENET_RGB = (np.array([0.485, 0.456, 0.406], np.float32),
                 np.array([0.229, 0.224, 0.225], np.float32))


@dataclass(frozen=True)
class _Mode:
    """Normalisation and preprocessing of one gaze model (ptgaze ``data/configs``)."""
    camera: tuple        # normalised camera (fx, cx, cy)
    size: tuple          # normalised patch (width, height)
    distance: float      # normalised camera distance in metres
    centre: str          # face-centre definition: "nose" or "mouth"; "eyes" per eye


MODES = {
    "eth_xgaze": _Mode(camera=(960.0, 112.0, 112.0), size=(224, 224), distance=0.6,
                       centre="nose"),
    "mpiifacegaze": _Mode(camera=(1600.0, 112.0, 112.0), size=(224, 224), distance=1.0,
                          centre="mouth"),
    "mpiigaze": _Mode(camera=(960.0, 30.0, 18.0), size=(60, 36), distance=0.6,
                      centre="eyes"),
}


def default_camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics for an uncalibrated camera: focal length = image width.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        np.ndarray: ``(3, 3)`` camera matrix with the principal point at the centre.
    """
    return np.array([[width, 0.0, width // 2], [0.0, width, height // 2], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fit_head(landmarks_px: np.ndarray, camera_matrix: np.ndarray):
    """Fit the 3D face template to face-mesh points.

    Args:
        landmarks_px (np.ndarray): ``(468, 2)`` face-mesh points in (undistorted) pixels.
        camera_matrix (np.ndarray): ``(3, 3)`` intrinsics.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: The head rotation ``(3, 3)`` and the
            template in camera coordinates ``(468, 3)``, or ``None`` if the fit failed.
    """
    ok, rvec, tvec = cv2.solvePnP(
        face_model.LANDMARKS, landmarks_px.astype(np.float64), camera_matrix,
        np.zeros((1, 5)), np.zeros(3), np.array([0.0, 0.0, 1.0]),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return None
    head_rot = Rotation.from_rotvec(np.asarray(rvec, dtype=float).reshape(3)).as_matrix()
    model3d = face_model.LANDMARKS @ head_rot.T + np.asarray(tvec, dtype=float).reshape(3)
    return head_rot, model3d


def normalize(image_bgr: np.ndarray, head_rot: np.ndarray, centre: np.ndarray,
              camera_matrix: np.ndarray, mode: _Mode, grayscale: bool = False):
    """Warp the region around ``centre`` into the normalised camera of ``mode``.

    Args:
        image_bgr (np.ndarray): The (undistorted) frame.
        head_rot (np.ndarray): ``(3, 3)`` head rotation.
        centre (np.ndarray): 3D point to centre on, in camera coordinates (metres).
        camera_matrix (np.ndarray): ``(3, 3)`` intrinsics of the frame.
        mode (_Mode): The model's normalisation.
        grayscale (bool, optional): Return a histogram-equalised grey patch (MPIIGaze).
            Defaults to ``False``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The patch, the ``(3, 3)`` normalising
            rotation, and the normalised head pose ``(pitch, yaw)`` in radians.
    """
    z_axis = _unit(centre)
    y_axis = _unit(np.cross(z_axis, head_rot[:, 0]))
    x_axis = _unit(np.cross(y_axis, z_axis))
    normalizing_rot = np.vstack([x_axis, y_axis, z_axis])

    fx, cx, cy = mode.camera
    norm_camera = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    scale = np.diag([1.0, 1.0, mode.distance / np.linalg.norm(centre)])
    warp = norm_camera @ scale @ normalizing_rot @ np.linalg.inv(camera_matrix)
    patch = cv2.warpPerspective(image_bgr, warp, mode.size)
    if grayscale:
        patch = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))

    head_z = (normalizing_rot @ head_rot)[:, 2]
    head_pose = np.array([np.arcsin(head_z[1]), np.arctan2(head_z[0], head_z[2])])
    return patch, normalizing_rot, head_pose


def _angles_to_vector(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    return -np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                      np.cos(pitch) * np.cos(yaw)], axis=-1)


class GazeModel:
    """One ptgaze gaze model plus the normalisation around it.

    Args:
        weights_path (str): Path to the ``.safetensors`` checkpoint.
        mode (str): ``"eth_xgaze"``, ``"mpiifacegaze"`` or ``"mpiigaze"``.
        device (str): PyTorch device string.
    """

    def __init__(self, weights_path: str, mode: str, device: str):
        from safetensors.torch import load_file

        self.mode = MODES[mode]
        self.name = mode
        self.device = device
        if mode == "eth_xgaze":
            import timm
            self.model = timm.create_model("resnet18", num_classes=2)
        elif mode == "mpiifacegaze":
            from .networks import MPIIFaceGazeNet
            self.model = MPIIFaceGazeNet()
        else:
            from .networks import MPIIGazeNet
            self.model = MPIIGazeNet()
        self.model.load_state_dict(load_file(weights_path))
        self.model.to(device).eval()

    def _face_tensor(self, patch: np.ndarray) -> np.ndarray:
        x = patch.astype(np.float32) / 255.0
        if self.name == "eth_xgaze":            # trained on RGB
            mean, std = _IMAGENET_RGB
            x = x[:, :, ::-1]
        else:                                   # MPIIFaceGaze: BGR, BGR-ordered stats
            mean, std = _IMAGENET_RGB[0][::-1], _IMAGENET_RGB[1][::-1]
        return ((x - mean) / std).transpose(2, 0, 1)

    @torch.no_grad()
    def estimate(self, image_bgr: np.ndarray, landmarks: List[np.ndarray],
                 camera_matrix: np.ndarray) -> List[Optional[np.ndarray]]:
        """Estimate the gaze direction of several faces in one frame.

        Args:
            image_bgr (np.ndarray): The (undistorted) frame.
            landmarks (list[np.ndarray]): Per face, ``(468, 2)`` face-mesh points.
            camera_matrix (np.ndarray): ``(3, 3)`` intrinsics of the frame.

        Returns:
            list[np.ndarray | None]: Per face, the unit gaze vector in camera
                coordinates (x right, y down, z forward) -- for MPIIGaze the normalised
                mean of the two eyes -- or ``None`` when the head fit failed.
        """
        heads = [fit_head(points, camera_matrix) for points in landmarks]
        fitted = [i for i, head in enumerate(heads) if head is not None]
        out: List[Optional[np.ndarray]] = [None] * len(landmarks)
        if not fitted:
            return out

        if self.name == "mpiigaze":
            images, poses, rotations = [], [], []
            for i in fitted:
                head_rot, model3d = heads[i]
                for indices, mirror in ((face_model.REYE_INDICES, True),
                                        (face_model.LEYE_INDICES, False)):
                    patch, rot, pose = normalize(image_bgr, head_rot,
                                                 model3d[indices].mean(axis=0),
                                                 camera_matrix, self.mode, grayscale=True)
                    if mirror:  # the model uses the left-eye convention
                        patch, pose = patch[:, ::-1], pose * np.array([1.0, -1.0])
                    images.append(patch.astype(np.float32)[None] / 255.0)
                    poses.append(pose.astype(np.float32))
                    rotations.append(rot)
            angles = self.model(torch.from_numpy(np.ascontiguousarray(np.stack(images))).to(self.device),
                                torch.from_numpy(np.stack(poses)).to(self.device)).cpu().numpy()
            angles = angles.astype(np.float64)
            angles[0::2] *= np.array([1.0, -1.0])  # un-mirror the right eyes' yaw
            vectors = np.stack([_angles_to_vector(p, y) @ r
                                for (p, y), r in zip(angles, rotations)])
            per_face = vectors.reshape(len(fitted), 2, 3).mean(axis=1)
        else:
            patches, rotations = [], []
            for i in fitted:
                head_rot, model3d = heads[i]
                corners = np.concatenate([
                    face_model.REYE_INDICES, face_model.LEYE_INDICES,
                    face_model.NOSE_INDICES if self.mode.centre == "nose"
                    else face_model.MOUTH_INDICES])
                patch, rot, _ = normalize(image_bgr, head_rot, model3d[corners].mean(axis=0),
                                          camera_matrix, self.mode)
                patches.append(self._face_tensor(patch))
                rotations.append(rot)
            angles = self.model(torch.from_numpy(np.ascontiguousarray(np.stack(patches))).to(self.device))
            pitch, yaw = angles.cpu().numpy().astype(np.float64).T
            # Row vector times the normalising rotation = its inverse applied to a column.
            per_face = np.stack([v @ r for v, r in zip(_angles_to_vector(pitch, yaw), rotations)])

        for i, vector in zip(fitted, per_face):
            norm = np.linalg.norm(vector)
            if np.isfinite(norm) and norm > 0:
                out[i] = vector / norm
        return out
