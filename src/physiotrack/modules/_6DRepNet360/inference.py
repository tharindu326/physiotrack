"""6DRepNet360 head-orientation inference on face crops.

Hempel, Abdelrahman & Al-Hamadi, "Toward Robust and Unconstrained Full Range of
Rotation Head Pose Estimation", IEEE TIP 2024 (https://github.com/thohemp/6DRepNet360).
"""
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import utils
from .model import load_model


class HeadPoseEstimator:
    """Predicts yaw, pitch and roll from RGB face crops.

    Args:
        weights_path (str): Path to a 6DRepNet360 checkpoint.
        device (str): PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(self, weights_path: str, device: str):
        self.device = device
        self.model = load_model(weights_path, device=device)
        # The preprocessing 6DRepNet360 was trained and evaluated with.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def estimate(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        """Estimate head orientation for a batch of RGB face crops.

        Args:
            crops_rgb (list[np.ndarray]): ``(H, W, 3)`` uint8 RGB crops.

        Returns:
            np.ndarray: ``(N, 3)`` float array of ``(yaw, pitch, roll)`` in degrees, in
                the image-aligned convention :func:`utils.draw_axis` draws.
        """
        batch = torch.stack([self.transform(Image.fromarray(c)) for c in crops_rgb])
        rotations = self.model(batch.to(self.device))
        euler = utils.compute_euler_angles_from_rotation_matrices(rotations)
        euler = euler.cpu().numpy() * 180.0 / np.pi
        # Model axes -> image-aligned angles: pitch and yaw flip sign, roll does not.
        return np.stack([-euler[:, 1], -euler[:, 0], euler[:, 2]], axis=1)
