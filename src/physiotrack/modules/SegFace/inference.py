"""SegFace face-parsing inference wrapper.

A minimal inference-only port of SegFace (Narayan et al., 2024, MIT licensed —
see ``LICENSE`` in this directory). Only the Swin-Base / CelebAMask-HQ (19-class)
variant is vendored. The model runs on a single aligned face crop and returns a
dense class-index map of face parts.

    seg = SegFaceInference(checkpoint_path, device="cuda")
    parsing = seg.infer(face_bgr)        # (H, W) int class map, same size as input crop
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .models import SegFaceCeleb
from ...core.device import torch_device

# CelebAMask-HQ 19-class label set (index order matches the trained head).
CELEBA_CLASSES = [
    "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow", "r_brow",
    "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip", "hair", "eye_g", "hat",
    "ear_r", "neck_l",
]

# RGB palette (one colour per class), mirroring the upstream visualization.
CELEBA_PALETTE = np.array([
    [0, 0, 0], [128, 64, 0], [200, 80, 80], [0, 192, 0], [64, 0, 0],
    [192, 0, 0], [0, 128, 128], [128, 128, 128], [0, 0, 128], [128, 0, 128],
    [0, 128, 0], [64, 128, 0], [64, 0, 128], [192, 128, 0], [192, 0, 128],
    [128, 128, 0], [64, 128, 128], [192, 128, 128], [0, 64, 0],
], dtype=np.uint8)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegFaceInference:
    """Loads a SegFace (Swin-Base / CelebA) checkpoint and parses face crops."""

    def __init__(self, checkpoint_path, input_resolution=512, device="cpu"):
        self.input_resolution = int(input_resolution)
        self.device = torch.device(torch_device(device))

        self.model = SegFaceCeleb(self.input_resolution, "swin_base")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = (
            ckpt.get("state_dict_backbone")
            or ckpt.get("state_dict")
            or ckpt
        ) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    # -- preprocessing -------------------------------------------------------- #
    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_resolution, self.input_resolution),
                         interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    # -- inference ------------------------------------------------------------ #
    @torch.no_grad()
    def infer(self, face_bgr: np.ndarray) -> np.ndarray:
        """Parse a single face crop.

        Args:
            face_bgr: ``(H, W, 3)`` BGR face crop.

        Returns:
            ``(H, W)`` ``int`` class-index map (CelebA 19-class), resized back to
            the input crop's resolution with nearest-neighbour.
        """
        h, w = face_bgr.shape[:2]
        x = self._preprocess(face_bgr)
        # labels/dataset are accepted by the upstream signature but unused at inference.
        logits = self.model(x, None, None)
        logits = F.interpolate(logits, size=(self.input_resolution, self.input_resolution),
                               mode="bilinear", align_corners=False)
        parsing = torch.argmax(logits.softmax(dim=1), dim=1)[0].cpu().numpy().astype(np.int32)
        if (h, w) != parsing.shape:
            parsing = cv2.resize(parsing.astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return parsing

    @staticmethod
    def colorize(parsing: np.ndarray) -> np.ndarray:
        """Map a class-index parsing map to a BGR colour image."""
        rgb = CELEBA_PALETTE[np.clip(parsing, 0, len(CELEBA_PALETTE) - 1)]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
