"""
ZipDepth Inference Module
Provides monocular depth estimation from RGB images using ZipDepth models.

This wrapper mirrors the :class:`DepthAnythingV2Inference` interface so the
high-level :class:`physiotrack.Depth` predictor can drive either backend
through the same ``inference`` / ``inference_batch`` contract.
"""

import os
import time
from collections import deque
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .models import create_model

from ..._logging import get_logger
from ...core.device import torch_device
from ..._paths import weights_dir

logger = get_logger(__name__)


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Remove DDP (``module.``) and ``torch.compile`` (``_orig_mod.``) prefixes.

    Checkpoints saved from a wrapped model carry these key prefixes; both (in any
    order or nesting) are stripped so weights load into a plain model.
    """
    prefixes = ('module.', '_orig_mod.')

    def _clean(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p):]
                    changed = True
        return key

    return {_clean(k): v for k, v in state_dict.items()}


def _make_divisible(value: float, divisor: int) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (at least ``divisor``)."""
    return max(divisor, int(round(value / divisor) * divisor))


class ZipDepthInference:
    """Inference wrapper for ZipDepth monocular depth models.

    Attributes:
        model: The fused ZipDepth model instance.
        device: Resolved device string the model runs on (e.g. ``'cuda'``,
            ``'cpu'``, ``'mps'``).
        variant: Architecture size variant (currently always ``'base'``).
    """

    def __init__(self, model_path: str, model_config: dict, device: Union[str, int] = 'cuda',
                 input_size: int = 384, verbose: bool = True, ensure_multiple_of: int = 32):
        """Initialize ZipDepth inference.

        Args:
            model_path: Path to the model weights file (``.pth``).
            model_config: Config dict with keys ``'variant'``, ``'global_mode'``
                and ``'upsample_unfold'`` (the last selects the base vs NPU head).
            device: Device to run inference on (``0``, ``'cuda'``, ``'cpu'``, ``'mps'``).
            input_size: Target length of the image's shorter side at inference
                (aspect ratio is preserved). Defaults to ``384``.
            verbose: Whether to print initialization info.
            ensure_multiple_of: Round model input dimensions to this multiple.
                Defaults to ``32``.
        """
        if not os.path.isfile(model_path):
            model_name = os.path.basename(model_path)
            model_dir = str(weights_dir())
            raise ValueError(
                f"The model file '{model_name}' does not exist at {model_path}\n"
                f"It is normally auto-downloaded from the physiotrack HuggingFace "
                f"repo on first use. To place it manually, download it from:\n"
                f"  https://huggingface.co/tharindu326/physiotrack/resolve/main/{model_name}\n"
                f"And place it in: {os.path.abspath(model_dir)}/"
            )

        # Honour the requested device exactly (no silent CPU fallback).
        self.device = torch_device(device)
        self._is_cuda = self.device.startswith('cuda')

        self.variant = model_config['variant']
        self.global_mode = model_config['global_mode']
        self.upsample_unfold = model_config['upsample_unfold']
        self.input_size = input_size
        self.ensure_multiple_of = ensure_multiple_of
        self.verbose = verbose

        # Build architecture and load weights
        self.model = create_model(
            variant=self.variant,
            global_mode=self.global_mode,
            upsample_unfold=self.upsample_unfold,
        )

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = _strip_state_dict_prefixes(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose and unexpected:
            logger.debug("ZipDepth: ignored unexpected checkpoint keys: %s", unexpected)
        if self.verbose and missing:
            print(f"  ZipDepth: warning — missing keys (random init): {missing}")

        self.model = self.model.to(self.device).eval()
        # Re-parameterize (fold Conv+BN and RepVGG branches) for fast inference.
        self.model.fuse_for_inference()
        if self._is_cuda:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.verbose:
            head = 'NPU/mobile head' if not self.upsample_unfold else 'GPU head'
            logger.info("ZipDepth (%s, %s) loaded on %s", self.variant, head, self.device)

        # Inference time tracking
        self._inference_times = deque(maxlen=100)

    def _compute_target_size(self, h: int, w: int) -> Tuple[int, int]:
        """Model input dimensions for an ``(h, w)`` image, preserving aspect ratio."""
        scale = self.input_size / min(h, w)
        new_h = _make_divisible(h * scale, self.ensure_multiple_of)
        new_w = _make_divisible(w * scale, self.ensure_multiple_of)
        return new_h, new_w

    def _image2tensor(self, raw_image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """Convert a BGR uint8 image to a normalized model input tensor.

        Returns ``(tensor[1,3,H,W] in [0,1], orig_h, orig_w)``. ImageNet mean/std
        normalization is applied inside the model, not here.
        """
        h, w = raw_image.shape[:2]
        new_h, new_w = self._compute_target_size(h, w)

        resized = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(self.device)
        # BGR -> RGB, HWC -> CHW, add batch dim, scale to [0, 1]
        tensor = tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        if self._is_cuda:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor, h, w

    @torch.no_grad()
    def _infer_image(self, raw_image: np.ndarray) -> np.ndarray:
        """Run the model on one BGR image and return an ``(H, W)`` float32 depth map."""
        tensor, h, w = self._image2tensor(raw_image)
        depth = self.model(tensor)

        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=True)
        return depth[0, 0].cpu().float().numpy()

    def inference(self, image: np.ndarray, normalize: bool = False,
                  colormap: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Run depth estimation on a single image.

        Args:
            image: Input BGR image (HxWx3 numpy array).
            normalize: If True, normalize depth to the 0-255 uint8 range.
            colormap: If provided, also return a colorized depth map using this
                OpenCV colormap (e.g. ``'inferno'``, ``'viridis'``, ``'magma'``).

        Returns:
            If ``colormap`` is None: the raw depth map (HxW float32 numpy array).
            If ``colormap`` is provided: a tuple ``(raw_depth, colored_depth)``.
        """
        start_time = time.time()
        depth = self._infer_image(image)
        self._inference_times.append((time.time() - start_time) * 1000)

        if normalize:
            depth = self._normalize_depth(depth)

        if colormap is not None:
            return depth, self._apply_colormap(depth, colormap)

        return depth

    def inference_batch(self, images: List[np.ndarray], normalize: bool = False,
                        colormap: Optional[str] = None) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Run depth estimation on a batch (list) of images.

        Args:
            images: List of input BGR images.
            normalize: If True, normalize each depth map to 0-255.
            colormap: If provided, also return colored depth maps.

        Returns:
            List of results, each in the same format as :meth:`inference`.
        """
        return [self.inference(image, normalize, colormap) for image in images]

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize a depth map to the 0-255 uint8 range."""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth)
        return depth_normalized.astype(np.uint8)

    def _apply_colormap(self, depth: np.ndarray, colormap: str = 'inferno') -> np.ndarray:
        """Apply an OpenCV colormap to a depth map, returning a BGR uint8 image."""
        if depth.dtype != np.uint8:
            depth = self._normalize_depth(depth)

        colormap_dict = {
            'inferno': cv2.COLORMAP_INFERNO,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'bone': cv2.COLORMAP_BONE,
            'turbo': cv2.COLORMAP_TURBO,
        }
        cv_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_INFERNO)
        return cv2.applyColorMap(depth, cv_colormap)

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if len(self._inference_times) == 0:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    def get_avg_fps(self) -> float:
        """Get average FPS based on recorded inference times."""
        avg_time = self.get_avg_inference_time()
        return 1000.0 / avg_time if avg_time > 0 else 0.0
