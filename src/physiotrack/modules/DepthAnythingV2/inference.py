"""
DepthAnythingV2 Inference Module
Provides depth estimation from RGB images using DepthAnythingV2 models.
"""

import cv2
import torch
import os
import numpy as np
from typing import Optional, Tuple, Union, List
import time
from collections import deque

from .dpt import DepthAnythingV2

from ..._logging import get_logger
from ...core.device import torch_device
from ..._paths import weights_dir

logger = get_logger(__name__)


class DepthAnythingV2Inference:
    """
    Inference wrapper for DepthAnythingV2 depth estimation models.

    Attributes:
        model: The DepthAnythingV2 model instance
        device: Device to run inference on ('cuda', 'cpu', or 'mps')
        encoder: Encoder type ('vits', 'vitb', 'vitl')
    """

    def __init__(self, model_path: str, model_config: dict, device: Union[str, int] = 'cuda',
                 input_size: int = 518, verbose: bool = True):
        """
        Initialize DepthAnythingV2 inference.

        Args:
            model_path: Path to the model weights file
            model_config: Model configuration dict containing 'encoder', 'features', 'out_channels'
            device: Device to run inference on (0, 'cuda', 'cpu', 'mps')
            input_size: Input image size for inference (default: 518)
            verbose: Whether to print initialization info
        """
        # Check if model file exists
        if not os.path.isfile(model_path):
            model_name = os.path.basename(model_path)
            model_dir = str(weights_dir())
            raise ValueError(
                f"The model file '{model_name}' does not exist at {model_path}\n"
                f"Please download it manually from HuggingFace:\n"
                f"  - vits: https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth\n"
                f"  - vitb: https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth\n"
                f"  - vitl: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth\n"
                f"And place it in: {os.path.abspath(model_dir)}/"
            )

        # Honour the requested device exactly (no silent CPU fallback).
        self.device = torch_device(device)

        self.encoder = model_config['encoder']
        self.input_size = input_size
        self.verbose = verbose

        # Initialize model
        self.model = DepthAnythingV2(
            encoder=model_config['encoder'],
            features=model_config['features'],
            out_channels=model_config['out_channels']
        )

        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

        if self.verbose:
            logger.info("DepthAnythingV2 (%s) loaded on %s", self.encoder, self.device)

        # Inference time tracking
        self._inference_times = deque(maxlen=100)

    def inference(self, image: np.ndarray, normalize: bool = False,
                  colormap: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run depth estimation on a single image.

        Args:
            image: Input BGR image (HxWx3 numpy array)
            normalize: If True, normalize depth to 0-255 range
            colormap: If provided, return colored depth map using this colormap
                     (e.g., 'inferno', 'viridis', 'magma', 'plasma', 'jet')

        Returns:
            If colormap is None: Raw depth map (HxW numpy array, float32)
            If colormap is provided: Tuple of (raw_depth, colored_depth)
        """
        start_time = time.time()

        with torch.no_grad():
            depth = self.model.infer_image(image, self.input_size)

        inference_time = (time.time() - start_time) * 1000
        self._inference_times.append(inference_time)

        if normalize:
            depth = self._normalize_depth(depth)

        if colormap is not None:
            colored_depth = self._apply_colormap(depth, colormap)
            return depth, colored_depth

        return depth

    def inference_batch(self, images: List[np.ndarray], normalize: bool = False,
                        colormap: Optional[str] = None) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Run depth estimation on a batch of images.

        Args:
            images: List of input BGR images
            normalize: If True, normalize depth to 0-255 range
            colormap: If provided, return colored depth maps

        Returns:
            List of depth results (same format as inference())
        """
        results = []
        for image in images:
            result = self.inference(image, normalize, colormap)
            results.append(result)
        return results

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-255 range."""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth)
        return depth_normalized.astype(np.uint8)

    def _apply_colormap(self, depth: np.ndarray, colormap: str = 'inferno') -> np.ndarray:
        """Apply colormap to depth map."""
        # Normalize if not already uint8
        if depth.dtype != np.uint8:
            depth = self._normalize_depth(depth)

        # Map colormap name to OpenCV colormap
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
        colored = cv2.applyColorMap(depth, cv_colormap)
        return colored

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if len(self._inference_times) == 0:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    def get_avg_fps(self) -> float:
        """Get average FPS based on inference times."""
        avg_time = self.get_avg_inference_time()
        if avg_time > 0:
            return 1000.0 / avg_time
        return 0.0
