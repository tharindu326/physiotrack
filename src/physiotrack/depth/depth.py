"""
Depth Estimation Module
Provides high-level wrapper for depth estimation models.
"""

import logging

from ..models import Models
from ..modules import DepthAnythingV2Inference, ZipDepthInference
from ..results import DepthResult
import os
import numpy as np
from typing import Optional, Tuple, Union, List

from .._logging import get_logger
from ..core.predictor import PredictorMixin

logger = get_logger(__name__)


class DepthBase(PredictorMixin):
    """Shared implementation for the monocular depth presets.

    Resolves a depth model enum, wraps the corresponding backend (currently
    Depth-Anything-V2), and exposes the unified
    [`predict`][physiotrack.Depth] interface returning a
    [`DepthResult`][physiotrack.DepthResult].

    Not used directly; instantiate a ``Depth.*`` preset.

    Attributes:
        model (Models.Depth.*): The resolved model enum in use.
        depth_framework (str): Backend name, e.g. ``"DepthAnythingV2"`` or ``"ZipDepth"``.
        device (int | str): Inference device.
        input_size (int): Input resolution used for inference.
        verbose (bool): Whether initialization info is printed.
    """

    default_model = None

    def __init__(self, model=None, device='cpu', input_size: Optional[int] = None,
                 verbose: bool = True, **kwargs):
        """Configure a depth estimator.

        Args:
            model (Models.Depth.*, optional): A validated depth model enum, e.g.
                ``Models.Depth.DepthAnythingV2.vitl`` or ``Models.Depth.ZipDepth.base``.
                Defaults to ``None`` (uses the preset's class-level ``default_model``).
            device (int | str, optional): Inference device, e.g. ``'cpu'``,
                ``'cuda'``, ``'mps'`` or a device index like ``0``. Defaults to
                ``'cpu'``.
            input_size (int, optional): Input resolution used for inference. For
                Depth-Anything-V2 this is the square input size; for ZipDepth it
                is the length of the image's shorter side (aspect ratio preserved).
                Defaults to ``None``, which selects the model's native resolution
                (518 for Depth-Anything-V2, 384 for ZipDepth).
            verbose (bool, optional): Print initialization info. Defaults to
                ``True``.
            **kwargs (Any): Reserved for forward-compatibility; currently unused by
                the depth backends.

        Raises:
            ValueError: If no model can be resolved, or the model maps to an
                unsupported backend.

        Note:
            On first use the model weights are auto-downloaded from Hugging Face
            and cached.
        """
        if model is None:
            if self.default_model is None:
                raise ValueError(
                    f"{type(self).__name__} needs a model: pass model=<a Models.Depth "
                    f"member>, or use a preset such as Depth.DepthAnythingV2Base() that "
                    f"supplies one. Browse the options with Models.list(task='Depth')."
                )
            model = self.default_model

        Models.validate_depth_model(model)

        model_path = Models.resolve(model)

        self.minfo = Models.info(model)
        self.depth_framework = self.minfo['backend']
        logger.log(logging.INFO if verbose else logging.DEBUG,
                   'Initiating %s %s for depth estimation', self.depth_framework, model.name)

        # Get model configuration. When ``input_size`` is not specified, fall back
        # to the model's native resolution declared in its registry config.
        model_config = Models.get_depth_config(model)
        if input_size is None:
            input_size = model_config.get('input_size', 518)

        if self.depth_framework == 'DepthAnythingV2':
            self.depth_estimator = DepthAnythingV2Inference(
                model_path=model_path,
                model_config=model_config,
                device=device,
                input_size=input_size,
                verbose=verbose
            )
        elif self.depth_framework == 'ZipDepth':
            self.depth_estimator = ZipDepthInference(
                model_path=model_path,
                model_config=model_config,
                device=device,
                input_size=input_size,
                verbose=verbose
            )
        else:
            raise ValueError(f"Invalid depth model type: {self.depth_framework}")

        self.model = model
        self.device = device
        self.input_size = input_size
        self.verbose = verbose

    def predict(self, source) -> Union[DepthResult, List[DepthResult]]:
        """Estimate depth for one image or a batch of images.

        Args:
            source (np.ndarray | list[np.ndarray] | tuple[np.ndarray]): A single
                BGR image ``(H, W, 3)`` or a list/tuple of such frames for batch
                inference.

        Returns:
            DepthResult | list[DepthResult]: A
                [`DepthResult`][physiotrack.DepthResult] for a single frame, or a
                ``list[DepthResult]`` when ``source`` is a list/tuple. Access the
                raw ``(H, W)`` float map via ``result.depth``, a colorized BGR
                view via ``result.plot(colormap=...)``, and a ``0..1`` normalized
                map via ``result.normalized()``.

        Example:
            ```python
            import physiotrack as pt

            depth = pt.Depth.DepthAnythingV2Base(device=0)
            result = depth.predict(frame)        # or: depth(frame)
            raw = result.depth                   # (H, W) float depth
            colored = result.plot(colormap="inferno")
            ```

        See Also:
            [`DepthResult`][physiotrack.DepthResult]: the returned depth container.
        """
        frames, was_batch = self._as_frames(source)
        if was_batch:
            depths = self.depth_estimator.inference_batch(frames, False, None)
            return [DepthResult(orig_img=frame, depth=depth)
                    for frame, depth in zip(frames, depths)]
        depth = self.depth_estimator.inference(frames[0], False, None)
        return self._unwrap([DepthResult(orig_img=frames[0], depth=depth)], was_batch)

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if hasattr(self.depth_estimator, 'get_avg_inference_time'):
            return self.depth_estimator.get_avg_inference_time()
        return 0.0

    def get_avg_fps(self) -> float:
        """Get average FPS based on inference times."""
        if hasattr(self.depth_estimator, 'get_avg_fps'):
            return self.depth_estimator.get_avg_fps()
        return 0.0


class Depth:
    """Monocular depth predictors, grouped as ready-to-use presets.

    ``Depth`` is a namespace of nested predictor classes. Instantiate a preset,
    then call [`predict`][physiotrack.Depth] (or the instance directly)
    on a frame to get a [`DepthResult`][physiotrack.DepthResult].

    Presets:
        - [`DepthAnythingV2Small`][physiotrack.Depth.DepthAnythingV2Small]:
          ``vits`` — fastest, least accurate.
        - [`DepthAnythingV2Base`][physiotrack.Depth.DepthAnythingV2Base]:
          ``vitb`` — balanced.
        - [`DepthAnythingV2Large`][physiotrack.Depth.DepthAnythingV2Large]:
          ``vitl`` — most accurate.
        - [`ZipDepth`][physiotrack.Depth.ZipDepth]: ``base`` — lightweight
          (~6M params), fast, relative depth; GPU/server head.
        - [`ZipDepthNPU`][physiotrack.Depth.ZipDepthNPU]: ``npu`` — ZipDepth with
          an NPU/CPU/mobile-friendly upsampling head.
        - [`Custom`][physiotrack.Depth.Custom]: any validated depth model.

    Example:
        ```python
        import physiotrack as pt

        depth = pt.Depth.DepthAnythingV2Base(device=0)
        result = depth.predict(frame)            # or: depth(frame)
        raw = result.depth                       # raw float depth map (H, W)
        colored = result.plot(colormap="inferno")
        norm = result.normalized()               # 0..1
        ```

    Note:
        Weights are auto-downloaded from Hugging Face on first use and cached.

    See Also:
        [`DepthResult`][physiotrack.DepthResult]: depth output container.
    """

    def __new__(cls, *args, **kwargs):
        """Refuse direct instantiation of the preset namespace.

        Raises:
            TypeError: Always. ``Depth`` groups the presets; it is not itself a
                predictor, and instantiating it used to return an object with no
                model attached, which failed later with a confusing error.
        """
        presets = [n for n in vars(cls) if not n.startswith("_")
                   and isinstance(vars(cls)[n], type)]
        raise TypeError(
            f"Depth is a namespace of presets, not a predictor. Use one of: "
            f"{', '.join(f'Depth.{p}()' for p in presets)} "
            f"— for example Depth.DepthAnythingV2Base()."
        )

    class Custom(DepthBase):
        """Depth estimator backed by any user-specified validated depth model.

        Example:
            ```python
            import physiotrack as pt
            from physiotrack import Models

            depth = pt.Depth.Custom(model=Models.Depth.DepthAnythingV2.vitl)
            result = depth.predict(frame)
            ```
        """

        def __init__(self, model, device='cpu', input_size: Optional[int] = None,
                     verbose: bool = True, **kwargs):
            """Configure a custom depth estimator.

            Args:
                model (Models.Depth.*): A validated depth model enum, e.g.
                    ``Models.Depth.DepthAnythingV2.vitl`` or
                    ``Models.Depth.ZipDepth.base``.
                device (int | str, optional): Inference device, e.g. ``'cpu'``,
                    ``'cuda'``, ``'mps'`` or a device index. Defaults to ``'cpu'``.
                input_size (int, optional): Input resolution used for inference.
                    Defaults to ``None`` (the model's native resolution).
                verbose (bool, optional): Print initialization info. Defaults to
                    ``True``.
                **kwargs (Any): Forwarded to
                    [`DepthBase`][physiotrack.Depth].
            """
            Models.validate_depth_model(model)
            super().__init__(
                model=model,
                device=device,
                input_size=input_size,
                verbose=verbose,
                **kwargs
            )

    class DepthAnythingV2Small(DepthBase):
        """Depth-Anything-V2 Small estimator (faster, less accurate).

        Wraps ``Models.Depth.DepthAnythingV2.vits``. See
        [`DepthBase`][physiotrack.Depth] for constructor
        arguments.
        """
        default_model = Models.Depth.DepthAnythingV2.vits

    class DepthAnythingV2Base(DepthBase):
        """Depth-Anything-V2 Base estimator (balanced speed/accuracy).

        Wraps ``Models.Depth.DepthAnythingV2.vitb``. See
        [`DepthBase`][physiotrack.Depth] for constructor
        arguments.
        """
        default_model = Models.Depth.DepthAnythingV2.vitb

    class DepthAnythingV2Large(DepthBase):
        """Depth-Anything-V2 Large estimator (most accurate).

        Wraps ``Models.Depth.DepthAnythingV2.vitl``. See
        [`DepthBase`][physiotrack.Depth] for constructor
        arguments.
        """
        default_model = Models.Depth.DepthAnythingV2.vitl

    class ZipDepth(DepthBase):
        """ZipDepth estimator — lightweight monocular depth (GPU/server head).

        Wraps ``Models.Depth.ZipDepth.base``, a ~6M-parameter model that returns
        a relative (affine-invariant) depth map, much faster and smaller than
        Depth-Anything-V2. See [`DepthBase`][physiotrack.Depth] for constructor
        arguments.
        """
        default_model = Models.Depth.ZipDepth.base

    class ZipDepthNPU(DepthBase):
        """ZipDepth estimator with the NPU/CPU/mobile-friendly upsampling head.

        Wraps ``Models.Depth.ZipDepth.npu``. Shares ZipDepth's encoder/decoder
        weights but uses an ONNX/mobile-friendly upsampling head instead of the
        unfold-based one. See [`DepthBase`][physiotrack.Depth] for constructor
        arguments.
        """
        default_model = Models.Depth.ZipDepth.npu
