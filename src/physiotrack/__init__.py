"""Physiotrack — a Python toolkit for contactless human understanding.

Quick start::

    import physiotrack as pt

    det = pt.Detection.Person()
    result = det.predict(frame)          # -> pt.Result
    annotated = result.plot()

Every image predictor (Detection, Pose, Segmentation, Depth, Face) exposes
``.predict()`` (and is callable), returning a unified result object whose
``.plot()`` draws the overlay. Full API reference:
https://tharindu326.github.io/physiotrack/
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("physiotrack")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

# --- Model registry -----------------------------------------------------------
from ._logging import set_log_level
from ._paths import migrate_weight_cache
from .models import Models

# --- Unified result objects ---------------------------------------------------
from .results import (Result, DepthResult, Pose3DResult, TrackResult, FrameResult,
                      VideoResults, Instance, Keypoint, Keypoints, ResultMeta)

# The lazy names below are invisible to anything that reads this file without running
# it -- type checkers, IDE completion, and the mkdocstrings/griffe pass that builds the
# API reference. Declaring them under TYPE_CHECKING makes the public surface statically
# resolvable while keeping the import cost at zero: the block never executes at runtime,
# so ``__getattr__`` still does the real work on first access.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capture import Video
    from .depth import Depth
    from .detect import Detection
    from .face import (Face, FaceExpression, FaceLandmarks, FaceOrientation, FaceQuality,
                       FaceRegions, GazeEstimator, VRFace)
    from .modules._3DCPNet.inference import (apply_3dpcnet_transform,
                                             reverse_3dpcnet_transform)
    from .pose import Pose
    from .pose.canonicalizer import CanonicalView, PoseCanonicalizer, canonicalize_pose
    from .pose.evaluate import (calculate_mpjpe, calculate_pampjpe,
                                calculate_rotation_error, evaluate_canonicalization,
                                evaluate_pose_predictions)
    from .pose.pose3D import Pose3D
    from .segment import Segmentation
    from .trackers import Tracker, TrackerConfig


# Everything below is resolved on first attribute access rather than at import time.
#
# Each entry maps a public name to the module that defines it. Importing a predictor
# pulls in its model backend (torch, ultralytics, timm, matplotlib), which used to
# make a bare ``import physiotrack`` cost seconds and load thousands of modules even
# to read a docstring. Subpackages now import their backends from ``.modules``
# directly instead of from this module, so nothing here has to be eager.
_LAZY_ATTRS = {
    # high-level predictors
    "Detection": ".detect",
    "Segmentation": ".segment",
    "Pose": ".pose",
    "Pose3D": ".pose.pose3D",
    "Depth": ".depth",
    "Face": ".face",
    "VRFace": ".face",
    "FaceOrientation": ".face",
    "FaceLandmarks": ".face",
    "FaceExpression": ".face",
    "GazeEstimator": ".face",
    "FaceQuality": ".face",
    "FaceRegions": ".face",
    # tracking
    "Tracker": ".trackers",
    "TrackerConfig": ".trackers",
    # orchestrator
    "Video": ".capture",
    # pose post-processing
    "PoseCanonicalizer": ".pose.canonicalizer",
    "canonicalize_pose": ".pose.canonicalizer",
    "CanonicalView": ".pose.canonicalizer",
    "apply_3dpcnet_transform": ".modules._3DCPNet.inference",
    "reverse_3dpcnet_transform": ".modules._3DCPNet.inference",
    "evaluate_pose_predictions": ".pose.evaluate",
    "evaluate_canonicalization": ".pose.evaluate",
    "calculate_mpjpe": ".pose.evaluate",
    "calculate_pampjpe": ".pose.evaluate",
    "calculate_rotation_error": ".pose.evaluate",
}

# Subpackages reachable as attributes (``pt.signals.joint_angles``) without being
# imported up front.
_LAZY_SUBMODULES = ("signals", "detect", "segment", "pose", "depth", "face",
                    "trackers", "capture", "core", "utils", "models", "results")


def __getattr__(name):
    """Resolve public names on first access (PEP 562)."""
    import importlib

    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache so later lookups skip this path
        return value
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose the lazily-loaded names to ``dir()`` and tab completion."""
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_SUBMODULES))


__all__ = [
    # package metadata
    "__version__",
    # predictors
    "Detection", "Pose", "Pose3D", "Segmentation", "Depth",
    "Face", "VRFace", "FaceOrientation", "FaceLandmarks", "FaceExpression",
    "GazeEstimator", "FaceQuality", "FaceRegions",
    # tracking
    "Tracker", "TrackerConfig",
    # orchestrator
    "Video",
    # registry
    "Models",
    # logging
    "set_log_level",
    # weight cache
    "migrate_weight_cache",
    # results
    "Result", "DepthResult", "Pose3DResult", "TrackResult", "FrameResult",
    "VideoResults",
    "Instance", "Keypoint", "Keypoints", "ResultMeta",
    # signal extraction and analysis (rPPG/HRV/respiration, motion features, plots)
    "signals",
    # pose post-processing
    "PoseCanonicalizer", "canonicalize_pose", "CanonicalView",
    "apply_3dpcnet_transform", "reverse_3dpcnet_transform",
    "evaluate_pose_predictions", "evaluate_canonicalization",
    "calculate_mpjpe", "calculate_pampjpe", "calculate_rotation_error",
]
