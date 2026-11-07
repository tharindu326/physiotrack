"""Face detection and per-face analysis.

Detectors find faces ([`Face`][physiotrack.Face], [`VRFace`][physiotrack.VRFace]); the
face stages each add one property to every face and chain in any order their
requirements allow:

- [`FaceOrientation`][physiotrack.FaceOrientation] -- head yaw / pitch / roll.
- [`FaceLandmarks`][physiotrack.FaceLandmarks] -- 478-point face mesh.
- [`FaceExpression`][physiotrack.FaceExpression] -- facial expression category.
- [`GazeEstimator`][physiotrack.GazeEstimator] -- 3D gaze (needs the face mesh).
- [`FaceQuality`][physiotrack.FaceQuality] -- brightness, sharpness, size.
- [`FaceRegions`][physiotrack.FaceRegions] -- visible face parts (skin, eyes, ...).
"""
from .base import FaceStage, check_stage_order
from .detect import Face, VRFace
from .expression import FaceExpression
from .face_orientation import FaceOrientation
from .gaze import GazeEstimator
from .landmarks import FaceLandmarks
from .quality import FaceQuality
from .regions import FaceRegions
from ..modules._6DRepNet360.utils import plot_pose_cube, draw_axis

__all__ = [
    "Face", "VRFace",
    "FaceStage", "check_stage_order",
    "FaceOrientation", "FaceLandmarks", "FaceExpression", "GazeEstimator", "FaceQuality",
    "FaceRegions",
    "plot_pose_cube", "draw_axis",
]
