"""
Public API for the physiotrack signals subsystem.

Re-exports the key signal-processing functions and classes so users can do::

    from physiotrack.signals import bandpass_filter, POS, RealTimePlotter
"""

from .filters import (
    bandpass_filter,
    zero_mean_std_norm,
    zero_mean_std_norm_1ch,
    notch_filter,
    highpass_filter,
    lowpass_filter,
    detrend_advanced,
    bandpass_firwin,
    signaltonoise_dB,
)
from .normalize import (
    min_max_normalize,
    z_score_normalize,
    robust_scale_normalize,
    max_abs_normalize,
    decimal_scaling_normalize,
    log_normalize,
    sigmoid_normalize,
    tanh_normalize,
    unit_vector_normalize,
    quantile_normalize,
    power_transform_normalize,
)
from .evaluate import (
    compute_plv,
    event_synchronization,
    phase_synchrony,
    compute_rmse,
    align_signals,
    normalized_cross_correlation,
    calculate_pearson_correlation,
    calculate_dtw_distance,
    hrv_errors,
)
from .keypoints import as_frame_records, as_keypoint_dicts
from .ppg import POS, CHROM, LGI, OMIT
from .ppg.constants import (
    HR_BAND, RESP_BAND, HRV_VLF_BAND, HRV_LF_BAND, HRV_HF_BAND,
    RPPG_METHODS, DEFAULT_RPPG_METHOD,
)
from .ppg.metrics import bvp_to_hr, bvp_snr, hr_errors, benchmark_rppg_methods
from .ppg.estimator import HeartRateEstimator
from .ppg.peaks import detect_pulse_peaks, bvp_to_rri
from .ppg.artifacts import find_rr_artifacts, correct_rr_artifacts
from .ppg.hrv import (
    hrv_time, hrv_frequency, hrv_nonlinear, compute_hrv,
    sample_entropy, approximate_entropy,
)
from .ppg.respiration import (
    respiration_rate_from_signal, respiration_from_pulse, respiration_from_rri,
)
from .motion.utils import (
    extract_keypoint_sequence_2d,
    extract_keypoint_sequence_3d,
    extract_keypoints_sequence,
    add_head_centroid,
    add_body_centroid,
    add_pelvic_centroid,
    resample_dataframe_by_interpolation,
)
from .motion.features import (
    get_relative_coordinates,
    compute_all_motion_features,
    compute_all_joint_angles,
    joint_angles,
    compute_rom_angles,
    get_keypoint_features,
    select_feature_data,
    respiration_from_motion,
)
from .face.features import (
    FACE_LANDMARK_TABLES,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    iris_position,
)
from .face.temporal import (
    face_feature_sequence,
    detect_blinks,
    blink_rate,
    mouth_movement,
    face_window_summary,
)

__all__ = [
    # filters
    "bandpass_filter",
    "zero_mean_std_norm",
    "zero_mean_std_norm_1ch",
    "notch_filter",
    "highpass_filter",
    "lowpass_filter",
    "detrend_advanced",
    "bandpass_firwin",
    "signaltonoise_dB",
    # normalize
    "min_max_normalize",
    "z_score_normalize",
    "robust_scale_normalize",
    "max_abs_normalize",
    "decimal_scaling_normalize",
    "log_normalize",
    "sigmoid_normalize",
    "tanh_normalize",
    "unit_vector_normalize",
    "quantile_normalize",
    "power_transform_normalize",
    # evaluate
    "compute_plv",
    "event_synchronization",
    "phase_synchrony",
    "compute_rmse",
    "align_signals",
    "normalized_cross_correlation",
    "calculate_pearson_correlation",
    "calculate_dtw_distance",
    "hrv_errors",
    # rPPG extraction
    "POS",
    "CHROM",
    "LGI",
    "OMIT",
    # rPPG analysis bands + method registry (single source of truth)
    "HR_BAND",
    "RESP_BAND",
    "HRV_VLF_BAND",
    "HRV_LF_BAND",
    "HRV_HF_BAND",
    "RPPG_METHODS",
    "DEFAULT_RPPG_METHOD",
    # rPPG HR metrics + estimator + skin extraction
    "bvp_to_hr",
    "bvp_snr",
    "hr_errors",
    "benchmark_rppg_methods",
    "HeartRateEstimator",
    # RR-interval extraction, artefact correction, HRV, respiration
    "detect_pulse_peaks",
    "bvp_to_rri",
    "find_rr_artifacts",
    "correct_rr_artifacts",
    "hrv_time",
    "hrv_frequency",
    "hrv_nonlinear",
    "compute_hrv",
    "sample_entropy",
    "approximate_entropy",
    "respiration_rate_from_signal",
    "respiration_from_pulse",
    "respiration_from_rri",
    "respiration_from_motion",
    # motion: keypoint sequences & centroids
    "extract_keypoint_sequence_2d",
    "extract_keypoint_sequence_3d",
    "extract_keypoints_sequence",
    "add_head_centroid",
    "add_body_centroid",
    "add_pelvic_centroid",
    "resample_dataframe_by_interpolation",
    # motion: features
    "get_relative_coordinates",
    "compute_all_motion_features",
    "compute_all_joint_angles",
    "joint_angles",
    "compute_rom_angles",
    "get_keypoint_features",
    "select_feature_data",
    # face: per-frame landmark geometry and per-face sequences
    "FACE_LANDMARK_TABLES",
    "eye_aspect_ratio",
    "mouth_aspect_ratio",
    "iris_position",
    "face_feature_sequence",
    "detect_blinks",
    "blink_rate",
    "mouth_movement",
    "face_window_summary",
    # result <-> dict adapters
    "as_keypoint_dicts",
    "as_frame_records",
]


# Resolved on first access rather than at import time (PEP 562).
#
# The filters, metrics, HRV, respiration and motion-feature code above is pure
# NumPy/SciPy/pandas. These two groups are not: the OpenCV overlay panels pull in cv2
# and matplotlib, and the SegFace-based skin ROI pulls the whole deep-learning stack
# (torch and ultralytics). Importing them eagerly made `from physiotrack.signals
# import joint_angles` cost seconds and load torch, which the DSP layer never uses.
# The lazy names below are invisible to anything that reads this file without running
# it -- type checkers, IDE completion, and the mkdocstrings/griffe pass that builds the
# API reference. Declaring them under TYPE_CHECKING makes the public surface statically
# resolvable while keeping the import cost at zero: the block never executes at runtime,
# so ``__getattr__`` still does the real work on first access.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plotting import (
        HRVPlotter,
        HeartRatePlotter,
        JointAnglePlotter,
        KeypointMotionPlotter,
        RPPGPlotter,
        RealTimePlotter,
        RespirationPlotter,
    )
    from .ppg.skin import FaceParsing, FaceSkinExtractor

_LAZY_ATTRS = {
    # overlay panels (cv2 + matplotlib)
    "RealTimePlotter": ".plotting",
    "KeypointMotionPlotter": ".plotting",
    "JointAnglePlotter": ".plotting",
    "HeartRatePlotter": ".plotting",
    "RPPGPlotter": ".plotting",
    "HRVPlotter": ".plotting",
    "RespirationPlotter": ".plotting",
    # SegFace skin/face parsing (torch)
    "FaceSkinExtractor": ".ppg.skin",
    "FaceParsing": ".ppg.skin",
}

# The lazily-resolved names are declared once, here, so __all__ cannot drift from
# the map above.
__all__ += list(_LAZY_ATTRS)


def __getattr__(name):
    """Resolve the plotting and skin-ROI names on first access."""
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache so later lookups skip this path
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
