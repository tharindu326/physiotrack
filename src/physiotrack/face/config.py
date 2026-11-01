from dataclasses import dataclass


@dataclass
class FaceAnalysisConfig:
    """Configuration for the modular face-analysis pipeline."""

    tracking: bool = True
    head_pose: bool = True
    landmarks: bool = True
    quality: bool = True
    eyes: bool = True
    blink: bool = True
    gaze: bool = True
    gaze_estimation: bool = False
    mouth: bool = True
    mouth_motion: bool = True
    emotion: bool = True
    regions: bool = True
    temporal: bool = True

    tracker_type: str = "ocsort"

    blink_threshold: float = 0.22
    min_closed_frames: int = 3

    gaze_estimation_mode: str = "eth-xgaze"
    gaze_estimation_min_iou: float = 0.10

    temporal_window_sec: float = 5.0

    emotion_model: str = "enet_b0_8_best_afew"
    emotion_engine: str = "onnx"

    def validate(self):
        """Validate configuration values and module dependencies."""
        if self.blink_threshold <= 0:
            raise ValueError(
                "blink_threshold must be greater than zero"
            )

        if self.min_closed_frames < 1:
            raise ValueError(
                "min_closed_frames must be at least 1"
            )

        if not (
            0.0
            <= self.gaze_estimation_min_iou
            <= 1.0
        ):
            raise ValueError(
                "gaze_estimation_min_iou must be between 0 and 1"
            )

        if self.gaze_estimation_mode not in {
            "eth-xgaze",
            "mpiigaze",
            "mpiifacegaze",
        }:
            raise ValueError(
                "Unsupported gaze_estimation_mode: "
                f"{self.gaze_estimation_mode}"
            )

        if self.temporal_window_sec <= 0:
            raise ValueError(
                "temporal_window_sec must be greater than zero"
            )

        if self.tracker_type not in {
            "ocsort",
            "bytetrack",
            "strongsort",
            "boosttrack",
        }:
            raise ValueError(
                f"Unsupported tracker_type: {self.tracker_type}"
            )

        if self.eyes and not self.landmarks:
            raise ValueError(
                "eyes requires landmarks=True"
            )

        if self.blink and not self.eyes:
            raise ValueError(
                "blink requires eyes=True"
            )

        if self.gaze and not self.landmarks:
            raise ValueError(
                "gaze requires landmarks=True"
            )

        if self.mouth and not self.landmarks:
            raise ValueError(
                "mouth requires landmarks=True"
            )

        if self.mouth_motion and not self.mouth:
            raise ValueError(
                "mouth_motion requires mouth=True"
            )

        if self.temporal and not self.tracking:
            raise ValueError(
                "temporal requires tracking=True"
            )