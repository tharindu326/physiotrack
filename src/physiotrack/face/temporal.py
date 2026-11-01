from collections import Counter, defaultdict, deque

import numpy as np


class FaceTemporalAggregator:
    """Aggregate per-frame face features over a temporal window."""

    def __init__(self, fps, window_sec=5.0):
        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        if window_sec <= 0:
            raise ValueError("window_sec must be greater than zero")

        self.fps = float(fps)
        self.window_sec = float(window_sec)

        self.window_frames = max(
            1,
            int(round(self.fps * self.window_sec)),
        )

        self.buffers = defaultdict(
            lambda: deque(maxlen=self.window_frames)
        )

    @staticmethod
    def _numeric_summary(values):
        """Return basic statistics for valid numeric values."""
        valid_values = [
            float(value)
            for value in values
            if value is not None
            and np.isfinite(value)
        ]

        if not valid_values:
            return None

        array = np.asarray(
            valid_values,
            dtype=float,
        )

        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    def update(self, instance):
        """Add one tracked face instance to its temporal buffer."""
        if instance.id is None:
            return None

        if instance.face_features is None:
            return None

        orientation = None

        if instance.orientation is not None:
            orientation = {
                "yaw": instance.orientation.get(
                    "yaw"
                ),
                "pitch": instance.orientation.get(
                    "pitch"
                ),
                "roll": instance.orientation.get(
                    "roll"
                ),
            }

        self.buffers[instance.id].append(
            {
                "face_features": instance.face_features,
                "orientation": orientation,
            }
        )

        return self.summary(instance.id)

    def summary(self, person_id):
        """Return temporal descriptors for one tracked person."""
        frames = self.buffers.get(person_id)

        if not frames:
            return None

        def values(feature_name, key):
            return [
                frame["face_features"]
                .get(feature_name, {})
                .get(key)
                for frame in frames
                if frame["face_features"]
                .get(feature_name, {})
                .get("available", False)
            ]

        def orientation_values(key):
            return [
                frame["orientation"].get(key)
                for frame in frames
                if frame["orientation"] is not None
                and frame["orientation"].get(key) is not None
            ]

        emotions = [
            frame["face_features"]
            .get("emotion", {})
            .get("emotion")
            for frame in frames
            if frame["face_features"]
            .get("emotion", {})
            .get("available", False)
        ]

        emotions = [
            emotion
            for emotion in emotions
            if emotion is not None
        ]

        dominant_emotion = None

        if emotions:
            dominant_emotion = Counter(
                emotions
            ).most_common(1)[0][0]

        blink_events = sum(
            1
            for frame in frames
            if frame["face_features"]
            .get("blink", {})
            .get("blink", False)
        )

        return {
            "person_id": person_id,
            "window_frames": len(frames),
            "window_sec": float(
                len(frames) / self.fps
            ),
            "head_pose": {
                "yaw": self._numeric_summary(
                    orientation_values("yaw")
                ),
                "pitch": self._numeric_summary(
                    orientation_values("pitch")
                ),
                "roll": self._numeric_summary(
                    orientation_values("roll")
                ),
            },
            "eyes": {
                "mean_openness": self._numeric_summary(
                    values(
                        "eyes",
                        "mean_openness",
                    )
                ),
            },
            "gaze": {
                "mean_iris_x": self._numeric_summary(
                    values(
                        "gaze",
                        "mean_iris_x",
                    )
                ),
                "mean_iris_y": self._numeric_summary(
                    values(
                        "gaze",
                        "mean_iris_y",
                    )
                ),
            },
            "mouth": {
                "openness": self._numeric_summary(
                    values(
                        "mouth",
                        "mouth_openness",
                    )
                ),
                "movement": self._numeric_summary(
                    values(
                        "mouth_motion",
                        "mouth_movement",
                    )
                ),
            },
            "quality": {
                "brightness": self._numeric_summary(
                    values(
                        "quality",
                        "brightness",
                    )
                ),
                "sharpness": self._numeric_summary(
                    values(
                        "quality",
                        "sharpness",
                    )
                ),
                "face_area_ratio": self._numeric_summary(
                    values(
                        "quality",
                        "face_area_ratio",
                    )
                ),
            },
            "blink": {
                "events": int(blink_events),
            },
            "emotion": {
                "dominant": dominant_emotion,
            },
        }

    def reset(self, person_id=None):
        """Clear one person's history or all temporal history."""
        if person_id is None:
            self.buffers.clear()
        else:
            self.buffers.pop(person_id, None)