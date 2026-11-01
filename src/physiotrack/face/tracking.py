import numpy as np

from ..trackers import Tracker, TrackerConfig


class FaceTracker:
    """Tracking support for detected faces."""

    def __init__(self, tracker_type="ocsort", device="cpu"):
        config = TrackerConfig(
            tracker_type=tracker_type,
            classes=[0],
            device=device,
        )

        self.tracker = Tracker(config)

    def track(self, frame, face_result):
        """Track faces detected in one frame."""
        detections = []

        for face in face_result:
            if face.box is None:
                continue

            confidence = (
                face.confidence
                if face.confidence is not None
                else 1.0
            )

            class_id = (
                face.cls
                if face.cls is not None
                else 0
            )

            detections.append(
                [
                    *face.box,
                    confidence,
                    class_id,
                ]
            )

        if detections:
            detections = np.asarray(detections, dtype=np.float32)
        else:
            detections = np.empty((0, 6), dtype=np.float32)

        return self.tracker.track(frame, detections)