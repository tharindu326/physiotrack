from numbers import Integral


class BlinkDetector:
    """Detect blinks from temporal eye-openness values."""

    def __init__(
        self,
        threshold,
        fps,
        min_closed_frames=2,
    ):
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1")

        self.threshold = float(threshold)
        self.fps = float(fps)
        self.min_closed_frames = int(min_closed_frames)

        self._states = {}

    def _get_state(self, person_id):
        if person_id not in self._states:
            self._states[person_id] = {
                "closed_frames": 0,
                "blink_count": 0,
                "frame_count": 0,
                "first_frame_index": None,
                "last_frame_index": None,
            }

        return self._states[person_id]

    def update(self, openness, person_id=0, frame_index=None):
        """Update one person's blink state at an optional video frame index.

        frame_index is a zero-based index in the current video. When omitted,
        consecutive calls for that person are treated as consecutive frames,
        preserving the original standalone component interface.
        """
        if frame_index is not None and (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, Integral)
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")

        # A missing tracking identity cannot safely own a temporal history.
        # Never store unrelated untracked faces under the shared None key.
        if person_id is None:
            return {
                "eye_state": "unknown",
                "blink": False,
                "blink_count": 0,
                "blink_duration": None,
                "blink_rate": None,
            }

        state = self._get_state(person_id)
        previous_index = state["last_frame_index"]
        current_index = (
            int(frame_index)
            if frame_index is not None
            else (0 if previous_index is None else previous_index + 1)
        )

        if previous_index is not None:
            if current_index <= previous_index:
                raise ValueError(
                    "frame_index must increase for each person"
                )
            if current_index > previous_index + 1:
                # Unobserved frames cannot form one continuous eye closure.
                state["closed_frames"] = 0

        if state["first_frame_index"] is None:
            state["first_frame_index"] = current_index
        state["last_frame_index"] = current_index
        state["frame_count"] += 1

        blink = False
        blink_duration = None

        if openness is None:
            # Do not bridge a closed-eye sequence across missing landmarks.
            state["closed_frames"] = 0

            return {
                "eye_state": "unknown",
                "blink": False,
                "blink_count": state["blink_count"],
                "blink_duration": None,
                "blink_rate": self._blink_rate(state),
            }

        if openness < self.threshold:
            state["closed_frames"] += 1
            eye_state = "closed"

        else:
            eye_state = "open"

            if state["closed_frames"] >= self.min_closed_frames:
                blink = True
                state["blink_count"] += 1
                blink_duration = (
                    state["closed_frames"] / self.fps
                )

            state["closed_frames"] = 0

        return {
            "eye_state": eye_state,
            "blink": blink,
            "blink_count": state["blink_count"],
            "blink_duration": blink_duration,
            "blink_rate": self._blink_rate(state),
        }

    def _blink_rate(self, state):
        # Inclusive video-frame span: missing detections still advance time.
        elapsed_frames = (
            state["last_frame_index"]
            - state["first_frame_index"]
            + 1
        )
        elapsed_minutes = (elapsed_frames / self.fps) / 60.0

        if elapsed_minutes == 0:
            return 0.0

        return (
            state["blink_count"]
            / elapsed_minutes
        )

    def reset(self, person_id=None):
        """Reset blink history."""
        if person_id is None:
            self._states.clear()
        else:
            self._states.pop(person_id, None)