class MouthMovement:
    """Estimate temporal mouth movement from mouth-openness values."""

    def __init__(self, fps):
        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.fps = float(fps)
        self._states = {}

    def _get_state(self, person_id):
        if person_id not in self._states:
            self._states[person_id] = {
                "previous_openness": None,
                "previous_valid_frame": None,
                "frame_index": 0,
            }

        return self._states[person_id]

    def update(self, openness, person_id=0):
        """Update mouth movement using one frame."""
        # Do not share motion history between faces lacking a tracking ID.
        if person_id is None:
            return {
                "mouth_movement": None,
                "mouth_velocity": None,
            }

        state = self._get_state(person_id)
        state["frame_index"] += 1

        if openness is None:
            state["previous_openness"] = None
            state["previous_valid_frame"] = None

            return {
                "mouth_movement": None,
                "mouth_velocity": None,
            }

        previous = state["previous_openness"]
        previous_frame = state["previous_valid_frame"]

        if previous is None or previous_frame is None:
            movement = 0.0
            velocity = 0.0
        else:
            movement = abs(openness - previous)

            frame_gap = (
                state["frame_index"]
                - previous_frame
            )

            elapsed_time = (
                frame_gap / self.fps
            )

            velocity = (
                movement / elapsed_time
                if elapsed_time > 0
                else 0.0
            )

        state["previous_openness"] = openness
        state["previous_valid_frame"] = state["frame_index"]

        return {
            "mouth_movement": movement,
            "mouth_velocity": velocity,
        }

    def reset(self, person_id=None):
        """Reset mouth movement history."""
        if person_id is None:
            self._states.clear()
        else:
            self._states.pop(person_id, None)