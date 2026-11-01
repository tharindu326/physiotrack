import math


class EyeOpenness:
    """Estimate eye openness from facial landmarks."""

    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)

    @staticmethod
    def _distance(p1, p2, image_size):
        width, height = image_size

        dx = (p1.x - p2.x) * width
        dy = (p1.y - p2.y) * height

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    def _eye_ratio(self, landmarks, indices, image_size):
        p1, p2, p3, p4, p5, p6 = [
            landmarks[i] for i in indices
        ]

        vertical_1 = self._distance(
            p2, p6, image_size
        )
        vertical_2 = self._distance(
            p3, p5, image_size
        )
        horizontal = self._distance(
            p1, p4, image_size
        )

        if horizontal == 0:
            return None

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    def predict(self, landmarks, image_size):
        """Calculate openness for the left and right eyes."""
        left = self._eye_ratio(
            landmarks,
            self.LEFT_EYE,
            image_size,
        )

        right = self._eye_ratio(
            landmarks,
            self.RIGHT_EYE,
            image_size,
        )

        if left is None or right is None:
            mean = None
        else:
            mean = (left + right) / 2.0

        return {
            "left_openness": left,
            "right_openness": right,
            "mean_openness": mean,
        }
