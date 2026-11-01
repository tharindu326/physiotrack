import math


class MouthOpenness:
    """Estimate mouth openness from facial landmarks."""

    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    UPPER_LIP = 13
    LOWER_LIP = 14

    @staticmethod
    def _distance(p1, p2, image_size):
        width, height = image_size

        dx = (p1.x - p2.x) * width
        dy = (p1.y - p2.y) * height

        distance_px = math.sqrt(
            dx * dx +
            dy * dy
        )

        return distance_px / width

    def predict(self, landmarks, image_size):
        """Calculate normalized mouth openness."""
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]

        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]

        mouth_width = self._distance(
            left, right, image_size
        )
        mouth_height = self._distance(
            upper, lower, image_size
        )

        if mouth_width == 0:
            return {
                "mouth_openness": None,
                "mouth_width": None,
                "mouth_height": None,
            }

        openness = mouth_height / mouth_width

        return {
            "mouth_openness": openness,
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
        }
