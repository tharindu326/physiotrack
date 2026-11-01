import math
from statistics import median

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarks:
    """Facial landmark extraction using MediaPipe."""

    def __init__(self, model_path, num_faces=1):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def predict(self, frame):
        """Extract facial landmarks from one image."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect(mp_image)

        return result.face_landmarks

    @staticmethod
    def _square_box(frame, box):
        """Create a square face box while keeping it inside the frame."""
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = map(float, box)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return None

        side = max(box_width, box_height)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        side = min(side, width, height)

        sx1 = int(round(cx - side / 2.0))
        sy1 = int(round(cy - side / 2.0))
        sx2 = sx1 + int(round(side))
        sy2 = sy1 + int(round(side))

        if sx1 < 0:
            sx2 -= sx1
            sx1 = 0

        if sy1 < 0:
            sy2 -= sy1
            sy1 = 0

        if sx2 > width:
            shift = sx2 - width
            sx1 -= shift
            sx2 = width

        if sy2 > height:
            shift = sy2 - height
            sy1 -= shift
            sy2 = height

        sx1 = max(0, sx1)
        sy1 = max(0, sy1)

        if sx2 <= sx1 or sy2 <= sy1:
            return None

        return sx1, sy1, sx2, sy2

    @staticmethod
    def _match_geometry(points, box):
        """Measure spatial agreement of a landmark set with an input face box.

        The input points and box are in full-frame pixel coordinates. A valid
        match must have its median center and at least half its points inside
        the original (not square-expanded) detector box.
        """
        x1, y1, x2, y2 = map(float, box)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0 or not points:
            return None

        center_x = median(point[0] for point in points)
        center_y = median(point[1] for point in points)
        if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
            return None

        inside = sum(
            x1 <= point_x <= x2 and y1 <= point_y <= y2
            for point_x, point_y in points
        )
        fraction_inside = inside / len(points)
        if fraction_inside < 0.5:
            return None

        # Distances are normalized by the respective detector-box dimensions.
        dx = (center_x - (x1 + x2) / 2.0) / box_width
        dy = (center_y - (y1 + y2) / 2.0) / box_height
        return math.hypot(dx, dy), fraction_inside

    def predict_face(self, frame, box, other_boxes=None):
        """Extract landmarks spatially associated with the requested face.

        Returns None when the crop contains only a nonmatching or ambiguous
        candidate. Optional other_boxes permits comparison with other detected
        faces without changing the standalone single-face interface.
        """
        square_box = self._square_box(frame, box)
        if square_box is None:
            return None

        x1, y1, x2, y2 = square_box
        crop = frame[y1:y2, x1:x2]
        face_landmarks = self.predict(crop)
        if not face_landmarks:
            return None

        frame_height, frame_width = frame.shape[:2]
        crop_width = x2 - x1
        crop_height = y2 - y1
        matches = []

        for landmarks in face_landmarks:
            if not landmarks:
                continue

            try:
                positions = [
                    (x1 + float(point.x) * crop_width,
                     y1 + float(point.y) * crop_height)
                    for point in landmarks
                ]
            except (TypeError, ValueError, AttributeError):
                continue

            if not all(
                math.isfinite(px) and math.isfinite(py)
                for px, py in positions
            ):
                continue

            match = self._match_geometry(positions, box)
            if match is None:
                continue

            target_distance, target_fraction = match
            associated_with_neighbor = False
            for other_box in (other_boxes if other_boxes is not None else ()):
                other_match = self._match_geometry(positions, other_box)
                if other_match is not None:
                    other_distance, _ = other_match
                    if other_distance <= target_distance + 1e-9:
                        associated_with_neighbor = True
                        break

            if not associated_with_neighbor:
                matches.append((target_distance, -target_fraction, landmarks, positions))

        if not matches:
            return None

        matches.sort(key=lambda item: (item[0], item[1]))
        if len(matches) > 1:
            best, next_best = matches[0], matches[1]
            if (abs(best[0] - next_best[0]) <= 1e-9
                    and abs(best[1] - next_best[1]) <= 1e-9):
                return None

        _, _, landmarks, positions = matches[0]
        for point, (px, py) in zip(landmarks, positions):
            point.x = px / frame_width
            point.y = py / frame_height
        return landmarks

    def close(self):
        """Release the MediaPipe landmarker."""
        self.landmarker.close()