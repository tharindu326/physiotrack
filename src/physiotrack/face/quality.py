import cv2


class FaceQuality:
    """Estimate basic quality indicators for detected faces."""

    def predict(self, frame, face_result):
        """Calculate quality indicators for each detected face."""
        height, width = frame.shape[:2]
        frame_area = height * width

        results = []

        for face in face_result:
            if face.box is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in face.box]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            brightness = float(gray.mean() / 255.0)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            face_area = (x2 - x1) * (y2 - y1)
            face_area_ratio = float(face_area / frame_area)

            confidence = (
                float(face.confidence)
                if face.confidence is not None
                else None
            )

            results.append(
                {
                    "confidence": confidence,
                    "brightness": brightness,
                    "sharpness": sharpness,
                    "face_area_ratio": face_area_ratio,
                }
            )

        return results