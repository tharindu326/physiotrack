import cv2

import physiotrack as pt
from physiotrack.face import FaceAnalysisConfig


def main():
    video_path = "input.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    config = FaceAnalysisConfig(
        emotion=True,
        regions=True,
        temporal=True,
    )

    pipeline = pt.FaceAnalysis(
        fps=fps,
        config=config,
        device="cpu",
    )

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            result = pipeline.predict(frame)

            for face in result:
                features = face.face_features

                print(
                    "frame:",
                    frame_index,
                    "person_id:",
                    face.id,
                    "emotion:",
                    features["emotion"]["emotion"],
                    "eye_openness:",
                    features["eyes"]["mean_openness"],
                    "mouth_openness:",
                    features["mouth"]["mouth_openness"],
                )

            frame_index += 1

    finally:
        cap.release()
        pipeline.close()


if __name__ == "__main__":
    main()