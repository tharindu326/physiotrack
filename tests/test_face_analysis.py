
from types import SimpleNamespace

import numpy as np
import pytest

from physiotrack.face.analysis import FaceAnalysis
from physiotrack.face.blink import BlinkDetector
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.mouth_motion import MouthMovement
from physiotrack.results import Instance, Result


class DummyDetector:
    def predict(self, frame):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=None,
                    box=np.array(
                        [10, 10, 50, 50],
                        dtype=float,
                    ),
                    confidence=0.9,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


class DummyTracker:
    def track(self, frame, detection_result):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=1,
                    box=np.array(
                        [10, 10, 50, 50],
                        dtype=float,
                    ),
                    confidence=0.9,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


class DummyOrientation:
    def predict(self, frame, boxes):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    box=np.asarray(
                        boxes[0],
                        dtype=float,
                    ),
                    orientation={
                        "pitch": 1.0,
                        "yaw": 2.0,
                        "roll": 3.0,
                    },
                )
            ],
            task="face",
        )


class DummyLandmarks:
    def predict_face(self, frame, box):
        return [
            SimpleNamespace(
                x=0.5,
                y=0.5,
                z=0.0,
            )
            for _ in range(478)
        ]

    def close(self):
        pass


class DummyQuality:
    def predict(self, frame, faces):
        return [
            {
                "confidence": 0.9,
                "brightness": 0.5,
                "sharpness": 100.0,
                "face_area_ratio": 0.16,
            }
        ]


class DummyEyes:
    def predict(self, landmarks, image_size):
        return {
            "left_openness": 0.3,
            "right_openness": 0.3,
            "mean_openness": 0.3,
        }


class DummyBlink:
    def update(self, openness, person_id=0):
        return {
            "eye_state": "open",
            "blink": False,
            "blink_count": 0,
            "blink_duration": None,
            "blink_rate": 0.0,
        }

    def reset(self):
        pass


class DummyGaze:
    def predict(self, landmarks, image_size):
        return {
            "right_iris_x": 0.5,
            "right_iris_y": 0.0,
            "left_iris_x": 0.5,
            "left_iris_y": 0.0,
            "mean_iris_x": 0.5,
            "mean_iris_y": 0.0,
        }


class DummyGazeEstimator:
    def __init__(self):
        self.initialized = True
        self.was_closed = False
        self.last_boxes = None
        self.last_min_iou = None

    def predict_faces(
        self,
        image,
        boxes,
        min_iou=0.10,
    ):
        self.last_boxes = np.asarray(
            boxes,
            dtype=float,
        ).copy()

        self.last_min_iou = float(
            min_iou
        )

        return [
            {
                "available": True,
                "gaze_vector": [
                    -0.1,
                    0.2,
                    -0.97,
                ],
                "pitch": 11.5,
                "yaw": -5.9,
                "association_iou": 0.85,
            }
            for _ in self.last_boxes
        ]

    def close(self):
        self.was_closed = True
        self.initialized = False


class DummyMouth:
    def predict(self, landmarks, image_size):
        return {
            "mouth_openness": 0.2,
            "mouth_width": 0.4,
            "mouth_height": 0.08,
        }


class DummyMouthMotion:
    def update(self, openness, person_id=0):
        return {
            "mouth_movement": 0.0,
            "mouth_velocity": 0.0,
        }

    def reset(self):
        pass


class DummyEmotion:
    def predict(self, crop):
        return {
            "emotion": "Neutral",
            "confidence": 0.8,
            "scores": {
                "Neutral": 0.8,
            },
        }


class DummyRegions:
    def predict(self, frame, boxes=None):
        mask = np.ones(
            (40, 40),
            dtype=bool,
        )

        return {
            "result": None,
            "faces": [
                {
                    "box": np.array(
                        [10, 10, 50, 50],
                        dtype=int,
                    ),
                    "regions": {
                        "skin": mask,
                    },
                }
            ],
        }


class DummyTemporal:
    def update(self, instance):
        return {
            "person_id": instance.id,
            "window_frames": 1,
            "window_sec": 0.04,
            "head_pose": {
                "yaw": {
                    "mean": 2.0,
                    "std": 0.0,
                    "min": 2.0,
                    "max": 2.0,
                },
                "pitch": {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                },
                "roll": {
                    "mean": 3.0,
                    "std": 0.0,
                    "min": 3.0,
                    "max": 3.0,
                },
            },
            "eyes": {},
            "gaze": {},
            "mouth": {},
            "quality": {},
            "blink": {
                "events": 0,
            },
            "emotion": {
                "dominant": "Neutral",
            },
        }

    def reset(self):
        pass


def make_pipeline(
    config=None,
    gaze_estimation=None,
):
    if gaze_estimation is None:
        gaze_estimation = (
            DummyGazeEstimator()
        )

    return FaceAnalysis(
        detector=DummyDetector(),
        tracker=DummyTracker(),
        orientation=DummyOrientation(),
        landmarks=DummyLandmarks(),
        quality=DummyQuality(),
        eyes=DummyEyes(),
        blink=DummyBlink(),
        gaze=DummyGaze(),
        gaze_estimation=gaze_estimation,
        mouth=DummyMouth(),
        mouth_motion=DummyMouthMotion(),
        emotion=DummyEmotion(),
        regions=DummyRegions(),
        temporal=DummyTemporal(),
        config=config,
        fps=25,
        device="cpu",
    )


def test_analysis_returns_structured_face_instance():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    instance = result[0]

    assert instance.id == 1

    assert instance.orientation == {
        "pitch": 1.0,
        "yaw": 2.0,
        "roll": 3.0,
    }

    assert instance.face_features is not None

    assert set(
        instance.face_features.keys()
    ) == {
        "landmarks",
        "quality",
        "eyes",
        "blink",
        "gaze",
        "mouth",
        "mouth_motion",
        "emotion",
        "regions",
        "temporal",
    }


def test_analysis_preserves_feature_values():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    features = result[0].face_features

    assert features["landmarks"]["available"] is True
    assert features["landmarks"]["count"] == 478

    assert features["quality"]["available"] is True

    assert features["eyes"]["available"] is True
    assert features["eyes"]["mean_openness"] == 0.3

    assert features["blink"]["available"] is True

    assert features["gaze"]["available"] is True
    assert features["gaze"]["mean_iris_x"] == 0.5

    assert features["mouth"]["available"] is True
    assert features["mouth"]["mouth_openness"] == 0.2

    assert features["mouth_motion"]["available"] is True

    assert features["emotion"]["available"] is True
    assert features["emotion"]["emotion"] == "Neutral"

    assert features["regions"]["available"] is True
    assert features["regions"]["skin_pixel_count"] == 1600

    assert features["temporal"]["available"] is True

    assert (
        features["temporal"]["summary"]["person_id"]
        == 1
    )


def test_analysis_returns_empty_result_when_tracker_has_no_faces():
    class EmptyTracker:
        def track(self, frame, detection_result):
            return Result(
                orig_img=frame,
                instances=[],
                task="face",
            )

    pipeline = make_pipeline()

    pipeline.tracker = EmptyTracker()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 0
    assert result.task == "face"


def test_reset_temporal_state_calls_temporal_components():
    class FlagBlink(DummyBlink):
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    class FlagMouth(DummyMouthMotion):
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    class FlagTemporal(DummyTemporal):
        def __init__(self):
            self.was_reset = False

        def reset(self, person_id=None):
            self.was_reset = True

    pipeline = make_pipeline()

    pipeline.blink = FlagBlink()
    pipeline.mouth_motion = FlagMouth()
    pipeline.temporal = FlagTemporal()

    pipeline.reset_temporal_state()

    assert pipeline.blink.was_reset is True
    assert pipeline.mouth_motion.was_reset is True
    assert pipeline.temporal.was_reset is True


def test_config_can_disable_optional_modules():
    config = FaceAnalysisConfig(
        emotion=False,
        regions=False,
    )

    pipeline = make_pipeline(
        config=config
    )

    assert pipeline.emotion is None
    assert pipeline.regions is None

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    features = result[0].face_features

    assert features["emotion"]["available"] is False
    assert features["emotion"]["emotion"] is None

    assert features["regions"]["available"] is False
    assert features["regions"]["classes"] == []


def test_config_can_disable_landmark_dependent_modules():
    config = FaceAnalysisConfig(
        landmarks=False,
        eyes=False,
        blink=False,
        gaze=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    pipeline = make_pipeline(
        config=config
    )

    assert pipeline.landmarks is None
    assert pipeline.eyes is None
    assert pipeline.blink is None
    assert pipeline.gaze is None
    assert pipeline.mouth is None
    assert pipeline.mouth_motion is None

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    features = result[0].face_features

    assert features["landmarks"]["available"] is False
    assert features["landmarks"]["count"] == 0

    assert features["eyes"]["available"] is False
    assert features["blink"]["available"] is False
    assert features["gaze"]["available"] is False
    assert features["mouth"]["available"] is False
    assert features["mouth_motion"]["available"] is False
    assert features["emotion"]["available"] is False
    assert features["regions"]["available"] is False
    assert features["temporal"]["available"] is False


def test_gaze_estimation_is_absent_when_disabled():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    features = result[
        0
    ].face_features

    assert (
        "gaze_estimation"
        not in features
    )

    assert (
        pipeline.gaze_estimation
        is None
    )


def test_gaze_estimation_adds_structured_features_when_enabled():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    assert len(result) == 1

    features = result[
        0
    ].face_features

    assert (
        "gaze_estimation"
        in features
    )

    gaze_estimation = features[
        "gaze_estimation"
    ]

    assert (
        gaze_estimation[
            "available"
        ]
        is True
    )

    assert gaze_estimation[
        "gaze_vector"
    ] == [
        -0.1,
        0.2,
        -0.97,
    ]

    assert (
        gaze_estimation[
            "pitch"
        ]
        == 11.5
    )

    assert (
        gaze_estimation[
            "yaw"
        ]
        == -5.9
    )

    assert (
        gaze_estimation[
            "association_iou"
        ]
        == 0.85
    )


def test_gaze_estimation_receives_pipeline_face_boxes():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    pipeline.predict(
        frame
    )

    np.testing.assert_allclose(
        dummy_gaze_estimator.last_boxes,
        np.array(
            [
                [
                    10.0,
                    10.0,
                    50.0,
                    50.0,
                ]
            ]
        ),
    )

    assert (
        dummy_gaze_estimator.last_min_iou
        == pytest.approx(
            config.gaze_estimation_min_iou
        )
    )


def test_gaze_estimation_does_not_require_landmarks():
    config = FaceAnalysisConfig(
        landmarks=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=True,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    assert pipeline.landmarks is None
    assert pipeline.gaze is None

    assert (
        pipeline.gaze_estimation
        is dummy_gaze_estimator
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    features = result[
        0
    ].face_features

    assert (
        features[
            "landmarks"
        ][
            "available"
        ]
        is False
    )

    assert (
        features[
            "gaze"
        ][
            "available"
        ]
        is False
    )

    assert (
        features[
            "gaze_estimation"
        ][
            "available"
        ]
        is True
    )


def test_close_releases_gaze_estimator():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    assert (
        dummy_gaze_estimator.was_closed
        is False
    )

    pipeline.close()

    assert (
        dummy_gaze_estimator.was_closed
        is True
    )

    assert (
        dummy_gaze_estimator.initialized
        is False
    )


def test_analysis_rejects_invalid_config_type():
    with pytest.raises(
        TypeError,
        match="config must be a FaceAnalysisConfig instance",
    ):
        FaceAnalysis(
            detector=DummyDetector(),
            config={},
            fps=25,
        )

def test_analysis_default_blink_parameters_match_config():
    pipeline = FaceAnalysis(
        detector=DummyDetector(),
        tracker=DummyTracker(),
        orientation=DummyOrientation(),
        landmarks=DummyLandmarks(),
        quality=DummyQuality(),
        eyes=DummyEyes(),
        blink=None,
        gaze=DummyGaze(),
        mouth=DummyMouth(),
        mouth_motion=DummyMouthMotion(),
        emotion=DummyEmotion(),
        regions=DummyRegions(),
        temporal=DummyTemporal(),
        fps=25,
        device="cpu",
    )

    assert pipeline.config.blink_threshold == pytest.approx(
        0.22
    )
    assert pipeline.config.min_closed_frames == 3

    assert pipeline.blink.threshold == pytest.approx(
        0.22
    )
    assert pipeline.blink.min_closed_frames == 3

def test_missing_tracked_frame_breaks_temporal_continuity():
    class GapTracker:
        def __init__(self):
            self.calls = 0

        def track(self, frame, detection_result):
            self.calls += 1

            if self.calls == 2:
                return Result(
                    orig_img=frame,
                    instances=[],
                    task="face",
                )

            return Result(
                orig_img=frame,
                instances=[
                    Instance(
                        id=1,
                        box=np.array(
                            [10, 10, 50, 50],
                            dtype=float,
                        ),
                        confidence=0.9,
                        cls=0,
                        cls_name="face",
                    )
                ],
                task="face",
            )

    class SequenceMouth(DummyMouth):
        def __init__(self):
            self.values = iter(
                [
                    0.1,
                    0.4,
                ]
            )

        def predict(self, landmarks, image_size=None):
            openness = next(
                self.values
            )

            return {
                "mouth_openness": openness,
                "mouth_width": 0.4,
                "mouth_height": (
                    openness * 0.4
                ),
            }

    class RecordingTemporal(DummyTemporal):
        def __init__(self):
            self.reset_person_ids = []

        def reset(self, person_id=None):
            self.reset_person_ids.append(
                person_id
            )

    blink = BlinkDetector(
        threshold=0.22,
        fps=25,
        min_closed_frames=1,
    )

    # Establish one completed blink before the tracked-frame gap.
    blink.update(
        0.1,
        person_id=1,
        frame_index=0,
    )
    completed = blink.update(
        0.3,
        person_id=1,
        frame_index=1,
    )

    assert completed["blink_count"] == 1

    mouth_motion = MouthMovement(
        fps=25
    )

    temporal = RecordingTemporal()

    pipeline = make_pipeline()

    pipeline.tracker = GapTracker()
    pipeline.blink = blink
    pipeline._frame_index = 2  # Preseeded blink samples used video frames 0-1.
    pipeline.mouth = SequenceMouth()
    pipeline.mouth_motion = mouth_motion
    pipeline.temporal = temporal

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    first = pipeline.predict(
        frame
    )

    missing = pipeline.predict(
        frame
    )

    returned = pipeline.predict(
        frame
    )

    assert len(first) == 1
    assert len(missing) == 0
    assert len(returned) == 1

    returned_features = (
        returned[0].face_features
    )

    # The gap must not create a blink or erase the prior cumulative count.
    assert returned_features[
        "blink"
    ][
        "blink"
    ] is False

    assert returned_features[
        "blink"
    ][
        "blink_count"
    ] == 1

    # Mouth movement after the gap must start a new contiguous segment.
    assert returned_features[
        "mouth_motion"
    ][
        "mouth_movement"
    ] == 0.0

    assert returned_features[
        "mouth_motion"
    ][
        "mouth_velocity"
    ] == 0.0

    # Temporal aggregation must not bridge the unobserved frame.
    assert temporal.reset_person_ids == [
        1
    ]


def test_face_analysis_emotion_rgb_integration_b01():
    """B01: verify RGB conversion through the real FaceAnalysis path."""

    from physiotrack.face.emotion import FaceEmotion

    class RecordingRecognizer:
        idx_to_emotion_class = {
            0: "Neutral",
        }

        def predict_emotions(self, face_image, logits=False):
            self.received_image = face_image.copy()
            self.received_logits = logits

            return (
                ["Neutral"],
                np.array(
                    [[1.0]],
                    dtype=float,
                ),
            )

    recognizer = RecordingRecognizer()

    emotion = FaceEmotion.__new__(FaceEmotion)
    emotion.recognizer = recognizer

    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=True,
        regions=False,
        temporal=False,
    )

    pipeline = make_pipeline(
        config=config,
    )

    pipeline.emotion = emotion

    # DummyDetector supplies a face box at [10, 10, 50, 50].
    frame_bgr = np.zeros(
        (80, 80, 3),
        dtype=np.uint8,
    )

    frame_bgr[10:50, 10:50] = [
        10,
        20,
        200,
    ]

    original_frame = frame_bgr.copy()

    result = pipeline.predict(
        frame_bgr
    )

    assert len(result) == 1

    features = result[0].face_features

    assert features["emotion"]["available"] is True
    assert features["emotion"]["emotion"] == "Neutral"
    assert features["emotion"]["confidence"] == 1.0

    assert features["emotion"]["scores"] == {
        "Neutral": 1.0,
    }

    # The detector crop must be 40 x 40 pixels.
    assert recognizer.received_image.shape == (
        40,
        40,
        3,
    )

    # The recognizer must receive RGB, not BGR.
    np.testing.assert_array_equal(
        recognizer.received_image[0, 0],
        np.array(
            [200, 20, 10],
            dtype=np.uint8,
        ),
    )

    assert recognizer.received_logits is False

    # Neither FaceEmotion nor the pipeline may modify the input frame.
    np.testing.assert_array_equal(
        frame_bgr,
        original_frame,
    )


def test_b05_face_analysis_uses_video_time_across_absent_person_frames():
    class TimelineTracker:
        def __init__(self):
            self.index = -1

        def track(self, frame, detection_result):
            self.index += 1
            visible_ids = (
                [1, 2] if self.index in (0, 1)
                else [2] if self.index in (2, 3, 4, 5, 6)
                else [1, 2]
            )
            return Result(
                orig_img=frame,
                instances=[
                    Instance(
                        id=person_id,
                        box=np.array([10, 10, 50, 50], dtype=float),
                        confidence=0.9,
                        cls=0,
                        cls_name="face",
                    )
                    for person_id in visible_ids
                ],
                task="face",
            )

    class TimelineEyes:
        def __init__(self):
            self.openness = 0.3

        def predict(self, landmarks, image_size):
            return {
                "left_openness": self.openness,
                "right_openness": self.openness,
                "mean_openness": self.openness,
            }

    config = FaceAnalysisConfig(
        tracking=True,
        head_pose=False,
        landmarks=True,
        quality=False,
        eyes=True,
        blink=True,
        min_closed_frames=1,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )
    eyes = TimelineEyes()
    pipeline = FaceAnalysis(
        detector=DummyDetector(),
        tracker=TimelineTracker(),
        landmarks=DummyLandmarks(),
        eyes=eyes,
        config=config,
        fps=10,
        min_closed_frames=1,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Person 1 blinks on frame 1 and is absent during frames 2 through 6.
    eyes.openness = 0.1
    pipeline.predict(frame)
    eyes.openness = 0.3
    completed = pipeline.predict(frame)
    assert completed[0].face_features["blink"]["blink_count"] == 1
    for _ in range(5):
        missing = pipeline.predict(frame)
        assert [face.id for face in missing] == [2]
    returned = pipeline.predict(frame)
    by_id = {face.id: face.face_features["blink"] for face in returned}
    assert by_id[1]["blink_count"] == 1
    assert by_id[1]["blink_rate"] == pytest.approx(75.0)  # 8 frames / 10 FPS
    assert by_id[2]["blink_count"] == 1
    assert by_id[2]["blink_rate"] == pytest.approx(75.0)
    assert pipeline._frame_index == 8
    pipeline.reset_temporal_state()
    assert pipeline._frame_index == 0
    restarted = pipeline.predict(frame)
    for face in restarted:
        assert face.face_features["blink"]["blink_count"] == 0
        assert face.face_features["blink"]["blink_rate"] == 0.0


def test_b02_multiple_untracked_faces_have_no_shared_temporal_state():
    class TwoUntrackedFaces:
        def predict(self, frame):
            return Result(
                orig_img=frame,
                instances=[
                    Instance(
                        id=None,
                        box=np.array(box, dtype=float),
                        confidence=0.9,
                        cls=0,
                        cls_name="face",
                    )
                    for box in ([5, 5, 40, 40], [50, 5, 90, 40])
                ],
                task="face",
            )

    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=True,
        quality=False, eyes=True, blink=True, gaze=False,
        gaze_estimation=False, mouth=True, mouth_motion=True,
        emotion=False, regions=False, temporal=False,
    )
    pipeline = FaceAnalysis(
        detector=TwoUntrackedFaces(), landmarks=DummyLandmarks(),
        eyes=DummyEyes(), mouth=DummyMouth(), config=config, fps=25,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(3):
        result = pipeline.predict(frame)
        assert len(result) == 2
        for face in result:
            assert face.id is None
            assert face.face_features["eyes"]["available"] is True
            assert face.face_features["mouth"]["available"] is True
            assert face.face_features["blink"]["available"] is False
            assert face.face_features["blink"]["blink_rate"] is None
            assert face.face_features["mouth_motion"]["available"] is False
            assert face.face_features["mouth_motion"]["mouth_velocity"] is None
    assert pipeline.blink._states == {}
    assert pipeline.mouth_motion._states == {}


def test_b03_face_analysis_associates_independent_regions_for_overlapping_faces():
    """Full FaceAnalysis path must associate uncrossed counts by face box."""
    from physiotrack.face.regions import FaceRegions
    from physiotrack.segment import Segmentation

    boxes = ([10, 10, 70, 70], [40, 40, 90, 90])

    class OverlapDetector:
        def __init__(self, ordered_boxes):
            self.ordered_boxes = ordered_boxes

        def predict(self, frame):
            return Result(
                orig_img=frame,
                task="face",
                instances=[Instance(id=None, box=np.array(box, dtype=float),
                                    confidence=0.9, cls=0, cls_name="face")
                           for box in self.ordered_boxes],
            )

    class ControlledSegFace:
        def infer(self, crop):
            return np.full(crop.shape[:2], int(crop[0, 0, 0]), dtype=np.int32)

    segmenter = Segmentation.Face.__new__(Segmentation.Face)
    segmenter.segmentor = ControlledSegFace()
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=True,
        temporal=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10, 10, 0] = 2
    frame[40, 40, 0] = 10
    expected = {
        tuple(boxes[0]): {"skin": 60 * 60},
        tuple(boxes[1]): {"nose": 50 * 50},
    }

    for order in (boxes, list(reversed(boxes))):
        pipeline = FaceAnalysis(detector=OverlapDetector(order),
                                regions=FaceRegions(segmenter=segmenter),
                                config=config, device="cpu")
        result = pipeline.predict(frame)
        assert len(result) == 2
        for instance in result:
            features = instance.face_features["regions"]
            assert features["available"] is True
            assert features["pixel_counts"] == expected[tuple(int(v) for v in instance.box)]
            assert features["association_iou"] == pytest.approx(1.0)
            assert instance.id is None


def test_face_analysis_uses_landmark_face_association_without_changing_injected_api():
    """Check per-face association through the real pipeline and both box orders."""
    from physiotrack.face.landmarks import FaceLandmarks

    target_box = [40, 10, 60, 90]
    neighbor_box = [10, 20, 35, 50]

    class TwoFaces:
        def __init__(self, boxes):
            self.boxes = boxes

        def predict(self, frame):
            return Result(
                orig_img=frame, task="face", instances=[
                    Instance(id=None, box=np.array(box, dtype=float),
                             confidence=0.9, cls=0, cls_name="face")
                    for box in self.boxes
                ],
            )

    native = FaceLandmarks.__new__(FaceLandmarks)
    observed_crops = []

    def controlled_backend(crop):
        observed_crops.append(crop.shape)
        # A face detected near the left edge of the square crop.
        return [[SimpleNamespace(
            x=(22.5 + (index % 5 - 2) * 0.1 - 10) / 80,
            y=(35 + (index % 7 - 3) * 0.1 - 10) / 80,
            z=0.0,
        ) for index in range(478)]]

    native.predict = controlled_backend
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=True, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for order in ([target_box, neighbor_box], [neighbor_box, target_box]):
        pipeline = FaceAnalysis(
            detector=TwoFaces(order), landmarks=native,
            config=config, device="cpu",
        )
        result = pipeline.predict(frame)
        assert len(result) == 2
        by_box = {
            tuple(int(value) for value in face.box): face.face_features
            for face in result
        }
        assert by_box[tuple(target_box)]["landmarks"] == {
            "available": False, "count": 0,
        }
        assert by_box[tuple(neighbor_box)]["landmarks"] == {
            "available": True, "count": 478,
        }
    assert observed_crops

    # Existing two-argument injected tools must remain supported.
    pipeline = FaceAnalysis(
        detector=TwoFaces([target_box, neighbor_box]),
        landmarks=DummyLandmarks(), config=config, device="cpu",
    )
    result = pipeline.predict(frame)
    assert all(face.face_features["landmarks"]["available"] for face in result)


def test_head_pose_association_preserves_box_indices_when_first_crop_is_skipped():
    """A skipped first crop must not receive the second face's pose."""
    class ControlledDetector:
        def predict(self, frame):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=101, box=np.array([10, 10, 10, 10], dtype=float),
                         confidence=0.9, cls=0, cls_name="face"),
                Instance(id=202, box=np.array([50, 10, 80, 40], dtype=float),
                         confidence=0.9, cls=0, cls_name="face"),
            ])

    class SkippingOrientation:
        def predict(self, frame, boxes):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=1, box=np.asarray(boxes[1], dtype=int),
                         orientation={"yaw": 22., "pitch": 12., "roll": 2.}),
            ])

    config = FaceAnalysisConfig(
        tracking=False, head_pose=True, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )
    pipeline = FaceAnalysis(detector=ControlledDetector(),
                            orientation=SkippingOrientation(), config=config)
    result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert [face.id for face in result] == [101, 202]
    assert result[0].orientation is None
    assert result[1].orientation == {"yaw": 22., "pitch": 12., "roll": 2.}


def test_head_pose_association_keeps_identity_with_reordered_outputs_and_boxes():
    """An explicit backend index is independent of result-list and track order."""
    boxes = [[5, 5, 30, 40], [55, 10, 90, 60]]

    class Detector:
        def predict(self, frame):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=100 + i, box=np.asarray(box, dtype=float),
                         confidence=0.9, cls=0, cls_name="face")
                for i, box in enumerate(boxes)
            ])

    class ReorderedOrientation:
        def predict(self, frame, requested_boxes):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=i, box=np.asarray(requested_boxes[i], dtype=int),
                         orientation={"yaw": float(i + 1), "pitch": 0., "roll": 0.})
                for i in (1, 0)
            ])

    config = FaceAnalysisConfig(
        tracking=False, head_pose=True, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )
    pipeline = FaceAnalysis(detector=Detector(),
                            orientation=ReorderedOrientation(), config=config)
    result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert [instance.orientation["yaw"] for instance in result] == [1., 2.]


def test_head_pose_association_idless_injected_results_use_unique_boxes_only():
    """Injected orientation modules can omit backend indices without misrouting."""
    config = FaceAnalysisConfig(
        tracking=False, head_pose=True, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )

    class Detector:
        def predict(self, frame):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=None, box=np.array(box, dtype=float),
                         confidence=0.9, cls=0, cls_name="face")
                for box in [[5, 5, 40, 40], [55, 5, 90, 40]]
            ])

    class InjectedOrientation:
        def predict(self, frame, boxes):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=None, box=np.asarray(boxes[i], dtype=float),
                         orientation={"yaw": float(i), "pitch": 0., "roll": 0.})
                for i in (1, 0)
            ])

    pipeline = FaceAnalysis(detector=Detector(),
                            orientation=InjectedOrientation(), config=config)
    result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert [face.orientation["yaw"] for face in result] == [0., 1.]

    # A contradictory explicit index/box pair must not be assigned by position.
    class ContradictoryOrientation(InjectedOrientation):
        def predict(self, frame, boxes):
            return Result(orig_img=frame, task="face", instances=[
                Instance(id=0, box=np.asarray(boxes[1], dtype=float),
                         orientation={"yaw": 99., "pitch": 0., "roll": 0.}),
            ])

    pipeline.orientation = ContradictoryOrientation()
    result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert all(face.orientation is None for face in result)

    assert not FaceAnalysis._orientation_boxes_match(
        [10, 10, 10, 30], [10, 10, 10, 30]
    )
    assert not FaceAnalysis._orientation_boxes_match(
        [float("nan"), 0, 20, 20], [0, 0, 20, 20]
    )


def test_eye_and_blink_availability_requires_real_eye_measurements():
    """A face with landmarks can still have unmeasurable eye geometry."""
    from physiotrack.face.eyes import EyeOpenness

    pipeline = make_pipeline()
    pipeline.eyes = EyeOpenness()
    pipeline.blink = BlinkDetector(threshold=0.22, fps=25, min_closed_frames=1)

    # DummyLandmarks places every landmark at (0.5, 0.5): both eye widths
    # are zero, so EyeOpenness correctly returns three missing ratios.
    features = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))[0].face_features
    assert features["landmarks"]["available"] is True
    assert features["eyes"] == {
        "left_openness": None,
        "right_openness": None,
        "mean_openness": None,
        "available": False,
    }
    assert features["blink"]["available"] is False
    assert features["blink"]["eye_state"] == "unknown"
    assert features["blink"]["blink"] is False
    assert pipeline.blink._states[1]["closed_frames"] == 0


@pytest.mark.parametrize("measurements", [
    {"left_openness": None, "right_openness": 0.3, "mean_openness": None},
    {"left_openness": 0.3, "right_openness": 0.3},
    {"left_openness": 0.3, "right_openness": 0.3, "mean_openness": float("nan")},
    {"left_openness": float("inf"), "right_openness": 0.3, "mean_openness": 0.3},
    {"left_openness": True, "right_openness": 0.3, "mean_openness": 0.3},
])
def test_missing_or_nonfinite_eye_values_are_not_available(measurements):
    class InjectedEyes:
        def predict(self, landmarks, image_size):
            return dict(measurements)

    pipeline = make_pipeline()
    pipeline.eyes = InjectedEyes()
    pipeline.blink = BlinkDetector(threshold=0.22, fps=25, min_closed_frames=1)
    features = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))[0].face_features
    assert features["eyes"]["available"] is False
    for key in ("left_openness", "right_openness", "mean_openness"):
        value = features["eyes"][key]
        assert value is None or np.isfinite(value)
    assert features["blink"]["available"] is False
    assert features["blink"]["eye_state"] == "unknown"
    assert features["blink"]["blink"] is False


def test_missing_eye_sample_interrupts_blink_closure_without_erasing_history():
    class SequentialEyes:
        values = (0.1, None, 0.5, 0.0)
        index = 0

        def predict(self, landmarks, image_size):
            value = self.values[self.index]
            self.index += 1
            return {
                "left_openness": value,
                "right_openness": value,
                "mean_openness": value,
            }

    pipeline = make_pipeline()
    pipeline.eyes = SequentialEyes()
    pipeline.blink = BlinkDetector(threshold=0.22, fps=25, min_closed_frames=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = pipeline.predict(frame)[0].face_features
    missing = pipeline.predict(frame)[0].face_features
    opened = pipeline.predict(frame)[0].face_features
    closed_zero = pipeline.predict(frame)[0].face_features

    assert first["blink"]["eye_state"] == "closed"
    assert missing["eyes"]["available"] is False
    assert missing["blink"]["available"] is False
    assert missing["blink"]["eye_state"] == "unknown"
    assert opened["eyes"]["available"] is True
    assert opened["blink"]["available"] is True
    assert opened["blink"]["blink"] is False
    assert opened["blink"]["blink_count"] == 0
    assert closed_zero["eyes"]["available"] is True
    assert closed_zero["eyes"]["mean_openness"] == 0.0
    assert closed_zero["blink"]["eye_state"] == "closed"
