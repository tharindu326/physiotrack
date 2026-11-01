
import numpy as np

from physiotrack.face.emotion import FaceEmotion
from physiotrack.face.face_orientation import FaceOrientation
from physiotrack.face.regions import FaceRegions
from physiotrack.results import Instance, Result


class DummyEmotionRecognizer:
    idx_to_emotion_class = {
        0: "Anger",
        1: "Contempt",
        2: "Disgust",
        3: "Fear",
        4: "Happiness",
        5: "Neutral",
        6: "Sadness",
        7: "Surprise",
    }

    def predict_emotions(self, face_image, logits=False):
        return (
            ["Happiness"],
            np.array(
                [[
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.80,
                    0.05,
                    0.03,
                    0.02,
                ]],
                dtype=float,
            ),
        )


def test_emotion_predict_formats_output():
    emotion = FaceEmotion.__new__(FaceEmotion)
    emotion.recognizer = DummyEmotionRecognizer()

    face = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    result = emotion.predict(face)

    assert result["emotion"] == "Happiness"

    assert np.isclose(
        result["confidence"],
        0.80,
    )

    assert set(
        result["scores"].keys()
    ) == {
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    }


class DummySegmenter:
    def predict(self, frame, boxes=None):
        seg_map = np.zeros(
            (100, 100),
            dtype=np.int32,
        )

        seg_map[20:60, 20:60] = 1
        seg_map[30:40, 30:40] = 2

        instances = [
            Instance(
                box=np.array(
                    [20, 20, 60, 60],
                    dtype=float,
                )
            )
        ]

        result = Result(
            orig_img=frame,
            instances=instances,
            task="segmentation",
        )

        result.seg_map = seg_map

        result.names = {
            0: "background",
            1: "skin",
            2: "l_eye",
        }

        return result


def test_regions_extracts_per_face_masks():
    regions = FaceRegions(
        segmenter=DummySegmenter()
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    output = regions.predict(
        frame,
        boxes=[
            [20, 20, 60, 60]
        ],
    )

    assert len(output["faces"]) == 1

    face = output["faces"][0]

    assert "skin" in face["regions"]
    assert "l_eye" in face["regions"]

    assert face["regions"]["skin"].shape == (
        40,
        40,
    )

    assert face["regions"]["l_eye"].shape == (
        40,
        40,
    )


def test_regions_returns_empty_faces_without_segmentation_map():
    class NoMapSegmenter:
        def predict(self, frame, boxes=None):
            result = Result(
                orig_img=frame,
                instances=[],
                task="segmentation",
            )

            result.seg_map = None
            result.names = None

            return result

    regions = FaceRegions(
        segmenter=NoMapSegmenter()
    )

    frame = np.zeros(
        (50, 50, 3),
        dtype=np.uint8,
    )

    output = regions.predict(frame)

    assert output["faces"] == []


class DummyOrientationModel:
    def predict(self, source, boxes=None):
        instances = []

        for box in boxes:
            instances.append(
                Instance(
                    box=np.asarray(
                        box,
                        dtype=float,
                    ),
                    orientation={
                        "pitch": 1.0,
                        "yaw": 2.0,
                        "roll": 3.0,
                    },
                )
            )

        return Result(
            orig_img=source,
            instances=instances,
            task="face",
        )


def test_orientation_result_preserves_pose_values():
    orientation = FaceOrientation.__new__(
        FaceOrientation
    )

    orientation.model = DummyOrientationModel()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = orientation.model.predict(
        frame,
        boxes=[
            [10, 10, 40, 40],
            [50, 50, 90, 90],
        ],
    )

    assert len(result) == 2

    for instance in result:
        assert instance.orientation == {
            "pitch": 1.0,
            "yaw": 2.0,
            "roll": 3.0,
        }


def test_emotion_predict_converts_bgr_to_rgb_without_mutation():
    """B01: verify RGB conversion and preservation of source pixels."""

    class RecordingRecognizer(DummyEmotionRecognizer):
        def predict_emotions(self, face_image, logits=False):
            self.received_image = face_image.copy()
            self.received_logits = logits

            return super().predict_emotions(
                face_image,
                logits=logits,
            )

    recognizer = RecordingRecognizer()

    emotion = FaceEmotion.__new__(FaceEmotion)
    emotion.recognizer = recognizer

    face_bgr = np.array(
        [
            [
                [10, 20, 200],
                [0, 100, 255],
            ]
        ],
        dtype=np.uint8,
    )

    original_image = face_bgr.copy()

    result = emotion.predict(face_bgr)

    expected_rgb = np.array(
        [
            [
                [200, 20, 10],
                [255, 100, 0],
            ]
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(
        recognizer.received_image,
        expected_rgb,
    )

    np.testing.assert_array_equal(
        face_bgr,
        original_image,
    )

    assert recognizer.received_logits is False
    assert result["emotion"] == "Happiness"

    assert np.isclose(
        result["confidence"],
        0.80,
    )


def test_b03_overlapping_face_regions_use_independent_crop_predictions():
    """Face masks must not inherit another face's labels in shared pixels."""
    from physiotrack.segment import Segmentation

    class ControlledSegFace:
        def infer(self, crop):
            label = int(crop[0, 0, 0])
            parsing = np.full(crop.shape[:2], label, dtype=np.int32)
            parsing[0, 1] = 0
            return parsing

    segmenter = Segmentation.Face.__new__(Segmentation.Face)
    segmenter.segmentor = ControlledSegFace()
    regions = FaceRegions(segmenter=segmenter)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box_a = [10, 10, 70, 70]
    box_b = [40, 40, 90, 90]
    frame[10, 10, 0] = 2   # skin
    frame[40, 40, 0] = 10  # nose (within A's box)

    expected = {
        tuple(box_a): ("skin", (60, 60), 60 * 60 - 1),
        tuple(box_b): ("nose", (50, 50), 50 * 50 - 1),
    }
    for boxes in ([box_a, box_b], [box_b, box_a]):
        output = regions.predict(frame, boxes=boxes)
        result = output["result"]
        assert result.seg_map.shape == (100, 100)
        assert len(result.face_seg_maps) == len(result) == 2
        assert len(output["faces"]) == 2
        for face in output["faces"]:
            name, shape, count = expected[tuple(face["box"])]
            assert list(face["regions"]) == [name]
            assert face["regions"][name].shape == shape
            assert int(face["regions"][name].sum()) == count


def test_b03_face_regions_cropped_boxes_and_native_map_alignment():
    from physiotrack.segment import Segmentation

    class ControlledSegFace:
        def infer(self, crop):
            return np.full(crop.shape[:2], 2, dtype=np.int32)

    segmenter = Segmentation.Face.__new__(Segmentation.Face)
    segmenter.segmentor = ControlledSegFace()
    regions = FaceRegions(segmenter=segmenter)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    output = regions.predict(frame, boxes=[[-5, 0, 25, 30],
                                                [10, 10, 45, 45],
                                                [10, 10, 10, 20]])
    assert len(output["result"]) == 2
    assert len(output["result"].face_seg_maps) == 2
    assert [tuple(face["box"]) for face in output["faces"]] == [
        (0, 0, 25, 30), (10, 10, 40, 40)
    ]
    assert [int(face["regions"]["skin"].sum()) for face in output["faces"]] == [
        25 * 30, 30 * 30
    ]


def test_head_pose_backend_keeps_original_index_after_skipping_invalid_crop():
    """6DRepNet360 labels a retained prediction with its original input index."""
    from collections import deque
    import torch
    from physiotrack.modules._6DRepNet360.inference import HeadPoseEstimator

    estimator = HeadPoseEstimator.__new__(HeadPoseEstimator)
    estimator.render_pose = False
    estimator.inference_times = deque(maxlen=100)
    estimator._inference_batch = lambda crops: [
        torch.eye(3, dtype=torch.float32) for _ in crops
    ]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, backend_result = estimator.predict(
        frame, [[10, 10, 10, 10], [50, 10, 80, 40]],
    )
    assert [pose["id"] for pose in backend_result["detections"]] == [1]
    assert backend_result["detections"][0]["bbox"] == [50, 10, 80, 40]

    converted = FaceOrientation._to_result(frame, backend_result)
    assert len(converted) == 1
    assert converted[0].id == 1
    np.testing.assert_array_equal(converted[0].box, [50, 10, 80, 40])
    assert all(k in converted[0].orientation for k in ("yaw", "pitch", "roll"))
