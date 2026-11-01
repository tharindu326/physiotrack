from types import SimpleNamespace

import numpy as np

from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.face.tracking import FaceTracker
from physiotrack.results import Instance, Result


class DummyTracker:
    def __init__(self):
        self.calls = []

    def track(self, frame, detections):
        self.calls.append(
            {
                "frame": frame,
                "detections": detections,
            }
        )

        return "tracked"


def test_face_tracker_converts_result_to_detection_array():
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.tracker = DummyTracker()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    face_result = Result(
        orig_img=frame,
        instances=[
            Instance(
                box=np.array(
                    [10, 20, 30, 40],
                    dtype=float,
                ),
                confidence=0.8,
                cls=0,
            ),
            Instance(
                box=np.array(
                    [50, 60, 80, 90],
                    dtype=float,
                ),
                confidence=0.9,
                cls=0,
            ),
        ],
        task="face",
    )

    result = tracker.track(
        frame,
        face_result,
    )

    assert result == "tracked"
    assert len(tracker.tracker.calls) == 1

    detections = tracker.tracker.calls[0]["detections"]

    assert detections.shape == (2, 6)

    assert np.allclose(
        detections[0],
        [10, 20, 30, 40, 0.8, 0],
    )

    assert np.allclose(
        detections[1],
        [50, 60, 80, 90, 0.9, 0],
    )


def test_face_tracker_handles_empty_detection_result():
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.tracker = DummyTracker()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    face_result = Result(
        orig_img=frame,
        instances=[],
        task="face",
    )

    tracker.track(
        frame,
        face_result,
    )

    detections = tracker.tracker.calls[0]["detections"]

    assert detections.shape == (0, 6)


def test_face_tracker_uses_defaults_for_missing_confidence_and_class():
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.tracker = DummyTracker()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    face_result = Result(
        orig_img=frame,
        instances=[
            Instance(
                box=np.array(
                    [10, 20, 30, 40],
                    dtype=float,
                ),
                confidence=None,
                cls=None,
            )
        ],
        task="face",
    )

    tracker.track(
        frame,
        face_result,
    )

    detections = tracker.tracker.calls[0]["detections"]

    assert np.allclose(
        detections[0],
        [10, 20, 30, 40, 1.0, 0],
    )


def test_square_box_returns_square_inside_frame():
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    box = FaceLandmarks._square_box(
        frame,
        [50, 20, 100, 80],
    )

    assert box is not None

    x1, y1, x2, y2 = box

    assert x2 - x1 == y2 - y1
    assert 0 <= x1 < x2 <= 200
    assert 0 <= y1 < y2 <= 100


def test_square_box_clamps_to_frame_edges():
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    box = FaceLandmarks._square_box(
        frame,
        [-20, -10, 40, 50],
    )

    assert box is not None

    x1, y1, x2, y2 = box

    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 100
    assert y2 <= 100

    assert x2 - x1 == y2 - y1


def test_square_box_rejects_invalid_box():
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    assert FaceLandmarks._square_box(
        frame,
        [20, 20, 20, 40],
    ) is None

    assert FaceLandmarks._square_box(
        frame,
        [20, 20, 40, 20],
    ) is None


def test_predict_face_maps_crop_landmarks_back_to_frame():
    landmarker = FaceLandmarks.__new__(
        FaceLandmarks
    )

    detected = [
        SimpleNamespace(
            x=0.25,
            y=0.50,
            z=0.0,
        ),
        SimpleNamespace(
            x=0.75,
            y=0.50,
            z=0.0,
        ),
    ]

    def fake_predict(crop):
        return [detected]

    landmarker.predict = fake_predict

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    landmarks = landmarker.predict_face(
        frame,
        [20, 20, 60, 60],
    )

    assert landmarks is not None
    assert len(landmarks) == 2

    assert np.isclose(
        landmarks[0].x,
        0.30,
    )

    assert np.isclose(
        landmarks[0].y,
        0.40,
    )

    assert np.isclose(
        landmarks[1].x,
        0.50,
    )

    assert np.isclose(
        landmarks[1].y,
        0.40,
    )


def test_predict_face_returns_none_when_no_landmarks_are_found():
    landmarker = FaceLandmarks.__new__(
        FaceLandmarks
    )

    landmarker.predict = lambda crop: []

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = landmarker.predict_face(
        frame,
        [20, 20, 60, 60],
    )

    assert result is None

def _controlled_landmark_group(square_box, absolute_center, count=478):
    """Represent one known face in crop-normalized backend coordinates."""
    x1, y1, x2, y2 = square_box
    cx, cy = absolute_center
    return [
        SimpleNamespace(
            x=(cx + (index % 7 - 3) * 0.2 - x1) / (x2 - x1),
            y=(cy + (index % 5 - 2) * 0.2 - y1) / (y2 - y1),
            z=0.0,
        )
        for index in range(count)
    ]


def test_predict_face_rejects_nearby_face_inside_expanded_square():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target_box = [40, 10, 60, 90]
    neighbor_box = [10, 20, 35, 50]
    square = FaceLandmarks._square_box(frame, target_box)
    assert square == (10, 10, 90, 90)
    assert target_box[0] >= neighbor_box[2]

    landmarker = FaceLandmarks.__new__(FaceLandmarks)
    landmarker.predict = lambda crop: [
        _controlled_landmark_group(square, (22.5, 35.0))
    ]
    assert landmarker.predict_face(frame, target_box) is None


def test_predict_face_selects_target_in_either_backend_order():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target_box = [40, 10, 60, 90]
    neighbor_box = [10, 20, 35, 50]
    square = FaceLandmarks._square_box(frame, target_box)
    neighbor = _controlled_landmark_group(square, (22.5, 35.0))
    target = _controlled_landmark_group(square, (50.0, 50.0))
    landmarker = FaceLandmarks.__new__(FaceLandmarks)

    for returned in ([neighbor, target], [target, neighbor]):
        landmarker.predict = lambda crop, returned=returned: returned
        found = landmarker.predict_face(
            frame, target_box, other_boxes=[neighbor_box]
        )
        assert found is target
        assert np.isclose(np.median([point.x for point in found]), 0.50)
        assert np.isclose(np.median([point.y for point in found]), 0.50)


def test_predict_face_rejects_candidate_better_matching_adjacent_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target_box = [30, 10, 70, 90]
    neighbor_box = [50, 10, 90, 90]
    square = FaceLandmarks._square_box(frame, target_box)
    landmarker = FaceLandmarks.__new__(FaceLandmarks)
    # Candidate lies in both overlapping boxes, but is closer to the other.
    landmarker.predict = lambda crop: [
        _controlled_landmark_group(square, (66.0, 50.0))
    ]
    assert landmarker.predict_face(
        frame, target_box, other_boxes=[neighbor_box]
    ) is None


def test_predict_face_rejects_indistinguishable_candidates():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target_box = [20, 20, 60, 60]
    square = FaceLandmarks._square_box(frame, target_box)
    landmarker = FaceLandmarks.__new__(FaceLandmarks)
    landmarker.predict = lambda crop: [
        _controlled_landmark_group(square, (40.0, 40.0)),
        _controlled_landmark_group(square, (40.0, 40.0)),
    ]
    assert landmarker.predict_face(frame, target_box) is None


def test_predict_face_rejects_nonfinite_candidate():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarker = FaceLandmarks.__new__(FaceLandmarks)
    landmarker.predict = lambda crop: [[
        SimpleNamespace(x=float("nan"), y=0.5, z=0.0)
    ]]
    assert landmarker.predict_face(frame, [20, 20, 60, 60]) is None
