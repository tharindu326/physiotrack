from types import SimpleNamespace

import numpy as np
import pytest

from physiotrack.face.gaze_estimation import GazeEstimator


class DummyPTGazeEstimator:
    def __init__(
        self,
        faces=None,
        detections_per_call=None,
    ):
        self.faces = (
            list(faces)
            if faces is not None
            else []
        )

        self.detections_per_call = (
            [
                list(detections)
                for detections in detections_per_call
            ]
            if detections_per_call is not None
            else None
        )

        self.detect_calls = 0
        self.estimate_calls = 0
        self.estimated_faces = []
        self.was_closed = False

    def detect_faces(
        self,
        image,
    ):
        self.detect_calls += 1

        if self.detections_per_call is None:
            return self.faces

        detection_index = (
            self.detect_calls
            - 1
        )

        if (
            detection_index
            < len(
                self.detections_per_call
            )
        ):
            return self.detections_per_call[
                detection_index
            ]

        return []

    def estimate_gaze(
        self,
        image,
        face,
    ):
        self.estimate_calls += 1

        self.estimated_faces.append(
            face
        )

    def close(self):
        self.was_closed = True


def make_face(
    bbox,
    gaze_vector,
    name=None,
):
    return SimpleNamespace(
        bbox=np.asarray(
            bbox,
            dtype=np.float64,
        ),
        gaze_vector=np.asarray(
            gaze_vector,
            dtype=np.float64,
        ),
        name=name,
    )


def make_estimator(
    faces=None,
    detections_per_call=None,
):
    estimator = GazeEstimator(
        mode="eth-xgaze",
        device="cpu",
    )

    backend = DummyPTGazeEstimator(
        faces=faces,
        detections_per_call=(
            detections_per_call
        ),
    )

    estimator._estimator = backend

    return (
        estimator,
        backend,
    )


def test_normalize_vector_returns_unit_vector():
    vector = np.array(
        [
            3.0,
            4.0,
            0.0,
        ]
    )

    normalized = (
        GazeEstimator._normalize_vector(
            vector
        )
    )

    np.testing.assert_allclose(
        normalized,
        np.array(
            [
                0.6,
                0.8,
                0.0,
            ]
        ),
    )

    assert np.linalg.norm(
        normalized
    ) == pytest.approx(
        1.0
    )


def test_normalize_vector_rejects_invalid_size():
    with pytest.raises(
        ValueError,
        match="Expected a 3D gaze vector",
    ):
        GazeEstimator._normalize_vector(
            [
                1.0,
                2.0,
            ]
        )


def test_normalize_vector_rejects_non_finite_values():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        GazeEstimator._normalize_vector(
            [
                1.0,
                np.nan,
                -1.0,
            ]
        )


def test_normalize_vector_rejects_zero_length():
    with pytest.raises(
        ValueError,
        match="zero length",
    ):
        GazeEstimator._normalize_vector(
            [
                0.0,
                0.0,
                0.0,
            ]
        )


def test_vector_to_angles_forward_vector():
    pitch, yaw = (
        GazeEstimator._vector_to_angles(
            np.array(
                [
                    0.0,
                    0.0,
                    -1.0,
                ]
            )
        )
    )

    assert pitch == pytest.approx(
        0.0
    )

    assert yaw == pytest.approx(
        0.0
    )


def test_ptgaze_bbox_conversion():
    bbox = np.array(
        [
            [
                10,
                20,
            ],
            [
                50,
                80,
            ],
        ]
    )

    converted = (
        GazeEstimator._ptgaze_bbox_to_xyxy(
            bbox
        )
    )

    np.testing.assert_allclose(
        converted,
        np.array(
            [
                10.0,
                20.0,
                50.0,
                80.0,
            ]
        ),
    )


def test_ptgaze_bbox_conversion_rejects_wrong_shape():
    with pytest.raises(
        ValueError,
        match="shape",
    ):
        GazeEstimator._ptgaze_bbox_to_xyxy(
            np.array(
                [
                    10,
                    20,
                    50,
                    80,
                ]
            )
        )


def test_box_iou_identical_boxes():
    box = np.array(
        [
            10.0,
            20.0,
            50.0,
            80.0,
        ]
    )

    iou = GazeEstimator._box_iou(
        box,
        box,
    )

    assert iou == pytest.approx(
        1.0
    )


def test_box_iou_non_overlapping_boxes():
    box_a = np.array(
        [
            0.0,
            0.0,
            10.0,
            10.0,
        ]
    )

    box_b = np.array(
        [
            20.0,
            20.0,
            30.0,
            30.0,
        ]
    )

    iou = GazeEstimator._box_iou(
        box_a,
        box_b,
    )

    assert iou == pytest.approx(
        0.0
    )


def test_box_iou_partial_overlap():
    box_a = np.array(
        [
            0.0,
            0.0,
            10.0,
            10.0,
        ]
    )

    box_b = np.array(
        [
            5.0,
            5.0,
            15.0,
            15.0,
        ]
    )

    iou = GazeEstimator._box_iou(
        box_a,
        box_b,
    )

    expected = (
        25.0
        / 175.0
    )

    assert iou == pytest.approx(
        expected
    )


def test_predict_faces_detects_once_per_frame():
    face_a = make_face(
        bbox=[
            [
                10,
                10,
            ],
            [
                50,
                50,
            ],
        ],
        gaze_vector=[
            0.0,
            0.0,
            -1.0,
        ],
    )

    face_b = make_face(
        bbox=[
            [
                60,
                10,
            ],
            [
                100,
                50,
            ],
        ],
        gaze_vector=[
            0.1,
            0.0,
            -0.99,
        ],
    )

    estimator, backend = (
        make_estimator(
            faces=[
                face_a,
                face_b,
            ]
        )
    )

    image = np.zeros(
        (
            120,
            120,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ],
            [
                60,
                10,
                100,
                50,
            ],
        ],
    )

    assert len(results) == 2

    assert (
        backend.detect_calls
        == 1
    )

    assert (
        backend.estimate_calls
        == 2
    )


def test_predict_faces_preserves_target_order():
    face_left = make_face(
        bbox=[
            [
                10,
                10,
            ],
            [
                50,
                50,
            ],
        ],
        gaze_vector=[
            -0.2,
            0.0,
            -0.98,
        ],
        name="left",
    )

    face_right = make_face(
        bbox=[
            [
                70,
                10,
            ],
            [
                110,
                50,
            ],
        ],
        gaze_vector=[
            0.2,
            0.0,
            -0.98,
        ],
        name="right",
    )

    estimator, backend = (
        make_estimator(
            faces=[
                face_right,
                face_left,
            ]
        )
    )

    image = np.zeros(
        (
            120,
            120,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ],
            [
                70,
                10,
                110,
                50,
            ],
        ],
    )

    assert len(results) == 2

    assert (
        backend.estimated_faces[
            0
        ].name
        == "left"
    )

    assert (
        backend.estimated_faces[
            1
        ].name
        == "right"
    )

    assert (
        results[
            0
        ][
            "yaw"
        ]
        < 0
    )

    assert (
        results[
            1
        ][
            "yaw"
        ]
        > 0
    )


def test_predict_faces_uses_one_to_one_matching():
    face = make_face(
        bbox=[
            [
                10,
                10,
            ],
            [
                50,
                50,
            ],
        ],
        gaze_vector=[
            0.0,
            0.0,
            -1.0,
        ],
    )

    estimator, backend = (
        make_estimator(
            faces=[
                face
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ],
            [
                12,
                12,
                48,
                48,
            ],
        ],
        min_iou=0.10,
    )

    assert len(results) == 2

    available_count = sum(
        result[
            "available"
        ]
        for result in results
    )

    assert (
        available_count
        == 1
    )

    assert (
        backend.detect_calls
        == 1
    )

    assert (
        backend.estimate_calls
        == 1
    )


def test_predict_faces_returns_unavailable_when_all_detection_attempts_fail():
    estimator, backend = (
        make_estimator(
            detections_per_call=[
                [],
                [],
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ]
        ],
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result[
            "available"
        ]
        is False
    )

    assert (
        result[
            "gaze_vector"
        ]
        is None
    )

    assert (
        result[
            "pitch"
        ]
        is None
    )

    assert (
        result[
            "yaw"
        ]
        is None
    )

    assert (
        result[
            "association_iou"
        ]
        is None
    )

    assert (
        backend.detect_calls
        == 2
    )

    assert (
        backend.estimate_calls
        == 0
    )


def test_predict_faces_uses_crop_fallback_after_full_frame_detection_failure():
    crop_face = make_face(
        bbox=[
            [
                0,
                0,
            ],
            [
                40,
                40,
            ],
        ],
        gaze_vector=[
            0.2,
            0.1,
            -0.97,
        ],
        name="crop_face",
    )

    estimator, backend = (
        make_estimator(
            detections_per_call=[
                [],
                [
                    crop_face
                ],
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ]
        ],
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result[
            "available"
        ]
        is True
    )

    assert (
        result[
            "association_iou"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        result[
            "gaze_vector"
        ]
        is not None
    )

    assert (
        len(
            result[
                "gaze_vector"
            ]
        )
        == 3
    )

    assert (
        np.linalg.norm(
            np.asarray(
                result[
                    "gaze_vector"
                ],
                dtype=np.float64,
            )
        )
        == pytest.approx(
            1.0
        )
    )

    assert (
        backend.detect_calls
        == 2
    )

    assert (
        backend.estimate_calls
        == 1
    )

    assert (
        backend.estimated_faces[
            0
        ].name
        == "crop_face"
    )


def test_predict_faces_crop_fallback_rejects_match_below_min_iou():
    crop_face = make_face(
        bbox=[
            [
                0,
                0,
            ],
            [
                10,
                10,
            ],
        ],
        gaze_vector=[
            0.2,
            0.1,
            -0.97,
        ],
        name="weak_crop_face",
    )

    estimator, backend = (
        make_estimator(
            detections_per_call=[
                [],
                [
                    crop_face
                ],
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ]
        ],
        min_iou=0.50,
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result[
            "available"
        ]
        is False
    )

    assert (
        result[
            "gaze_vector"
        ]
        is None
    )

    assert (
        result[
            "pitch"
        ]
        is None
    )

    assert (
        result[
            "yaw"
        ]
        is None
    )

    assert (
        result[
            "association_iou"
        ]
        == pytest.approx(
            0.0625
        )
    )

    assert (
        result[
            "association_iou"
        ]
        < 0.50
    )

    assert (
        backend.detect_calls
        == 2
    )

    assert (
        backend.estimate_calls
        == 0
    )


def test_predict_faces_crop_fallback_preserves_multiple_target_order():
    first_crop_face = make_face(
        bbox=[
            [
                0,
                0,
            ],
            [
                40,
                40,
            ],
        ],
        gaze_vector=[
            -0.2,
            0.0,
            -0.98,
        ],
        name="first",
    )

    second_crop_face = make_face(
        bbox=[
            [
                0,
                0,
            ],
            [
                40,
                40,
            ],
        ],
        gaze_vector=[
            0.2,
            0.0,
            -0.98,
        ],
        name="second",
    )

    estimator, backend = (
        make_estimator(
            detections_per_call=[
                [],
                [
                    first_crop_face
                ],
                [
                    second_crop_face
                ],
            ]
        )
    )

    image = np.zeros(
        (
            120,
            120,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                10,
                10,
                50,
                50,
            ],
            [
                70,
                10,
                110,
                50,
            ],
        ],
    )

    assert len(results) == 2

    assert (
        results[
            0
        ][
            "available"
        ]
        is True
    )

    assert (
        results[
            1
        ][
            "available"
        ]
        is True
    )

    assert (
        results[
            0
        ][
            "yaw"
        ]
        < 0
    )

    assert (
        results[
            1
        ][
            "yaw"
        ]
        > 0
    )

    assert (
        results[
            0
        ][
            "association_iou"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        results[
            1
        ][
            "association_iou"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert [
        face.name
        for face in (
            backend.estimated_faces
        )
    ] == [
        "first",
        "second",
    ]

    assert (
        backend.detect_calls
        == 3
    )

    assert (
        backend.estimate_calls
        == 2
    )


def test_predict_faces_returns_best_iou_for_unmatched_target():
    face = make_face(
        bbox=[
            [
                0,
                0,
            ],
            [
                20,
                20,
            ],
        ],
        gaze_vector=[
            0.0,
            0.0,
            -1.0,
        ],
    )

    estimator, backend = (
        make_estimator(
            faces=[
                face
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    results = estimator.predict_faces(
        image=image,
        boxes=[
            [
                15,
                15,
                35,
                35,
            ]
        ],
        min_iou=0.50,
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result[
            "available"
        ]
        is False
    )

    assert (
        result[
            "association_iou"
        ]
        is not None
    )

    assert (
        0.0
        < result[
            "association_iou"
        ]
        < 0.50
    )

    assert (
        backend.detect_calls
        == 1
    )

    assert (
        backend.estimate_calls
        == 0
    )


def test_predict_faces_rejects_invalid_threshold():
    estimator, _ = (
        make_estimator()
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="min_iou",
    ):
        estimator.predict_faces(
            image=image,
            boxes=[
                [
                    10,
                    10,
                    50,
                    50,
                ]
            ],
            min_iou=1.1,
        )


def test_predict_faces_rejects_invalid_box_shape():
    estimator, _ = (
        make_estimator()
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        estimator.predict_faces(
            image=image,
            boxes=[
                [
                    10,
                    10,
                    50,
                ]
            ],
        )


def test_predict_faces_rejects_invalid_box_coordinates():
    estimator, _ = (
        make_estimator()
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="invalid coordinates",
    ):
        estimator.predict_faces(
            image=image,
            boxes=[
                [
                    50,
                    50,
                    10,
                    10,
                ]
            ],
        )


def test_predict_faces_requires_initialization():
    estimator = GazeEstimator(
        mode="eth-xgaze",
        device="cpu",
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        RuntimeError,
        match="not initialized",
    ):
        estimator.predict_faces(
            image=image,
            boxes=[
                [
                    10,
                    10,
                    50,
                    50,
                ]
            ],
        )


def test_predict_face_uses_multi_face_api():
    face = make_face(
        bbox=[
            [
                10,
                10,
            ],
            [
                50,
                50,
            ],
        ],
        gaze_vector=[
            0.0,
            0.0,
            -1.0,
        ],
    )

    estimator, backend = (
        make_estimator(
            faces=[
                face
            ]
        )
    )

    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    result = estimator.predict_face(
        image=image,
        box=[
            10,
            10,
            50,
            50,
        ],
    )

    assert (
        result[
            "available"
        ]
        is True
    )

    assert (
        result[
            "association_iou"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        backend.detect_calls
        == 1
    )

    assert (
        backend.estimate_calls
        == 1
    )


def test_close_releases_backend():
    estimator, backend = (
        make_estimator()
    )

    assert (
        estimator.initialized
        is True
    )

    estimator.close()

    assert (
        backend.was_closed
        is True
    )

    assert (
        estimator.initialized
        is False
    )

def test_predict_follows_single_source_contract():
    estimator, backend = make_estimator(
        faces=[]
    )

    image = np.zeros(
        (
            20,
            30,
            3,
        ),
        dtype=np.uint8,
    )

    result = estimator.predict(
        image
    )

    assert isinstance(
        result,
        dict,
    )
    assert result["available"] is False
    assert backend.detect_calls == 1


def test_predict_follows_batch_source_contract():
    estimator, backend = make_estimator(
        faces=[]
    )

    frames = [
        np.zeros(
            (
                20,
                30,
                3,
            ),
            dtype=np.uint8,
        ),
        np.zeros(
            (
                24,
                36,
                3,
            ),
            dtype=np.uint8,
        ),
    ]

    results = estimator.predict(
        frames
    )

    assert isinstance(
        results,
        list,
    )
    assert len(results) == 2
    assert all(
        result["available"] is False
        for result in results
    )
    assert backend.detect_calls == 2


def test_predict_accepts_image_path(tmp_path):
    import cv2

    estimator, backend = make_estimator(
        faces=[]
    )

    image_path = tmp_path / "gaze_input.png"
    image = np.zeros(
        (
            20,
            30,
            3,
        ),
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(image_path),
        image,
    )

    result = estimator.predict(
        image_path
    )

    assert isinstance(
        result,
        dict,
    )
    assert result["available"] is False
    assert backend.detect_calls == 1