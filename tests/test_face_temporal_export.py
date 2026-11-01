import json

import numpy as np

from physiotrack.face.export import FaceResultExporter
from physiotrack.face.temporal import FaceTemporalAggregator
from physiotrack.results import Instance, Result


def make_instance(
    person_id,
    yaw,
    eye,
    mouth,
    blink,
    emotion,
):
    return Instance(
        id=person_id,
        orientation={
            "yaw": yaw,
            "pitch": 0.0,
            "roll": 0.0,
        },
        face_features={
            "eyes": {
                "available": True,
                "mean_openness": eye,
            },
            "gaze": {
                "available": True,
                "mean_iris_x": 0.5,
                "mean_iris_y": 0.5,
            },
            "mouth": {
                "available": True,
                "mouth_openness": mouth,
            },
            "mouth_motion": {
                "available": True,
                "mouth_movement": 0.05,
            },
            "quality": {
                "available": True,
                "brightness": 0.6,
                "sharpness": 100.0,
                "face_area_ratio": 0.1,
            },
            "blink": {
                "available": True,
                "blink": blink,
            },
            "emotion": {
                "available": True,
                "emotion": emotion,
            },
        },
    )


def test_temporal_summary():
    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    samples = [
        (0.0, 0.3, 0.1, False, "Neutral"),
        (10.0, 0.2, 0.2, True, "Happiness"),
        (20.0, 0.4, 0.3, False, "Happiness"),
    ]

    summary = None

    for yaw, eye, mouth, blink, emotion in samples:
        summary = aggregator.update(
            make_instance(
                1,
                yaw,
                eye,
                mouth,
                blink,
                emotion,
            )
        )

    assert summary["window_frames"] == 3
    assert summary["window_sec"] == 0.3
    assert summary["head_pose"]["yaw"]["mean"] == 10.0
    assert np.isclose(
        summary["eyes"]["mean_openness"]["mean"],
        0.3,
    )
    assert np.isclose(
        summary["mouth"]["openness"]["mean"],
        0.2,
    )
    assert summary["blink"]["events"] == 1
    assert summary["emotion"]["dominant"] == "Happiness"


def test_sliding_window_stays_bounded():
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    summary = None

    for frame_index in range(10):
        summary = aggregator.update(
            make_instance(
                1,
                float(frame_index),
                0.3,
                0.2,
                False,
                "Neutral",
            )
        )

    assert summary["window_frames"] == 6
    assert summary["window_sec"] == 3.0
    assert summary["head_pose"]["yaw"]["mean"] == 6.5


def test_people_have_independent_temporal_buffers():
    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    summary_1 = aggregator.update(
        make_instance(
            1,
            10.0,
            0.3,
            0.2,
            False,
            "Neutral",
        )
    )

    summary_2 = aggregator.update(
        make_instance(
            2,
            -10.0,
            0.4,
            0.1,
            True,
            "Sadness",
        )
    )

    assert summary_1["person_id"] == 1
    assert summary_2["person_id"] == 2

    assert summary_1["head_pose"]["yaw"]["mean"] == 10.0
    assert summary_2["head_pose"]["yaw"]["mean"] == -10.0


def test_frame_records_preserve_person_id_and_metadata():
    frame = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    instances = [
        make_instance(
            1,
            1.0,
            0.3,
            0.1,
            False,
            "Neutral",
        ),
        make_instance(
            2,
            2.0,
            0.4,
            0.2,
            False,
            "Happiness",
        ),
    ]

    result = Result(
        orig_img=frame,
        instances=instances,
        task="face",
    )

    records = FaceResultExporter.frame_records(
        result,
        frame_index=5,
        timestamp=0.2,
    )

    assert len(records) == 2

    assert records[0]["frame_index"] == 5
    assert records[0]["timestamp"] == 0.2

    assert {
        record["person_id"]
        for record in records
    } == {1, 2}


def test_export_json_and_csv(tmp_path):
    records = [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "person_id": 1,
            "head_pose": {
                "yaw": {
                    "mean": 1.5,
                }
            },
        }
    ]

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    FaceResultExporter.save_json(
        records,
        json_path,
    )

    FaceResultExporter.save_csv(
        records,
        csv_path,
    )

    assert json_path.exists()
    assert csv_path.exists()

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded[0]["person_id"] == 1

    csv_text = csv_path.read_text(
        encoding="utf-8"
    )

    assert "head_pose.yaw.mean" in csv_text
    assert "person_id" in csv_text