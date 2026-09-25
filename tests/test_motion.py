"""Motion keypoint extraction and joint-angle unit consistency."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import physiotrack as pt
from physiotrack.signals import (
    extract_keypoint_sequence_2d, extract_keypoint_sequence_3d, add_pelvic_centroid,
    compute_all_joint_angles, joint_angles, compute_rom_angles,
    FACE_LANDMARK_TABLES, blink_rate, detect_blinks, eye_aspect_ratio,
    face_feature_sequence, face_window_summary, iris_position, mouth_aspect_ratio,
    mouth_movement,
)
from physiotrack.signals.motion.features import (
    compute_joint_angle_2d, compute_joint_angle_3d,
)

DATA = Path(__file__).resolve().parent / "data"


def _frame_with_hips():
    return [{
        "frame_id": 0, "timestamp": 0.0,
        "instances": [{
            "id": 1,
            "keypoints": [
                {"id": 11, "x": 10.0, "y": 20.0, "confidence": 0.9},
                {"id": 12, "x": 30.0, "y": 20.0, "confidence": 0.9},
            ],
            "keypoints3D": [
                {"id": 11, "x": 1.0, "y": 2.0, "z": 3.0},
                {"id": 12, "x": 3.0, "y": 2.0, "z": 1.0},
            ],
        }],
    }]


def test_extract_pelvic_centroid_2d_no_keyerror():
    data = add_pelvic_centroid(_frame_with_hips(), "coco_wholebody")
    df = extract_keypoint_sequence_2d(data, keypoint_id=135)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 20.0 and df.iloc[0]["y"] == 20.0


def test_extract_pelvic_centroid_3d_no_keyerror():
    data = add_pelvic_centroid(_frame_with_hips(), "coco_wholebody")
    df = extract_keypoint_sequence_3d(data, keypoint_id=135)
    assert len(df) == 1
    assert df.iloc[0]["x"] == 2.0 and df.iloc[0]["z"] == 2.0


# --- Joint angles are degrees everywhere ------------------------------------------
# Regression guard: compute_joint_angle_2d/3d once returned radians while
# joint_angles() converted to degrees and compute_all_joint_angles() did not, so the
# two public paths disagreed by a factor of 180/pi on identical geometry.

@pytest.mark.parametrize("fn, a, b, c, expected", [
    (compute_joint_angle_2d, (1, 0), (0, 0), (0, 1), 90.0),
    (compute_joint_angle_2d, (-1, 0), (0, 0), (1, 0), 180.0),
    (compute_joint_angle_2d, (1, 0), (0, 0), (1, 1), 45.0),
    (compute_joint_angle_3d, (1, 0, 0), (0, 0, 0), (0, 1, 0), 90.0),
    (compute_joint_angle_3d, (-1, 0, 0), (0, 0, 0), (1, 0, 0), 180.0),
])
def test_interior_angle_returns_degrees(fn, a, b, c, expected):
    assert fn(a, b, c) == pytest.approx(expected)


@pytest.mark.parametrize("fn, a, b, c", [
    (compute_joint_angle_2d, (0, 0), (0, 0), (1, 0)),
    (compute_joint_angle_3d, (0, 0, 0), (0, 0, 0), (1, 0, 0)),
])
def test_interior_angle_degenerate_segment_is_nan(fn, a, b, c):
    assert math.isnan(fn(a, b, c))


def test_joint_angles_and_dataframe_path_agree_in_degrees():
    # A right angle at the left elbow (shoulder 5, elbow 7, wrist 9).
    kps = [
        {"id": 5, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 7, "x": 1.0, "y": 0.0, "confidence": 1.0},
        {"id": 9, "x": 1.0, "y": 1.0, "confidence": 1.0},
    ]
    per_frame = joint_angles(kps, joints=["leftElbow"])["leftElbow"]

    wide = pd.DataFrame([{"5_x": 0.0, "5_y": 0.0, "7_x": 1.0, "7_y": 0.0,
                         "9_x": 1.0, "9_y": 1.0}])
    sequence = compute_all_joint_angles(wide)["ang_2d_leftElbow"].iloc[0]

    assert per_frame == pytest.approx(90.0)
    assert sequence == pytest.approx(90.0)
    assert per_frame == pytest.approx(sequence)


# --- signals accept the predictor result objects, not only serialized dicts --------

def _right_angle_elbow_parts():
    """A right angle at the left elbow (COCO ids 5 shoulder, 7 elbow, 9 wrist)."""
    import physiotrack as pt

    dicts = [
        {"id": 5, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 7, "x": 1.0, "y": 0.0, "confidence": 1.0},
        {"id": 9, "x": 1.0, "y": 1.0, "confidence": 1.0},
    ]
    keypoints = pt.Keypoints(dicts, "COCO")
    instance = pt.Instance(id=1, box=np.array([0, 0, 10, 10], np.float32),
                           keypoints=keypoints)
    result = pt.Result(orig_img=np.zeros((32, 32, 3), np.uint8), instances=[instance],
                       task="pose", architecture="COCO")
    return dicts, keypoints, instance, result


def test_joint_angles_accepts_result_objects_and_dicts_alike():
    dicts, keypoints, instance, result = _right_angle_elbow_parts()
    for source in (dicts, keypoints, instance, result):
        got = joint_angles(source, joints=["leftElbow"])
        assert got["leftElbow"] == pytest.approx(90.0), f"failed for {type(source).__name__}"


def test_joint_angles_rejects_ambiguous_multi_instance_result():
    import physiotrack as pt

    _, _, instance, _ = _right_angle_elbow_parts()
    two = pt.Result(orig_img=np.zeros((32, 32, 3), np.uint8),
                    instances=[instance, instance], task="pose", architecture="COCO")
    # Choosing a subject implicitly is exactly the class of bug this guards against.
    with pytest.raises(ValueError, match="one subject"):
        joint_angles(two, joints=["leftElbow"])


def test_joint_angles_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Expected Keypoints"):
        joint_angles(42)


def test_rom_angles_accepts_an_instance():
    _, _, instance, _ = _right_angle_elbow_parts()
    # No hip keypoints present, so the result is empty -- but it must not raise.
    assert compute_rom_angles(instance) == {}


def test_rom_angles_are_degrees_and_use_neutral_offset():
    # Thigh (hip 11 -> knee 13) straight down, trunk reference at shoulder 5 straight
    # up: the raw hip angle is 180 deg, so flexion reads scale*180 + 180 == 0 at neutral.
    kps = [
        {"id": 11, "x": 0.0, "y": 0.0, "confidence": 1.0},
        {"id": 5, "x": 0.0, "y": -1.0, "confidence": 1.0},
        {"id": 13, "x": 0.0, "y": 1.0, "confidence": 1.0},
    ]
    out = compute_rom_angles(kps, movements=["leftHipFlexion"])
    assert out["leftHipFlexion"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Face signals: landmark geometry and per-face sequences
# --------------------------------------------------------------------------- #

def _face_points(layout, eye_open=2.0, mouth=(3.0, 5.0), iris=(6.5, 1.0), angle=0.0):
    """Synthetic landmarks with closed-form measures.

    Each eye is 10 px wide with its lids ``eye_open`` above and below the corner line,
    so EAR = (2*2*eye_open) / (2*10) = eye_open / 5. The mouth is 40 px wide with the
    inner lips ``mouth[0]`` above and ``mouth[1]`` below the midline, so MAR =
    sum(mouth) / 40. The iris sits at ``iris`` relative to the image-left eye corner, so
    its position is (iris_x / 10, iris_y / 10). Everything is rotated by ``angle``.
    """
    table = FACE_LANDMARK_TABLES[layout]
    pts = {}
    for eye, x0 in (("right_eye", 0.0), ("left_eye", 30.0)):
        p1, p2, p3, p4, p5, p6 = table[eye]
        pts[p1], pts[p4] = (x0, 0.0), (x0 + 10.0, 0.0)
        pts[p2], pts[p6] = (x0 + 3.0, -eye_open), (x0 + 3.0, eye_open)
        pts[p3], pts[p5] = (x0 + 7.0, -eye_open), (x0 + 7.0, eye_open)
    for eye, x0 in (("right_iris", 0.0), ("left_iris", 30.0)):
        if eye in table:
            centre, a, b = table[eye]
            pts[a], pts[b] = (x0, 0.0), (x0 + 10.0, 0.0)
            pts[centre] = (x0 + iris[0], iris[1])
    left, right = table["mouth_corners"]
    upper, lower = table["mouth_opening"]
    pts[left], pts[right] = (0.0, 30.0), (40.0, 30.0)
    pts[upper], pts[lower] = (20.0, 30.0 - mouth[0]), (20.0, 30.0 + mouth[1])
    c, s = math.cos(angle), math.sin(angle)
    return [{"id": i, "x": c * x - s * y + 100.0, "y": s * x + c * y + 100.0}
            for i, (x, y) in pts.items()]


def _mesh(layout="FACEMESH", **kwargs):
    return pt.Keypoints(_face_points(layout, **kwargs), layout)


class TestFaceGeometry:
    @pytest.mark.parametrize("layout", ["FACEMESH", "WHOLEBODY"])
    def test_closed_form_values_in_both_layouts(self, layout):
        mesh = _mesh(layout, eye_open=1.5, mouth=(3.0, 5.0))
        ear = eye_aspect_ratio(mesh)
        assert ear["left"] == pytest.approx(0.3) and ear["right"] == pytest.approx(0.3)
        assert ear["mean"] == pytest.approx(0.3)
        assert mouth_aspect_ratio(mesh) == pytest.approx(0.2)

    def test_iris_position_in_eye_widths(self):
        pos = iris_position(_mesh(iris=(6.5, 1.0)))
        assert pos["x"] == pytest.approx(0.65) and pos["y"] == pytest.approx(0.1)
        assert pos["left_x"] == pytest.approx(pos["right_x"])

    def test_measures_are_rotation_invariant(self):
        upright = _mesh(eye_open=1.2, iris=(4.0, -0.5))
        tilted = _mesh(eye_open=1.2, iris=(4.0, -0.5), angle=math.radians(35))
        assert eye_aspect_ratio(tilted)["mean"] == pytest.approx(eye_aspect_ratio(upright)["mean"])
        assert mouth_aspect_ratio(tilted) == pytest.approx(mouth_aspect_ratio(upright))
        assert iris_position(tilted)["x"] == pytest.approx(0.4)

    def test_real_face_meshes_match_the_reference_implementation(self):
        # Two MediaPipe meshes from the bundled selfie, with the values the thesis-branch
        # EyeOpenness / MouthOpenness / GazeDescriptor gave on the same points.
        fixture = json.loads((DATA / "facemesh_selfie.json").read_text(encoding="utf-8"))
        for face in fixture["faces"]:
            mesh = pt.Keypoints([{"id": i, "x": x, "y": y}
                                 for i, (x, y) in enumerate(face["points"])], "FACEMESH")
            expected = face["expected"]
            ear, iris = eye_aspect_ratio(mesh), iris_position(mesh)
            assert ear["left"] == pytest.approx(expected["ear_left"], abs=1e-12)
            assert ear["right"] == pytest.approx(expected["ear_right"], abs=1e-12)
            assert mouth_aspect_ratio(mesh) == pytest.approx(expected["mar"], abs=1e-12)
            assert iris["x"] == pytest.approx(expected["iris_x"], abs=1e-12)
            assert iris["y"] == pytest.approx(expected["iris_y"], abs=1e-12)

    def test_instances_results_and_dicts_are_accepted(self):
        face = pt.Instance(keypoints=_mesh())
        one = pt.Result(orig_img=None, task="face", instances=[face])
        dicts = _face_points("FACEMESH")
        assert eye_aspect_ratio(face) == eye_aspect_ratio(one) == \
            eye_aspect_ratio(dicts, layout="FACEMESH")
        assert eye_aspect_ratio(pt.Instance()) == {"left": None, "right": None, "mean": None}
        assert mouth_aspect_ratio(pt.Instance()) is None

    def test_layout_errors_are_explicit(self):
        dicts = _face_points("FACEMESH")
        with pytest.raises(ValueError, match="carry no layout"):
            eye_aspect_ratio(dicts)
        with pytest.raises(ValueError, match="contradicts"):
            eye_aspect_ratio(_mesh(), layout="WHOLEBODY")
        with pytest.raises(ValueError, match="No face landmarks"):
            eye_aspect_ratio(dicts, layout="COCO")
        with pytest.raises(ValueError, match="no iris points"):
            iris_position(_mesh("WHOLEBODY"))
        two = pt.Result(orig_img=None, task="face",
                        instances=[pt.Instance(keypoints=_mesh())] * 2)
        with pytest.raises(ValueError, match="one face"):
            eye_aspect_ratio(two)

    def test_missing_points_give_none(self):
        partial = [kp for kp in _face_points("FACEMESH") if kp["id"] != 33]
        ear = eye_aspect_ratio(partial, layout="FACEMESH")
        assert ear["right"] is None and ear["left"] is not None and ear["mean"] is None


ABSENT = "absent"


def _run(ears, fps=30.0, ids=(1,), times=None, mars=None, yaws=None, expressions=None):
    """A synthetic Video run: per frame one face per id.

    ``ears[i]`` is the first face's EAR; ``None`` means no face mesh and ``ABSENT``
    means the face is not in the frame. Other ids have open eyes.
    """
    frames = []
    times = np.arange(len(ears)) / fps if times is None else times
    for index, t in enumerate(times):
        faces = []
        for face_id in ids:
            ear = ears[index] if face_id == ids[0] else 0.3
            if ear == ABSENT:
                continue
            mar = mars[index] if mars is not None else 0.1
            mesh = None if ear is None else _mesh(eye_open=5.0 * ear, mouth=(0.0, 40.0 * mar))
            faces.append(pt.Instance(
                id=face_id, box=np.array([0, 0, 50, 50], np.float32), keypoints=mesh,
                orientation={"yaw": yaws[index] if yaws is not None else 1.0,
                             "pitch": 2.0, "roll": 3.0},
                expression=({"label": expressions[index], "confidence": 1.0, "scores": {}}
                            if expressions is not None else None)))
        meta = pt.ResultMeta(frame_index=index, timestamp=float(t))
        frames.append(pt.FrameResult(
            result=pt.Result(orig_img=None, instances=[], task="track", meta=meta),
            faces=pt.Result(orig_img=None, instances=faces, task="face",
                            architecture="FACEMESH")))
    return pt.VideoResults(frames)


# A scenario run through the thesis-branch BlinkDetector / MouthMovement /
# FaceTemporalAggregator (Multimodal-Sensing-Lab/physiotrack@3d5dd33, driven like its
# analysis pipeline): closures at the start, too short, long, cut by a missing mesh,
# after a missing mesh, and after the face left the view.
C, O = 0.1, 0.3
SCENARIO = ([C] * 4 + [O] * 5 + [C] * 2 + [O] * 4 + [C] * 20 + [O] * 3 + [C] * 3 + [None]
            + [C] * 3 + [O] * 2 + [ABSENT] * 3 + [C] * 5 + [O] * 3)
THESIS_BLINKS = [(4, 4 / 30), (35, 20 / 30), (45, 3 / 30), (55, 5 / 30)]  # (reopen frame, s)


class TestBlinks:
    def test_matches_the_reference_implementation(self):
        blinks = detect_blinks(_run(SCENARIO))
        assert (blinks.end * 30).round().astype(int).tolist() == [f for f, _ in THESIS_BLINKS]
        assert blinks.duration.tolist() == pytest.approx([d for _, d in THESIS_BLINKS])
        assert (blinks.min_ear == C).all()

    def test_rate_counts_the_frames_the_face_was_present(self):
        present = sum(e != ABSENT for e in SCENARIO)
        assert blink_rate(_run(SCENARIO)) == pytest.approx(4 / (present / 30.0 / 60.0))
        assert math.isnan(blink_rate(_run([0.3])))

    def test_agrees_with_an_independent_find_peaks_implementation(self, rng):
        from scipy.signal import find_peaks

        for _ in range(20):
            lengths = rng.randint(1, 25, size=8).tolist()
            ear = [O] * int(rng.randint(3, 20))
            for n in lengths:
                ear += [C] * n + [O] * int(rng.randint(3, 20))
            ear = np.array(ear)
            ear[ear > 0.2] += rng.uniform(0.0, 0.05, size=(ear > 0.2).sum())  # open-eye noise
            # Oracle: closures are flat-topped peaks of -EAR; their plateau is the run.
            _, props = find_peaks(-ear, height=-0.22, plateau_size=1)
            expected = props["plateau_sizes"][props["plateau_sizes"] >= 3] / 30.0
            got = detect_blinks(_run(ear.tolist()))
            assert got.duration.to_numpy() == pytest.approx(expected)

    def test_subsampled_video_keeps_its_blinks(self):
        # Video(fps=20) on a 30 fps source keeps frames 1,2,3,5,6,8,...: uneven steps
        # must not be mistaken for gaps.
        ear = [O] * 10 + [C] * 4 + [O] * 10 + [C] * 3 + [O] * 10
        times = np.array([i + i // 2 for i in range(len(ear))]) / 30.0
        assert len(detect_blinks(_run(ear, times=times))) == 2

    def test_faces_are_followed_by_id(self):
        run = _run([O] * 5 + ([C] * 5 + [O] * 5) * 3, ids=(4, 9))
        assert len(detect_blinks(run, detection_id=4)) == 3
        assert len(detect_blinks(run, detection_id=9)) == 0
        with pytest.raises(ValueError, match="pass detection_id"):
            detect_blinks(run)

    def test_iterators_are_accepted(self):
        run = _run(SCENARIO)
        assert blink_rate(iter(run)) == pytest.approx(blink_rate(run))
        assert len(detect_blinks(frame for frame in run)) == 4

    def test_parameters_and_timestamps_are_validated(self):
        with pytest.raises(ValueError, match="threshold"):
            detect_blinks(_run(SCENARIO), threshold=0)
        with pytest.raises(ValueError, match="min_closed_frames"):
            blink_rate(_run(SCENARIO), min_closed_frames=0)
        record = _run([O, C]).to_dict_list()
        record[1]["timestamp"] = None
        with pytest.raises(ValueError, match="no timestamp"):
            detect_blinks(record)


class TestMouthMovement:
    def test_matches_the_reference_definition(self):
        # |MAR_t - MAR_prev| and its backward rate; 0 on the first frame and after a gap.
        mars = [0.1, 0.2, 0.4, 0.4, 0.1, 0.1, 0.3, 0.2]
        ears = [O, O, O, O, O, None, O, O]
        out = mouth_movement(_run(ears, mars=mars))
        assert out.mar_movement.tolist()[:5] == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.3])
        assert out.mar_velocity.tolist()[:5] == pytest.approx([0.0, 3.0, 6.0, 0.0, 9.0])
        assert math.isnan(out.mar_movement[5])
        assert out.mar_movement.tolist()[6:] == pytest.approx([0.0, 0.1])

    def test_velocity_uses_the_actual_frame_times(self, rng):
        times = np.cumsum(rng.uniform(0.025, 0.045, size=40))
        mars = 0.3 + 0.1 * np.sin(times)
        out = mouth_movement(_run([O] * 40, times=times, mars=mars))
        expected = np.abs(np.diff(mars)) / np.diff(times)
        assert out.mar_velocity.to_numpy()[1:] == pytest.approx(expected)


class TestWindowSummary:
    def test_matches_the_reference_aggregator(self):
        yaws = [float(i % 7) - 3.0 for i in range(len(SCENARIO))]
        labels = [["Neutral", "Happiness", "Happiness"][i % 3] for i in range(len(SCENARIO))]
        mars = [round(0.1 + 0.3 * abs(math.sin(i / 3.0)), 4) for i in range(len(SCENARIO))]
        summary = face_window_summary(_run(SCENARIO, yaws=yaws, mars=mars,
                                           expressions=labels), window=0.5)
        assert len(summary) == sum(e != ABSENT for e in SCENARIO)
        assert summary.window_frames.max() == 15            # round(0.5 s * 30 fps)
        # The window restarts when the face leaves the view (frames 47-49).
        after = summary[summary.frame == 50].iloc[0]
        assert after.window_frames == 1 and after.yaw_mean == pytest.approx(yaws[50])
        # The 15-frame window ending at frame 45 holds the reopening frames 35 and 45.
        row = summary[summary.frame == 45].iloc[0]
        assert row.blink_events == 2 and row.dominant_expression == "Happiness"
        window = [e for e in SCENARIO[31:46] if e is not None]
        assert row.ear_mean == pytest.approx(np.mean(window))
        assert row.ear_std == pytest.approx(np.std(window))
        assert (row.yaw_min, row.yaw_max) == (-3.0, 3.0)

    def test_untracked_faces_are_not_summarised(self):
        run = _run([O] * 5)
        for frame in run:
            frame.faces.instances = [f.replace(id=None) for f in frame.faces]
        assert face_window_summary(run).empty
        with pytest.raises(ValueError, match="window"):
            face_window_summary(run, window=0)


def test_face_feature_sequence_tabulates_every_face():
    df = face_feature_sequence(_run([O, C, O], ids=(1, 2)))
    assert len(df) == 6 and df.frame.tolist() == [0, 0, 1, 1, 2, 2]
    assert set(df.detection_id) == {1, 2}
    face1 = df[df.detection_id == 1]
    assert face1.ear.tolist() == pytest.approx([O, C, O])
    assert face1.mar.tolist() == pytest.approx([0.1] * 3)
    assert face1.iris_x.tolist() == pytest.approx([0.65] * 3)
    assert (face1[["x1", "y1", "x2", "y2"]].to_numpy() == [0, 0, 50, 50]).all()
    assert (df.yaw == 1.0).all() and df.expression.isna().all()
    assert face_feature_sequence(_run([])).empty
