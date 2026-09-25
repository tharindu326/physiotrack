"""The result-object serialization contract.

One vocabulary across the object model and its serialized form, and a genuine
round-trip: ``from_dict(to_dict(x))`` must reproduce ``x``. Previously each result type
used different keys (``detections`` / ``tracks``), renamed fields on the way out
(``box`` -> ``bbox``, ``orientation`` -> ``pose``), dropped ``cls_name`` and ``mask``
entirely, and had no way back at all, so a written JSON file could not be reloaded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import physiotrack as pt


def _instance():
    keypoints = pt.Keypoints(
        [
            {"id": 5, "x": 1.0, "y": 2.0, "confidence": 0.9},
            {"id": 7, "x": 3.0, "y": 4.0, "confidence": 0.8},
        ],
        "COCO",
    )
    return pt.Instance(
        id=7,
        box=np.array([1, 2, 3, 4], np.float32),
        confidence=0.77,
        cls=0,
        cls_name="person",
        keypoints=keypoints,
        orientation={"yaw": 1.0, "pitch": 2.0, "roll": 3.0},
        mask=np.ones((4, 4), bool),
    )


def _result():
    return pt.Result(orig_img=np.zeros((8, 8, 3), np.uint8), instances=[_instance()],
                     task="pose", architecture="COCO")


# --- one vocabulary ----------------------------------------------------------------

def test_serialized_keys_match_the_attribute_names():
    data = _result().to_dict()
    assert "instances" in data and "detections" not in data
    instance = data["instances"][0]
    # The attribute is `box`, so the key is `box`; likewise `orientation`.
    assert "box" in instance and "bbox" not in instance
    assert "orientation" in instance and "pose" not in instance


def test_cls_name_is_not_dropped():
    assert _result().to_dict()["instances"][0]["cls_name"] == "person"


def test_track_result_uses_the_same_key_as_result():
    track = pt.TrackResult(instances=[_instance()], orig_img=np.zeros((8, 8, 3), np.uint8))
    data = track.to_dict()
    assert "instances" in data and "tracks" not in data


# --- round-trip --------------------------------------------------------------------

def test_result_round_trips():
    original = _result()
    restored = pt.Result.from_dict(original.to_dict())
    a, b = original[0], restored[0]

    assert (restored.task, restored.architecture) == (original.task, original.architecture)
    assert (b.id, b.confidence, b.cls, b.cls_name) == (a.id, a.confidence, a.cls, a.cls_name)
    assert np.array_equal(np.asarray(b.box), np.asarray(a.box))
    assert b.orientation == a.orientation
    assert len(b.keypoints) == len(a.keypoints)
    # Keypoint names are rebuilt from the recorded architecture.
    assert b.keypoints[0].name == a.keypoints[0].name == "left_shoulder"


def test_masks_are_flagged_but_omitted_by_default():
    data = _result().to_dict()
    assert data["instances"][0]["has_mask"] is True
    assert "mask" not in data["instances"][0]
    assert pt.Result.from_dict(data)[0].mask is None


def test_masks_round_trip_when_arrays_are_requested():
    restored = pt.Result.from_dict(_result().to_dict(include_arrays=True))
    assert restored[0].mask is not None
    assert restored[0].mask.shape == (4, 4)


def test_to_json_is_valid_json_and_writes_a_file(tmp_path):
    result = _result()
    assert isinstance(json.loads(result.to_json()), dict)

    path = tmp_path / "result.json"
    assert result.to_json(path) is None
    assert pt.Result.from_dict(json.loads(path.read_text(encoding="utf-8")))[0].id == 7


def test_from_dict_rejects_a_payload_without_a_task():
    with pytest.raises(KeyError, match="task"):
        pt.Result.from_dict({"instances": []})


# --- depth -------------------------------------------------------------------------

def test_depth_records_that_it_is_relative_not_metric():
    depth = pt.DepthResult(orig_img=np.zeros((4, 4, 3), np.uint8),
                           depth=np.arange(16, dtype=float).reshape(4, 4))
    data = depth.to_dict()
    assert data["relative"] is True
    assert data["shape"] == [4, 4]


def test_depth_array_round_trips_only_when_requested():
    depth = pt.DepthResult(orig_img=np.zeros((4, 4, 3), np.uint8),
                           depth=np.arange(16, dtype=float).reshape(4, 4))
    # Without the array the map genuinely cannot be recovered, so this must not
    # silently produce an empty or zero-filled result.
    with pytest.raises(KeyError, match="include_arrays"):
        pt.DepthResult.from_dict(depth.to_dict())

    restored = pt.DepthResult.from_dict(depth.to_dict(include_arrays=True))
    assert np.array_equal(restored.depth, depth.depth)


# --- Video output: FrameResult / VideoResults ---------------------------------------
# Video.run() used to return plain dicts, which discarded the Instance/Keypoints object
# model at exactly the point most users enter the library. These pin the replacement.

def _frame_result(frame_index=0, timestamp=0.0, vitals=None):
    result = pt.Result(orig_img=np.zeros((8, 8, 3), np.uint8), instances=[_instance()],
                       task="pose", architecture="COCO",
                       meta=pt.ResultMeta(frame_index=frame_index, timestamp=timestamp,
                                          fps=30.0))
    return pt.FrameResult(result=result, vitals=vitals)


def test_frame_result_behaves_like_its_instances():
    frame = _frame_result()
    assert len(frame) == 1
    assert frame[0].id == 7
    assert [i.id for i in frame] == [7]
    # The object model is intact: keypoints are still named, not raw dicts.
    assert frame[0].keypoints.by_name("left_shoulder") is not None


def test_frame_result_exposes_metadata():
    frame = _frame_result(frame_index=12, timestamp=0.4)
    assert (frame.meta.frame_index, frame.meta.timestamp, frame.meta.fps) == (12, 0.4, 30.0)


def test_frame_result_vitals_accessors():
    frame = _frame_result(vitals={"hr": 72.0, "snr": -3.0})
    assert (frame.hr, frame.snr) == (72.0, -3.0)
    # Absent vitals must read as None rather than raising.
    assert _frame_result().hr is None


def test_frame_result_round_trips():
    original = _frame_result(frame_index=3, timestamp=0.1, vitals={"hr": 60.0})
    restored = pt.FrameResult.from_dict(original.to_dict(), architecture="COCO")
    assert restored.meta.frame_index == 3
    assert restored.hr == 60.0
    assert restored[0].keypoints.by_name("left_shoulder") is not None


def test_video_results_is_a_sequence_that_serializes(tmp_path):
    results = pt.VideoResults([_frame_result(i, i / 30.0) for i in range(3)])
    assert len(results) == 3
    assert [f.meta.frame_index for f in results] == [0, 1, 2]
    # list semantics, including slicing
    assert len(results[:2]) == 2

    path = tmp_path / "run.json"
    results.to_json(path)
    reloaded = pt.VideoResults.from_dict_list(
        json.loads(path.read_text(encoding="utf-8")), architecture="COCO")
    assert len(reloaded) == 3
    assert reloaded[1].meta.frame_index == 1


def test_signals_accept_video_results_and_dicts_alike():
    from physiotrack.signals import as_frame_records

    results = pt.VideoResults([_frame_result(i, i / 30.0) for i in range(2)])
    from_objects = as_frame_records(results)
    from_dicts = as_frame_records(results.to_dict_list())

    assert from_objects == from_dicts
    assert [r["frame_id"] for r in from_objects] == [0, 1]


def test_as_frame_records_rejects_unsupported_elements():
    from physiotrack.signals import as_frame_records

    with pytest.raises(TypeError, match="FrameResult"):
        as_frame_records([object()])


# --- face analysis on the result model -----------------------------------------------

def _face_mesh(n=478, offset=0.0):
    return pt.Keypoints([{"id": i, "x": float(i) + offset, "y": 2.0 * i} for i in range(n)],
                        "FACEMESH")


def _face(**fields):
    return pt.Instance(box=np.array([10, 20, 60, 90], np.float32), confidence=0.8, cls=0,
                       cls_name="face", **fields)


def test_facemesh_keypoints_are_named_and_carry_no_confidence():
    mesh = _face_mesh()
    assert mesh.by_name("left_iris_center").id == 473
    assert mesh.by_name("right_eye_outer").id == 33
    assert mesh[5].name == "facemesh_5" and mesh[5].confidence is None
    assert np.isnan(mesh.conf).all()
    assert "conf=" not in repr(mesh[5])


def test_unknown_keypoint_architecture_is_rejected():
    # Ids of different layouts overlap, so silently falling back to COCO names would
    # give face-mesh point 1 the name of a body joint.
    with pytest.raises(ValueError, match="Unknown keypoint architecture"):
        pt.Keypoints([{"id": 1, "x": 0.0, "y": 0.0}], "HALPE")


def test_wholebody_face_points_use_the_subjects_sides():
    from physiotrack.pose.config import COCO_WHOLEBODY

    # iBUG-68 36-41 / 17-21 are the subject's right eye / eyebrow, like body id 2.
    assert COCO_WHOLEBODY["59"] == "face_right_eye_0"
    assert COCO_WHOLEBODY["65"] == "face_left_eye_0"
    assert COCO_WHOLEBODY["40"] == "face_right_eyebrow_0"
    assert COCO_WHOLEBODY["2"] == "right_eye"


def test_face_fields_round_trip():
    face = _face(id=3, orientation={"yaw": 1.0, "pitch": 2.0, "roll": 3.0},
                 expression={"label": "Neutral", "confidence": 0.7, "scores": {"Neutral": 0.7}},
                 gaze={"pitch": 5.0, "yaw": -4.0, "vector": [0.0, 0.1, -0.99]},
                 quality={"brightness": 0.5, "sharpness": 120.0, "area_ratio": 0.01})
    restored = pt.Instance.from_dict(face.to_dict())
    for name in ("id", "orientation", "expression", "gaze", "quality", "cls_name"):
        assert getattr(restored, name) == getattr(face, name)
    assert "expression='Neutral'" in repr(face)


def test_face_mesh_is_serialized_only_on_request():
    # 478 points per face per frame would make a video's JSON hundreds of MB.
    face = _face(keypoints=_face_mesh())
    brief = face.to_dict()
    assert brief["has_keypoints"] is True and "keypoints" not in brief
    full = face.to_dict(include_arrays=True)
    assert len(full["keypoints"]) == 478 and "confidence" not in full["keypoints"][0]
    restored = pt.Instance.from_dict(full, architecture="FACEMESH")
    assert restored.keypoints.by_name("left_iris_center").x == 473.0
    # Body keypoints stay in the default output.
    assert "keypoints" in _instance().to_dict()


def test_instance_replace_copies_and_validates():
    face = _face(id=1)
    tagged = face.replace(id=9, quality={"brightness": 0.1})
    assert (tagged.id, face.id) == (9, 1)
    assert tagged.box is face.box and tagged.quality == {"brightness": 0.1}
    with pytest.raises(TypeError, match="no field"):
        face.replace(colour="red")


def test_facemesh_is_drawn_with_face_contours_not_the_body_skeleton():
    img = np.zeros((200, 200, 3), np.uint8)
    # COCO-17 edge (0, 5) would join mesh points 0 and 5; the face contours do not.
    mesh = pt.Keypoints([{"id": i, "x": 20.0, "y": 20.0} for i in range(478)], "FACEMESH")
    mesh.by_id(5).x, mesh.by_id(5).y = 180.0, 180.0
    drawn = pt.Result(orig_img=img, task="face", architecture="FACEMESH",
                      instances=[pt.Instance(keypoints=mesh)]).plot()
    assert drawn[100, 100].sum() == 0


def test_frame_result_carries_faces_through_serialization():
    faces = pt.Result(orig_img=None, task="face", architecture="FACEMESH",
                      instances=[_face(id=7, keypoints=_face_mesh(),
                                       orientation={"yaw": 1.0, "pitch": 0.0, "roll": 0.0})])
    frame = pt.FrameResult(
        result=pt.Result(orig_img=None, instances=[pt.Instance(id=7)], task="track",
                         meta=pt.ResultMeta(frame_index=4, timestamp=0.2)),
        faces=faces)
    data = frame.to_dict(include_arrays=True)
    assert data["task"] == "track" and data["faces"]["task"] == "face"
    restored = pt.FrameResult.from_dict(json.loads(json.dumps(data)))
    assert restored.result.task == "track"
    assert restored.faces.architecture == "FACEMESH"
    assert restored.faces[0].id == 7 and len(restored.faces[0].keypoints) == 478
    assert "faces=1" in repr(restored)


# --- box geometry ---------------------------------------------------------------------

def test_box_iou_matches_hand_computed_overlaps():
    from physiotrack.core.boxes import box_iou

    iou = box_iou([[0, 0, 10, 10], [0, 0, 0, 5]], [[5, 0, 15, 10], [0, 0, 10, 10]])
    assert iou.shape == (2, 2)
    assert iou[0] == pytest.approx([50 / 150, 1.0])
    assert iou[1].tolist() == [0.0, 0.0]           # a zero-area box overlaps nothing
    assert box_iou(np.empty((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_crop_square_pads_at_the_border_and_maps_back():
    from physiotrack.core.boxes import crop_square

    frame = np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)
    crop, (left, top), side = crop_square(frame, [0, 10, 20, 50], scale=1.5)
    assert crop.shape == (60, 60) and side == 60
    assert (left, top) == (-20, 0)
    assert (crop[:, :20] == 0).all()                 # padding outside the frame
    # A crop pixel maps back with frame_xy = offset + crop_xy.
    assert crop[30, 25] == frame[top + 30, left + 25]
    with pytest.raises(ValueError, match="without area"):
        crop_square(frame, [5, 5, 5, 9])


def test_clip_box_rounds_and_clips():
    from physiotrack.core.boxes import clip_box

    assert clip_box([-3.4, 2.6, 50.2, 9.0], (8, 40)) == (0, 3, 40, 8)
    assert clip_box([50, 50, 60, 60], (8, 40)) is None


def test_assign_ids_matches_faces_to_containing_subjects_one_to_one():
    from physiotrack.core.boxes import assign_ids

    subjects = [[0, 0, 100, 200], [150, 0, 250, 200]]
    faces = [[160, 10, 200, 50], [20, 10, 60, 50], [400, 400, 420, 420]]
    assert assign_ids(faces, subjects, [11, 22]) == [22, 11, None]
    # Two faces in one person box: only the better-covered face gets the id.
    assert assign_ids([[10, 10, 40, 40], [90, 10, 130, 40]], [[0, 0, 100, 100]], [5]) == [5, None]
    assert assign_ids(np.empty((0, 4)), subjects, [11, 22]) == []


# --- Video: faces through the pipeline -----------------------------------------------
# Stub detectors stand in for YOLO so no weights load; the tracker is real.

_CLIP = (Path(__file__).resolve().parents[1]
         / "examples" / "face_tracking" / "data" / "students_face_tracking.mp4")


class _Rows:
    def __init__(self, rows):
        self.data = self
        self._rows = rows

    def cpu(self):
        return self

    def numpy(self):
        return self._rows


class _YoloResult:
    def __init__(self, rows):
        self.boxes = _Rows(rows)


class _PersonDetector:
    """Two fixed person boxes per frame; no detect_batch, so Video's per-frame path runs."""
    rows = np.array([[40, 40, 120, 200, 0.9, 0], [200, 60, 280, 220, 0.8, 0]], np.float32)

    def detect(self, frame, **kwargs):
        return [_YoloResult(self.rows)], frame


class _FaceDetector(_PersonDetector):
    """Face boxes inside the two person boxes, as a face detector preset reports them."""
    task = "face"
    rows = np.array([[60, 45, 100, 85, 0.9, 0], [220, 65, 260, 105, 0.8, 0]], np.float32)

    def predict(self, frames):
        return [pt.Result(orig_img=f, task="face", instances=[
            pt.Instance(box=r[:4].copy(), confidence=float(r[4]), cls=0) for r in self.rows])
            for f in frames]

    def get_avg_fps(self):
        return 0.0

    def get_avg_inference_time(self):
        return 0.0


def _tracker():
    return pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0],
                                       enable_subject_lock=False))


def _quality_stage():
    from physiotrack.face import FaceQuality
    return FaceQuality()


def test_video_exports_tracked_instances_without_a_pose_estimator():
    results = pt.Video(source=_CLIP, detector=_PersonDetector(), tracker=_tracker(),
                       fps=5).run()
    tracked = [inst for frame in results for inst in frame if inst.id is not None]
    assert tracked and all(len(inst.box) == 4 for inst in tracked)
    assert results[-1].result.task == "track"
    assert results[-1].faces is None


def test_video_gives_faces_the_track_id_of_their_person():
    results = pt.Video(source=_CLIP, detector=_PersonDetector(), tracker=_tracker(),
                       face=_FaceDetector(), face_stages=[_quality_stage()], fps=5).run()
    last = results[-1]
    person_of = {int(inst.box[0]): inst.id for inst in last}          # x1 -> track id
    assert sorted(f.id for f in last.faces) == sorted(person_of.values())
    assert [f.id for f in last.faces] == [person_of[40], person_of[200]]
    assert all(f.quality is not None for f in last.faces)
    assert "faces" in last.to_dict()


def test_a_face_detector_as_detector_makes_the_tracked_subjects_the_faces():
    results = pt.Video(source=_CLIP, detector=_FaceDetector(), tracker=_tracker(),
                       face_stages=[_quality_stage()], fps=5).run()
    last = results[-1]
    assert [f.id for f in last.faces] == [i.id for i in last]
    assert all(f.id is not None and f.quality is not None for f in last.faces)


def test_video_rejects_ambiguous_or_missing_face_sources():
    with pytest.raises(ValueError, match="already a face detector"):
        pt.Video(source=_CLIP, detector=_FaceDetector(), face=_FaceDetector())
    with pytest.raises(ValueError, match="face_stages need faces"):
        pt.Video(source=_CLIP, detector=_PersonDetector(), face_stages=[_quality_stage()])
    with pytest.raises(ValueError, match="mixed with other detectors"):
        pt.Video(source=_CLIP, detector=[_FaceDetector(), _PersonDetector()])


def test_every_detector_contributes_detections():
    video = pt.Video(source=_CLIP, detector=[_PersonDetector(), _PersonDetector()])
    frame = np.zeros((240, 320, 3), np.uint8)
    (_, detections), = video.process_batch_detections([frame])
    assert len(detections) == 2 and all(len(d) == 2 for d in detections)


def test_vr_detector_points_at_the_published_head_checkpoint():
    assert pt.Models.Detection.YOLO.VR.m_vr.value == "yolo11m_VR_head.pt"
    assert list(pt.Models.Detection.YOLO.VR.__members__) == ["m_vr"]
