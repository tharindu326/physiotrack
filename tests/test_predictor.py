"""The shared input contract for image predictors.

Every predictor accepted "a frame or a list of frames" and each re-implemented the
batch check; none accepted a path, so the most obvious first line anyone writes --
``det.predict("photo.jpg")`` -- failed with an opaque error from inside a backend. These
tests pin what a source may be, how batching is decided, and that the contract is the
same for every predictor.

The input layer and the class wiring are covered without loading models. The face
stages are the exception: ``TestFaceStagesOnRealWeights`` runs their small models (a few
to tens of MB) against values produced by the upstream implementations.
"""
from pathlib import Path

import numpy as np
import pytest

from physiotrack.core.predictor import PredictorMixin, as_frames, load_image


@pytest.fixture
def image_file(tmp_path):
    """A real 8x6 image on disk."""
    import cv2

    path = tmp_path / "frame.png"
    img = np.zeros((6, 8, 3), np.uint8)
    img[2:4, 3:5] = (10, 20, 30)
    cv2.imwrite(str(path), img)
    return path


class TestLoadImage:
    def test_reads_a_file(self, image_file):
        img = load_image(image_file)
        assert img.shape == (6, 8, 3)
        assert tuple(img[2, 3]) == (10, 20, 30)

    def test_accepts_a_string_path(self, image_file):
        assert load_image(str(image_file)).shape == (6, 8, 3)

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.png"):
            load_image(tmp_path / "nope.png")

    def test_undecodable_file_points_at_video(self, tmp_path):
        # The most likely mistake is handing a video to an image predictor, so the
        # message should say where video belongs instead of just "could not decode".
        bad = tmp_path / "clip.mp4"
        bad.write_bytes(b"not really a video")
        with pytest.raises(ValueError, match="physiotrack.Video"):
            load_image(bad)


class TestAsFrames:
    def test_single_array_is_not_a_batch(self):
        frames, was_batch = as_frames(np.zeros((4, 4, 3), np.uint8))
        assert len(frames) == 1 and was_batch is False

    def test_grayscale_array_is_accepted(self):
        frames, was_batch = as_frames(np.zeros((4, 4), np.uint8))
        assert frames[0].shape == (4, 4) and was_batch is False

    def test_single_path_is_not_a_batch(self, image_file):
        frames, was_batch = as_frames(image_file)
        assert frames[0].shape == (6, 8, 3) and was_batch is False

    def test_list_is_a_batch(self):
        frames, was_batch = as_frames([np.zeros((4, 4, 3), np.uint8)] * 3)
        assert len(frames) == 3 and was_batch is True

    def test_single_element_list_is_still_a_batch(self):
        # So a caller that batches never needs a special case for n == 1.
        frames, was_batch = as_frames([np.zeros((4, 4, 3), np.uint8)])
        assert len(frames) == 1 and was_batch is True

    def test_batch_of_paths(self, image_file):
        frames, was_batch = as_frames([image_file, str(image_file)])
        assert len(frames) == 2 and was_batch is True
        assert all(f.shape == (6, 8, 3) for f in frames)

    def test_mixed_arrays_and_paths(self, image_file):
        frames, was_batch = as_frames([np.zeros((6, 8, 3), np.uint8), image_file])
        assert len(frames) == 2 and was_batch is True

    def test_tuple_is_a_batch(self):
        _, was_batch = as_frames((np.zeros((4, 4, 3), np.uint8),))
        assert was_batch is True

    def test_a_stack_of_frames_is_rejected_as_ambiguous(self):
        # (N, H, W, 3) could be a batch or a volume; requiring a list removes the guess.
        with pytest.raises(ValueError, match="pass a list"):
            as_frames(np.zeros((5, 4, 4, 3), np.uint8))

    def test_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty sequence"):
            as_frames([])

    def test_unsupported_type_names_what_is_accepted(self):
        with pytest.raises(TypeError, match="BGR array"):
            as_frames(42)

    def test_bad_batch_element_names_its_index(self):
        with pytest.raises(TypeError, match="element 1"):
            as_frames([np.zeros((4, 4, 3), np.uint8), 42])


class TestMixin:
    def test_call_forwards_to_predict(self):
        class P(PredictorMixin):
            def predict(self, source, **kwargs):
                return ("predicted", source, kwargs)

        p = P()
        assert p("x", conf=0.5) == p.predict("x", conf=0.5)

    def test_call_forwards_positional_arguments(self):
        # Face stages take the faces as a second argument: stage(frame, faces).
        class P(PredictorMixin):
            def predict(self, source, faces=None):
                return (source, faces)

        assert P()("frame", "faces") == ("frame", "faces")

    def test_unimplemented_predict_is_a_clear_error(self):
        class Bare(PredictorMixin):
            pass

        with pytest.raises(NotImplementedError, match="must implement predict"):
            Bare().predict(np.zeros((4, 4, 3), np.uint8))

    def test_unwrap_respects_the_batch_flag(self):
        assert PredictorMixin._unwrap(["a"], False) == "a"
        assert PredictorMixin._unwrap(["a"], True) == ["a"]


class TestEveryPredictorFollowsTheContract:
    """The uniformity itself, asserted over the real predictor classes."""

    @staticmethod
    def _bases():
        from physiotrack.depth.depth import DepthBase
        from physiotrack.detect.detect import _DetectionAPI
        from physiotrack.face import (FaceExpression, FaceLandmarks, FaceOrientation,
                                      FaceQuality, FaceRegions, FaceStage, GazeEstimator)
        from physiotrack.pose.pose import PoseBase
        from physiotrack.segment.segment import SegmentationBase
        return [_DetectionAPI, PoseBase, SegmentationBase, DepthBase, FaceStage,
                FaceOrientation, FaceLandmarks, FaceExpression, GazeEstimator, FaceQuality,
                FaceRegions]

    def test_all_inherit_the_mixin(self):
        for cls in self._bases():
            assert issubclass(cls, PredictorMixin), cls.__name__

    def test_none_redefines_call(self):
        # A per-class __call__ that only forwards is duplication waiting to drift.
        for cls in self._bases():
            assert "__call__" not in vars(cls), f"{cls.__name__} redefines __call__"

    def test_first_parameter_is_named_source(self):
        import inspect

        for cls in self._bases():
            params = list(inspect.signature(cls.predict).parameters)
            assert params[1] == "source", \
                f"{cls.__name__}.predict names its input {params[1]!r}, not 'source'"

    def test_no_public_predict_batch_remains(self):
        # A list passed to predict() covers it; a second public entry point is not needed.
        for cls in self._bases():
            assert not hasattr(cls, "predict_batch"), \
                f"{cls.__name__} still exposes predict_batch"

    def test_tracker_keeps_its_own_verb(self):
        """Tracking is not a per-image predictor and should not pretend to be.

        It is stateful and sequential, and consumes detections rather than pixels, so
        `track(frame, detections)` is the honest signature. This is asserted rather than
        left implicit so nobody "unifies" it into predict() by mistake.
        """
        from physiotrack.trackers.track import Tracker

        assert hasattr(Tracker, "track")
        assert not issubclass(Tracker, PredictorMixin)


# --------------------------------------------------------------------------- #
# Face stages
# --------------------------------------------------------------------------- #
def _stub_stage(provides="quality", requires=(), delay=0.0):
    """A stub face stage: numbers the faces it sees and records every call."""
    import time

    from physiotrack.face import FaceStage

    class Stub(FaceStage):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _infer_batch(self, frames, faces):
            time.sleep(delay)
            self.calls.append([len(f) for f in faces])
            return [[{"n": i} for i in range(len(f))] for f in faces]

    Stub.provides, Stub.requires = provides, requires
    return Stub()


def _faces(boxes, ids=None, **fields):
    import physiotrack as pt

    ids = ids or [None] * len(boxes)
    return pt.Result(orig_img=np.zeros((100, 100, 3), np.uint8), task="face", instances=[
        pt.Instance(box=np.asarray(b, np.float32), id=i, confidence=0.9, **fields)
        for b, i in zip(boxes, ids)])


class TestFaceStageContract:
    frame = np.zeros((100, 100, 3), np.uint8)

    def test_output_is_aligned_and_keeps_ids_and_earlier_fields(self):
        stage = _stub_stage()
        faces = _faces([[0, 0, 10, 10], [20, 20, 40, 40]], ids=[7, 3],
                       orientation={"yaw": 1.0, "pitch": 2.0, "roll": 3.0})
        out = stage(self.frame, faces)
        assert out.task == "face" and [f.id for f in out] == [7, 3]
        assert [f.quality for f in out] == [{"n": 0}, {"n": 1}]
        assert all(f.orientation == {"yaw": 1.0, "pitch": 2.0, "roll": 3.0} for f in out)
        assert all(f.confidence == pytest.approx(0.9) for f in out)

    def test_input_result_is_not_mutated(self):
        faces = _faces([[0, 0, 10, 10]])
        _stub_stage()(self.frame, faces)
        assert faces[0].quality is None

    def test_face_without_area_keeps_its_instance_with_none(self):
        stage = _stub_stage()
        out = stage(self.frame, _faces([[0, 0, 10, 10], [5, 5, 5, 20], [30, 30, 50, 50]]))
        assert stage.calls == [[2]]                      # only the two real boxes
        assert [f.quality for f in out] == [{"n": 0}, None, {"n": 1}]

    def test_faces_outside_the_frame_are_not_analysed(self):
        stage = _stub_stage()
        out = stage(self.frame, _faces([[-60, 10, -20, 50], [120, 10, 160, 50],
                                        [90, 90, 130, 130]]))
        assert stage.calls == [[1]]                      # only the partly visible box
        assert [f.quality for f in out] == [None, None, {"n": 0}]

    def test_none_means_the_whole_image(self):
        out = _stub_stage()(self.frame)
        assert len(out) == 1 and out[0].box.tolist() == [0, 0, 100, 100]

    def test_accepts_plain_boxes_and_track_results(self):
        import physiotrack as pt

        stage = _stub_stage()
        assert len(stage(self.frame, np.array([[0, 0, 10, 10], [1, 1, 5, 5]]))) == 2
        assert len(stage(self.frame, np.empty((0, 4)))) == 0
        tracked = pt.TrackResult(instances=[pt.Instance(id=4, box=np.array([0, 0, 9, 9.]))],
                                 orig_img=self.frame, rendered=self.frame, raw=[])
        assert stage(self.frame, tracked)[0].id == 4

    def test_batch_needs_one_entry_per_image(self):
        stage = _stub_stage()
        out = stage([self.frame, self.frame], [_faces([[0, 0, 9, 9]]), None])
        assert [len(r) for r in out] == [1, 1] and stage.calls == [[1, 1]]
        with pytest.raises(ValueError, match="one entry per image"):
            stage([self.frame, self.frame], [_faces([[0, 0, 9, 9]])])

    def test_timings_are_recorded(self):
        stage = _stub_stage(delay=0.02)
        assert stage.get_avg_fps() == 0.0
        stage([self.frame, self.frame], [_faces([[0, 0, 10, 10]])] * 2)
        # 20 ms for a batch of two frames is about 10 ms per frame.
        assert 5.0 < stage.get_avg_inference_time() < 1000.0
        assert stage.get_avg_fps() == pytest.approx(1000.0 / stage.get_avg_inference_time())

    def test_stage_order_is_checked(self):
        from physiotrack.face import check_stage_order

        mesh = _stub_stage(provides="landmarks")
        gaze = _stub_stage(provides="gaze", requires=("landmarks",))
        check_stage_order([mesh, gaze])
        with pytest.raises(ValueError, match="needs landmarks"):
            check_stage_order([gaze, mesh])
        with pytest.raises(ValueError, match="put FaceLandmarks before it"):
            check_stage_order([gaze])
        with pytest.raises(TypeError, match="not a face stage"):
            check_stage_order([object()])

    def test_gaze_refuses_faces_without_a_mesh(self):
        from physiotrack.face import GazeEstimator

        stage = object.__new__(GazeEstimator)   # the check needs no model
        with pytest.raises(ValueError, match="FACEMESH"):
            stage._check_input(_faces([[0, 0, 10, 10]]))


# The real models on a bundled photo with fixed face boxes, so these checks do not
# depend on the face detector. Expected values for the vendored backends were produced
# once by the upstream implementations -- emotiefflib 1.1.1's ONNX recognizer and
# ptgaze 0.3.0's normalisation and model -- on the same inputs.
_SELFIE = (Path(__file__).resolve().parents[1]
           / "examples/face_detection/data/selfie/two_person_selfie.jpg")
_SELFIE_BOXES = np.array([[439, 641, 827, 1191], [828, 636, 1244, 1229]], np.float32)


@pytest.fixture(scope="module")
def selfie():
    return load_image(_SELFIE)


class TestFaceStagesOnRealWeights:
    def test_orientation_regression(self, selfie):
        import physiotrack as pt

        faces = pt.FaceOrientation()(selfie, _SELFIE_BOXES)
        got = [[f.orientation[k] for k in ("yaw", "pitch", "roll")] for f in faces]
        assert np.allclose(got, [[11.39, 9.44, -3.96], [-9.02, 8.10, -7.83]], atol=0.5)

    def test_landmarks_are_a_478_point_mesh_with_anatomical_sides(self, selfie):
        import physiotrack as pt

        stage = pt.FaceLandmarks()
        faces = stage(selfie, _SELFIE_BOXES)
        stage.close()
        assert faces.architecture == "FACEMESH"
        for face in faces:
            mesh = face.keypoints
            assert len(mesh) == 478 and mesh.architecture == "FACEMESH"
            x1, y1, x2, y2 = face.box
            left, right = mesh.by_name("left_iris_center"), mesh.by_name("right_iris_center")
            # The subject's left eye appears on the image right.
            assert x1 < right.x < left.x < x2 and y1 < left.y < y2
            assert 0.15 < pt.signals.eye_aspect_ratio(face)["mean"] < 0.45
        stage.close()                                   # releasing twice is harmless

    def test_expression_matches_emotiefflib(self, selfie):
        import physiotrack as pt

        stage = pt.FaceExpression()
        faces = stage(selfie, _SELFIE_BOXES)
        upstream = [[0.0, 0.0, 0.00012, 0.0, 0.999241, 0.0, 0.000639, 0.0],
                    [3e-06, 3e-06, 0.007739, 0.0, 0.99224, 0.0, 1.5e-05, 0.0]]
        for face, expected in zip(faces, upstream):
            assert face.expression["label"] == "Happiness"
            got = [face.expression["scores"][label] for label in stage.labels]
            assert np.allclose(got, expected, atol=1e-4)

    def test_multitask_and_seven_class_expression_models(self, selfie):
        import physiotrack as pt

        box = _SELFIE_BOXES[:1]
        mtl = pt.FaceExpression(model=pt.Models.Face.Expression.enet_b0_8_va_mtl)
        e = mtl(selfie, box)[0].expression
        scores = [e["scores"][label] for label in mtl.labels]
        assert np.allclose(scores, [0.0, 0.0002, 6.2e-05, 0.0, 0.999732, 1e-06, 0.0, 4e-06],
                           atol=1e-4)
        assert (e["valence"], e["arousal"]) == pytest.approx((0.792673, 0.20151), abs=1e-4)
        seven = pt.FaceExpression(model=pt.Models.Face.Expression.enet_b2_7)
        e = seven(selfie, box)[0].expression
        assert len(seven.labels) == 7 and "Contempt" not in seven.labels
        assert e["label"] == "Happiness" and "valence" not in e

    @staticmethod
    def _template_faces(selfie):
        """A face mesh projected from the 3D face template at a known head pose."""
        import cv2
        from scipy.spatial.transform import Rotation

        import physiotrack as pt
        from physiotrack.modules.Gaze import default_camera_matrix
        from physiotrack.modules.Gaze import face_model

        h, w = selfie.shape[:2]
        rotation = Rotation.from_euler("xyz", [10, -15, 5], degrees=True).as_matrix()
        points = face_model.LANDMARKS @ rotation.T + np.array([0.02, 0.01, 0.6])
        pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                   default_camera_matrix(w, h), np.zeros(5))[0].reshape(-1, 2)
        mesh = pt.Keypoints([{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(pixels)],
                            "FACEMESH")
        return pt.Result(orig_img=selfie, task="face", architecture="FACEMESH", instances=[
            pt.Instance(box=np.r_[pixels.min(0), pixels.max(0)].astype(np.float32),
                        keypoints=mesh)])

    @pytest.mark.parametrize("member, expected", [
        ("eth_xgaze_resnet18", [-0.213939, 0.1201, -0.969436]),
        ("mpiifacegaze_resnet_simple", [-0.041427, 0.187613, -0.981369]),
        ("mpiigaze_resnet_preact", [-0.083482, 0.115266, -0.98982]),   # mean of both eyes
    ])
    def test_gaze_matches_ptgaze(self, selfie, member, expected):
        import physiotrack as pt

        model = getattr(pt.Models.Face.Gaze, member)
        gaze = pt.GazeEstimator(model=model)(selfie, self._template_faces(selfie))[0].gaze
        assert np.allclose(gaze["vector"], expected, atol=1e-3)
        x, y, z = gaze["vector"]
        assert gaze["yaw"] == pytest.approx(np.degrees(np.arctan2(x, -z)))
        assert gaze["pitch"] == pytest.approx(np.degrees(np.arctan2(y, np.hypot(x, z))))

    def test_gaze_undistorts_with_calibration(self, selfie):
        import physiotrack as pt
        from physiotrack.modules.Gaze import default_camera_matrix

        faces = self._template_faces(selfie)
        camera = default_camera_matrix(selfie.shape[1], selfie.shape[0])
        plain = pt.GazeEstimator(camera_matrix=camera)(selfie, faces)[0].gaze
        zero = pt.GazeEstimator(camera_matrix=camera, dist_coeffs=np.zeros(5))(selfie, faces)
        assert np.allclose(zero[0].gaze["vector"], plain["vector"], atol=1e-6)
        with pytest.raises(ValueError, match="camera_matrix"):
            pt.GazeEstimator(dist_coeffs=np.zeros(5))

    def test_regions_report_the_face_parts_of_each_box(self, selfie):
        import physiotrack as pt

        faces = pt.FaceRegions()(selfie, _SELFIE_BOXES)
        for face in faces:
            fractions = face.regions["fractions"]
            assert 0.5 < fractions["skin"] < 0.9                 # frontal, unoccluded faces
            assert {"nose", "mouth"} <= set(fractions) and sum(fractions.values()) <= 1.0
            x1, y1, x2, y2 = face.box
            assert sum(face.regions["pixel_counts"].values()) <= (x2 - x1) * (y2 - y1) + 1

    def test_quality_is_model_free(self, selfie):
        import physiotrack as pt

        for face in pt.FaceQuality()(selfie, _SELFIE_BOXES):
            q = face.quality
            assert 0 < q["brightness"] < 1 and q["sharpness"] > 0
            x1, y1, x2, y2 = face.box
            assert q["area_ratio"] == pytest.approx(
                (x2 - x1) * (y2 - y1) / (selfie.shape[0] * selfie.shape[1]), rel=1e-3)

    def test_stages_chain_and_serialize(self, selfie):
        import physiotrack as pt

        faces = pt.FaceQuality()(selfie, pt.FaceOrientation()(selfie, _SELFIE_BOXES))
        data = faces.to_dict()
        assert {"orientation", "quality"} <= set(data["instances"][0])
        assert pt.Result.from_dict(data)[1].orientation == faces[1].orientation
