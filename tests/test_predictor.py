"""The shared input contract for image predictors.

Every predictor accepted "a frame or a list of frames" and each re-implemented the
batch check; none accepted a path, so the most obvious first line anyone writes --
``det.predict("photo.jpg")`` -- failed with an opaque error from inside a backend. These
tests pin what a source may be, how batching is decided, and that the contract is the
same for every predictor.

The models themselves are not loaded here (each is a large download); the input layer and
the class wiring are what these tests cover.
"""

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
        frames, was_batch = as_frames(
            [np.zeros((4, 4, 3), np.uint8)] * 3
        )
        assert len(frames) == 3 and was_batch is True

    def test_single_element_list_is_still_a_batch(self):
        # So a caller that batches never needs a special case for n == 1.
        frames, was_batch = as_frames(
            [np.zeros((4, 4, 3), np.uint8)]
        )
        assert len(frames) == 1 and was_batch is True

    def test_batch_of_paths(self, image_file):
        frames, was_batch = as_frames(
            [image_file, str(image_file)]
        )
        assert len(frames) == 2 and was_batch is True
        assert all(f.shape == (6, 8, 3) for f in frames)

    def test_mixed_arrays_and_paths(self, image_file):
        frames, was_batch = as_frames(
            [
                np.zeros((6, 8, 3), np.uint8),
                image_file,
            ]
        )
        assert len(frames) == 2 and was_batch is True

    def test_tuple_is_a_batch(self):
        _, was_batch = as_frames(
            (np.zeros((4, 4, 3), np.uint8),)
        )
        assert was_batch is True

    def test_a_stack_of_frames_is_rejected_as_ambiguous(self):
        # (N, H, W, 3) could be a batch or a volume; requiring a list removes the guess.
        with pytest.raises(ValueError, match="pass a list"):
            as_frames(
                np.zeros((5, 4, 4, 3), np.uint8)
            )

    def test_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty sequence"):
            as_frames([])

    def test_unsupported_type_names_what_is_accepted(self):
        with pytest.raises(TypeError, match="BGR array"):
            as_frames(42)

    def test_bad_batch_element_names_its_index(self):
        with pytest.raises(TypeError, match="element 1"):
            as_frames(
                [
                    np.zeros((4, 4, 3), np.uint8),
                    42,
                ]
            )


class TestMixin:
    def test_call_forwards_to_predict(self):
        class P(PredictorMixin):
            def predict(self, source, **kwargs):
                return ("predicted", source, kwargs)

        p = P()

        assert p(
            "x",
            conf=0.5,
        ) == p.predict(
            "x",
            conf=0.5,
        )

    def test_unimplemented_predict_is_a_clear_error(self):
        class Bare(PredictorMixin):
            pass

        with pytest.raises(
            NotImplementedError,
            match="must implement predict",
        ):
            Bare().predict(
                np.zeros((4, 4, 3), np.uint8)
            )

    def test_unwrap_respects_the_batch_flag(self):
        assert PredictorMixin._unwrap(
            ["a"],
            False,
        ) == "a"

        assert PredictorMixin._unwrap(
            ["a"],
            True,
        ) == ["a"]


class TestEveryPredictorFollowsTheContract:
    """The uniformity itself, asserted over the real predictor classes."""

    @staticmethod
    def _bases():
        from physiotrack.depth.depth import DepthBase
        from physiotrack.detect.detect import _DetectionAPI
        from physiotrack.face.analysis import FaceAnalysis
        from physiotrack.face.face_orientation import FaceOrientation
        from physiotrack.face.gaze_estimation import GazeEstimator
        from physiotrack.pose.pose import PoseBase
        from physiotrack.segment.segment import SegmentationBase

        return [
            _DetectionAPI,
            PoseBase,
            SegmentationBase,
            DepthBase,
            FaceOrientation,
            GazeEstimator,
            FaceAnalysis,
        ]

    def test_all_inherit_the_mixin(self):
        for cls in self._bases():
            assert issubclass(
                cls,
                PredictorMixin,
            ), cls.__name__

    def test_none_redefines_call(self):
        # A per-class __call__ that only forwards is duplication waiting to drift.
        for cls in self._bases():
            assert "__call__" not in vars(
                cls
            ), f"{cls.__name__} redefines __call__"

    def test_first_parameter_is_named_source(self):
        import inspect

        for cls in self._bases():
            params = list(
                inspect.signature(
                    cls.predict
                ).parameters
            )

            assert params[1] == "source", (
                f"{cls.__name__}.predict names its input "
                f"{params[1]!r}, not 'source'"
            )

    def test_no_public_predict_batch_remains(self):
        # A list passed to predict() covers it; a second public entry point is not needed.
        for cls in self._bases():
            assert not hasattr(
                cls,
                "predict_batch",
            ), f"{cls.__name__} still exposes predict_batch"

    def test_tracker_keeps_its_own_verb(self):
        """Tracking is not a per-image predictor and should not pretend to be.

        It is stateful and sequential, and consumes detections rather than pixels, so
        `track(frame, detections)` is the honest signature. This is asserted rather than
        left implicit so nobody "unifies" it into predict() by mistake.
        """
        from physiotrack.trackers.track import Tracker

        assert hasattr(
            Tracker,
            "track",
        )

        assert not issubclass(
            Tracker,
            PredictorMixin,
        )