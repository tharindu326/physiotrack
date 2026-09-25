"""The shared input contract for every image predictor.

`Detection`, `Pose`, `Segmentation`, `Depth` and `FaceOrientation` each accepted
"a frame or a list of frames" and each re-implemented the same two steps: decide whether
the input was a batch, and dispatch accordingly. None of them accepted a *path*, so the
first thing anyone writes -- ``det.predict("photo.jpg")`` -- failed with an opaque error
from deep inside a backend, and `FaceOrientation` had grown a separate public
`predict_batch()` because its `predict()` could not take a list at all.

`PredictorMixin` holds the contract once: what a source may be, how it is loaded, and how
single-versus-batch is decided. Predictors keep their own inference internals.
"""
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

__all__ = ["PredictorMixin", "load_image", "as_frames"]

_IMAGE_SUFFIXES = frozenset({
    ".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".avif",
    ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
    ".tiff", ".tif", ".exr", ".hdr", ".pic",
})


def load_image(path: Union[str, os.PathLike]) -> np.ndarray:
    """Read an image file into a BGR array.

    Args:
        path (str | os.PathLike): Path to an image file.

    Returns:
        np.ndarray: The image as ``(H, W, 3)`` BGR.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file exists but cannot be decoded -- most often because it is
            a video rather than an image, or the extension does not match the contents.
    """
    import cv2

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such image file: {p}")

    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        hint = ""
        if p.suffix.lower() not in _IMAGE_SUFFIXES:
            hint = (f" The extension {p.suffix!r} is not a still-image format OpenCV "
                    f"reads; for video use physiotrack.Video(source=...).")
        raise ValueError(f"Could not decode {p} as an image.{hint}")
    return image


def as_frames(source: Any) -> Tuple[List[np.ndarray], bool]:
    """Normalise any accepted predictor input into a list of BGR frames.

    Args:
        source (str | os.PathLike | np.ndarray | Sequence): One image, or a sequence of
            them. Each element may be a BGR array or a path to an image file.

    Returns:
        tuple[list[np.ndarray], bool]: The frames, and whether the caller passed a batch.
            The flag is what decides between returning a single result and a list, so a
            one-element list still yields a one-element list of results.

    Raises:
        TypeError: If ``source`` (or an element of it) is neither an array nor a path.
        ValueError: If ``source`` is an empty sequence, or an array is not a single
            ``(H, W, 3)`` / ``(H, W)`` image.
        FileNotFoundError: If a path does not exist.
    """
    if isinstance(source, np.ndarray):
        if source.ndim not in (2, 3):
            raise ValueError(
                f"Expected a single image of shape (H, W, 3) or (H, W), got an array "
                f"with shape {source.shape}. To run on several frames, pass a list."
            )
        return [source], False

    if isinstance(source, (str, os.PathLike)):
        return [load_image(source)], False

    if isinstance(source, Sequence):
        items = list(source)
        if not items:
            raise ValueError("Received an empty sequence; there is nothing to predict on.")
        frames = []
        for i, item in enumerate(items):
            if isinstance(item, np.ndarray):
                frames.append(item)
            elif isinstance(item, (str, os.PathLike)):
                frames.append(load_image(item))
            else:
                raise TypeError(
                    f"Batch element {i} is a {type(item).__name__}; each element must be "
                    f"a BGR array or a path to an image file."
                )
        return frames, True

    raise TypeError(
        f"Cannot predict on a {type(source).__name__}. Pass a BGR array (H, W, 3), a "
        f"path to an image file, or a sequence of either. For video, use "
        f"physiotrack.Video(source=...)."
    )


class PredictorMixin:
    """The uniform call contract shared by every image predictor.

    Provides :meth:`__call__` and the input normalisation behind :meth:`predict`, so all
    predictors accept the same things and are invocable the same way.
    """

    def predict(self, source, **kwargs):
        """Run the model on one image or a batch of images.

        Args:
            source (str | os.PathLike | np.ndarray | Sequence): A single BGR image
                ``(H, W, 3)``, a path to an image file, or a sequence of either for
                batch inference.
            **kwargs (Any): Predictor-specific options.

        Returns:
            Result | list[Result]: One result for a single image, or a list of results
                -- one per frame, in order -- when ``source`` is a sequence.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict()."
        )

    def __call__(self, source, *args, **kwargs):
        """Alias for :meth:`predict`, so ``model(frame)`` works like ``model.predict(frame)``.

        Args:
            source (str | os.PathLike | np.ndarray | Sequence): See :meth:`predict`.
            *args (Any): Forwarded to :meth:`predict` -- e.g. the face boxes of a face
                stage, ``stage(frame, faces)``.
            **kwargs (Any): Forwarded to :meth:`predict`.

        Returns:
            Result | list[Result]: See :meth:`predict`.
        """
        return self.predict(source, *args, **kwargs)

    @staticmethod
    def _as_frames(source: Any) -> Tuple[List[np.ndarray], bool]:
        """Normalise ``source`` via :func:`as_frames`."""
        return as_frames(source)

    @staticmethod
    def _unwrap(results: List[Any], was_batch: bool):
        """Return ``results`` for a batch call, or its single element otherwise."""
        return results if was_batch else results[0]
