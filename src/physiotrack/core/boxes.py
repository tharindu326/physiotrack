"""Bounding-box geometry shared by the predictors, the tracker and the video pipeline.

All boxes are ``[x1, y1, x2, y2]`` in pixel coordinates of the frame they were
detected in, with ``(x1, y1)`` the top-left corner.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["box_iou", "clip_box", "crop_square", "assign_ids"]


def box_iou(boxes_a, boxes_b) -> np.ndarray:
    """Pairwise intersection-over-union of two sets of boxes.

    Args:
        boxes_a (array-like): ``(N, 4)`` or a single ``(4,)`` box.
        boxes_b (array-like): ``(M, 4)`` or a single ``(4,)`` box.

    Returns:
        np.ndarray: ``(N, M)`` IoU matrix in ``[0, 1]``. Degenerate (zero-area) boxes
            have IoU 0 with everything.

    Example:
        ```python
        from physiotrack.core.boxes import box_iou

        box_iou([0, 0, 10, 10], [[5, 0, 15, 10]])   # array([[0.3333]])
        ```
    """
    a = np.atleast_2d(np.asarray(boxes_a, dtype=np.float64))[:, :4]
    b = np.atleast_2d(np.asarray(boxes_b, dtype=np.float64))[:, :4]
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def clip_box(box, shape: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
    """Round a box to integer pixels and clip it to an image.

    Args:
        box (array-like): ``[x1, y1, x2, y2]``.
        shape (Sequence[int]): The image shape ``(H, W, ...)``.

    Returns:
        tuple[int, int, int, int] | None: The clipped box, or ``None`` when nothing of
            it lies inside the image.
    """
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_square(frame: np.ndarray, box, scale: float = 1.0
                ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Crop a square, centred on a box, padding with black where it leaves the frame.

    The side is ``scale * max(box width, box height)``. Unlike clipping, padding keeps
    the crop square and centred on the box even at the frame border, so a point found in
    the crop maps back exactly with ``frame_xy = offset + crop_xy``.

    Args:
        frame (np.ndarray): Image ``(H, W)`` or ``(H, W, C)``.
        box (array-like): ``[x1, y1, x2, y2]`` with a positive width and height.
        scale (float, optional): Side length relative to the longer box side. Defaults
            to ``1.0``.

    Returns:
        tuple[np.ndarray, tuple[int, int], int]: The ``(side, side[, C])`` crop, the
            frame coordinates ``(x, y)`` of its top-left pixel (negative when it starts
            outside the frame), and the side length in pixels.

    Raises:
        ValueError: If the box has no area or ``scale`` is not positive.
    """
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Cannot crop a box without area: {[x1, y1, x2, y2]}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    side = max(1, int(round(max(x2 - x1, y2 - y1) * scale)))
    left = int(round((x1 + x2) / 2.0 - side / 2.0))
    top = int(round((y1 + y2) / 2.0 - side / 2.0))
    height, width = frame.shape[:2]

    # The part of the square inside the frame (possibly empty), then pad it back out.
    x0, x1 = min(max(left, 0), width), min(max(left + side, 0), width)
    y0, y1 = min(max(top, 0), height), min(max(top + side, 0), height)
    pad_left, pad_top = min(side, max(0, -left)), min(side, max(0, -top))
    pad_right, pad_bottom = side - (x1 - x0) - pad_left, side - (y1 - y0) - pad_top
    inside = frame[y0:y1, x0:x1]
    if inside.size == 0:  # the square lies entirely outside the frame
        crop = np.zeros((side, side) + frame.shape[2:], dtype=frame.dtype)
    else:
        crop = cv2.copyMakeBorder(inside, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=0)
    return crop, (left, top), side


def assign_ids(face_boxes, subject_boxes, subject_ids, min_coverage: float = 0.5
               ) -> list:
    """Give each face the id of the tracked subject whose box contains it.

    Coverage is the fraction of the face box that lies inside a subject box. Faces and
    subjects are matched one-to-one by maximising total coverage (Hungarian algorithm),
    so two faces never share a subject.

    Args:
        face_boxes (array-like): ``(F, 4)`` face boxes.
        subject_boxes (array-like): ``(S, 4)`` tracked subject (person) boxes.
        subject_ids (Sequence[int]): The ``S`` track ids, aligned with ``subject_boxes``.
        min_coverage (float, optional): Minimum fraction of a face inside a subject box
            for the pair to count as a match. Defaults to ``0.5``.

    Returns:
        list[int | None]: One entry per face: the matched subject id, or ``None``.
    """
    from scipy.optimize import linear_sum_assignment

    faces = np.atleast_2d(np.asarray(face_boxes, dtype=np.float64)).reshape(-1, 4)
    subjects = np.atleast_2d(np.asarray(subject_boxes, dtype=np.float64)).reshape(-1, 4)
    ids = [None] * len(faces)
    if len(faces) == 0 or len(subjects) == 0:
        return ids

    top_left = np.maximum(faces[:, None, :2], subjects[None, :, :2])
    bottom_right = np.minimum(faces[:, None, 2:], subjects[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    face_area = np.prod(np.clip(faces[:, 2:] - faces[:, :2], 0.0, None), axis=1)
    coverage = np.divide(wh[..., 0] * wh[..., 1], face_area[:, None],
                         out=np.zeros((len(faces), len(subjects))),
                         where=face_area[:, None] > 0)

    # Pairs below the threshold must not take part in the matching: otherwise a weak
    # pair can pull a face away from the subject box that fully contains it.
    coverage[coverage < min_coverage] = 0.0
    rows, cols = linear_sum_assignment(coverage, maximize=True)
    for face_index, subject_index in zip(rows, cols):
        if coverage[face_index, subject_index] > 0.0:
            ids[face_index] = int(subject_ids[subject_index])
    return ids
