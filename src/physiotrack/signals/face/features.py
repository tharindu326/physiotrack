"""Per-frame facial geometry measured from face landmarks.

Eye aspect ratio, mouth aspect ratio and iris position are pure geometry on landmark
pixel coordinates, so they work on either landmark layout the library produces: the
478-point MediaPipe face mesh (``"FACEMESH"``, from
[`FaceLandmarks`][physiotrack.FaceLandmarks]) and the 68 face points of COCO-WholeBody
pose (``"WHOLEBODY"``, ids 23-90 in iBUG-68 order). The two layouts number their points
differently -- id 33 is an eye corner in one and a jaw point in the other -- so the
layout is always taken from the keypoints themselves, or must be named explicitly for
plain dicts.

Validation: the measures are checked against closed-form values on synthetic
landmark geometry, and on two real MediaPipe face meshes (a committed fixture) against
the project's face-analysis validation implementation, in ``tests/test_motion.py``.
"""
import math

__all__ = ["FACE_LANDMARK_TABLES", "eye_aspect_ratio", "mouth_aspect_ratio",
           "iris_position"]

# Landmark ids per layout. Eyes list the six points (p1..p6) of Soukupova & Cech: p1/p4
# are the eye corners, p2/p3 the upper lid and p6/p5 the lower lid directly below them.
# "Right"/"left" are the subject's own. Mouth: the two outer lip corners, and the inner
# lip points at the midline. Iris: (iris centre, image-left eye corner, image-right eye
# corner) -- only the face mesh has iris points.
FACE_LANDMARK_TABLES = {
    "FACEMESH": {
        "right_eye": (33, 160, 158, 133, 153, 144),
        "left_eye": (362, 385, 387, 263, 373, 380),
        "mouth_corners": (61, 291),
        "mouth_opening": (13, 14),
        "right_iris": (468, 33, 133),
        "left_iris": (473, 362, 263),
    },
    "WHOLEBODY": {
        # iBUG-68 36-41 (right eye) and 42-47 (left eye), offset by 23.
        "right_eye": (59, 60, 61, 62, 63, 64),
        "left_eye": (65, 66, 67, 68, 69, 70),
        # iBUG 48 / 54 outer corners, 62 / 66 inner-lip midline.
        "mouth_corners": (71, 77),
        "mouth_opening": (85, 89),
    },
}


def _points(source, layout):
    """Return ``({id: (x, y)}, layout)`` for one face's landmarks.

    The layout comes from a ``Keypoints`` object (directly, via an ``Instance``, or via
    a single-instance ``Result``); plain dicts carry no layout, so it must be given.
    """
    keypoints = source
    instances = getattr(source, "instances", None)
    if instances is not None:
        if len(instances) != 1:
            raise ValueError(
                f"This measurement is defined for one face, but the result holds "
                f"{len(instances)}. Pass a single face, e.g. `result[0]`."
            )
        keypoints = instances[0].keypoints
    elif hasattr(source, "keypoints") and not isinstance(source, dict):
        keypoints = source.keypoints
    elif isinstance(source, dict):
        keypoints = source.get("keypoints")

    if keypoints is None:
        return {}, layout
    own = getattr(keypoints, "architecture", None)
    if own is not None:
        if layout is not None and layout != own:
            raise ValueError(f"layout={layout!r} contradicts the keypoints' own {own!r}.")
        layout = own
    if layout is None:
        raise ValueError(
            "Plain keypoint dicts carry no layout; pass layout='FACEMESH' or "
            "layout='WHOLEBODY'."
        )
    if layout not in FACE_LANDMARK_TABLES:
        raise ValueError(
            f"No face landmarks in layout {layout!r}; face geometry is defined for "
            f"{', '.join(FACE_LANDMARK_TABLES)}."
        )
    points = {}
    for kp in keypoints:
        if isinstance(kp, dict):
            points[kp["id"]] = (float(kp["x"]), float(kp["y"]))
        else:
            points[kp.id] = (float(kp.x), float(kp.y))
    return points, layout


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eye_ratio(points, ids):
    if not all(i in points for i in ids):
        return None
    p1, p2, p3, p4, p5, p6 = (points[i] for i in ids)
    width = _distance(p1, p4)
    if width == 0:
        return None
    return (_distance(p2, p6) + _distance(p3, p5)) / (2.0 * width)


def eye_aspect_ratio(keypoints, layout=None):
    """Eye aspect ratio (EAR) of each eye: lid opening over eye width.

    ``EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)`` from Soukupova & Cech,
    "Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016. It is roughly
    constant (about 0.25-0.35) while the eye is open and drops towards 0 as it closes,
    and it is invariant to face size and in-plane rotation.

    Args:
        keypoints (Keypoints | Instance | Result | dict | list[dict]): One face's
            landmarks: ``"FACEMESH"`` keypoints from
            [`FaceLandmarks`][physiotrack.FaceLandmarks], ``"WHOLEBODY"`` pose
            keypoints, a face ``Instance`` / single-face ``Result`` carrying either, or
            their serialized dicts.
        layout (str, optional): ``"FACEMESH"`` or ``"WHOLEBODY"``. Required for plain
            dicts; otherwise read from the keypoints. Defaults to ``None``.

    Returns:
        dict[str, float | None]: ``{"left", "right", "mean"}``; an entry is ``None``
            when its landmarks are missing (``mean`` needs both eyes). All ``None``
            when the face has no landmarks.

    Raises:
        ValueError: If the layout is unknown, missing for plain dicts, or has no face
            landmarks (e.g. ``"COCO"``), or a multi-face ``Result`` is passed.

    Example:
        ```python
        import physiotrack as pt

        faces = pt.FaceLandmarks()(frame, pt.Face()(frame))
        pt.signals.eye_aspect_ratio(faces[0])      # {'left': 0.29, 'right': 0.31, 'mean': 0.30}
        ```

    See Also:
        [`detect_blinks`][physiotrack.signals.detect_blinks]: blinks from the EAR series.
    """
    points, layout = _points(keypoints, layout)
    if not points:
        return {"left": None, "right": None, "mean": None}
    table = FACE_LANDMARK_TABLES[layout]
    left = _eye_ratio(points, table["left_eye"])
    right = _eye_ratio(points, table["right_eye"])
    mean = None if left is None or right is None else (left + right) / 2.0
    return {"left": left, "right": right, "mean": mean}


def mouth_aspect_ratio(keypoints, layout=None):
    """Mouth aspect ratio (MAR): inner-lip opening over mouth width.

    The gap between the upper and lower inner lip at the midline divided by the distance
    between the outer mouth corners -- the library's mouth measure, built like the eye
    aspect ratio (several variants exist in the literature; this one uses one vertical
    pair). It is 0 with the lips closed and grows as the mouth opens, independent of face
    size; it was validated against the FELT reference on RAVDESS speech (see the face
    validation guide).

    Args:
        keypoints (Keypoints | Instance | Result | dict | list[dict]): One face's
            landmarks (see [`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio]).
        layout (str, optional): ``"FACEMESH"`` or ``"WHOLEBODY"``; required for plain
            dicts. Defaults to ``None``.

    Returns:
        float | None: The ratio, or ``None`` when the landmarks are missing or the mouth
            width is zero.

    Raises:
        ValueError: As for [`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio].

    Example:
        ```python
        mar = pt.signals.mouth_aspect_ratio(faces[0])
        ```

    See Also:
        [`mouth_movement`][physiotrack.signals.mouth_movement]: its frame-to-frame change.
    """
    points, layout = _points(keypoints, layout)
    table = FACE_LANDMARK_TABLES.get(layout, {})
    ids = table.get("mouth_corners", ()) + table.get("mouth_opening", ())
    if not points or not all(i in points for i in ids):
        return None
    left, right = (points[i] for i in table["mouth_corners"])
    upper, lower = (points[i] for i in table["mouth_opening"])
    width = _distance(left, right)
    return None if width == 0 else _distance(upper, lower) / width


def _iris_in_eye(points, ids):
    if not all(i in points for i in ids):
        return None, None
    iris, a, b = (points[i] for i in ids)
    ux, uy = b[0] - a[0], b[1] - a[1]
    width = math.hypot(ux, uy)
    if width == 0:
        return None, None
    ux, uy = ux / width, uy / width
    dx, dy = iris[0] - a[0], iris[1] - a[1]
    # Along the eye axis, and along its normal (-uy, ux), which points down the image.
    return (dx * ux + dy * uy) / width, (-dx * uy + dy * ux) / width


def iris_position(keypoints, layout=None):
    """Position of each iris within its eye, in eye widths.

    For each eye the iris centre is projected onto the line through the two eye corners.
    ``x`` runs from 0 at the image-left corner to 1 at the image-right corner (about
    0.5 when looking straight ahead) and ``y`` is the perpendicular offset, positive
    downwards, both in units of the eye's width. Because both eyes use the same image
    direction the two can be averaged. This is a 2D, head-relative gaze cue that needs
    no camera model; for a 3D gaze direction use
    [`GazeEstimator`][physiotrack.GazeEstimator].

    Args:
        keypoints (Keypoints | Instance | Result | dict | list[dict]): One face's
            ``"FACEMESH"`` landmarks -- the only layout with iris points.
        layout (str, optional): ``"FACEMESH"``; required for plain dicts. Defaults to
            ``None``.

    Returns:
        dict[str, float | None]: ``{"left_x", "left_y", "right_x", "right_y", "x",
            "y"}``, where ``x`` / ``y`` average both eyes; entries are ``None`` when
            their landmarks are missing.

    Raises:
        ValueError: If the layout has no iris points (``"WHOLEBODY"``), or as for
            [`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio].

    Example:
        ```python
        pos = pt.signals.iris_position(faces[0])
        looking_right = pos["x"] > 0.6
        ```
    """
    points, layout = _points(keypoints, layout)
    table = FACE_LANDMARK_TABLES[layout] if layout in FACE_LANDMARK_TABLES else {}
    if layout is not None and "left_iris" not in table:
        raise ValueError(f"The {layout} layout has no iris points; use FACEMESH landmarks.")
    empty = {"left_x": None, "left_y": None, "right_x": None, "right_y": None,
             "x": None, "y": None}
    if not points:
        return empty
    left_x, left_y = _iris_in_eye(points, table["left_iris"])
    right_x, right_y = _iris_in_eye(points, table["right_iris"])
    both = left_x is not None and right_x is not None
    return {
        "left_x": left_x, "left_y": left_y, "right_x": right_x, "right_y": right_y,
        "x": (left_x + right_x) / 2.0 if both else None,
        "y": (left_y + right_y) / 2.0 if both else None,
    }
