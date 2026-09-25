"""Face signals over a video: per-face table, blinks, mouth movement, window summaries.

These read the faces of a [`Video`][physiotrack.Video] run
([`FrameResult.faces`][physiotrack.FrameResult]) and follow each face by its ``id`` --
the track id of the subject it belongs to. Every processed frame counts: a frame in
which a face is absent, or has no face mesh, is a gap for that face. A gap always
interrupts a blink, a movement estimate and a summary window rather than being bridged,
so frames skipped by ``Video(fps=...)`` subsampling are never mistaken for gaps. Times
are the frames' timestamps; a record without one is rejected.

The definitions -- blink counting, blink rate, mouth movement and the window summary --
are those of the project's face-analysis validation (see the face validation guide),
and are cross-checked against that implementation in ``tests/test_motion.py``.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .features import eye_aspect_ratio, iris_position, mouth_aspect_ratio

__all__ = ["face_feature_sequence", "detect_blinks", "blink_rate", "mouth_movement",
           "face_window_summary"]


def _frames(data):
    """Return ``[(time, frame_index, faces)]`` for every frame, faces as a ``Result``."""
    from ...results import Result

    frames = []
    for position, frame in enumerate(data):
        if isinstance(frame, dict):
            faces = frame.get("faces")
            faces = Result.from_dict(faces) if faces is not None else None
            index, time = frame.get("frame_id"), frame.get("timestamp")
        else:
            faces = frame.faces
            index, time = frame.meta.frame_index, frame.meta.timestamp
        if time is None or index is None:
            raise ValueError(
                f"Frame record {position} has no timestamp or frame index; face signals "
                f"are measured in seconds and need both (Video.run records them)."
            )
        frames.append((float(time), int(index), faces))
    return frames


def _frame_interval(frames):
    """Mean time between processed frames -- the processing rate's reciprocal."""
    if len(frames) < 2:
        return float("nan")
    return (frames[-1][0] - frames[0][0]) / (len(frames) - 1)


def _face_values(face):
    """The per-face measures, keyed like the table columns."""
    has_mesh = face.keypoints is not None
    ear = eye_aspect_ratio(face) if has_mesh else {}
    iris = (iris_position(face) if has_mesh and face.keypoints.architecture == "FACEMESH"
            else {})
    o, e, g = face.orientation or {}, face.expression or {}, face.gaze or {}
    q, r = face.quality or {}, face.regions or {}
    box = face.box if face.box is not None else [None] * 4
    vector = g.get("vector") or [None] * 3
    row = {
        "detection_id": face.id,
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
        "confidence": face.confidence,
        "yaw": o.get("yaw"), "pitch": o.get("pitch"), "roll": o.get("roll"),
        "ear_left": ear.get("left"), "ear_right": ear.get("right"), "ear": ear.get("mean"),
        "mar": mouth_aspect_ratio(face) if has_mesh else None,
        "iris_left_x": iris.get("left_x"), "iris_left_y": iris.get("left_y"),
        "iris_right_x": iris.get("right_x"), "iris_right_y": iris.get("right_y"),
        "iris_x": iris.get("x"), "iris_y": iris.get("y"),
        "expression": e.get("label"), "expression_confidence": e.get("confidence"),
        "valence": e.get("valence"), "arousal": e.get("arousal"),
        "gaze_pitch": g.get("pitch"), "gaze_yaw": g.get("yaw"),
        "gaze_x": vector[0], "gaze_y": vector[1], "gaze_z": vector[2],
        "brightness": q.get("brightness"), "sharpness": q.get("sharpness"),
        "area_ratio": q.get("area_ratio"),
        "skin_fraction": (r["fractions"].get("skin", 0.0) if r else None),
    }
    for label, score in (e.get("scores") or {}).items():
        row[f"expression_{label}"] = score
    return row


def face_feature_sequence(data):
    """Tabulate every face of a video run, one row per face per frame.

    Collects what the Video's face stages produced -- box, orientation, expression,
    gaze, quality, regions -- together with the landmark geometry
    ([`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio],
    [`mouth_aspect_ratio`][physiotrack.signals.mouth_aspect_ratio],
    [`iris_position`][physiotrack.signals.iris_position]) computed from each face's
    mesh. Measures a run did not produce are ``NaN`` / ``None``. For per-face windowed
    statistics use [`face_window_summary`][physiotrack.signals.face_window_summary].

    Args:
        data (VideoResults | Iterable[FrameResult] | list[dict]): The output of
            [`Video.run`][physiotrack.Video.run], or its serialized frame dicts. Face
            meshes are only in the dicts when they were saved with
            ``include_arrays=True``.

    Returns:
        pandas.DataFrame: Columns ``time`` (s), ``frame``, ``detection_id`` (the face's
            ``id``, ``None`` without a tracker), ``x1``..``y2`` (box, px),
            ``confidence``, ``yaw`` / ``pitch`` / ``roll`` (deg), ``ear_left`` /
            ``ear_right`` / ``ear``, ``mar``, ``iris_left_x`` .. ``iris_y`` (eye widths),
            ``expression`` / ``expression_confidence`` / ``expression_<class>``,
            ``valence`` / ``arousal``, ``gaze_pitch`` / ``gaze_yaw`` (deg), ``gaze_x`` /
            ``gaze_y`` / ``gaze_z``, ``brightness`` / ``sharpness`` / ``area_ratio``,
            ``skin_fraction``; rows in frame order.

    Raises:
        ValueError: If a frame record has no timestamp or frame index.

    Example:
        ```python
        import physiotrack as pt

        results = pt.Video("clip.mp4", detector=pt.Face(), tracker=pt.Tracker(),
                           face_stages=[pt.FaceLandmarks()]).run()
        df = pt.signals.face_feature_sequence(results)
        df.groupby("detection_id").ear.describe()
        ```
    """
    rows = []
    for time, index, faces in _frames(data):
        for face in (faces or []):
            rows.append({"time": time, "frame": index, **_face_values(face)})
    df = pd.DataFrame(rows, columns=None if rows else ["time", "frame", "detection_id"])
    text = {"detection_id", "expression"}
    numeric = [c for c in df.columns if c not in text]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    return df


def _track(frames, detection_id, value):
    """One face over every processed frame.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The frame times, ``value(face)`` per
            frame (``NaN`` where the face is absent or the value is ``None``), and
            whether the face was present in each frame.
    """
    values, present = np.full(len(frames), np.nan), np.zeros(len(frames), bool)
    for position, (time, _, faces) in enumerate(frames):
        matches = [f for f in (faces or []) if detection_id is None or f.id == detection_id]
        if detection_id is None and len(matches) > 1:
            ids = sorted({f.id for f in matches}, key=str)
            raise ValueError(
                f"Frame at t={time:.3f}s has {len(matches)} faces (ids {ids}); pass "
                f"detection_id to choose one. Faces get ids from a tracker."
            )
        if matches:
            present[position] = True
            v = value(matches[0])
            if v is not None:
                values[position] = float(v)
    return np.array([f[0] for f in frames]), values, present


def _mean_ear(face):
    return eye_aspect_ratio(face)["mean"] if face.keypoints is not None else None


def _mar(face):
    return mouth_aspect_ratio(face) if face.keypoints is not None else None


def _check_blink_parameters(threshold, min_closed_frames):
    if not threshold > 0:
        raise ValueError(f"threshold must be greater than zero, got {threshold}.")
    if int(min_closed_frames) != min_closed_frames or min_closed_frames < 1:
        raise ValueError(f"min_closed_frames must be an integer >= 1, got {min_closed_frames}.")


def _blink_events(times, ear, threshold, min_closed_frames):
    """Blinks in an EAR series (``NaN`` = gap), as ``(start, end, duration, min_ear, end_index)``."""
    events, closed_start = [], None
    for i, value in enumerate(ear):
        if np.isnan(value):                 # a gap never bridges a closure
            closed_start = None
        elif value < threshold:
            closed_start = i if closed_start is None else closed_start
        else:
            if closed_start is not None and i - closed_start >= min_closed_frames:
                events.append((times[closed_start], times[i], times[i] - times[closed_start],
                               float(ear[closed_start:i].min()), i))
            closed_start = None
    return events


def detect_blinks(data, detection_id=None, threshold=0.22, min_closed_frames=3):
    """Detect blinks of one face from its eye-aspect-ratio series.

    A blink is at least ``min_closed_frames`` consecutive frames with mean EAR below
    ``threshold``, counted when the eye reopens. A gap -- a frame without the face or
    without its mesh -- interrupts a closure, which is then not counted. The duration
    runs from the first closed frame to the reopening frame (closed frames times the
    frame interval). The EAR threshold (Soukupová & Čech, CVWW 2016) and the defaults
    ``threshold=0.22``, ``min_closed_frames=3`` are the settings validated on MPEBlink
    (see the face validation guide); they suit frontal faces at 25-30 fps. EAR depends
    on the viewing angle, so tune ``threshold`` per camera setup.

    Args:
        data (VideoResults | Iterable[FrameResult] | list[dict]): A Video run whose face
            stages included [`FaceLandmarks`][physiotrack.FaceLandmarks].
        detection_id (int, optional): The face to analyse. Defaults to ``None``, which
            is valid only when every frame has at most one face.
        threshold (float, optional): Mean EAR below which the eyes count as closed.
            Defaults to ``0.22``.
        min_closed_frames (int, optional): Fewest consecutive closed frames that make
            a blink. Defaults to ``3``.

    Returns:
        pandas.DataFrame: One row per blink with ``start`` and ``end`` (s),
            ``duration`` (s) and ``min_ear`` (the deepest closure).

    Raises:
        ValueError: If ``detection_id`` is ``None`` but a frame holds several faces, a
            frame record has no timestamp, or a parameter is out of range.

    Example:
        ```python
        blinks = pt.signals.detect_blinks(results, detection_id=1)
        print(len(blinks), blinks.duration.mean())
        ```
    """
    _check_blink_parameters(threshold, min_closed_frames)
    times, ear, _ = _track(_frames(data), detection_id, _mean_ear)
    events = _blink_events(times, ear, threshold, min_closed_frames)
    return pd.DataFrame([e[:4] for e in events], columns=["start", "end", "duration", "min_ear"])


def blink_rate(data, detection_id=None, threshold=0.22, min_closed_frames=3):
    """Blinks per minute of one face over the time it was present.

    The rate divides the blinks of
    [`detect_blinks`][physiotrack.signals.detect_blinks] by the time the face was in
    view -- its frames times the mean processing interval -- including frames where its
    mesh was not found. Frames without the face do not count.

    Args:
        data (VideoResults | Iterable[FrameResult] | list[dict]): As for
            [`detect_blinks`][physiotrack.signals.detect_blinks].
        detection_id (int, optional): The face to analyse. Defaults to ``None``.
        threshold (float, optional): As for ``detect_blinks``. Defaults to ``0.22``.
        min_closed_frames (int, optional): As for ``detect_blinks``. Defaults to ``3``.

    Returns:
        float: Blinks per minute, or ``NaN`` when the run has fewer than two frames or
            the face was never present.

    Raises:
        ValueError: As for [`detect_blinks`][physiotrack.signals.detect_blinks].

    Example:
        ```python
        rate = pt.signals.blink_rate(results, detection_id=1)   # e.g. 17.2
        ```
    """
    _check_blink_parameters(threshold, min_closed_frames)
    frames = _frames(data)
    times, ear, present = _track(frames, detection_id, _mean_ear)
    observed = present.sum() * _frame_interval(frames)
    if not observed > 0:
        return float("nan")
    return len(_blink_events(times, ear, threshold, min_closed_frames)) / (observed / 60.0)


def _movement(times, mar):
    """|MAR_t - MAR_prev| and its rate, restarting at 0 after every gap."""
    movement, velocity = np.full(mar.size, np.nan), np.full(mar.size, np.nan)
    previous = None
    for i, value in enumerate(mar):
        if np.isnan(value):
            previous = None
            continue
        if previous is None:
            movement[i] = velocity[i] = 0.0
        else:
            movement[i] = abs(value - mar[previous])
            velocity[i] = movement[i] / (times[i] - times[previous])
        previous = i
    return movement, velocity


def mouth_movement(data, detection_id=None):
    """Frame-to-frame movement of one face's mouth.

    The movement is the absolute change of the
    [`mouth_aspect_ratio`][physiotrack.signals.mouth_aspect_ratio] since the previous
    frame, and the velocity that change divided by the time between the two frames. The
    first frame after a gap has movement and velocity ``0``. These are the measures
    validated on FELT / RAVDESS speech (see the face validation guide).

    Args:
        data (VideoResults | Iterable[FrameResult] | list[dict]): A Video run whose face
            stages included [`FaceLandmarks`][physiotrack.FaceLandmarks].
        detection_id (int, optional): The face to analyse. Defaults to ``None``, which
            is valid only when every frame has at most one face.

    Returns:
        pandas.DataFrame: One row per processed frame with ``time`` (s), ``mar``,
            ``mar_movement`` (MAR units) and ``mar_velocity`` (MAR units per second);
            ``NaN`` where the face or its mesh is missing.

    Raises:
        ValueError: If ``detection_id`` is ``None`` but a frame holds several faces, or
            a frame record has no timestamp.

    Example:
        ```python
        motion = pt.signals.mouth_movement(results, detection_id=1)
        speaking = motion.mar_velocity.rolling(15).mean() > 0.5
        ```
    """
    times, mar, _ = _track(_frames(data), detection_id, _mar)
    movement, velocity = _movement(times, mar)
    return pd.DataFrame({"time": times, "mar": mar, "mar_movement": movement,
                         "mar_velocity": velocity})


# Measures summarised per window: column -> per-face value.
_SUMMARISED = ("yaw", "pitch", "roll", "ear", "iris_x", "iris_y", "mar", "mar_movement",
               "brightness", "sharpness", "area_ratio")


def face_window_summary(data, window=5.0, threshold=0.22, min_closed_frames=3):
    """Sliding-window statistics of every face, one row per face per frame.

    For each face and frame, summarises the face's last ``window`` seconds of
    observations -- the most recent ``round(window / interval)`` frames in which it
    was present, restarting whenever it leaves the view -- with the mean, standard
    deviation (population), minimum and maximum of each measure, the number of blinks
    that ended in the window, and the most frequent expression. Frames without a
    measure (e.g. no mesh) are skipped for that measure only. This is the temporal
    aggregation of the project's face-analysis validation (see the face validation
    guide).

    Args:
        data (VideoResults | Iterable[FrameResult] | list[dict]): A Video run with face
            stages.
        window (float, optional): Window length in seconds. Defaults to ``5.0``.
        threshold (float, optional): Blink EAR threshold, as for
            [`detect_blinks`][physiotrack.signals.detect_blinks]. Defaults to ``0.22``.
        min_closed_frames (int, optional): Blink minimum closure, as for
            ``detect_blinks``. Defaults to ``3``.

    Returns:
        pandas.DataFrame: Columns ``time`` (s), ``frame``, ``detection_id``,
            ``window_frames``, ``window_sec``, ``<measure>_mean`` / ``_std`` / ``_min``
            / ``_max`` for yaw, pitch, roll, ear, iris_x, iris_y, mar, mar_movement,
            brightness, sharpness and area_ratio (``NaN`` when the window has none),
            ``blink_events`` and ``dominant_expression``. Faces without an ``id`` are
            not summarised.

    Raises:
        ValueError: If ``window`` or a blink parameter is out of range, or a frame record
            has no timestamp.

    Example:
        ```python
        summary = pt.signals.face_window_summary(results, window=5.0)
        summary[summary.detection_id == 1][["time", "ear_mean", "blink_events"]]
        ```
    """
    if not window > 0:
        raise ValueError(f"window must be greater than zero, got {window}.")
    _check_blink_parameters(threshold, min_closed_frames)
    frames = _frames(data)
    interval = _frame_interval(frames)
    size = max(1, int(round(window / interval))) if np.isfinite(interval) else 1
    ids = sorted({f.id for _, _, faces in frames for f in (faces or []) if f.id is not None},
                 key=str)

    rows = []
    for rank, face_id in enumerate(ids):
        times, ear, present = _track(frames, face_id, _mean_ear)
        _, mar, _ = _track(frames, face_id, _mar)
        movement, _ = _movement(times, mar)
        ended = np.zeros(len(frames), bool)
        for event in _blink_events(times, ear, threshold, min_closed_frames):
            ended[event[4]] = True
        by_frame = [next((f for f in (faces or []) if f.id == face_id), None)
                    for _, _, faces in frames]

        buffer = []
        for position, (time, index, _) in enumerate(frames):
            face = by_frame[position]
            if face is None:          # the face left the view: restart its window
                buffer = []
                continue
            values = _face_values(face)
            values["mar_movement"] = None if np.isnan(movement[position]) else movement[position]
            values["blink"] = bool(ended[position])
            buffer = (buffer + [values])[-size:]
            row = {"_rank": rank, "time": time, "frame": index, "detection_id": face_id,
                   "window_frames": len(buffer), "window_sec": len(buffer) * interval}
            for name in _SUMMARISED:
                series = np.array([v[name] for v in buffer if v[name] is not None], float)
                series = series[np.isfinite(series)]
                row.update({f"{name}_mean": series.mean() if series.size else np.nan,
                            f"{name}_std": series.std() if series.size else np.nan,
                            f"{name}_min": series.min() if series.size else np.nan,
                            f"{name}_max": series.max() if series.size else np.nan})
            row["blink_events"] = sum(v["blink"] for v in buffer)
            labels = [v["expression"] for v in buffer if v["expression"] is not None]
            row["dominant_expression"] = Counter(labels).most_common(1)[0][0] if labels else None
            rows.append(row)
    if not rows:
        return pd.DataFrame(rows)
    table = pd.DataFrame(rows).sort_values(["frame", "_rank"], kind="stable")
    return table.drop(columns="_rank").reset_index(drop=True)
