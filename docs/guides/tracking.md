# Multi-object Tracking

Tracking turns per-frame detection boxes into **persistent tracks** — each subject
keeps the same integer `id` across frames, so you can follow a person through a
clip, draw movement trails, or lock onto a single subject. Physiotrack wraps four
interchangeable tracking backends behind one API: you feed a frame plus that
frame's detections and get back a [`TrackResult`][physiotrack.TrackResult] whose
instances carry stable ids.

Tracking is detection-driven: run a [`Detection`][physiotrack.Detection] model
first, then pass its boxes to the tracker. For a fully wired video pipeline
(detect → pose → track → overlay) see the [Video Pipeline guide](video.md).

## Quick start

```python
import numpy as np
import cv2
import physiotrack as pt

det = pt.Detection.Person(device=0)
tracker = pt.Tracker(pt.TrackerConfig(tracker="ocsort", classes=[0]))

cap = cv2.VideoCapture("clip.mp4")
while True:
    ok, frame = cap.read()            # BGR frame (H, W, 3)
    if not ok:
        break

    # Build the (N, 6) detection array: [x1, y1, x2, y2, conf, cls]
    result = det.predict(frame)
    detections = np.array(
        [[*inst.box, inst.confidence, inst.cls] for inst in result],
        dtype=np.float32,
    ).reshape(-1, 6)

    track_result = tracker.track(frame, detections)   # -> pt.TrackResult
    print(track_result.ids)                            # e.g. [1, 2, 5]

    annotated = track_result.plot()                    # tracker overlay (BGR)
    cv2.imshow("tracks", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
```

!!! warning "`track()` expects an `(N, 6)` NumPy array"
    [`Tracker.track`][physiotrack.Tracker.track] filters detections by class with
    `detections[:, 5]`, so it needs a NumPy array shaped `(N, 6)` —
    `[x1, y1, x2, y2, conf, cls]` — **not** a `dict`. Do not pass
    `det.predict(frame).to_dict()` (that returns a dict). Build the array from the
    [`Result`][physiotrack.Result] instances as shown above.

## Available backends

The backend is chosen by
[`TrackerConfig.tracker_type`][physiotrack.TrackerConfig] (case-insensitive; you
can also pass the friendly alias `tracker=`). OC-SORT is the default.

| `tracker_type` | Backend | Appearance model | Notes |
| --- | --- | --- | --- |
| `"ocsort"` | OC-SORT | none | **Default.** Observation-centric SORT; fast, motion-only. |
| `"bytetrack"` | ByteTrack | none | IOU + high/low score association; robust to low-confidence boxes. |
| `"strongsort"` | StrongSORT | OSNet ReID | Kalman motion **plus** appearance re-identification (runs a neural net on `device`). |
| `"boosttrack"` | BoostTrack | none | IOU / Mahalanobis / shape similarity with detection-confidence boosting. |

=== "OC-SORT (default)"

    ```python
    cfg = pt.TrackerConfig(tracker="ocsort", classes=[0])
    tracker = pt.Tracker(cfg)
    ```

=== "ByteTrack"

    ```python
    cfg = pt.TrackerConfig(tracker="bytetrack", classes=[0])
    tracker = pt.Tracker(cfg)
    ```

=== "StrongSORT"

    ```python
    # Appearance ReID; set device="cuda" for the OSNet network.
    cfg = pt.TrackerConfig(tracker="strongsort", classes=[0], device="cuda")
    tracker = pt.Tracker(cfg)
    ```

=== "BoostTrack"

    ```python
    cfg = pt.TrackerConfig(tracker="boosttrack", classes=[0])
    tracker = pt.Tracker(cfg)
    ```

!!! info "Only the selected backend's hyper-parameters are used"
    Every backend has its own set of fields on the config (all prefixed with the
    backend name). The others remain settable but are ignored at runtime, so it is
    safe to keep a single [`TrackerConfig`][physiotrack.TrackerConfig] around and
    just flip `tracker_type`.

## The detect → track loop

Call [`track`][physiotrack.Tracker.track] **once per frame, in order**:

```python
track_result = tracker.track(frame, detections)
```

- `frame` — the current BGR frame `(H, W, 3)`. Used by appearance-based backends
  (StrongSORT) and for rendering the overlay.
- `detections` — an `(N, 6)` array, rows `[x1, y1, x2, y2, conf, cls]`. Rows whose
  `cls` is not in [`TrackerConfig.classes`][physiotrack.TrackerConfig] are dropped
  before tracking (default `classes=[0]`, the COCO "person" class).

!!! tip "A Tracker is stateful — one instance per video stream"
    The tracker holds Kalman state, track history, id counters and (optionally)
    the locked subject between calls. Create **one**
    [`Tracker`][physiotrack.Tracker] per video and feed it frames in order. Reusing
    a tracker across unrelated clips will carry stale state; construct a fresh one
    for each stream.

## Working with results

[`track`][physiotrack.Tracker.track] returns a
[`TrackResult`][physiotrack.TrackResult] that behaves like a sequence of tracked
[`Instance`][physiotrack.Instance] objects:

```python
track_result = tracker.track(frame, detections)

track_result.ids            # list[int] — persistent ids this frame, e.g. [1, 2, 5]
track_result.boxes          # np.ndarray (M, 4) — [x1, y1, x2, y2] rows
len(track_result)           # number of active tracks

for inst in track_result:   # iterate instances
    print(inst.id, inst.box, inst.cls, inst.confidence)

annotated = track_result.plot()          # tracker's own rich overlay (or draw yourself)
data = track_result.to_dict()            # {"task": "track", "instances": [...]} — JSON-friendly
```

`plot()` with no argument returns the tracker's own `rendered` overlay (boxes,
labels, trails, locked-subject box). Pass a `frame` to draw simple `ID <n>` boxes on a
copy of your own frame instead: `track_result.plot(frame, color=(0, 255, 0))`. See
[Result objects](../api/results.md) for the full result API.

## Configuration groups

[`TrackerConfig`][physiotrack.TrackerConfig] carries ~46 fields. They fall into a
few groups; a summary follows — see the full [Tracking API](../api/tracking.md)
for every field and default.

| Group | Representative fields | What it controls |
| --- | --- | --- |
| General | `tracker_type`, `classes`, `device`, `trail_length` | Backend choice, which classes to track, compute device, history length. |
| Overlay / trails | `show_detection_boxes`, `show_original_tracks`, `show_locked_subject`, `show_tracking_tail`, `show_all_trails`, `tail_opacity`, `colors` | Which boxes/labels/trails the built-in overlay draws and in what colours. |
| Subject lock | `enable_subject_lock`, `required_consecutive_frames`, `inconsistent_motion_threshold`, `subject_reinit_iou_threshold` | The single-subject isolation heuristic (see below). |
| ByteTrack | `bytetrack_track_thresh`, `bytetrack_match_thresh`, `bytetrack_track_buffer`, `bytetrack_frame_rate` | ByteTrack association thresholds and buffering. |
| StrongSORT | `strongsort_reid_weights`, `strongsort_max_dist`, `strongsort_max_age`, `strongsort_ema_alpha`, … | ReID weights + motion/appearance matching. |
| OC-SORT | `ocsort_det_thresh`, `ocsort_max_age`, `ocsort_min_hits`, `ocsort_iou_thresh`, `ocsort_inertia`, `ocsort_use_byte`, … | OC-SORT association and lifetime. |
| BoostTrack | `boosttrack_det_thresh`, `boosttrack_lambda_iou`, `boosttrack_lambda_mhd`, `boosttrack_lambda_shape`, `boosttrack_max_age`, … | BoostTrack similarity weights and boosting. |

Settings can be passed as constructor kwargs or set as attributes afterwards (both
may be mixed); an unknown keyword raises `TypeError`, guarding against typos:

```python
cfg = pt.TrackerConfig(tracker="ocsort", classes=[0])  # constructor kwargs
cfg.debug_mode = True                                   # attribute assignment
print(cfg)                                              # inspect resolved settings
```

??? note "Subject lock — following a single subject"
    With `enable_subject_lock=True` the tracker locks onto one stable subject
    using a stability + IOU heuristic: a track must persist for
    `required_consecutive_frames` frames to be promoted, is re-matched by IOU
    against `subject_reinit_iou_threshold`, and is dropped after
    `inconsistent_motion_threshold` consecutive low-IOU frames (or after enough
    misses). When active, the overlay draws the locked subject in blue with a movement
    trail (`show_locked_subject` / `show_tracking_tail`, both on by default), and
    the locked subject is exposed as `tracker.locked_subject_id` /
    `tracker.locked_subject_box`. This is what the `examples/tracker_aided_pose_video.py`
    demo enables so pose is run only on the isolated subject.

## See also

- [Tracking API reference](../api/tracking.md) — [`Tracker`][physiotrack.Tracker],
  [`TrackerConfig`][physiotrack.TrackerConfig] and all ~46 config fields.
- [Video Pipeline guide](video.md) — run detection + tracking + pose end-to-end.
- [Detection guide](detection.md) — produce the boxes the tracker consumes.
- [Result objects](../api/results.md) — [`TrackResult`][physiotrack.TrackResult]
  and [`Instance`][physiotrack.Instance].
- [Model Zoo](../model-zoo.md) — detection weights that feed the tracker.
