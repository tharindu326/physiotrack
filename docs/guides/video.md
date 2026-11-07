# Video Pipeline

[`Video`][physiotrack.Video] is the high-level orchestrator that runs the **full
inference pipeline** over a clip, camera device or RTSP stream. You attach the
predictors you want — detection, pose, tracking, segmentation, face analysis,
depth — and it drives them frame-by-frame (optionally in batches), composites every
enabled model's output onto each frame, writes an annotated MP4, and returns the
per-frame results as structured data.

Every stage is optional: the same class covers a bare pose-only pass and the
complete multi-model pipeline. Nothing runs until you call
[`run`][physiotrack.Video.run].

![Full inference overlay: pose, tracking, segmentation, depth and side panels composited on one frame](../images/full_inference_overlay.png)

## What Video orchestrates

The per-batch pipeline order is:

**detection → tracking → pose → segmentation → faces (detection, subject ids, face
stages) → depth → overlay / compositing**

You attach models by passing already-constructed predictor instances to the
constructor; any left as `None` are simply skipped. On top of the model overlays,
`Video` can build side panels — a keypoint-motion plot, joint-angle / ROM grids, a
ROM skeleton canvas, a top-down radar view, a depth view and an ego-video view.

!!! warning "Pairing a detector with pose"
    When you supply both a custom `detector` **and** a `pose` estimator, the pose
    estimator must be a `Pose.Custom` instance (so it consumes the external boxes).
    Otherwise `run()` raises `ValueError`. A pose-only pass (no `detector`) can use
    any pose preset, e.g. `Pose.VRStudent`.

## Quick start

The minimal pose-only pass (from `examples/pose_video.py`):

```python
from physiotrack import Pose, Video
from pathlib import Path

pose = Pose.VRStudent(verbose=False, device=0)

video = Video(
    source="clip.mp4",
    pose=pose,
    output_dir="output",
    verbose=True,
)

results = video.run(
    "output/clip_poses.mp4",     # annotated video (optional)
    "output/clip_result.json",   # per-frame JSON dump (optional)
)
print(f"Processed {len(results)} frames")
```

## Constructing a Video

Pass predictor **instances** (or `None` to skip a stage); `detector` and
`segmenter` also accept a `list` to run several of the same kind. The constructor
opens the source immediately (to read fps / resolution / frame count) but processes
nothing until `run()`.

```python
import physiotrack as pt

pose = pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.b_wholebody, device=0)
detector = pt.Detection.VRStudent(device=0)

cfg = pt.TrackerConfig(tracker="ocsort", classes=[0])
tracker = pt.Tracker(config=cfg)

video = pt.Video(
    source="clip.mp4",       # file path, rtsp:// URL, or int camera index
    pose=pose,
    detector=detector,       # requires a Pose.Custom estimator (see above)
    tracker=tracker,
    output_dir="output",
    batch_size=4,
    verbose=True,
)
```

### Key options

All are keyword-only constructor arguments; see the full
[Video API](../api/video.md) for the exhaustive list and defaults.

| Group | Arguments | What it controls |
| --- | --- | --- |
| Models | `detector`, `pose`, `segmenter`, `tracker`, `face`, `face_stages`, `depth`, `ego_video` | Which predictors run; `None` skips a stage. `detector` / `segmenter` accept a list. |
| Source / output | `source`, `output_dir`, `fps` | Input (file / RTSP / camera index), output directory, target processing frame rate (subsampling). |
| Geometry | `resize`, `rotate`, `orient` | `resize=(w, h)`; `rotate` = 90° CW; `orient` = explicit `0/90/180/270` fix for phone clips. |
| Radar / floor map | `floor_map`, `floor_map_background`, `floor_map_rotation` | Four `(x, y)` floor corners enable the top-down radar view (needs `tracker` + `pose`). |
| Depth view | `depth_colormap` | Matplotlib colormap for the depth panel (`"inferno"`, `"viridis"`, `"jet"`, …). |
| Motion plot | `plot_keypoint`, `plot_keypoint_name` | COCO keypoint id to plot as a live signal panel (needs `pose`). |
| Kinematics | `plot_angles`, `angle_joints`, `rom`, `rom_render` | Live joint-angle panel and clinical range-of-motion overlays / skeleton canvas (need `pose`). |
| Vitals | `rppg`, `hrv`, `respiration`, `respiration_source`, `rppg_method`, `rppg_roi`, `rppg_window_sec` | Contactless heart rate / HRV (rPPG) and respiration panels + `vitals` JSON. See below. |
| Runtime | `batch_size`, `verbose`, `show_fps`, `show` | Frames per batch, logging, FPS stats, live OpenCV preview window. |

### Vitals (rPPG heart rate · HRV · respiration)

Physiological panels are opt-in and each has a clear signal source:

| Panel | Enable | Signal source | Needs |
| --- | --- | --- | --- |
| Heart rate (rPPG) | `rppg=True` | Blood-volume pulse from a SegFace skin segmentation | nothing extra (SegFace is built in) |
| HRV | `hrv=True` | Same pulse, longer (60 s) window | same as `rppg` |
| Respiration | `respiration=True` | `respiration_source="pulse"` (rPPG amplitude) **or** `"motion"` (shoulder/torso motion, reuses pose keypoints) | skin ROI for `pulse`; `pose` for `motion` |

- **`rppg_method`** — extraction algorithm: `"POS"` (default), `"CHROM"`, `"LGI"`, `"OMIT"`.
- **`rppg_window_sec`** — rPPG sliding-window length (s). `None` (default) auto-selects
  `60` with `hrv` (needed for stable HRV) else `15`. The first reading appears after
  ~60% of the window fills, so use a smaller value (e.g. `10`) for **short clips** so the
  panels populate sooner.
- **`rppg_roi`** — the skin **segmentation** the pulse is sampled from (always a
  segmentation mask, never a raw face box). Defaults to `None`, which builds a SegFace
  [`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor] (it finds faces itself, so
  **no face detector is needed**). Override with a custom mask provider — a callable
  `roi(frame) -> mask` or an object with `skin_mask(frame, boxes=None) -> mask`, which
  receives the frame's face boxes when the pipeline finds faces (e.g. your own
  face-neck segmentation model). Pass a `FaceSkinExtractor(device=...)` to control its
  device.

!!! tip "VR headsets / occluded faces"
    By default rPPG segments the **face skin** (SegFace). When the upper face is covered
    (e.g. a VR HMD), swap in a **face-neck segmentation model** as `rppg_roi` so heart
    rate and HRV are recovered from the visible neck / lower-face skin, and set
    `respiration_source="motion"` so respiration comes from the shoulders. HR / HRV /
    respiration are independent building blocks — mix the sources per what your footage
    shows.

```python
import physiotrack as pt

# Default: rPPG on the SegFace face-skin segmentation, respiration from motion.
video = pt.Video(
    source="session.mp4",
    pose=pt.Pose.Person(),
    rppg=True, hrv=True,                        # rppg_roi=None -> SegFace skin (built in)
    respiration=True, respiration_source="motion",
)
video.run(output_video="out.mp4")

# VR / occluded face: sample the neck / lower-face skin from your own segmentation.
def face_neck_mask(frame):
    return my_face_neck_segmentor(frame)        # -> boolean (H, W) skin mask

video = pt.Video(
    source="hmd_session.mp4",
    pose=pt.Pose.Person(),
    rppg=True, hrv=True,
    rppg_roi=face_neck_mask,                     # neck/lower-face skin ROI
    respiration=True, respiration_source="motion",
)
video.run(output_video="out.mp4")
```

### Faces

`face_stages=[...]` runs [face stages](face.md) on every face, in order. The faces come
from a `face` detector (`pt.Face()` / `pt.VRFace()`), or — when `detector` is itself a
face detector — they are the tracked subjects.

```python
video = pt.Video(
    source="session.mp4",
    detector=pt.Detection.Person(), tracker=pt.Tracker(),
    face=pt.VRFace(),                                         # faces of tracked people
    face_stages=[pt.FaceOrientation(model=pt.Models.Face.Orientation.VR),
                 pt.FaceLandmarks()],
)
results = video.run()
blinks = pt.signals.detect_blinks(results, detection_id=1)   # person 1's blinks
```

With a tracker, a face takes the track id of the person box that contains most of it
(one face per person), so face signals follow a person across frames; without one, face
ids are `None`. Faces are detected on the clean frames, drawn onto the output, and
returned as [`FrameResult.faces`][physiotrack.FrameResult]. When rPPG is on, its skin
segmentation reuses these face boxes instead of detecting faces again.

!!! tip "Batching and tracking"
    `batch_size` sets how many frames each pipeline step processes together (values
    below `1` are clamped to `1`). Tracking always runs **frame-by-frame**
    regardless of batch size, because a tracker is stateful. See the
    [Tracking guide](tracking.md).

## Running the pipeline

### `run()`

[`run`][physiotrack.Video.run] processes the whole source and returns a
[`VideoResults`][physiotrack.VideoResults], a typed sequence with one
[`FrameResult`][physiotrack.FrameResult] per processed frame. It optionally writes an
annotated MP4 (H.264 when available, otherwise MPEG-4) and a JSON dump.

```python
results = video.run(
    output_video="output/clip.mp4",   # optional
    output_json="output/clip.json",   # optional
)
```

Each `FrameResult` is iterable over its typed `Instance` objects and carries metadata.
Its serialized dictionary always has `frame_id` (int), `timestamp` (float seconds),
`task` and `instances` (list), plus optional pipeline fields:

| Key | Present when | Contents |
| --- | --- | --- |
| `frame_id` | always | Frame index (int). |
| `timestamp` | always | Seconds from start (float). |
| `task` | always | `"pose"`, `"track"` or `"detect"` — which stage the instances come from. |
| `instances` | always | Per-subject fields from the richest attached stage: pose instances with keypoints, else tracked instances with persistent `id`s, else bare detections. Empty when no detection, tracking, or pose stage is attached. |
| `track_box` | a tracker is attached and has a locked box | The tracked subject box `[x1, y1, x2, y2]`. |
| `faces` | a face source (`face=` or a face `detector`) is attached | The frame's faces as a `task="face"` result: each face's `id`, `box` and the fields its face stages added. Face meshes are included with `include_arrays=True`. |
| `vitals` | rPPG / HRV / respiration are on | `hr`, `snr`, `hrv`, `respiration`. |

You can also pass a `progress_callback(frame_id, total_frames, pose_results)` to
`run()` for live progress reporting.

### `batch_run()`

[`batch_run`][physiotrack.Video.batch_run] reuses this pipeline's configuration
across several input files, writing `<name>_processed.mp4` / `<name>_result.json`
into an output directory and returning a `dict` mapping each file stem to its
[`VideoResults`][physiotrack.VideoResults] sequence:

```python
results = video.batch_run(
    ["clip_a.mp4", "clip_b.mp4"],
    output_dir="output",
    save_videos=True,
    save_json=True,
)
```

## Complete example

A pose + detector + tracker pass with the subject-lock overlay, adapted from
`examples/tracker_aided_pose_video.py`:

```python
from physiotrack import Pose, Video, Models, Detection, Tracker, TrackerConfig
from pathlib import Path

# Pose estimator with an explicit detector + tracker.
pose_estimator = Pose.Custom(
    model=Models.Pose.ViTPose.WholeBody.b_wholebody, verbose=False, device=0
)
detector = Detection.VRStudent(
    model=Models.Detection.YOLO.VRSTUDENT.m_vrstudent, verbose=False, device=0
)

tracker_config = TrackerConfig()
tracker_config.tracker_type = "ocsort"
tracker_config.classes = [0]
tracker_config.enable_subject_lock = True     # lock onto one subject
tracker = Tracker(config=tracker_config)

input_video = "BV_S17_cut1.mp4"
output_directory = "output"
video_name = Path(input_video).stem

video_processor = Video(
    source=input_video,
    pose=pose_estimator,
    detector=detector,       # Pose.Custom consumes these boxes
    tracker=tracker,
    fps=None,                # process every frame at source fps
    resize=None,
    rotate=False,
    output_dir=output_directory,
    verbose=True,
)

video_output_path = Path(output_directory) / f"{video_name}_poses.mp4"
json_output_path = Path(output_directory) / f"{video_name}_result.json"

detection_data = video_processor.run(video_output_path, json_output_path)
print(f"Successfully processed video with {len(detection_data)} frames")
```

!!! tip "Live preview and FPS stats"
    Pass `show=True` to display the annotated frames in an OpenCV window while
    processing (press `q` to quit), and `show_fps=True` to print real-time and
    end-of-run component-wise FPS. Both are off by default.

!!! note "Sources: files, cameras and streams"
    `source` accepts a file path, an `rtsp://` URL, or an integer camera-device
    index (e.g. `0`). Files have a known frame count (so `verbose=True` shows a
    progress bar); RTSP streams and cameras do not.

## See also

- [Video API reference](../api/video.md) — the full [`Video`][physiotrack.Video]
  constructor and methods.
- [Pose guide](pose.md) — pose estimators to pass as `pose`.
- [Detection guide](detection.md) — detectors to pass as `detector`.
- [Multi-object Tracking guide](tracking.md) — configuring the `tracker`.
- [Result objects](../api/results.md) — the per-frame result structures.
