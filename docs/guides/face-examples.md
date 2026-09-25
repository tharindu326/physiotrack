# Face Detection & Tracking Examples

PhysioTrack includes runnable, self-contained examples with small synthetic
inputs. They are designed to answer three practical questions: does the model run,
what did it return, and how can those results be inspected outside Python?

| Example | Input | Main outputs |
| --- | --- | --- |
| [`examples/face_detection`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_detection) | four scene images | annotated PNGs, per-image JSON, `summary.csv`, `run.json` |
| [`examples/face_tracking`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_tracking) | one 10-second clip | annotated MP4, per-frame JSON, track CSV |
| [`examples/face_analysis`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_analysis) | the selfie image and the clip | every face stage on an image; per-face blinks and mouth motion over the video |

The generated `results/` directories are ignored by Git. Run the scripts locally,
inspect their outputs, and commit only deliberate documentation assets—not an entire
inference run.

!!! info "Synthetic example assets"
    The contributor confirms that the bundled visuals were generated through Google
    Gemini on August 13, 2026, with their people and scene imagery created using Nano
    Banana 2 (Gemini 3.1 Flash Image). They depict no real living people and are
    included for research, evaluation, testing, and documentation. The media is
    dedicated under CC0 1.0 Universal for any copyright and related rights held by
    the contributor; code and documentation remain GPL-3.0-or-later.
    See each example's `README.md` and `data/MEDIA.yml` for the complete notice,
    model scope, SynthID status, and checksums.

## Detect faces in images

From the repository root, after installing PhysioTrack:

```bash
python examples/face_detection/detect_faces.py
```

This processes the bundled selfie, point-of-view, crowd, and VR scenes on CPU. The
model is constructed once and reused for every image. Use CUDA, a different model
size, or your own input like this:

```bash
python examples/face_detection/detect_faces.py --device cuda
python examples/face_detection/detect_faces.py --model n_face --input path/to/images
python examples/face_detection/detect_faces.py --input path/to/one_image.jpg
```

Each annotated image has normal face boxes and confidence labels plus a top-left
panel containing:

- `Faces detected`: the number of [`Instance`][physiotrack.Instance] objects in
  this image's [`Result`][physiotrack.Result];
- `Detector`: the exact checkpoint filename, such as `yolov11m-face.pt`;
- `Device`: the requested compute device.

![Face detections in the synthetic point-of-view exercise scene](../images/exercise_class_pov.jpg)

*Example output from the bundled point-of-view scene: seven retained face boxes,
confidence labels, and the run-context panel. This is a qualitative illustration,
not a ground-truth accuracy result.*

### Detection output

```text
examples/face_detection/results/
├── annotated/<scene>__<image>.png
├── predictions/<scene>__<image>.json
├── summary.csv
└── run.json
```

`summary.csv` has one row per input image:

| Field | Meaning |
| --- | --- |
| `image` | relative input path |
| `width`, `height` | decoded image dimensions in pixels |
| `faces_detected` | number of boxes retained after confidence filtering and NMS |
| `mean_confidence`, `minimum_confidence` | summaries of the retained face confidences; blank when none were found |

Each per-image JSON is exactly [`Result.to_json()`][physiotrack.Result.to_json], so it
reloads with [`Result.from_dict`][physiotrack.Result.from_dict]:

```json
{
  "task": "face",
  "instances": [
    {"box": [100.0, 120.0, 400.0, 520.0], "confidence": 0.97, "cls": 0, "cls_name": "face"}
  ],
  "names": {"0": "face"}
}
```

Coordinates are `[x1, y1, x2, y2]` pixels. `run.json` records the registry path of the
model, the device, the thresholds, the image and face counts, and the detector's mean
inference time.

### Compare CPU and GPU

The companion benchmark runs the same image, weights, thresholds, and entry point on
both devices:

```bash
python examples/face_detection/compare_cpu_gpu.py
python examples/face_detection/compare_cpu_gpu.py --model n_face --repeats 20
```

It requires CUDA. Each device receives separate warm-up runs; CUDA is synchronized
around every measured `predict()` call so queued kernels are included in the elapsed
time. The output directory `examples/face_detection/results/cpu_vs_gpu/` contains:

| Output | Contents |
| --- | --- |
| `cpu.png`, `gpu.png` | final annotated result and mean timing for each device |
| `side_by_side.jpg` | the two annotated results on one canvas, the compact documentation preview |
| `comparison.json` | timing statistics per device, the speed-up, and the agreement of the two box sets (one-to-one IoU matching) |

![CPU and CUDA face-detection comparison](../images/side_by_side.jpg)

*Example run with `yolov11m-face.pt`: CPU (left) and CUDA (right) retained the
same seven face boxes. The displayed mean timings of 152.2 ms and 15.1 ms belong only to the
machine and configuration used for this run; reproduce the benchmark on your own
hardware before drawing performance conclusions.*

The reported speed-up is local evidence, not a portable benchmark: GPU model, CPU,
PyTorch/CUDA versions, image size, thermal state, and repeat count all affect it.

## Track faces in a video

Run the complete bundled clip:

```bash
python examples/face_tracking/track_faces.py
```

Add `--show` only when a desktop window is available:

```bash
python examples/face_tracking/track_faces.py --device cuda
python examples/face_tracking/track_faces.py --input path/to/video.mp4 --show
```

This is **tracking by detection**, and it is plain composition of two predictors
through the core [`Video`][physiotrack.Video] pipeline — the same pattern as
`examples/pose_video.py` and `examples/tracker_aided_pose_video.py`:

```python
detector = pt.Face(model=..., conf=..., iou=..., device=...)
tracker  = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))
video    = pt.Video(source=..., detector=detector, tracker=tracker, output_dir=...)
results  = video.run(output_video, output_json)
```

`Video` runs the face detector per frame, feeds the boxes to the stateful tracker,
draws IDs and trails, writes the annotated video (H.264 when available, MPEG-4
otherwise), and returns one [`FrameResult`][physiotrack.FrameResult] per frame whose
instances carry persistent track `id`s (`task="track"`). The script then derives a
per-track CSV from those results in a few lines.

The face count and active-track count can differ. A new detector box may need a few
frames before a tracker reports it, and a tracker may temporarily retain an object
through a missed detection.

### Tracking output

| Output | Contents |
| --- | --- |
| `*_tracked.mp4` | boxes, temporary IDs, and trails; source audio is not copied |
| `*_result.json` | the serialized [`VideoResults`][physiotrack.VideoResults] — one record per frame with `frame_id`, `timestamp`, and tracked `instances` |
| `*_tracks.csv` | one row per active track per frame: time, ID, box, confidence and class |

One JSON frame record has this shape:

```json
{
  "frame_id": 42,
  "timestamp": 1.75,
  "task": "track",
  "instances": [
    {"box": [100.0, 120.0, 180.0, 220.0], "confidence": 0.94, "cls": 0, "id": 1}
  ]
}
```

This is the same per-frame schema every `Video` pipeline produces (see the
[Video guide](video.md)), so the output feeds any tooling that already consumes
PhysioTrack results.

!!! warning "A track ID is not identity recognition"
    An ID is a temporary association inside one tracker run. It can change after a
    long occlusion, exit/re-entry, or association error. It is neither a name nor a
    biometric identity and must not be described as face recognition.

## How to interpret the examples

The bundled media is synthetic and carries no ground-truth boxes or tracks. A missed
face, extra box, or ID switch is useful qualitative evidence for debugging, but it is
not enough to calculate accuracy. Use a labelled benchmark and a fixed evaluation
protocol for claims about precision, recall, average precision, or tracking metrics;
the [face validation guide](face-validation.md) explains the separation.

## Analyse faces

The [face analysis examples](https://github.com/tharindu326/physiotrack/tree/main/examples/face_analysis)
go beyond boxes and tracks. `analyze_image.py` chains every
[face stage](face.md) on the selfie and prints each face's head pose, expression,
eye and mouth aspect ratios, iris position and gaze. `analyze_video.py` runs the stages
inside `Video` on the tracking clip and reports each face's blinks, blink rate and
mouth movement with the [face signals](signals.md#face-signals):

```bash
pip install "physiotrack[face]"
python examples/face_analysis/analyze_image.py
python examples/face_analysis/analyze_video.py --gaze
```
