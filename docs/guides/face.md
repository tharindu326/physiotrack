# Face Analysis

Find faces and analyse each one: head orientation, a 478-point face mesh, facial
expression, 3D gaze and image quality. From the face mesh the
[signals](signals.md#face-signals) functions measure the eyes, mouth and iris, and over
a video they turn those measures into blinks, blink rate and mouth movement.

Face analysis is built from two kinds of object:

- **Face detectors** — [`Face`][physiotrack.Face] and the VR-tuned
  [`VRFace`][physiotrack.VRFace] — return a [`Result`][physiotrack.Result] with
  `task="face"` whose instances are face boxes.
- **Face stages** each add one property to every face they are given, and return the
  same faces with that field filled in. They chain, and the same objects plug into
  [`Video`][physiotrack.Video] as `face_stages=[...]`.

| Stage | Adds | Model |
| --- | --- | --- |
| [`FaceOrientation`][physiotrack.FaceOrientation] | `orientation` — yaw / pitch / roll (deg) | 6DRepNet360 |
| [`FaceLandmarks`][physiotrack.FaceLandmarks] | `keypoints` — 478-point `"FACEMESH"` | MediaPipe Face Landmarker |
| [`FaceExpression`][physiotrack.FaceExpression] | `expression` — AffectNet category (+ valence / arousal) | EmotiEffLib EfficientNets |
| [`GazeEstimator`][physiotrack.GazeEstimator] | `gaze` — 3D direction + angles | ETH-XGaze / MPIIFaceGaze / MPIIGaze |
| [`FaceQuality`][physiotrack.FaceQuality] | `quality` — brightness, sharpness, size | none |
| [`FaceRegions`][physiotrack.FaceRegions] | `regions` — visible face parts per box (skin, eyes, ...) | SegFace |

!!! info "Install the face extra"
    Detection, orientation and quality need nothing beyond the base install.
    `FaceLandmarks`, `FaceExpression` and `GazeEstimator` need
    `pip install "physiotrack[face]"`, which adds MediaPipe, ONNX Runtime and
    safetensors.

## Quick start

```python
import physiotrack as pt
from physiotrack.core.predictor import load_image

image = load_image("photo.jpg")

faces = pt.Face()(image)                          # Result(task="face"): face boxes
faces = pt.FaceOrientation()(image, faces)        # + orientation
faces = pt.FaceLandmarks()(image, faces)          # + 478-point face mesh
faces = pt.FaceExpression()(image, faces)         # + expression
faces = pt.GazeEstimator()(image, faces)          # + gaze (needs the mesh)
faces = pt.FaceQuality()(image, faces)            # + quality

for face in faces:
    print(face.orientation, face.expression["label"], face.gaze["yaw"])
    print(pt.signals.eye_aspect_ratio(face))      # {'left': .., 'right': .., 'mean': ..}

faces.save("faces.png")                           # boxes, head axes, mesh, gaze, labels
```

## Available presets

### Face detectors

| Preset | Backend model | Description |
| --- | --- | --- |
| [`Face`][physiotrack.Face] | `Models.Detection.YOLO.FACE.m_face` | General-purpose YOLO face detector (`n_face`, `m_face`, `l_face`). |
| [`VRFace`][physiotrack.VRFace] | `Models.Detection.YOLO.VRFACE.l_vrface` | YOLOv12l-face tuned for VR headsets; robust to HMD occlusion. |

Both take the standard detector arguments (`conf=0.25`, `iou=0.45`, `classes`,
`device`, `verbose`).

### Face-stage models

| Model enum | Stage | Description |
| --- | --- | --- |
| `Models.Face.Orientation.default` | `FaceOrientation` | 6DRepNet360 (300W-LP + Panoptic). |
| `Models.Face.Orientation.VR` | `FaceOrientation` | VR-tuned variant for headset wearers. |
| `Models.Face.Landmarks.face_landmarker` | `FaceLandmarks` | MediaPipe Face Landmarker, 468 + 10 iris points. |
| `Models.Face.Expression.enet_b0_8_best_afew` | `FaceExpression` | EfficientNet-B0, 8 categories, fine-tuned on AFEW (default). |
| `Models.Face.Expression.enet_b0_8_best_vgaf` | `FaceExpression` | EfficientNet-B0, 8 categories, fine-tuned on VGAF. |
| `Models.Face.Expression.enet_b0_8_va_mtl` | `FaceExpression` | EfficientNet-B0, 8 categories plus valence and arousal. |
| `Models.Face.Expression.enet_b2_8` / `enet_b2_7` | `FaceExpression` | EfficientNet-B2, 8 or 7 (no Contempt) categories. |
| `Models.Face.Gaze.eth_xgaze_resnet18` | `GazeEstimator` | Full face, ETH-XGaze; robust to large head rotations (default). |
| `Models.Face.Gaze.mpiifacegaze_resnet_simple` | `GazeEstimator` | Full face, MPIIFaceGaze. |
| `Models.Face.Gaze.mpiigaze_resnet_preact` | `GazeEstimator` | Each eye, MPIIGaze; the face's gaze is the mean of its eyes. |
| `Models.Segmentation.SegFace.Face.swinb_celeba_512` | `FaceRegions` | SegFace face parsing, 19 CelebAMask-HQ classes. |

All weights download into the model cache on first use; the third-party ones come from
their publishers at a pinned revision and are verified by SHA-256. See the
[Model Zoo](../model-zoo.md).

!!! warning "Non-commercial weights"
    The expression weights are trained on AffectNet and the gaze weights on ETH-XGaze,
    MPIIGaze and MPIIFaceGaze, whose licences allow research use only. See
    `THIRD_PARTY_LICENSES.md`.

## Key options

Every stage is called the same way:

```python
stage.predict(source, faces=None)      # or stage(source, faces)
```

- `source` is a BGR image, an image path, or a list of either (a batch).
- `faces` is what to analyse: a face `Result` (from a detector or an earlier stage), a
  tracker's `TrackResult`, or `(N, 4)` boxes. For a batch, pass one entry per image.
  `None` treats the whole image as one face.
- The output keeps the input faces in order — their `id`, `box` and every field an
  earlier stage filled in — and adds this stage's field. A face the model could not
  analyse (e.g. no mesh found) keeps its instance with the field `None`.

| Stage | Constructor options |
| --- | --- |
| `FaceOrientation` | `model`, `device` |
| `FaceLandmarks` | `model` (MediaPipe runs on the CPU) |
| `FaceExpression` | `model`, `device` (CUDA needs `onnxruntime-gpu`) |
| `GazeEstimator` | `model`, `device`, `camera_matrix=None`, `dist_coeffs=None` |
| `FaceQuality` | — |
| `FaceRegions` | `model`, `device` |

Each stage analyses a face on a fixed, validated crop: 1.2x the box for orientation,
1.25x for landmarks (square, padded at the frame border), the box itself for
expression, quality and regions. A face whose box lies outside the frame is skipped.

A stage that needs another's output says so: `GazeEstimator` requires the face mesh,
so it must come after `FaceLandmarks`, and raises a clear error otherwise.

!!! note "Units and conventions"
    - `orientation` — yaw, pitch, roll in degrees, as drawn by the head axes.
    - `gaze["vector"]` — unit direction in camera coordinates (x right, y down,
      z forward; looking at the camera is `[0, 0, -1]`). `gaze["yaw"]` is positive
      towards the image right and `gaze["pitch"]` positive downwards, in degrees.
      These are camera-frame gaze angles, not head angles.
    - `expression` — `{"label", "confidence", "scores"}` with probabilities over
      `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`. The
      categories describe the visible facial configuration, not what a person feels.
    - `quality` — `brightness` in `[0, 1]`, `sharpness` as the Laplacian variance of
      the crop resized to 112 × 112 px, `area_ratio` of face box to frame.
    - Face-mesh keypoints carry `confidence=None`: MediaPipe reports none.

!!! note "Gaze needs the camera"
    Gaze estimation fits a 3D head to the face mesh, which depends on the camera's
    focal length. Without `camera_matrix` a pinhole camera with focal length equal to
    the image width is assumed; pass calibrated intrinsics, and `dist_coeffs` for a
    lens with distortion, for better accuracy.

## Working with results

```python
face = faces[0]
face.box                               # (4,) face box
face.orientation["yaw"]
mesh = face.keypoints                  # Keypoints, architecture "FACEMESH"
mesh.by_name("left_iris_center")       # named points: eyes, iris, lips, nose tip, chin
mesh.xy                                # (478, 2) pixel coordinates

faces.to_dict()                        # face meshes flagged, omitted (large)
faces.to_dict(include_arrays=True)     # ... or included
pt.Result.from_dict(data)              # back to objects
```

Face-mesh points use MediaPipe's numbering and the subject's own left and right. Only
the points the library measures have semantic names; the rest are `facemesh_<id>`.

### Faces in a video

Pass the stages to [`Video`][physiotrack.Video]. Faces come from one of two places:

```python
# 1. Faces of tracked people: each face gets the track id of its person.
video = pt.Video("clip.mp4",
                 detector=pt.Detection.Person(), tracker=pt.Tracker(),
                 pose=..., face=pt.Face(),
                 face_stages=[pt.FaceOrientation(), pt.FaceLandmarks()])

# 2. The faces themselves are the tracked subjects.
video = pt.Video("clip.mp4", detector=pt.Face(), tracker=pt.Tracker(),
                 face_stages=[pt.FaceLandmarks(), pt.FaceExpression()])

results = video.run()
results[0].faces                        # Result(task="face") of frame 0
```

A face is matched to the person box that contains most of it, one face per person.
Without a tracker, face `id`s are `None`. The faces are drawn onto the output video
and saved under `"faces"` in its JSON.

Then use the face signals on the `VideoResults`:

```python
df = pt.signals.face_feature_sequence(results)          # one row per face per frame
blinks = pt.signals.detect_blinks(results, detection_id=1)
rate = pt.signals.blink_rate(results, detection_id=1)    # blinks / min
mouth = pt.signals.mouth_movement(results, detection_id=1)
windows = pt.signals.face_window_summary(results, window=5.0)   # per-face 5-s stats
```

See [Face signals](signals.md#face-signals) for the definitions.

### Facial regions

[`FaceRegions`][physiotrack.FaceRegions] parses each face with SegFace and reports the
share of its box per face part — how much is visible skin, hair, glasses, a hat:

```python
faces = pt.FaceRegions()(image, faces)
faces[0].regions["fractions"].get("skin", 0.0)  # share of the box that is skin
```

For the frame-level parsing and its colour overlay, use
[`Segmentation.Face`][physiotrack.Segmentation].

## Recipes & tips

!!! tip "Detect once, analyse many"
    Pass one detector result through every stage rather than re-detecting. In a
    `Video`, the rPPG skin segmentation also reuses the face boxes.

!!! warning "Small or turned faces"
    The face mesh needs a reasonably frontal face of about 50 px or more. Smaller or
    strongly turned faces often get `keypoints=None`, and so no eye, mouth, iris or
    gaze measures.

!!! tip "Release MediaPipe early"
    A `FaceLandmarks` stage releases its MediaPipe landmarker automatically when it is
    garbage-collected or the program exits; call `landmarks.close()` to free it sooner.

## Runnable examples

- [`examples/face_analysis/`](https://github.com/tharindu326/physiotrack/tree/main/examples/face_analysis)
  — every stage on an image, and a video run with per-face blinks and mouth movement.
- [Face Detection & Tracking Examples](face-examples.md) — detection, CPU vs GPU,
  tracking, and their output schemas.
- [Face Detection & Tracking Validation](face-validation.md) — datasets, manifests and
  reporting boundaries.

## See also

- [Face API reference](../api/face.md) — every stage's parameters.
- [Face signals](signals.md#face-signals) — eye and mouth measures, blinks.
- [Result objects](../api/results.md) — `Instance` fields and serialization.
- [Video pipeline](video.md) — `face=` and `face_stages=`.
- [Model Zoo](../model-zoo.md) — weights and licences.
