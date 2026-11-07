# Core Concepts

Physiotrack has one job: make contactless human understanding **predictable**.
Learn the small set of conventions on this page and every subsystem —
detection, pose, segmentation, depth, face, signals — behaves the same way.

## The four-step rhythm

Every image predictor ([`Detection`][physiotrack.Detection],
[`Pose`][physiotrack.Pose], [`Segmentation`][physiotrack.Segmentation],
[`Depth`][physiotrack.Depth], [`Face`][physiotrack.Face]) follows the identical
pattern — modeled on Ultralytics / MediaPipe / scikit-learn:

```python
import physiotrack as pt

model  = pt.Detection.Person(conf=0.25, iou=0.45, device=0)   # 1. configure the MODEL
result = model.predict(image)                                 # 2. predict (or: model(image))
data   = result.boxes                                         # 3. read structured data
frame  = result.plot()                                        # 4. draw the overlay
```

Three rules make this hold everywhere:

!!! abstract "The three rules"
    1. **One verb.** Every predictor exposes `.predict(img)` and is callable.
       Batch with a list: `predict([img, img, ...]) -> list[Result]`.
    2. **One return type.** Every predictor returns a rich
       [`Result`][physiotrack.Result] object — never tuples in mixed orders.
    3. **Rendering lives on the result, not the model.**
       [`result.plot(...)`][physiotrack.Result.plot] draws the overlay;
       constructors only *configure* the model.

## Presets vs. `Custom`

Each predictor is a **namespace of ready-made presets** plus a `Custom` escape
hatch. Presets pin a validated model for a task; `Custom` lets you name any
registry entry.

```python
pt.Detection.Person()                                          # preset
pt.Detection.Custom(model=pt.Models.Detection.YOLO.VR.m_vr)    # any validated model
pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.l_wholebody)
```

| Predictor | Presets |
|-----------|---------|
| [`Detection`][physiotrack.Detection] | `.Person` · `.VR` · `.VRStudent` · `.Custom` (face detectors: [`Face`][physiotrack.Face], [`VRFace`][physiotrack.VRFace]) |
| [`Pose`][physiotrack.Pose] | `.Person` · `.VRStudent` · `.Custom` |
| [`Segmentation`][physiotrack.Segmentation] | `.Person` · `.VRHead` · `.BodyPart` · `.Face` · `.Custom` |
| [`Depth`][physiotrack.Depth] | `.DepthAnythingV2Small` · `Base` · `Large` · `.ZipDepth` · `.ZipDepthNPU` · `.Custom` |

`device` and `verbose` are accepted by the detection, pose, segmentation and depth
predictors. The [face stages](../guides/face.md) take `device` where they run a network
on it; `FaceLandmarks` (MediaPipe) and `FaceQuality` run on the CPU and take none. `conf`, `iou` and `classes`
apply to the **box-based** backends (detection, pose, instance segmentation) — a dense
predictor has no detections to threshold or filter, so `Depth` and `Segmentation.Face`
do not take them. Where such a predictor builds a detector internally, the detector's
thresholds are named for it: `Segmentation.Face(face_conf=…, face_iou=…)` configures the
face detector, not the parser. See each [API page](../api/index.md) for the exhaustive list.

### What `predict()` accepts

Every image predictor takes the same kinds of input and follows the same batching rule:

```python
det.predict(frame)                      # np.ndarray (H, W, 3) BGR  -> Result
det.predict("photo.jpg")                # a path is loaded for you  -> Result
det.predict([frame_a, frame_b])         # a sequence               -> list[Result]
det.predict(["a.png", "b.png"])         # paths work in batches too -> list[Result]
det(frame)                              # calling is an alias for predict
```

A sequence always returns a list — a one-element list gives a one-element list of
results — so code that batches does not need a special case for `n == 1`. For video,
use [`Video`][physiotrack.Video] rather than passing a video path here.

## The `Result` family

`.predict()` returns one object, whatever the task. What is populated depends on
the task, but the container and accessors are uniform.

| Task | Returns | Key attributes |
|------|---------|----------------|
| detect / pose / segment / face | [`Result`][physiotrack.Result] | `.boxes`, `.instances`, `.keypoints`, `.seg_map`, `.architecture`, `.plot()`, `.to_dict()` |
| depth | [`DepthResult`][physiotrack.DepthResult] | `.depth`, `.normalized()`, `.plot(colormap=...)` |
| track | [`TrackResult`][physiotrack.TrackResult] | `.instances`, `.ids`, `.boxes`, `.plot(frame)` |

A `Result` is **iterable** and **indexable** — iterating yields the per-detection
[`Instance`][physiotrack.Instance] objects:

```python
result = pt.Pose.Person().predict(frame)

len(result)                 # number of people
first  = result[0]          # -> Instance
for inst in result:         # iterate instances
    ...
d = result.to_dict()        # JSON-serializable dict (for logging / pipelines)
```

Read more on the [Result objects](../api/results.md) page.

## The object model: Instance → Keypoints → Keypoint

Each [`Instance`][physiotrack.Instance] is one detected subject, exposing
`.id`, `.box`, `.confidence`, `.cls` / `.cls_name`, `.mask`, `.orientation`, and
`.keypoints` (when applicable).

[`Keypoints`][physiotrack.Keypoints] is a queryable collection of
[`Keypoint`][physiotrack.Keypoint] objects — look them up by name or id, or take
array views:

```python
person = result[0]
kp = person.keypoints.by_name("left_wrist")   # or .by_id(9)
kp.x, kp.y, kp.confidence, kp.name

person.keypoints.xy      # (K, 2) ndarray
person.keypoints.xyz     # (K, 3) or None
person.keypoints.conf    # (K,) confidences
```

The keypoint names/ids follow the result's `architecture` — `"COCO"` (17) or
`"WHOLEBODY"` (133).

## The `Models` registry & auto-download

Weights are never hard-coded paths. They are addressed through the
[`Models`][physiotrack.Models] registry as
`Models.<Task>.<Backend>.<Variant>`, and **download automatically on first use**
(Hugging Face, or Ultralytics for YOLO/RT-DETR), then cache locally.

```python
from physiotrack import Models

Models.Detection.YOLO.PERSON.m_person       # YOLO person detector
Models.Pose.ViTPose.WholeBody.b_wholebody   # ViTPose whole-body
Models.Segmentation.SegFace.Face.swinb_celeba_512
Models.Depth.DepthAnythingV2.vitb
Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite
```

Feed any of these to a `.Custom(model=...)` predictor. Browse the full catalog in
the [Model Zoo](../model-zoo.md) and the [`Models` API](../api/models.md).

## Devices

Every predictor and the `Video` orchestrator take a `device` argument:

| Value | Runs on |
|-------|---------|
| `"cpu"` | CPU (default for most predictors) |
| `0`, `1`, … / `"cuda"` | CUDA GPU by index |
| `"mps"` | Apple Silicon |

GPU acceleration is recommended for video / real-time work.

## Batch vs. single

Pass one image to get one `Result`; pass a **list** to process a batch and get a
`list[Result]` back — the return shape mirrors the input:

```python
one   = model.predict(img)            # -> Result
many  = model.predict([img1, img2])   # -> [Result, Result]
```

The [`Video`][physiotrack.Video] pipeline batches internally via its
`batch_size` argument.

## The `Video` orchestrator

For streams and files, [`Video`][physiotrack.Video] composes any subset of the
subsystems into one end-to-end pipeline — capture, batching, per-frame inference,
overlay compositing, and writing an annotated MP4 + JSON. You attach
already-constructed predictors and call `.run()`:

```python
import physiotrack as pt

video = pt.Video(
    source="input.mp4",
    detector=pt.Detection.Person(),
    pose=pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.b_wholebody),
    tracker=pt.Tracker(config=pt.TrackerConfig()),
    output_dir="output",
)
data = video.run(output_video="out.mp4", output_json="out.json")
```

Beyond `detector`, `pose` and `tracker`, it accepts `segmenter=`, `depth=`,
`face=`, `face_stages=`, `floor_map=`, `ego_video=`, `plot_keypoint=`,
`plot_angles=`, `rom=` and more. See the [Video guide](../guides/video.md) and
the [`Video` API](../api/video.md).

## Where things live (imports)

```text
physiotrack             # predictors, Models, Video, Result family, canonicalization
physiotrack.signals     # rPPG (POS/CHROM/LGI/OMIT), HeartRateEstimator,
                        #   joint_angles, compute_rom_angles, filters, metrics, plotters
physiotrack.pose        # keypoint name maps (COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES)
physiotrack.face        # drawing helpers (draw_axis, plot_pose_cube)
```

You never reach into internal module paths — the top-level `physiotrack` package
and the named subpackages expose everything documented in the
[API Reference](../api/index.md).

## See also

- [Quickstart](quickstart.md) — the rhythm applied end-to-end.
- [Result objects](../api/results.md) — full accessor reference.
- [Model Registry](../api/models.md) — the registry API.
- [Guides](../guides/index.md) — one page per subsystem.
