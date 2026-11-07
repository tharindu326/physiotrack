# 2D Pose Estimation

Estimate 2D human keypoints from an image or video frame. Physiotrack's
[`Pose`][physiotrack.Pose] presets locate people and return their skeleton
landmarks — either the compact **COCO-17** body layout or the full
**COCO-WholeBody-133** layout (body + feet + face + hands). Use it whenever you
need where the joints are: rep counting, joint-angle / range-of-motion analysis,
gait, gesture, or as the front end for [3D lifting](pose3d.md).

Every prediction returns a unified [`Result`][physiotrack.Result] whose per-person
[`Instance`][physiotrack.Instance] objects carry a [`Keypoints`][physiotrack.Keypoints]
collection, and rendering is delegated to [`Result.plot`][physiotrack.Result.plot].

## Quick start

=== "Image"

    ```python
    import cv2
    import physiotrack as pt

    pose = pt.Pose.Person()               # whole-body ViTPose + person detector
    frame = cv2.imread("frame_1.png")
    result = pose.predict(frame)          # auto-detects people; or pose.predict(frame, boxes)

    print(f"{len(result)} people, architecture={result.architecture}")
    for person in result:
        nose = person.keypoints.by_name("nose")
        print(nose.x, nose.y, nose.confidence)

    cv2.imwrite("out.png", result.plot())
    ```

=== "Video"

    ```python
    from pathlib import Path
    from physiotrack import Pose, Video

    pose = Pose.VRStudent(device=0, verbose=False)

    video = Video(
        source="BV_S17_cut1.mp4",
        pose=pose,
        fps=None, resize=None, rotate=False,
        output_dir="output",
        verbose=True,
    )

    frames = video.run(
        Path("output/BV_S17_cut1_poses.mp4"),      # annotated video
        Path("output/BV_S17_cut1_result.json"),    # per-frame keypoints JSON
    )
    print(f"{len(frames)} frames; {sum(len(frame) for frame in frames)} pose instances")
    ```

    !!! tip "The JSON feeds 3D lifting"
        The `*_result.json` written here is exactly what
        [`Pose3D`][physiotrack.pose.pose3D.Pose3D] consumes to lift the poses to
        3D. See the [3D Pose guide](pose3d.md).

## Available presets

Pick a preset by use case; all return a `Result` with `task="pose"`.

| Preset | Person detector | Default pose model | Description |
| --- | --- | --- | --- |
| [`Pose.Person`][physiotrack.Pose] | `Detection.Person` (generic) | ViTPose WholeBody (`b_wholebody`, 133 kpts) | General whole-body pose for everyday footage. |
| [`Pose.VRStudent`][physiotrack.Pose] | `Detection.VRStudent` (VR-tuned) | ViTPose WholeBody (`b_wholebody`, 133 kpts) | Whole-body pose for VR-headset / studio capture. |
| [`Pose.Custom`][physiotrack.Pose] | `Detection.Person` (default) | **required** — any validated pose model | Bring your own backend/variant explicitly. |

See the [Model Zoo](../model-zoo.md) for every downloadable variant and its weights.

### Backends

The backend is inferred automatically from the model's metadata — you never
select it directly, you just pass a model enum to `Pose.Custom` (or accept the
preset default).

| Backend | Type | Boxes needed? | Models (`pt.Models.Pose.*`) |
| --- | --- | --- | --- |
| **ViTPose** | Top-down (needs person boxes) | Yes — auto-detected if omitted | `ViTPose.WholeBody.{s,b,l,h}_wholebody`, `ViTPose.COCO.{s,b,l,h}_coco` |
| **Sapiens** | Top-down (needs person boxes) | Yes — auto-detected if omitted | `Sapiens.WholeBody.{B03,B06,B1}_TS_COCOHB` |
| **YOLO** | Single-stage (self-contained) | No — detects people itself | `YOLO.COCO.{M11,L11}` |

```python
import physiotrack as pt

# Sapiens whole-body via Pose.Custom
pose = pt.Pose.Custom(
    pt.Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB, device="cuda",
)

# YOLO-Pose (single-stage COCO-17, no separate detector)
pose = pt.Pose.Custom(pt.Models.Pose.YOLO.COCO.M11, device="cuda")

# ViTPose body-only COCO-17
pose = pt.Pose.Custom(pt.Models.Pose.ViTPose.COCO.b_coco)
```

!!! info "COCO-17 vs WholeBody-133"
    The loaded model's layout is reported on `result.architecture`
    (`"WHOLEBODY"` or `"COCO"`) and propagated to every `Keypoints` object so
    landmarks get the right names. **COCO-17** models emit the 17 body joints;
    **WholeBody** models add feet, a 68-point face, and both hands.

## Key options

Constructor arguments shared by every preset (see
[`Pose`][physiotrack.Pose] for the exhaustive list):

| Argument | Default | Purpose |
| --- | --- | --- |
| `model` | preset default (`Custom`: required) | Pose model enum to load. |
| `device` | `'cpu'` | `'cpu'`, `'cuda'`, or a CUDA index like `0`. |
| `conf` / `iou` | `0.25` / `0.45` | YOLO-Pose confidence / NMS thresholds. |
| `detector_model` | `None` | Override the person detector (top-down backends). |
| `detector_conf` / `detector_iou` | `0.25` / `0.45` | Person-detector thresholds. |
| `classes` | `None` | Restrict detections to these class ids. |
| `verbose` | `False` | Backend logging. |

`predict(source, boxes=None)` (aliased by calling the instance, `pose(frame)`)
takes a single BGR frame `(H, W, 3)` or a list of frames for batch inference, and
an optional list of person boxes `[[x1, y1, x2, y2], ...]`.

!!! note "Auto-detection of people"
    Top-down backends (ViTPose, Sapiens) need person boxes. If you don't pass
    `boxes`, a person detector is created on demand and run first, then pose is
    estimated on each box. Pass your own `boxes` to skip detection (e.g. when you
    already track people). YOLO-Pose ignores `boxes` and finds people itself.

## Working with results

`predict` returns a [`Result`][physiotrack.Result] (or a `list[Result]` for a
batch). Iterate it for per-person [`Instance`][physiotrack.Instance] objects and
read their [`Keypoints`][physiotrack.Keypoints]:

```python
result = pose.predict(frame)

for person in result:                     # each person is an Instance
    kps = person.keypoints                # a Keypoints collection

    # Three lookup styles
    nose  = kps.by_name("nose")           # by joint name
    wrist = kps.by_id(9)                  # by skeleton id  -> left_wrist
    first = kps[0]                        # positional (skeleton order)
    print(nose.x, nose.y, nose.confidence)

    # Vectorized NumPy views
    xy   = kps.xy                         # (N, 2) pixel coords
    conf = kps.conf                       # (N,)   per-keypoint confidence
    # xyz = kps.xyz                       # (N, 3) or None if 2D-only

result.keypoints          # list[Keypoints], one per person with pose
result.boxes              # (M, 4) person boxes
result.to_dict()          # JSON-friendly dict (task + instances)
```

A missing landmark returns `None` from `by_name` / `by_id`, and unmapped ids are
named `"unknown_<id>"`. See [Result objects](../api/results.md) for the full
container API.

### WholeBody-133 keypoint index ranges

For WholeBody models the ids/names come from the COCO-WholeBody layout (ids
`0`–`135`, i.e. 133 detected points plus three derived centroids):

| Region | Ids | Count | Notes |
| --- | --- | --- | --- |
| Body | `0`–`16` | 17 | Standard COCO-17: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. |
| Feet | `17`–`22` | 6 | Left/right big toe, small toe, heel. |
| Face | `23`–`90` | 68 | Jaw `23`–`39`, eyebrows `40`–`49`, nose `50`–`58`, eyes `59`–`70`, mouth `71`–`90`. |
| Hands | `91`–`132` | 42 | 21 points per hand — left `91`–`111`, right `112`–`132`. |
| Derived | `133`–`135` | 3 | `head_centroid`, `body_centroid`, `pelvic_centroid` — added by [`add_head_centroid`][physiotrack.signals.add_head_centroid], [`add_body_centroid`][physiotrack.signals.add_body_centroid] and [`add_pelvic_centroid`][physiotrack.signals.add_pelvic_centroid]. `135` is the default `reference_point_id` for pelvis-relative coordinates. |

COCO-17 models expose only ids `0`–`16` (the body block above).

## Recipes & tips

!!! tip "Reuse boxes you already have"
    If you run detection or tracking upstream, pass the boxes straight into
    `pose.predict(frame, boxes)` to avoid re-detecting people.

!!! example "Plot without re-running inference"
    Rendering lives on the result, so you can draw the same prediction different
    ways: `result.plot()`, `result.plot(conf=True)`,
    `result.plot(boxes=False)` — no second forward pass.

!!! warning "First run downloads weights"
    The first time a validated model is used, its weights auto-download to the
    per-user cache (see [Model Zoo](../model-zoo.md#where-weights-are-cached)).
    Subsequent runs load from disk.

## See also

- [`Pose`][physiotrack.Pose] · [`Keypoints`][physiotrack.Keypoints] — API reference for the presets and keypoint container.
- [Pose API page](../api/pose.md) — full parameter tables.
- [Result objects](../api/results.md) — `Result`, `Instance`, `Keypoints`.
- [3D Pose & Canonicalization guide](pose3d.md) — lift these 2D keypoints to 3D.
- [Model Zoo](../model-zoo.md) — all downloadable pose backends and variants.
