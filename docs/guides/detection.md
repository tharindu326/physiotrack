# Detection

Object detection locates subjects in a frame and returns axis-aligned bounding
boxes with a confidence and class label. In Physiotrack it is the entry point for
most pipelines — person boxes feed pose and tracking, face boxes feed face
parsing and head-orientation. Use it whenever you need *where* something is before
deciding *what* to measure.

Detectors are exposed as ready-to-use presets on [`Detection`][physiotrack.Detection].
Instantiate a preset, call [`predict`][physiotrack.Detection] (or call the instance
directly), and read the returned [`Result`][physiotrack.Result].

## Quick start

```python
from physiotrack import Detection
import cv2

image = cv2.imread("frame_1.png")

detector = Detection.Person(conf=0.25, iou=0.45)   # person boxes (COCO class 0)
result = detector.predict(image)                    # or: detector(image)

boxes = result.boxes            # (N, 4) array of [x1, y1, x2, y2]
annotated = result.plot()       # BGR image with boxes drawn
cv2.imwrite("out.png", annotated)
```

## Available presets

Each preset pins a validated backbone; construction fails fast if you pass an
incompatible model. See the [Model Zoo](../model-zoo.md) for every variant.

| Preset | Backend | Classes | Description |
| --- | --- | --- | --- |
| [`Detection.Person`][physiotrack.Detection.Person] | YOLO | `[0]` (person) | People only; class filter is pinned to person. |
| [`Detection.VR`][physiotrack.Detection.VR] | YOLO | `VR-head` | VR-headset boxes; use `Segmentation.VRHead` when masks are needed. |
| [`Detection.VRStudent`][physiotrack.Detection.VRStudent] | YOLO | `VR-person` | Full-person boxes for people wearing a VR headset. |
| [`Detection.Custom`][physiotrack.Detection.Custom] | YOLO | any | Run any validated `Models.Detection.*` variant. |

!!! info "Faces"
    Face detectors live with the rest of face analysis: [`Face`][physiotrack.Face]
    and the VR-tuned [`VRFace`][physiotrack.VRFace] share this constructor and return
    a `Result` with `task="face"`. See the [Face analysis guide](face.md).

### VR objects and people

These presets answer different questions. `Detection.VR` locates the headset region,
`Detection.VRStudent` locates the full body of a VR-equipped person, and
`Detection.Person` locates people regardless of whether they wear a headset. Run the
[VR detection example](https://github.com/tharindu326/physiotrack/tree/main/examples/vr_detection)
to compare all three on the same synthetic lab scene.

Only a medium VR-head checkpoint (`yolo11m_VR_head.pt`) is currently published.
Large checkpoints are available for VR-person and generic-person detection. The
example's `--model-size largest` therefore uses medium for VR-head and large for the
other two, and reports every model in its panels and JSON output.

![VR-person and generic-person comparison](../images/comparison_person_vrperson.jpg)

*Focused output from the bundled synthetic scene. The large VR-person checkpoint
(top) retains two full people using headsets; the large generic-person checkpoint
(bottom) retains twelve people regardless of headset use. These counts describe
different detection tasks and are not accuracy scores.*

```python
from physiotrack import Detection, Models

# Custom preset: choose an explicit validated model
det = Detection.Custom(model=Models.Detection.YOLO.VR.m_vr, conf=0.3)
```

!!! note "Auto-download"
    On first use a preset's weights are pulled from Hugging Face and cached
    locally; later runs load from disk.

## Key options

Set defaults at construction; override any of `conf` / `iou` / `classes` per call.

| Option | Where | Default | Meaning |
| --- | --- | --- | --- |
| `conf` | constructor + `predict` | `0.25` | Objectness confidence threshold in `[0, 1]`. |
| `iou` | constructor + `predict` | `0.45` | NMS / IoU threshold in `[0, 1]`. |
| `classes` | constructor + `predict` | `None` | Restrict to these class ids (e.g. `[0]`). |
| `device` | constructor | `'cpu'` | `'cpu'`, `'cuda'`, `'mps'`, or an index like `0`. |
| `verbose` | constructor | `False` | Print backend inference logs. |

```python
detector = Detection.Person(device=0)          # run on the first CUDA device
# per-call overrides apply to this call only:
result = detector.predict(image, conf=0.5, iou=0.6)
```

See [`Detection`][physiotrack.Detection] for the full constructor signature.

## Working with results

`predict` returns a [`Result`][physiotrack.Result] for a single frame. It behaves
like a sequence of [`Instance`][physiotrack.Instance] objects and exposes a
vectorized `boxes` view.

```python
result = detector.predict(image)

len(result)              # number of detections
result.boxes             # (N, 4) float array of [x1, y1, x2, y2]

for inst in result:                      # iterate detections
    x1, y1, x2, y2 = inst.box            # (4,) pixel box
    print(inst.cls, inst.cls_name, inst.confidence)

data = result.to_dict()  # JSON-friendly: {"task": "detect", "instances": [...]}
```

Render an annotated copy with [`Result.plot`][physiotrack.Result.plot] — the source
frame is never modified:

```python
annotated = result.plot(conf=True, color=(0, 0, 255), thickness=2)
```

See [Result objects](../api/results.md) for every field and rendering toggle.

## Recipes

!!! example "Batch inference"
    Pass a list or tuple of frames to run them in one call; you get back a
    `list[Result]`, one per frame.

    ```python
    frames = [cv2.imread(p) for p in ("a.png", "b.png", "c.png")]
    results = detector.predict(frames)     # list[Result], same order as input
    counts = [len(r) for r in results]
    ```

!!! tip "Feed a tracker or pose model"
    A detector's boxes are the standard input to downstream stages. The
    [Tracker](tracking.md) needs an `(N, 6)` NumPy array of
    `[x1, y1, x2, y2, confidence, class]`, not the dictionary returned by
    `result.to_dict()`. Build it from the result instances as shown in the tracking
    guide, or let [Pose](pose.md) auto-detect people for you.

!!! warning "Class ids depend on the model"
    `classes` filters by the backbone's own class map. `Detection.Person` already
    pins `[0]`; for `Detection.Custom` inspect the model's labels before filtering.

## See also

- [VR-head and VR-person example](https://github.com/tharindu326/physiotrack/tree/main/examples/vr_detection) — compare VR regions, VR-equipped people, and all people.
- [`Detection` API reference](../api/detection.md) — full class and preset docs.
- [Result objects](../api/results.md) — `boxes`, `Instance`, `to_dict`, `plot`.
- [Model Zoo](../model-zoo.md) — available detection backbones.
- [Pose guide](pose.md) · [Tracking guide](tracking.md) — common next stages.
