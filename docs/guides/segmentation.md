# Segmentation

Segmentation labels every pixel with a class, producing dense masks instead of
boxes. Physiotrack unifies three backends behind one API: YOLO instance
segmentation (person, VR-head), Sapiens/Goliath body-part parsing, and SegFace
19-class face parsing. Use it when you need region shape or per-part maps —
skin regions for rPPG, body parts for biomechanics, or head/face masks for VR.

Every preset lives on [`Segmentation`][physiotrack.Segmentation]. Call
[`predict`][physiotrack.Segmentation] (or the instance directly) to get a
[`Result`][physiotrack.Result] whose `.seg_map` is an `(H, W)` class-index array;
`result.plot()` overlays the colorized mask.

## Quick start

```python
from physiotrack import Segmentation
import cv2

frame = cv2.imread("frame.png")

seg = Segmentation.Person(conf=0.24, iou=0.4)
result = seg.predict(frame)          # or: seg(frame)

seg_map = result.seg_map             # (H, W) int class-index map
overlay = result.plot()              # colorized segmentation blended over frame
cv2.imwrite("segmented.png", overlay)
```

## Available presets

Each preset selects a backend automatically from the model's registry metadata.
See the [Model Zoo](../model-zoo.md) for every variant.

| Preset | Backend | Output | Description |
| --- | --- | --- | --- |
| [`Segmentation.Person`][physiotrack.Segmentation.Person] | YOLO-seg | class map | Whole-person instance segmentation. |
| [`Segmentation.VRHead`][physiotrack.Segmentation.VRHead] | YOLO-seg | class map | VR head / face / neck parts. |
| [`Segmentation.BodyPart`][physiotrack.Segmentation.BodyPart] | Sapiens (Goliath) | class map | Fine-grained body-part parsing. |
| [`Segmentation.Face`][physiotrack.Segmentation.Face] | SegFace | class map + palette | 19-class face parsing (CelebAMask-HQ). |
| [`Segmentation.Custom`][physiotrack.Segmentation.Custom] | any | class map | Run any validated `Models.Segmentation.*`. |

=== "YOLO / Sapiens"

    ```python
    from physiotrack import Segmentation, Models

    # VR head/face/neck classes only
    seg = Segmentation.VRHead(conf=0.24, iou=0.4, classes=[0, 1, 2])

    # Sapiens Goliath body-part parsing (large model; GPU recommended)
    seg = Segmentation.BodyPart(device="cuda")

    # Custom: choose an explicit validated model
    seg = Segmentation.Custom(model=Models.Segmentation.YOLO.VRHEAD.M11)
    ```

=== "SegFace face parsing"

    ```python
    from physiotrack import Segmentation
    import cv2, numpy as np

    frame = cv2.imread("face.png")
    parser = Segmentation.Face(device=0)     # SegFace Swin-Base / CelebA-512

    # Auto-detects faces (like Pose auto-detects people), parses each crop
    result = parser.predict(frame)           # or pass boxes=[[x1,y1,x2,y2], ...]

    seg_map = result.seg_map                 # (H, W) face-part class indices
    present = [result.names[c] for c in np.unique(seg_map) if c != 0]
    cv2.imwrite("face_parsing.png", result.plot())
    ```

!!! note "SegFace runs on crops"
    Unlike the whole-frame segmenters, [`Segmentation.Face`][physiotrack.Segmentation.Face]
    operates on face crops. If you omit `boxes`, it lazily builds a
    [`Face`][physiotrack.Face] detector to find faces first (tune it with
    `face_conf` / `face_iou`). The result carries a full-frame `seg_map`, a
    `names` label map, and a 19-class `palette`.

!!! info "Auto-download"
    Weights are pulled from Hugging Face on first use and cached. Sapiens body-part
    models are large (multiple GB) — prefer a GPU device.

## Key options

| Option | Presets | Default | Meaning |
| --- | --- | --- | --- |
| `conf` | YOLO backend | `0.25` | Confidence threshold in `[0, 1]`. |
| `iou` | YOLO backend | `0.45` | NMS / IoU threshold in `[0, 1]`. |
| `classes` | YOLO backend | `None` | Restrict to these class ids. |
| `device` | all | `'cpu'` | `'cpu'`, `'cuda'`, `'mps'`, or an index like `0`. |
| `filter` | YOLO / Sapiens | `None` | Dict with `bbox_filter`, `detector_index`, `detector_class_filter` to keep segmentation only inside boxes. |
| `verbose` | all | `False` | Print backend inference logs. |

`Segmentation.Face` instead takes `face_detector`, `face_conf`, and `face_iou` to
control its auto-detection step. See [`Segmentation`][physiotrack.Segmentation] for
the full signatures.

## Working with results

The primary output is the class-index map `result.seg_map`, an `(H, W)` array where
each pixel holds a class id. `predict` accepts an optional `boxes` argument to keep
only the segmentation inside those regions (the rest is zeroed).

```python
result = seg.predict(frame)

seg_map = result.seg_map            # (H, W) class-index array
import numpy as np
np.unique(seg_map)                  # class ids present in the frame

# Restrict to boxes (e.g. from a detector)
result = seg.predict(frame, boxes=[[100, 80, 260, 400]])
```

For SegFace, each detected face is also recorded as an
[`Instance`][physiotrack.Instance] carrying its box, so you can slice per-face
part maps:

```python
for inst in result:                          # iterate detected faces
    x1, y1, x2, y2 = inst.box.astype(int)
    face_parts = result.seg_map[y1:y2, x1:x2]
```

Render with [`Result.plot`][physiotrack.Result.plot]; pass `masks=True` (the default)
to blend the colorized segmentation. When the result carries a `palette` (face
parsing), that palette is used; otherwise the default segmentation palette applies.

```python
overlay = result.plot(masks=True)            # colorized mask blended over the frame
```

See [Result objects](../api/results.md) for the full `seg_map` / `palette` /
`Instance` contract.

## Recipes

!!! example "Video loop"
    Reuse one segmenter across frames — construction (and download) happens once.

    ```python
    seg = Segmentation.VRHead(device="cuda", classes=[0, 1, 2])
    cap = cv2.VideoCapture("clip.mp4")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        overlay = seg.predict(frame).plot()
    ```

!!! tip "Batch inference"
    Pass a list of frames to get a `list[Result]` back, one per frame.

!!! warning "Backend picks the class set"
    Class ids in `seg_map` come from the chosen backbone. Person-seg, VR-head,
    Sapiens body parts, and SegFace face parts all use different label maps —
    check `result.names` (when provided) or the [Model Zoo](../model-zoo.md).

## See also

- [`Segmentation` API reference](../api/segmentation.md) — full class and preset docs.
- [Result objects](../api/results.md) — `seg_map`, `palette`, `plot(masks=True)`.
- [Model Zoo](../model-zoo.md) — YOLO-seg, Sapiens, and SegFace backbones.
- [Detection guide](detection.md) · [Signals guide](signals.md) — feed boxes in, or use skin masks for rPPG.
