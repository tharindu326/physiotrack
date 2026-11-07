# Quickstart

This page runs Physiotrack end-to-end: load an image with OpenCV, run
**detection** then **pose**, render the overlays, and finally process a **video** —
both a manual frame loop and the one-liner [`Video`][physiotrack.Video] pipeline.
Every snippet uses the real public API; adapt the file paths to your own media.

!!! note "Before you start"
    Install the package first ([Installation](installation.md)). The first call to
    any model downloads its weights automatically. All examples use
    `import physiotrack as pt` — the top-level entry point.

## 1. Detection on an image

Configure a predictor, call `.predict()`, then draw the result. `result.plot()`
returns an annotated BGR image you can write straight to disk with OpenCV.

```python
import cv2
import physiotrack as pt

frame = cv2.imread("frame.png")

det = pt.Detection.Person(conf=0.25, iou=0.45, device="cpu")   # configure
result = det.predict(frame)                                    # -> pt.Result  (or det(frame))
cv2.imwrite("boxes.png", result.plot())                        # annotated overlay

# Iterate the detected instances
for inst in result:
    print(inst.box, round(inst.confidence, 3), inst.cls_name)
```

!!! tip "`.predict(img)` or `model(img)`"
    Every predictor is callable, so `det(frame)` is shorthand for
    `det.predict(frame)`. Pass a **list** of images to batch:
    `det.predict([img1, img2]) -> list[Result]`.

See the [Detection guide](../guides/detection.md) and the
[`Detection`][physiotrack.Detection] API for all presets and options.

For a complete face-specific version with bundled selfie, POV, crowd and VR scenes,
run `python examples/face_detection/detect_faces.py`. The
[face examples guide](../guides/face-examples.md) explains its annotated PNG, CSV,
JSON and run-metadata outputs.

For a VR scene, run `python examples/vr_detection/detect_vr_people.py` to compare
VR-head boxes, full VR-person boxes, and generic person boxes on exactly the same
image. See the [VR objects and people](../guides/detection.md#vr-objects-and-people)
section for why their counts differ.

## 2. Pose on the same image

`Pose.Person()` auto-detects people first (no boxes needed), then estimates
keypoints. The result is iterable — one entry per person — and each person exposes
a [`Keypoints`][physiotrack.Keypoints] collection you query by name.

```python
import cv2
import physiotrack as pt

frame = cv2.imread("frame.png")

pose = pt.Pose.Person()
result = pose.predict(frame)                 # auto-detects people; or pose.predict(frame, boxes)

print(len(result), "people;", result.architecture)   # "WHOLEBODY" or "COCO"

for person in result:
    wrist = person.keypoints.by_name("left_wrist")
    if wrist:
        print(f"left_wrist: ({wrist.x:.0f}, {wrist.y:.0f}) conf={wrist.confidence:.2f}")

cv2.imwrite("pose.png", result.plot())
```

- **COCO** = 17 body keypoints; **WholeBody** = 133 (body + hands + face).
- Look up keypoints by name (`by_name("nose")`) or id (`by_id(9)`), or grab array
  views: `person.keypoints.xy`, `.conf`.

See the [Pose guide](../guides/pose.md) and the [`Pose`][physiotrack.Pose] API.

## 3. Process a video

=== "Manual frame loop"
    Predictors work on any single frame, so a plain OpenCV capture loop is all you
    need for a lightweight per-frame job:

    ```python
    import cv2
    import physiotrack as pt

    pose = pt.Pose.Person(device=0)
    cap = cv2.VideoCapture("input.mp4")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pose.predict(frame)
        cv2.imshow("pose", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

=== "`Video` pipeline (recommended)"
    The [`Video`][physiotrack.Video] orchestrator handles capture, batching, the
    frame loop, overlay compositing, and writing an annotated MP4 + JSON dump:

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
    print(f"processed {len(data)} frames")
    ```

    A single-model pipeline is just as valid — e.g. `pt.Video(source="clip.mp4",
    pose=pt.Pose.VRStudent()).run("out.mp4", "out.json")`.

!!! warning "Detector + `Pose.Custom`"
    When you combine an explicit `detector=` with a pose estimator in `Video`, the
    pose estimator must be a `Pose.Custom(...)` (the presets like `Pose.Person`
    run their own internal detector). If you pass no detector, use any preset.

!!! tip "Sideways phone clips"
    Phone videos often decode rotated (the angle lives in container metadata).
    Pass `orient=90` / `180` / `270` to `Video(...)` to rotate every frame upright;
    the default `orient=0` leaves frames untouched. Still images need nothing —
    OpenCV applies EXIF orientation on load.

## Next steps

- **Understand the pattern** you just used → [Core Concepts](concepts.md).
- **Go deeper per task** → [Guides](../guides/index.md):
  [Detection](../guides/detection.md) ·
  [Pose](../guides/pose.md) ·
  [Segmentation](../guides/segmentation.md) ·
  [Depth](../guides/depth.md) ·
  [Face](../guides/face.md) ·
  [Tracking](../guides/tracking.md) ·
  [Video](../guides/video.md) ·
  [Signals](../guides/signals.md).
- **Reading the output** → [Result objects](../api/results.md).
- **Pick different weights** → [Model Zoo](../model-zoo.md).
- **Run face detection/tracking examples** → [Face Examples](../guides/face-examples.md).
- **Compare VR-head, VR-person, and person detection** →
  [Detection](../guides/detection.md#vr-objects-and-people).
