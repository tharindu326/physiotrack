# Guides

Task-focused guides for every Physiotrack capability. Each predictor follows the
same shape: pick a **preset**, call `predict(frame)` (or the instance directly),
and read a [`Result`](../api/results.md) that renders its own overlay via
`result.plot()`. Weights are auto-downloaded from Hugging Face on first use and
cached.

!!! tip "One mental model, every task"
    All image predictors — [`Detection`][physiotrack.Detection],
    [`Pose`][physiotrack.Pose], [`Segmentation`][physiotrack.Segmentation],
    [`Face`][physiotrack.Face] — return a [`Result`][physiotrack.Result]. Depth
    returns a [`DepthResult`][physiotrack.DepthResult]. Learn the result object once
    and it transfers everywhere.

<div class="grid cards" markdown>

- :material-selection-drag:{ .lg .middle } **Detection**

    ---

    Person, face, and VR object boxes with confidence and class labels.

    [:octicons-arrow-right-24: Detection guide](detection.md) ·
    [VR example](https://github.com/tharindu326/physiotrack/tree/main/examples/vr_detection)

- :material-run-fast:{ .lg .middle } **Pose Estimation (2D)**

    ---

    COCO-17 and WholeBody-133 keypoints with auto person detection.

    [:octicons-arrow-right-24: Pose guide](pose.md)

- :material-axis-arrow:{ .lg .middle } **3D Pose & Canonicalization**

    ---

    Lift 2D keypoints to 3D and canonicalize pose geometry.

    [:octicons-arrow-right-24: Pose 3D guide](pose3d.md)

- :material-image-filter-center-focus:{ .lg .middle } **Segmentation**

    ---

    Person masks, VR-head parts, Sapiens body parts, and 19-class face parsing.

    [:octicons-arrow-right-24: Segmentation guide](segmentation.md)

- :material-layers-triple:{ .lg .middle } **Depth**

    ---

    Monocular dense depth with Depth-Anything-V2 (Small / Base / Large).

    [:octicons-arrow-right-24: Depth guide](depth.md)

- :material-face-recognition:{ .lg .middle } **Face & Head Orientation**

    ---

    Face detection, VR-face crops, and 6-DoF yaw/pitch/roll head pose.

    [:octicons-arrow-right-24: Face guide](face.md) ·
    [Runnable examples](face-examples.md) ·
    [Validation](face-validation.md)

- :material-vector-polyline:{ .lg .middle } **Tracking**

    ---

    Persistent multi-object ids across frames from any detector.

    [:octicons-arrow-right-24: Tracking guide](tracking.md)

- :material-video:{ .lg .middle } **Video Pipeline**

    ---

    Orchestrate detection, pose, tracking, and overlays over whole videos.

    [:octicons-arrow-right-24: Video guide](video.md)

- :material-heart-pulse:{ .lg .middle } **Signals (rPPG · Motion · ROM)**

    ---

    Heart rate from skin pixels, joint angles, and range-of-motion features.

    [:octicons-arrow-right-24: Signals guide](signals.md)

</div>

## See also

- [Model Zoo](../model-zoo.md) — every downloadable backbone by task and backend.
- [Result objects](../api/results.md) — the shared output contract for all tasks.
- [Quickstart](../getting-started/quickstart.md) — install and run your first prediction.
