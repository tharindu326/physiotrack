# Get Started

Everything you need to install Physiotrack and run your first prediction. If you
are new, work through these three pages in order — install the package, run a
hands-on example, then learn the handful of concepts that make the whole library
predictable.

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install from source, choose a PyTorch/CUDA build, and set up the optional video
    codec. Weights auto-download on first use.

    [:octicons-arrow-right-24: Install Physiotrack](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run detection and pose on an image, render the overlay, then process a video —
    frame-by-frame and with the one-liner `Video` pipeline.

    [:octicons-arrow-right-24: Hands-on quickstart](quickstart.md)

-   :material-lightbulb-on:{ .lg .middle } **Core Concepts**

    ---

    The unified design: `.predict()`, one `Result` type, `result.plot()`, the
    object model, the `Models` registry, and the `Video` orchestrator.

    [:octicons-arrow-right-24: Learn the concepts](concepts.md)

</div>

!!! tip "Already know the shape?"
    Every predictor is `configure → predict → read → plot`. Jump straight to the
    task [Guides](../guides/index.md) or the [API Reference](../api/index.md).

!!! example "Want a repository example with real files?"
    The [face detection and tracking examples](../guides/face-examples.md) include
    small synthetic inputs and save annotated media, CSV, and JSON output.
