<div align="center">

# Physiotrack

**Contactless human understanding: turning pixels into interpretable physiological and behavioral signals.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
![Status](https://img.shields.io/badge/status-under%20construction-FF0000)
[![Docs](https://img.shields.io/badge/docs-physiotrack-00897b.svg)](https://tharindu326.github.io/physiotrack/)

📖 **[Full documentation & API reference](https://tharindu326.github.io/physiotrack/)**

[Why Physiotrack](#why-physiotrack) ·
[Architecture](#architecture) ·
[Install](#installation) ·
[Quick start](#quick-start) ·
[Subsystems](#subsystem-guide) ·
[Models](#model-registry) ·
[Limitations](#scope--limitations) ·
[Citations](#citations)

</div>

---

**Physiotrack** is an open-source Python toolkit for contactless human understanding. It integrates
state-of-the-art computer-vision models (YOLO11, RT-DETR, ViTPose, Sapiens, Depth-Anything-V2,
ZipDepth, MotionBERT, 6DRepNet360, SegFace) into a **single, unified API** that extracts actionable, theory-linked
signals from ordinary **RGB video**, for healthcare, education, XR, and operator-support
systems. Developed at the **Center for Machine Vision and Signal Processing (CMVS), University of Oulu**.

<div align="center">

[![Physiotrack real-time inference demo](docs/images/vr_human_analysis.jpg)](https://youtu.be/DFVYfZCk3t4)

▶️ **[Watch the demo on YouTube](https://youtu.be/DFVYfZCk3t4)**

</div>

### Example outputs

<div align="center">

<img src="docs/images/full_inference_overlay.png" width="78%" alt="Full multi-task pipeline overlay" />

<sub>Full pipeline on one frame of a person in VR walking in a lab: tracked whole-body pose with a persistent id, live joint-angle &amp; clinical-ROM panels, the ROM skeleton, the wrist-motion plot, and colorized <i>relative</i> monocular depth. Respiration here comes from shoulder motion, reusing the pose keypoints.</sub>

<br/><br/>

<img src="docs/images/rppg_heartrate_overlay.png" width="40%" alt="Contactless rPPG heart-rate overlay" />

<sub>Contactless rPPG on an RGB face recording: one SegFace pass yields the face parsing and skin ROI that drive the live blood-volume-pulse signal and the derived heart rate (78.0 bpm at &minus;0.1 dB SNR here). Regenerate with <code>examples/evaluation/rppg_figure.py</code>.</sub>

</div>

## Why Physiotrack?

Foundational models are good at labeling *what* they see. Physiotrack is designed to help systems
understand *what it means*. It converts an ordinary RGB video stream into interpretable
human-state features: pose and motion patterns, posture symmetry, head orientation and gaze
stability, and rPPG-derived heart rate, heart-rate variability (HRV) and respiration rate.

The philosophy is simple: **"send meaning, not pixels"**, so downstream AI agents can reason about
human *states* instead of processing raw video.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Unified API](#the-unified-api)
- [Subsystem Guide](#subsystem-guide)
- [Model Registry](#model-registry)
- [Result Objects](#result-objects)
- [Face Analysis Module](#face-analysis-module)
- [Scope & Limitations](#scope--limitations)
- [Project Layout](#project-layout)
- [Citations](#citations)
- [Contributing](#contributing)
- [License](#license)
- [Authors & Acknowledgments](#authors)

---

## Architecture

Physiotrack is organized into **independent, composable subsystems**. Each one works standalone
through the same `predict()` API, and the `Video` orchestrator wires them into an end-to-end
pipeline. Every neural backend is resolved on demand through the **`Models` registry**, which
auto-downloads weights from Hugging Face.

The modules group into three tiers that mirror the *"send meaning, not pixels"* pipeline:
**🧰 enabling tools** (generic CV) → **🧍 human structure** (pose & kinematics) → **📡 human-state
signals** (the interpretable payload).

```mermaid
flowchart TB
    %% ---------------- Inputs ----------------
    subgraph IN["📥 Inputs · monocular RGB (BGR frames)"]
        direction LR
        I1["Video file<br/>MP4 / AVI / …"]
        I2["RTSP stream"]
        I3["Live camera<br/>device index"]
        I4["Single image"]
    end

    CAP["🎞️ <b>Video orchestrator</b> · capture.Video<br/><i>frame loop · resize · rotate · FPS subsample · batching</i><br/><i>per-frame pipeline: 6 model stages · sync ego-video</i>"]

    %% ====== TIER 1: enabling tools (general-purpose CV) ======
    subgraph TOOLS["🧰 General-purpose perception · enabling tools"]
        direction LR
        DET["🔍 <b>Detection</b> → boxes<br/><i>Person · Face · VR · VRStudent · Custom</i><br/>YOLO11 · RT-DETR"]
        TRK["🎯 <b>Tracking</b> → persistent IDs<br/>OC-SORT · ByteTrack<br/>StrongSORT · BoostTrack"]
        SEG["🎭 <b>Segmentation</b> → masks / class map<br/>YOLO11-seg · Sapiens-Goliath<br/>VR-head"]
        DEP["🌊 <b>Depth</b> · monocular<br/>Depth-Anything-V2 (S/B/L) · ZipDepth<br/><i>relative depth</i>"]
        FDET["😊 <b>Face detection</b> → boxes<br/>YOLO11-face · YOLO12-face (VR)"]
    end

    %% ====== TIER 2: human structure & pose ======
    subgraph HUMAN["🧍 Human structure &amp; pose"]
        direction LR
        POSE["🦴 <b>2D Pose</b> · top-down<br/>ViTPose (s/b/l/h) · Sapiens · YOLO11-pose<br/><i>COCO-17 · WholeBody-133</i>"]
        P3D["🧊 <b>3D Pose</b> · offline lift<br/>MotionBERT · DDHPose<br/><i>→ H36M-17</i>"]
        CAN["📐 <b>Canonicalization</b> · viewpoint-invariant<br/>3DPCNet (S2/S3/TC48) · GEOMETRIC<br/><i>front · back · left_side · right_side</i>"]
        FSEG["🧩 <b>Face parsing</b><br/>SegFace · Swin-Base<br/><i>19 CelebAMask-HQ classes</i>"]
    end

    %% ====== TIER 3: human-state signals (the payload) ======
    subgraph SIGNAL["📡 Human-state signals · physiological · motion · behavioral"]
        direction LR
        PPG["❤️ <b>rPPG → HR · HRV · RR</b> · physiological<br/>POS · CHROM · LGI · OMIT<br/><i>RR intervals · Lipponen-Tarvainen · Task-Force HRV</i>"]
        ANG["📐 <b>Joint angles &amp; ROM</b> · goniometry<br/>8 interior angles + clinical ROM<br/><i>flexion · extension · abd · add</i><br/><i>angle panel + ROM skeleton</i>"]
        MOT["🏃 <b>Motion features</b><br/>velocity · accel · trajectories<br/><i>centroids · filters · metrics</i>"]
        FORI["👁️ <b>Head orientation</b> · gaze<br/>6DRepNet360 · CMVS-FO-VR<br/><i>yaw · pitch · roll</i>"]
        RAD["🗺️ <b>Floor map / radar</b> · location<br/>4-corner homography<br/><i>bird's-eye trajectories</i>"]
    end

    %% ------------- Outputs -------------
    subgraph OUT["📤 Results &amp; visualization"]
        direction LR
        RES["<b>Result family</b><br/>Result · DepthResult · TrackResult · Pose3DResult<br/>Instance · Keypoints · FrameResult / VideoResults<br/>.plot() · .to_dict()"]
        VID["Annotated<br/>video"]
        JSON["JSON<br/>time-series"]
        RTP["Real-time<br/>plots"]
    end

    %% ------------- Model registry -------------
    REG[["🗂️ <b>Models registry</b> · Models.&lt;Task&gt;.&lt;Backend&gt;.&lt;Variant&gt;<br/>YOLO11/12 · RT-DETR · ViTPose · Sapiens · SegFace<br/>Depth-Anything-V2 · ZipDepth · MotionBERT · DDHPose · 3DPCNet · 6DRepNet360<br/><i>51 pretrained variants (52 selectable) · auto-download from Hugging Face</i>"]]

    %% ---------------- Flow ----------------
    I1 --> CAP
    I2 --> CAP
    I3 --> CAP
    I4 --> CAP
    CAP --> DET
    CAP --> DEP
    CAP --> FDET
    DET --> TRK
    DET -.boxes.-> POSE
    DET -.boxes.-> SEG
    POSE --> P3D --> CAN
    FDET --> FORI
    FDET -.boxes.-> FSEG
    TRK --> RAD
    POSE --> ANG
    POSE --> MOT
    CAN -.canonical kinematics.-> ANG
    FSEG -.skin ROI.-> PPG
    FDET -.face box · fallback.-> PPG

    DET --> RES
    TRK --> RES
    POSE --> RES
    SEG --> RES
    DEP --> RES
    FSEG --> RES
    P3D --> RES
    CAN --> RES
    PPG --> RES
    ANG --> RES
    MOT --> RES
    FORI --> RES
    RAD --> RES
    RES --> VID
    RES --> JSON
    RES --> RTP

    REG -.weights.-> TOOLS
    REG -.weights.-> HUMAN
    REG -.weights.-> FORI

    classDef input  fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef tools  fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef human  fill:#fff3e0,stroke:#e65100,color:#bf360c;
    classDef signal fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;
    classDef out    fill:#fce4ec,stroke:#ad1457,color:#880e4f;
    classDef orch   fill:#eceff1,stroke:#37474f,color:#263238;
    classDef reg    fill:#fffde7,stroke:#f9a825,color:#f57f17;

    class I1,I2,I3,I4 input;
    class DET,TRK,SEG,DEP,FDET tools;
    class POSE,P3D,CAN,FSEG human;
    class PPG,ANG,MOT,FORI,RAD signal;
    class RES,VID,JSON,RTP out;
    class CAP orch;
    class REG reg;
```

**Reading the diagram:** read it as *pixels → tools → human structure → signals*.
**🧰 Enabling tools** (detection, tracking, segmentation, depth, face detection) localize and parse the
image; **🧍 human structure** (2D/3D pose, canonicalization, face parsing) turns those into
body-specific estimates; **📡 human-state signals** (rPPG HR/RR, **joint angles & clinical ROM**,
motion, head orientation/gaze, location) are the interpretable payload. Each box lists its backend
options/variants (italics = key options or output format), selected through the
[`Models` registry](#model-registry). Solid arrows are per-frame data flow; dotted arrows show
detection boxes feeding pose/segmentation, skin regions feeding rPPG, canonical pose feeding the
angles, and weights from the registry. Every module also works standalone. Inputs are **monocular
RGB** (BGR frames) via OpenCV. Depth is *estimated* monocularly rather than sensed: there is no
RGB-D, infrared or thermal capture path. Where a clip comes from an RGB-D camera, only its colour
stream is used.

---

## Key Features

| Subsystem | What it does | Backends |
|-----------|--------------|----------|
| **Detection** | Multi-person / object boxes with confidences | YOLO11, RT-DETR, VR-specific |
| **Tracking** | Persistent IDs across frames, occlusion-robust | OC-SORT, ByteTrack, StrongSORT, BoostTrack |
| **Pose 2D** | Body keypoints (17 COCO or 133 whole-body) | ViTPose, Sapiens, YOLO11-Pose |
| **Pose 3D** | Lift 2D keypoints to 3D over time | MotionBERT, DDHPose |
| **Canonicalization** | Viewpoint-invariant 3D pose alignment | 3DPCNet, geometric |
| **Segmentation** | Pixel-level instance masks | YOLO-Seg, Sapiens, VR-Head |
| **Face parsing** | Face-part segmentation (19 classes) | SegFace (Swin-Base) |
| **Depth** | Monocular dense depth estimation | Depth-Anything-V2 (s/b/l), ZipDepth (base/npu) |
| **Face** | Face detection + 3D head orientation | YOLO-Face, 6DRepNet360, CMVS-FO-VR |
| **Signals** | rPPG heart rate, HRV, respiration + motion features | POS, CHROM, LGI, OMIT; RR intervals; Lipponen-Tarvainen artefact correction; Task-Force HRV |
| **Joint angles & ROM** | 8 anatomical joint angles + clinical range-of-motion (flexion/extension/abduction/adduction) as rows in the left-side angle panel, plus a clean full-room **skeleton canvas** | goniometry from pose |
| **Views** | Bird's-eye floor map, ego-video, depth & angle/ROM overlays | n/a |

**Inputs:** monocular RGB video — files, RTSP streams, live cameras, or single images. **Hardware:** CPU or CUDA GPU (acceleration
recommended for real-time use).

---

## Installation

**Requires Python 3.10 or newer.** Install the PyTorch build that matches your platform
first, then the package:

```bash
git clone https://github.com/tharindu326/physiotrack.git
cd physiotrack

# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ...or CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

Everything else resolves from `pyproject.toml`. Optional extras:

```bash
pip install -e ".[test]"     # test suite, including the NeuroKit2 reference implementation
pip install -e ".[docs]"     # MkDocs toolchain
pip install -e ".[pose3d]"   # smplx, for 3D mesh rendering in Pose3D
```

| | Supported |
| --- | --- |
| Python | 3.10, 3.11, 3.12 |
| OS | Linux, macOS, Windows |
| Hardware | CPU, or CUDA GPU (recommended for real-time use) |

Model weights are **not** bundled; they download automatically on first use through the
[`Models` registry](#model-registry) (Ultralytics weights via `ultralytics`, everything else from
Hugging Face).

#### Where weights are cached

Checkpoints are cached **outside** the installed package, so a read-only or containerised
install works, Docker layer caching is not defeated by a multi-gigabyte write into
`site-packages`, and several environments can share one download:

| | Location |
| --- | --- |
| `$PHYSIOTRACK_HOME` set | `$PHYSIOTRACK_HOME/weights` |
| `$XDG_CACHE_HOME` set | `$XDG_CACHE_HOME/physiotrack/weights` |
| Linux (default) | `~/.cache/physiotrack/weights` |
| macOS | `~/Library/Caches/physiotrack/weights` |
| Windows | `%LOCALAPPDATA%\physiotrack\weights` |

```bash
export PHYSIOTRACK_HOME=/shared/physiotrack   # share one cache across envs or containers
```

> **Upgrading from before 1.1?** Older versions downloaded into the package directory. Run
> `python -c "import physiotrack; physiotrack.migrate_weight_cache()"` once to move existing
> checkpoints into the cache instead of re-downloading them. Add `dry_run=True` to preview.

> **Windows / H.264:** OpenCV wheels often ship without an H.264 encoder. PhysioTrack detects
> this and falls back to MPEG-4 with a warning, so exports never fail silently — but for H.264
> output run `python install_openh264.py` once to place the OpenH264 runtime library.

### Testing

```bash
pip install -e ".[test]"
pytest
```

The suite validates the biomedical DSP against closed-form definitions and synthetic signals
with known ground truth, and cross-checks the HRV, artefact-correction and pulse-peak paths
against NeuroKit2 as an independent reference implementation. It must pass with **zero
skips**: `neurokit2` is part of the `test` extra precisely so those cross-checks cannot skip
silently and leave the strongest correctness claim unverified.

---

## Quick Start

```python
import cv2
import physiotrack as pt

frame = cv2.imread("examples/IMG_0121.mp4")  # or any BGR frame you have

# Detect people
det    = pt.Detection.Person(conf=0.25, device=0)
result = det.predict(frame)          # -> Result
cv2.imwrite("out.png", result.plot())

# Whole-body 2D pose (auto-detects people if no boxes given)
pose   = pt.Pose.Person()
people = pose.predict(frame)
wrist  = people[0].keypoints.by_name("left_wrist")
print(wrist.x, wrist.y, wrist.confidence)
```

---

## The Unified API

Everything user-facing is reached through one flat, predictable hierarchy. Import the entry
points from the top-level `physiotrack` package (or a named subsystem); you never need to reach
into internal module paths:

```text
physiotrack ─┬─ Detection.Person() / .Face() / .VR() / .VRStudent() / .Custom()
             ├─ Pose.Person() / .VRStudent() / .Custom()
             ├─ Pose3D(...) · canonicalize_pose(...) · PoseCanonicalizer
             ├─ Segmentation.Person() / .VRHead() / .BodyPart() / .Face() / .Custom()
             ├─ Depth.DepthAnythingV2Small() / Base() / Large() / .ZipDepth() / .ZipDepthNPU() / .Custom()
             ├─ Face · VRFace · FaceOrientation
             ├─ Tracker(config=TrackerConfig(...))
             ├─ Video(...)                              # end-to-end pipeline orchestrator
             ├─ Models.<Task>.<Backend>.<Variant>       # model registry (auto-download)
             └─ Result · DepthResult · TrackResult      # returned by every predictor
                          ↳ .boxes · .keypoints · .seg_map · .names · .plot() · .to_dict()

physiotrack.signals ─┬─ compute (plotter-free, use directly):
                     │    joint_angles() · compute_rom_angles() · motion features · respiration_from_motion()
                     │    rPPG: POS/CHROM/LGI/OMIT · HeartRateEstimator · bvp_to_hr · bvp_snr
                     │    pulse analysis: bvp_to_rri · correct_rr_artifacts · compute_hrv · respiration_from_pulse/rri
                     │    filters · agreement metrics (Pearson, RMSE, DTW, hrv_errors, …)
                     └─ overlays (optional, wrap the compute above):
                          JointAnglePlotter · RPPGPlotter · HeartRatePlotter · HRVPlotter · RespirationPlotter · KeypointMotionPlotter · RealTimePlotter
physiotrack.pose    ── keypoint name maps (COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES)
physiotrack.face    ── drawing helpers (draw_axis, plot_pose_cube)
```

Every image predictor (`Detection`, `Pose`, `Segmentation`, `Depth`, `Face`) follows the
**same pattern** (modeled on Ultralytics / MediaPipe / scikit-learn):

```python
model  = pt.Detection.Person(conf=0.25, iou=0.45, device=0)   # 1. configure the MODEL
result = model.predict(image)                                 # 2. predict  (or: model(image))
data   = result.boxes                                         # 3. read structured data
frame  = result.plot()                                        # 4. draw the overlay
```

Three rules make the whole library predictable:

1. **One verb.** Every predictor exposes `.predict(img)` and is callable. Batch with a list:
   `predict([img, img, ...]) -> list[Result]`.
2. **One return type.** Every predictor returns a rich [`Result`](#result-objects) object, not
   tuples in mixed orders.
3. **Rendering lives on the result, not the model.** `result.plot(...)` draws the overlay;
   constructors only configure the model.

---

## Subsystem Guide

<details open>
<summary><b>Detection</b></summary>

```python
from physiotrack import Detection, Models

det = Detection.Person()                       # also .Face() .VR() .VRStudent()
result = det.predict(image)
print(result.boxes)                            # (N, 4)
for inst in result:
    print(inst.box, inst.confidence, inst.cls_name)

det = Detection.Custom(model=Models.Detection.YOLO.VR.m_vr)   # custom weights
```
</details>

<details>
<summary><b>Pose estimation (2D)</b></summary>

```python
from physiotrack import Pose, Models

pose = Pose.Person()                           # or Pose.VRStudent(), Pose.Custom(model=...)
result = pose.predict(image)                   # auto-detects people if no boxes given
print(result.architecture)                     # "WHOLEBODY" or "COCO"

for person in result:
    wrist = person.keypoints.by_name("left_wrist")
    if wrist:
        print(wrist.x, wrist.y, wrist.confidence)

pose = Pose.Custom(model=Models.Pose.ViTPose.WholeBody.l_wholebody)
```

- **COCO**: 17 keypoints (body only). **WholeBody**: 133 keypoints (body + hands + face).
- When no bounding boxes are supplied, the pose estimator detects people with the default detector.
</details>

<details>
<summary><b>Pose 3D &amp; canonicalization</b></summary>

```python
from physiotrack import Models, Pose, Pose3D, Video, canonicalize_pose

results = Video(source="clip.mp4", pose=Pose.Person()).run()   # 2D pass

p3d = Pose3D(model=Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite, device="cpu")
poses = p3d.predict(results, fps=30)        # -> Pose3DResult, (N, 17, 3) H3.6M order
poses.by_name("left_wrist")                 # (N, 3) trajectory of one joint

# Viewpoint-invariant canonical form
canonical = canonicalize_pose(poses.poses, view="front")        # geometric (training-free) default
canonical = canonicalize_pose(                                 # learned 3DPCNet (recommended)
    poses.poses,
    model=Models.Pose3D.Canonicalizer.Models._3DPCNetTC48_byCam,
    view="front",
)

# Or canonicalize during lifting, and use predict_json() for the file-based workflow.
```

> Lifting is **sequence-level** (a temporal model needs a window of 2D frames per 3D frame)
> and single-subject. Coordinates are root-relative, not metric.
</details>

<details>
<summary><b>Segmentation, Depth, Face</b></summary>

```python
from physiotrack import Segmentation, Depth, VRFace, FaceOrientation, Models

seg = Segmentation.Person()
seg_map = seg.predict(image).seg_map           # (H, W) class map

# Face parsing (SegFace, 19 face-part classes). Faces are auto-detected if no
# boxes are given; pass boxes=[...] to parse specific faces.
parse = Segmentation.Face()
result = parse.predict(image)                  # -> Result(task="segment")
seg_map = result.seg_map                       # (H, W) face-part class map
annotated = result.plot()                      # overlay with the 19-class palette

depth = Depth.DepthAnythingV2Base()            # or Depth.ZipDepth() — lightweight, 6.8M params
d = depth.predict(image)
raw, colored = d.depth, d.plot(colormap="inferno")

face = VRFace()
boxes = face.predict(image).boxes
orient = FaceOrientation(model=Models.Pose3D.FaceOrientation.VR)
for inst in orient.predict(image, boxes):
    print(inst.orientation)                    # {"yaw": .., "pitch": .., "roll": ..}
```
</details>

<details>
<summary><b>Signals: rPPG / heart rate / HRV / respiration &amp; motion</b></summary>

The computation is **plotter-free** — use it directly; the overlay is an optional wrapper.

```python
from physiotrack.signals import POS, bvp_to_hr, bandpass_filter
from physiotrack.signals import FaceSkinExtractor, HeartRateEstimator

# Low level: one rPPG method on an RGB skin trace, shape (3, N) with rows R, G, B
bvp   = POS(fps=30).apply(rgb_trace)            # blood-volume-pulse candidate
clean = bandpass_filter(bvp, 0.75, 4.0, 30)
hr_bpm, times = bvp_to_hr(clean, fps=30)        # per-window HR (bpm) + window-centre times
latest = hr_bpm[-1]                             # a series, not a single number

# High level: SegFace face parsing -> rPPG on the skin (no plotter)
fs  = FaceSkinExtractor()                        # SegFace (detects faces itself)
est = HeartRateEstimator("POS", fps=30, window_sec=60)   # POS/CHROM/LGI/OMIT; bands configurable
mask, skin_canvas = fs.extract(frame)            # skin ROI mask + image-res skin canvas
est.update(frame, roi_mask=mask)                 # rPPG on the segmented skin; call per frame
print(est.hr, est.snr)                           # HR (bpm), de Haan SNR (dB)

# The same window also yields the downstream pulse analytics (no re-computation):
print(est.hrv())                                 # {RMSSD, SDNN, pNN50, SD1, SD2, LF/HF, ...}
print(est.respiration_rate())                    # breaths/min (RIAV; use "rsa" for RSA)

# Or step through the pulse-analysis chain explicitly:
from physiotrack.signals import bvp_to_rri, correct_rr_artifacts, compute_hrv
rri_ms, _   = bvp_to_rri(clean, fps=30)          # RR / inter-beat intervals (ms)
rri_ms, _   = correct_rr_artifacts(rri_ms)       # Lipponen-Tarvainen (2019) beat correction
hrv         = compute_hrv(rri_ms)                # time + frequency + non-linear HRV
# Note: correction re-derives the series from the corrected beats, so its length can
# change. Let compute_hrv rebuild the timestamps rather than reusing the originals.
```
(`update` also accepts a face `box` as a lightweight fallback when you don't run segmentation.)

For on-frame overlays, wrap a (shared) estimator with `RPPGPlotter` (BVP pulse), `HeartRatePlotter`
(bpm), `HRVPlotter` (HRV grid) and `RespirationPlotter` (breaths/min) — all read one
`HeartRateEstimator`, so the rPPG is computed once. See
[`examples/rppg_vitals.py`](examples/rppg_vitals.py) (all panels together); or enable them in the
pipeline with `Video(..., rppg=True, hrv=True, respiration=True)`. Respiration also has a pose-based route
(`respiration_from_motion`) that cross-validates the rPPG estimate. All HRV / artefact-correction
formulas are validated against closed-form definitions, and the HRV, artefact-correction and
pulse-peak paths are additionally cross-checked against NeuroKit2 as a reference implementation. Also includes motion features,
filters/normalizers, and signal-agreement metrics (`compute_rmse`,
`calculate_pearson_correlation`, `calculate_dtw_distance`, `hrv_errors`, …).
</details>

<details>
<summary><b>Joint angles &amp; clinical ROM (goniometry)</b></summary>

Two kinds of angle, both derived from pose keypoints:

- **Interior joint angles** — 8 anatomical angles (left/right **shoulder, elbow, hip, knee**), the
  angle *at* each joint. Good for any activity (e.g., gait-cycle joint-angle trajectories).
- **Clinical range-of-motion (ROM)** — named physiotherapy movements (**hip flexion / extension /
  abduction / adduction**), measured against a body reference axis. In the pipeline the values appear
  in the ROM grid, and the movements are drawn as **colour-coded goniometric arcs** on a clean
  **white-background skeleton panel** that mirrors the full room (so the person's position is
  preserved and the main frame stays uncluttered). Both sit on the left, under the joint-angle grid.
  Best for a controlled assessment (patient lying or standing).

The measurement is **plotter-free** — run it straight on pose keypoints:

```python
import physiotrack as pt
from physiotrack.signals import joint_angles, compute_rom_angles, JointAnglePlotter

det = pt.Pose.Person().predict(frame).to_dict()["instances"]
kps = det[0]["keypoints"]   # list of {"id", "x", "y", "confidence"} for one person
joint_angles(kps)        # {'leftElbow': 152, 'leftKnee': 174, ...}   interior angles
compute_rom_angles(kps)  # {'leftHipFlexion': 12, 'leftHipAbduction': 34, ...}  clinical ROM

# Optional overlay (wraps the same functions): joint-angle grid + ROM grid (2-col L|R)
plotter = JointAnglePlotter(rom=True)
plotter.update(result.to_dict()["instances"], frame_time=t)
frame = plotter.attach_to_frame(frame, position="top_left")
```

In the `Video` pipeline (see below): `plot_angles=True` shows the interior joint-angle grid;
`rom=True` (or a list like `["leftHipFlexion", "rightHipFlexion"]`) adds the clinical ROM grid and a
full-room **skeleton canvas** with color-matched arcs; `rom_render=False` keeps the ROM values in the
grid but hides the skeleton canvas. The angle/ROM grids and skeleton stack together on the left.
</details>

<details>
<summary><b>Tracking &amp; full video pipeline</b></summary>

```python
from physiotrack import Tracker, TrackerConfig, Video, Pose, Detection, Models

tracker = Tracker(config=TrackerConfig(tracker="ocsort", classes=[0]))

video = Video(
    source="input.mp4",
    detector=Detection.Person(),
    pose=Pose.Custom(model=Models.Pose.ViTPose.WholeBody.b_wholebody),
    tracker=tracker,
    output_dir="output",
)
data = video.run(output_video="out.mp4", output_json="out.json")
```

The `Video` orchestrator composes any subset of the pipeline. Besides `detector=`, `pose=` and
`tracker=` it accepts `segmenter=`, `depth=`, `face=`, `face_orientation=`, `floor_map=` (radar view),
`ego_video=`, `plot_keypoint=` (keypoint-motion plot), and the kinematics overlays
`plot_angles=True` (interior joint-angle panel, left), `rom=` (clinical ROM on a right-side skeleton
panel — `True` for the default hip set or a list of movements), and `rom_render=False` (compute ROM
without showing the skeleton panel). Overlay placement: the interior-angle panel sits on the **left**;
the motion plot, radar, depth, ego and **ROM skeleton** views stack on the **right**.

Phone clips often decode sideways/upside-down (the rotation is stored as metadata, not in the
pixels). Pass `orient=90/180/270` to rotate every frame upright; the default `orient=0` leaves frames
untouched. There is no auto/metadata mode — the angle is explicit because that metadata is unreliable
across builds. Still images need nothing here (OpenCV applies EXIF orientation on load).

```python
# Full pipeline with the interior-angle panel and the clinical ROM skeleton panel
Video(
    source="input.mp4",
    detector=Detection.Person(),
    pose=Pose.Person(),
    plot_angles=True,                 # interior joint-angle panel (left)
    rom=True,                         # hip flexion + abduction → ROM skeleton panel (left)
).run(output_video="out.mp4", output_json="out.json")
```
</details>

---

## Model Registry

All weights are addressed through the `Models` registry and auto-downloaded on first use:

```python
from physiotrack import Models

Models.Detection.YOLO.FACE.m_face           # YOLO face detector
Models.Detection.RTDETR.PERSON.x_person     # RT-DETR person detector
Models.Pose.ViTPose.WholeBody.b_wholebody   # ViTPose whole-body
Models.Pose.Sapiens.WholeBody.B1_TS_COCOHB  # Sapiens whole-body
Models.Segmentation.SegFace.Face.swinb_celeba_512  # SegFace face parsing (19 parts)
Models.Depth.DepthAnythingV2.vitb           # Depth-Anything-V2 base
Models.Depth.ZipDepth.base                  # ZipDepth (lightweight monocular depth)
Models.Pose3D.MotionBERT.mb_ft_h36m         # MotionBERT 3D lifter
Models.Pose3D.Canonicalizer.Models._3DPCNetS2   # 3DPCNet pose canonicalizer
Models.Pose3D.FaceOrientation.VR            # CMVS-FO-VR head-orientation
```

### Pose framework comparison

| Framework  | Variants         | Keypoints | Notes                                  |
|------------|------------------|-----------|----------------------------------------|
| YOLO-Pose  | COCO             | 17        | Fast, integrated detection + pose      |
| ViTPose    | COCO, WholeBody  | up to 133 | Transformer-based, high accuracy       |
| Sapiens    | WholeBody        | 133       | State-of-the-art whole-body estimation |

### Canonicalization models (3DPCNet)

The pose canonicalizer maps an arbitrary-viewpoint 3D pose to a **viewpoint-invariant canonical form**, so downstream kinematic analysis is robust to camera placement. Two model families are released (all use a Hybrid GCN-Transformer architecture); pick by your deployment scenario, or use `GEOMETRIC` for a training-free closed-form baseline.

| Registry member | Training data | Test split | MPJPE ↓ | PA-MPJPE ↓ | Rot. err ↓ |
|-----------------|---------------|------------|:------:|:----------:|:----------:|
| `_3DPCNetS2` | MMFi | TotalCapture (cross-dataset) | 49.4 | 38.5 | 4.47° |
| `_3DPCNetS3` | MMFi | TotalCapture (cross-dataset) | 48.4 | 37.1 | 4.20° |
| `_3DPCNetTC48_byCam` | TotalCapture (48 augmented cams) | held-out cams 41–48 | **44.1** | **27.6** | **0.45°** |
| `_3DPCNetTC48_byAction` | TotalCapture (48 augmented cams) | held-out action (rom3) | 46.2 | **27.6** | 1.24° |

> MPJPE / PA-MPJPE in mm, rotation error in degrees (lower is better), reported on the TotalCapture test set. `S2` / `S3` are two split configurations of the MMFi-trained model; `TC48_byCam` / `TC48_byAction` are trained directly on TotalCapture with 48 augmented camera angles and use camera- vs. action-disjoint test splits. The TC48 models cut rotation error by roughly an order of magnitude and reduce PA-MPJPE by ~10 mm versus the cross-dataset MMFi models.

### Model formats

- **Sapiens**: TorchScript `.pt2` (optimized inference)
- **ViTPose**: PyTorch `.pth` weights + matching config
- **YOLO / RT-DETR**: native format with built-in config

---

## Result Objects

| Task | Returns | Key attributes |
|------|---------|----------------|
| detect / pose / segment / face | `Result` | `.boxes`, `.instances`, `.keypoints`, `.seg_map`, `.architecture`, `.meta`, `.plot()`, `.to_dict()`, `.from_dict()` |
| depth | `DepthResult` | `.depth`, `.normalized()`, `.plot(colormap=...)` |
| track | `TrackResult` | `.instances`, `.ids`, `.boxes`, `.plot(frame)` |
| 3D lift (`Pose3D`) | `Pose3DResult` | `.poses` `(N,17,3)`, `.by_name(joint)`, `.fps`, `.view`, indexable/iterable by frame |
| `Video.run()` | `VideoResults` of `FrameResult` | per frame: `.meta`, `.vitals`, `.hr`, `.snr`, iterates its `Instance`s; `.to_json()` |

Each `Instance` exposes `.id`, `.box`, `.confidence`, `.cls` / `.cls_name`, `.keypoints`
(a `Keypoints` collection with `.by_name()` / `.by_id()`), `.mask`, and `.orientation` as
applicable. `Keypoints` also offers the array views `.xy`, `.xyz`, and `.conf`.

Every result carries a `.meta` ([`ResultMeta`](#result-objects)) recording where it came
from — frame index, timestamp, source frame rate, model, device — and `to_dict()` /
`from_dict()` round-trip, so a saved JSON can be reloaded into the object model rather than
hand-parsed.

---

## Face Analysis Module

The **Face Analysis Module** extends Physiotrack with a **modular, per-face analysis pipeline** exposed as `physiotrack.FaceAnalysis`. It integrates existing PhysioTrack face detection, tracking, head orientation, and semantic parsing with facial landmarks, eye and blink measurements, geometric and learned gaze, mouth dynamics, facial expression recognition, face-quality descriptors, sliding-window summaries, and structured result export. The implementation is in [`src/physiotrack/face/`](src/physiotrack/face/); the runnable video example is [`examples/face_analysis.py`](examples/face_analysis.py).

Its implementation covers **component integration, image geometry, face-to-result association, temporal state and output handling, and validation**. Detection and neural inference reuse pretrained backends rather than retraining YOLO-Face, 6DRepNet360, MediaPipe, SegFace, EmotiEffLib, or the optional ptgaze models. The face-analysis API operates separately from the existing general-purpose `Video` orchestrator; application code decodes video frames and calls `FaceAnalysis.predict` as demonstrated below.

### Execution architecture

The following diagram describes the implemented dependencies and per-frame data flow (conditional branches execute only when enabled). Input NumPy frames and OpenCV-decoded paths use **BGR** ordering; components perform RGB conversion at their respective inference boundaries. Boxes supplied to analysis components are the **selected face result**: tracker-associated boxes when tracking is enabled, otherwise detector boxes.

```mermaid
flowchart TB
    INPUT["BGR image / decoded video frame"] --> DET["Face detection: YOLO-Face"]
    DET --> SWITCH{"Tracking enabled?"}
    SWITCH -->|Yes| TRACK["FaceTracker: associated boxes and session-local IDs"]
    SWITCH -->|No| SELECT["Selected face boxes, IDs and detection metadata"]
    TRACK --> SELECT

    SELECT --> GAPS["Missing-ID handling: interrupt blink/mouth continuity; reset temporal windows"]
    GAPS --> FACES{"Any selected faces?"}
    FACES -->|No| RESULT["Result: task = face"]
    FACES -->|Yes| ROUTE["Analyze each selected face using its original box"]

    ROUTE --> POSE["Head pose: 6DRepNet360, if enabled"]
    ROUTE --> LEARNED["Learned gaze: ptgaze, if enabled"]
    ROUTE --> REGIONS["Semantic regions: SegFace, if enabled"]
    ROUTE --> QUALITY["Face quality: BGR face crop, if enabled"]
    ROUTE --> EMOTION["Expression: BGR face crop to RGB, if enabled"]
    ROUTE --> LANDMARKS["Face landmarks: MediaPipe, if enabled"]

    LANDMARKS --> EYES["Eye openness: landmark geometry"]
    EYES --> BLINK["Blink: FPS and non-null person ID"]
    LANDMARKS --> IRIS["Geometric gaze: relative iris position"]
    LANDMARKS --> MOUTH["Mouth openness: landmark geometry"]
    MOUTH --> MOTION["Mouth movement / velocity: FPS and non-null person ID"]

    ROUTE --> ASSEMBLE["Assemble per-face Instance and face_features"]
    POSE --> ASSEMBLE
    LEARNED --> ASSEMBLE
    REGIONS --> ASSEMBLE
    QUALITY --> ASSEMBLE
    EMOTION --> ASSEMBLE
    LANDMARKS --> ASSEMBLE
    EYES --> ASSEMBLE
    BLINK --> ASSEMBLE
    IRIS --> ASSEMBLE
    MOUTH --> ASSEMBLE
    MOTION --> ASSEMBLE

    ASSEMBLE --> TEMPORAL["Optional per-ID FaceTemporalAggregator"]
    TEMPORAL --> RESULT
    ASSEMBLE --> RESULT
    RESULT --> EXPORT["Separate, explicit FaceResultExporter: JSON / CSV"]
```

The orchestration is implemented in [`face/analysis.py`](src/physiotrack/face/analysis.py). Within a frame it performs detection, optional tracking and missing-identity handling; then head orientation, optional learned gaze and semantic regions; then, for each selected face, associated landmarks, quality, eye/blink, geometric gaze, mouth/motion, expression and region summaries; finally instance assembly and optional temporal updating. **Landmarks are shared** by the eye, geometric-gaze and mouth computations rather than separately re-inferred for each descriptor. Head-pose association retains source-box indices and checks original boxes; landmark assignment considers the requested and competing boxes; regions and learned gaze use their own box-association rules. This is a data-flow description, not a claim that every component runs on every configuration.

### Components and output contracts

| Component | Implementation and method | Frame-level result or behavior |
|---|---|---|
| Face detection | [`Face`](src/physiotrack/face/detect.py), YOLO face preset | Face bounding boxes, detection confidence and class metadata. |
| Face tracking | [`FaceTracker`](src/physiotrack/face/tracking.py), default OC-SORT | Frame-to-frame association and session-local face IDs; alternatives: ByteTrack, StrongSORT and BoostTrack. |
| Facial landmarks | [`FaceLandmarks`](src/physiotrack/face/landmarks.py), MediaPipe Face Landmarker | Associated 478-point landmark group when available; integrated `landmarks` summary records availability and count. |
| Head pose | [`FaceOrientation`](src/physiotrack/face/face_orientation.py), 6DRepNet360 | `Instance.orientation` containing yaw, pitch and roll in degrees, or `None` if unmatched. |
| Eye openness | [`EyeOpenness`](src/physiotrack/face/eyes.py) | Left/right and mean dimensionless eye-aspect ratios from pixel-consistent landmark distances. |
| Blink events | [`BlinkDetector`](src/physiotrack/face/blink.py) | Per-ID eye state, completed blink-event pulses, cumulative count, event duration and cumulative blinks/min. |
| Geometric gaze | [`GazeDescriptor`](src/physiotrack/face/gaze.py) | Normalized iris positions relative to the corresponding eye geometry, **not** absolute screen gaze. |
| Learned gaze | [`GazeEstimator`](src/physiotrack/face/gaze_estimation.py), optional ptgaze | Associated 3D unit gaze vector, pitch/yaw angles, and association IoU; independent of the geometric descriptor. |
| Mouth openness | [`MouthOpenness`](src/physiotrack/face/mouth.py) | Landmark-derived width-normalized mouth dimensions and dimensionless openness. |
| Mouth dynamics | [`MouthMovement`](src/physiotrack/face/mouth_motion.py) | Per-ID absolute change in openness and change per second, with missing-value continuity reset. |
| Expression recognition | [`FaceEmotion`](src/physiotrack/face/emotion.py), EmotiEffLib ONNX | Predicted class, reported confidence and scores; default model `enet_b0_8_best_afew` provides eight classes. |
| Facial regions | [`FaceRegions`](src/physiotrack/face/regions.py), SegFace | Per-face semantic classes, mask-pixel counts, skin fraction and association IoU; based on 19 classes including background. |
| Face quality | [`FaceQuality`](src/physiotrack/face/quality.py) | Independent detector confidence, normalized brightness, Laplacian-variance sharpness and relative face area. |
| Temporal summaries | [`FaceTemporalAggregator`](src/physiotrack/face/temporal.py) | Per-ID bounded-window numeric summaries, blink-event count and most frequent valid expression. |
| JSON/CSV export | [`FaceResultExporter`](src/physiotrack/face/export.py) | Explicit per-frame and temporal-window records; nested JSON or flattened CSV. |

**Interpretation and implementation boundaries.** Face-quality indicators are **not** merged into a calibrated quality score or an automatic acceptance gate. EmotiEffLib expression scores are backend outputs, not established psychological probabilities. The returned geometric gaze coordinates, learned gaze angles and head-pose angles have different meanings. The current temporal aggregator covers head pose, mean eye openness, geometric iris coordinates, mouth openness and movement, brightness/sharpness/face area, blink event pulses and dominant expression; it **does not** aggregate learned gaze, region masks or summaries, mouth velocity or cumulative blink rate. Per-face region masks are generated from independent crop-local maps, while the full-frame segmentation composite is retained for compatibility.

#### Configuration and identity semantics

The main switchboard is [`FaceAnalysisConfig`](src/physiotrack/face/config.py). Tracking, head pose, landmarks, quality, eyes, blink, geometric gaze, mouth, mouth motion, expression, regions and temporal processing default to `True`; **learned gaze estimation defaults to `False`**. The default tracker is `ocsort`, the default blink threshold is `0.22` with **3** minimum closed frames in `FaceAnalysis`, and the temporal window is `5.0` seconds. Default learned gaze mode is `eth-xgaze`, with minimum face-association IoU `0.10`.

- `eyes` and geometric `gaze` require `landmarks`; `blink` requires `eyes`; `mouth_motion` requires `mouth`; and `temporal` configuration requires `tracking`. `gaze_estimation` has its own optional backend and does not require `FaceLandmarks`.
- Constructing the blink detector, mouth movement and temporal aggregator also requires a positive `fps`. **Blink and mouth-motion histories require a non-`None` person ID**; no ID is inferred from the face's position in the result list. In the absence of an ID, instantaneous supported measurements can remain available while those two history outputs are unavailable.
- A blink is counted upon reopening after a sufficient closed-eye run; interrupted closures do not become completed events. The cumulative blink rate uses the inclusive elapsed interval of analyzed frames for each ID. A missing person interrupts pending blink/mouth continuity and resets that person's aggregation window.
- Track IDs are **association labels**, not verified biometric identities. `reset_temporal_state()` does not reset the tracker; independent recordings should use independent analyzer/tracker sessions.
- Disabled/missing measurements retain component-specific availability and null/default semantics. `gaze_estimation` is **omitted**, rather than inserted as an unavailable field, when learned gaze is disabled.

### Installation and use

Follow the [main installation instructions](#installation) for the base package. Learned gaze requires the separately declared optional dependency:

```bash
pip install -e ".[gaze]"     # optional learned gaze: ptgaze==0.3.0
```

The default face detector, landmarker, expression recognizer and other selected neural components resolve their weights according to their respective model registries/backends. The optional ptgaze backend may download its pretrained checkpoint. Network access and model availability are needed for first-time initialization unless the relevant assets are already cached.

**Independent still image:** disable the video-state components. The example uses an ordinary image file, not a synthetic frame sequence.

```python
import physiotrack as pt
from physiotrack.face import FaceAnalysisConfig

config = FaceAnalysisConfig(
    tracking=False,
    blink=False,
    mouth_motion=False,
    temporal=False,
)
pipeline = pt.FaceAnalysis(config=config, device="cpu")
try:
    result = pipeline.predict("image.jpg")  # OpenCV reads BGR
    for face in result:
        print(face.box, face.orientation, face.face_features)
finally:
    pipeline.close()
```

**Ordered video frames and native export:** supply the actual decoded-video FPS; the application, not `FaceAnalysis`, owns video decoding and output writing. A single input returns one `Result`, whereas a list of inputs returns a list of results processed sequentially by the same stateful instance.

```python
import cv2
import physiotrack as pt
from physiotrack.face import FaceAnalysisConfig, FaceResultExporter

cap = cv2.VideoCapture("input.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open input.mp4")
fps = float(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    cap.release()
    raise ValueError("A positive source-video FPS is required")

config = FaceAnalysisConfig(gaze_estimation=False)
pipeline = None
frame_rows, window_rows = [], []
try:
    pipeline = pt.FaceAnalysis(config=config, fps=fps, device="cpu")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.predict(frame)
        timestamp = frame_index / fps
        frame_rows.extend(FaceResultExporter.frame_records(result, frame_index, timestamp))
        window_rows.extend(FaceResultExporter.window_records(result, frame_index, timestamp))
        frame_index += 1
finally:
    cap.release()
    if pipeline is not None:
        pipeline.close()

FaceResultExporter.save_json(frame_rows, "output/face_frames.json")
FaceResultExporter.save_csv(window_rows, "output/face_windows.csv")
```

For an existing minimal example, see [`examples/face_analysis.py`](examples/face_analysis.py). To enable the optional learned gaze model, install the `gaze` extra and set `FaceAnalysisConfig(gaze_estimation=True, gaze_estimation_mode="eth-xgaze")`. Do not treat a list of **unrelated still images** as a meaningful tracked video sequence: state is reused between calls unless intentionally reset or disabled. The export API does not automatically save files after each prediction.

### Validation design and scientific evidence

Validation is organized by component under [`validation/`](validation/) with documented input protocols, runnable evaluation scripts and recorded numeric outputs. The validation methodology separates **external benchmark accuracy**, **isolated real-component execution**, **deterministic correctness/regression checks**, **end-to-end integration**, **runtime measurement**, and **controlled robustness assessment**. These are different forms of evidence: successful execution and output availability do **not** establish predictive accuracy, and runtime throughput is not an accuracy measure.

**Reported benchmark and controlled-reference results** are taken from the linked component `results/` CSV/Markdown files; each figure reflects its named evaluation protocol and population, not a new overall face-analysis accuracy score.

| Validated stage | Evaluation population and measure | Recorded result | Numerical source |
|---|---|---|---|
| Face detection | WIDER FACE official validation, Easy / Medium / Hard AP | **0.959001 / 0.949087 / 0.872082** | [`wider_face_thesis_table.csv`](validation/face_detection/results/wider_face_thesis_table.csv) |
| Face tracking | ECCV 2016 eight-video detector-plus-tracker evaluation | **MOTA 44.04%; IDF1 6.53%; 2,302 ID switches** overall | [`eccv16_tracking_thesis_table.csv`](validation/face_tracking/results/eccv16_tracking_thesis_table.csv) |
| Facial landmarks | 300-W, 600 Indoor/Outdoor images, 68-point benchmark mapping | **588/600** successful; **4.65% mean NME** | [`300w_landmark_thesis_table.csv`](validation/face_landmarks/results/300w_landmark_thesis_table.csv) |
| Head pose | AFLW coarse-pose reference, absolute yaw / pitch / roll error | **11.1036° / 13.7046° / 13.7529° MAE**; **12.8537°** overall | [`aflw_head_pose_thesis_table.csv`](validation/head_pose/results/aflw_head_pose_thesis_table.csv) |
| Eye openness and blink | MPEBlink 2.0 official test: 212 videos, 687 person sequences | **Eye-openness ROC AUC 0.6608**; blink-event **F1 28.62%**, recall **25.63%** | [`mpeblink_test_thesis_table.csv`](validation/eye_openness_and_blink/results/mpeblink_test_thesis_table.csv) |
| Learned gaze | MPIIFaceGaze with eth-xgaze; 37,629 successful of 37,667 annotations | Mean **8.2419°** 3D angular error | [`mpiifacegaze_ethxgaze_thesis_table.csv`](validation/gaze_estimation/results/mpiifacegaze_ethxgaze_thesis_table.csv) |
| Mouth openness | FELT/RAVDESS speech subset; 158,286 annotated frame comparisons | **MAE 0.032706** openness-ratio units; **Pearson r 0.932273** | [`felt_ravdess_mouth_openness_thesis_table.csv`](validation/mouth_openness/results/felt_ravdess_mouth_openness_thesis_table.csv) |
| Mouth movement and velocity | FELT/RAVDESS; 156,846 valid consecutive-frame transitions | Movement **MAE 0.0115167**; velocity **MAE 0.345156** openness/s | [`felt_ravdess_mouth_movement_velocity_thesis_table.csv`](validation/mouth_movement_velocity/results/felt_ravdess_mouth_movement_velocity_thesis_table.csv) |
| Facial expressions | CAER-S annotated target-face test subset, 13,942 crops | **Accuracy 26.75%; macro F1 26.00%** | [`caers_emotion_metrics.csv`](validation/emotion_recognition/results/caers_emotion_metrics.csv) |
| Semantic facial regions | CelebAMask-HQ official CelebA-based test partition, 2,824 images | **All-class mIoU 81.5328%; pixel accuracy 95.5849%** | [`celebamaskhq_summary_table.csv`](validation/face_regions/results/celebamaskhq_summary_table.csv) |
| Face quality | Controlled brightness, Gaussian-blur and bounding-box scale changes over 300-W | Descriptor response and expected monotonicity checked; **no supervised quality-accuracy claim** | [`face_quality_metrics.csv`](validation/face_quality/results/face_quality_metrics.csv) |
| Temporal aggregation | Eight deterministic validation cases | **150/150 checks passed**, no recorded numeric discrepancy | [`temporal_aggregation_metrics.csv`](validation/temporal_aggregation/results/temporal_aggregation_metrics.csv) |

In particular, the low **tracking IDF1**, **blink-event F1**, and **CAER-S expression accuracy** are retained as measured: they must not be replaced by component-execution availability rates, nor generalized to other datasets or deployment conditions. The AFLW pose reference is a coarse benchmark and is not a calibrated clinical pose reference. The face-quality tests evaluate implementation-consistent response to controlled transformations, not a learned quality classifier. For metrics, ground-truth protocols, split definitions, per-sample data and benchmark-specific citations, use the corresponding component README files below.

| Validation package | Protocol and executable entry points |
|---|---|
| [`face_detection`](validation/face_detection/README_WIDER_FACE.txt) | WIDER FACE evaluation, prediction files, isolated execution and plotting. |
| [`face_tracking`](validation/face_tracking/README_ECCV2016_FACE_TRACKING.txt) | Detector-plus-tracker MOT metrics, sequence-level analysis and association tests. |
| [`face_landmarks`](validation/face_landmarks/README_300W_FACIAL_LANDMARKS.txt) | 300-W evaluation, landmark assignment regression and per-image outputs. |
| [`head_pose`](validation/head_pose/README_AFLW_HEAD_POSE.txt) | AFLW pose evaluation and source-box association regression. |
| [`eye_openness_and_blink`](validation/eye_openness_and_blink/README_MPEBLINK_EYE_OPENNESS_AND_BLINK.txt) | MPEBlink validation-split threshold calibration, test-set event scoring and state checks. |
| [`gaze_estimation`](validation/gaze_estimation/README_GAZE_ESTIMATION.txt) | MPIIFaceGaze eth-xgaze vector-angular evaluation and per-participant records. |
| [`mouth_openness`](validation/mouth_openness/README_MOUTH_OPENNESS.txt) | FELT/RAVDESS openness-ratio agreement and actor-level outputs. |
| [`mouth_movement_velocity`](validation/mouth_movement_velocity/README_MOUTH_MOVEMENT_VELOCITY.txt) | FELT/RAVDESS temporal derivative agreement by actor and trial. |
| [`emotion_recognition`](validation/emotion_recognition/README_EMOTION_RECOGNITION.txt) | CAER-S target-face classification metrics and class-level error analysis. |
| [`face_regions`](validation/face_regions/README_CELEBAMASKHQ_FACE_REGIONS.txt) | CelebAMask-HQ pixel IoU/Dice and crop-overlap regression. |
| [`face_quality`](validation/face_quality/README_FACE_QUALITY.txt) | Deterministic indicator and perturbation-response checks. |
| [`temporal_aggregation`](validation/temporal_aggregation/README_TEMPORAL_AGGREGATION.txt) | Independent numerical recomputation, per-ID windows and state/reset correctness. |
| [`integration`](validation/integration/README_INTEGRATION.txt) | Single-/multi-person videos, images, configuration, native export and runtime. |
| [`robustness`](validation/robustness/README_ROBUSTNESS.txt) | Seven controlled still-image conditions against one baseline, reported descriptively. |

#### Integration, performance and controlled robustness

The [`integration`](validation/integration/README_INTEGRATION.txt) checks include toggled feature configurations, learned/geometric gaze coexistence and association, real full-frame multi-person execution, per-person temporal consistency, still-image handling, independent JSON/CSV export and full-video processing. Its recorded artifacts include:

- **Static-image integration:** **21/21 images passed**, **25 face records**, with time-dependent video features explicitly treated as not applicable ([`image_e2e_summary.json`](validation/integration/results/image_e2e/image_e2e_summary.json)).
- **Multi-person end-to-end:** **344 processed frames** and **688 face instances** in the recorded two-person video ([`multi_person_full_video_summary.json`](validation/integration/results/multi_person_e2e/multi_person_full_video_summary.json)).
- **Native export:** **429 frame records** and **429 window records** for the recorded 429-frame single-person video ([`face_pipeline_native_export_summary.json`](validation/integration/results/native_export/face_pipeline_native_export_summary.json)).
- **Whole-project end-to-end:** **568/568 processed frames**, with numerical mouth-motion availability in all 568 face samples and 567 nonzero motion samples ([`whole_project_e2e_summary.json`](validation/integration/results/whole_project_e2e/whole_project_e2e_summary.json)).

The **full-pipeline runtime** run enabled detection, tracking, landmarks, head pose, quality, eyes/blink, geometric and learned gaze, mouth/motion, expressions, semantic regions and temporal processing. Across **three recordings / 1,341 frames / 1,685 face instances**, it recorded **1,639.21 ms mean processing time per frame**, **1,342.39 ms median** and **0.610 processing frames/s**. This is the recorded **CPU-only Windows execution environment**, not a universal speed guarantee; per-frame processing time excludes decoding and export writes, while initialization is measured separately. See [`face_pipeline_runtime_summary.json`](validation/integration/results/runtime_performance/face_pipeline_runtime_summary.json) and the [per-video results](validation/integration/results/runtime_performance/face_pipeline_runtime_results.csv).

The [`robustness`](validation/robustness/README_ROBUSTNESS.txt) evaluation compares a baseline image with **six deterministic modifications** (dim/overexposed lighting, Gaussian/motion blur, small-face scaling and partial occlusion). Face detection succeeded in **7/7 cases** in that controlled experiment. The recorded changes in detector confidence, sharpness, head/gaze angles, mouth openness and expression are **descriptive case-specific observations**, not a general robustness or accuracy percentage; temporal video components were not tested on those still images. See [`robustness_summary.json`](validation/robustness/results/robustness_summary.json).

### Reproducibility, tests and media provenance

The validation packages store their protocol and expected dataset layout in their respective `README_*.txt` files; they provide component-level evaluators, aggregation/plotting scripts, independent execution tests, and tabular artifacts. Benchmark datasets and reference media are **external inputs** and are not installed by `pip install -e .`. Run each package's documented commands only after preparing its required data and model weights. The per-component outputs permit verification at the recorded sample/sequence level rather than relying solely on the summary table.

For source-level regression coverage the project also includes [`test_face_analysis.py`](tests/test_face_analysis.py), [`test_face_features.py`](tests/test_face_features.py), [`test_face_tracking_landmarks.py`](tests/test_face_tracking_landmarks.py), [`test_face_heavy_modules.py`](tests/test_face_heavy_modules.py), [`test_face_temporal_export.py`](tests/test_face_temporal_export.py), and [`test_gaze_estimation.py`](tests/test_gaze_estimation.py). With the test dependencies installed, these are executed through the existing `pytest` workflow. Dedicated validation regression scripts also cover landmark/pose association, region-map overlap and blink availability. A successful test or pipeline execution is evidence of software correctness on the exercised cases, **not** a substitute for external accuracy benchmarks.

**Media and redistribution:** Some benchmark-derived and externally sourced images/videos are intentionally excluded from Git. This does **not** exclude the independently generated numerical results, CSV/JSON files, scripts, statistical tables or quantitative plots. Synthetic demonstrations complement but are kept separate from benchmark evidence; their detailed input inventory, source attributions, licensing decisions and publication status are recorded in [`VALIDATION_MEDIA_LICENSES.md`](VALIDATION_MEDIA_LICENSES.md). This README intentionally does not duplicate that media inventory.

---

## Scope & Limitations

**PhysioTrack is research software. It is not a medical device**, has not been evaluated by
any regulatory body, and must not be used for diagnosis, screening, triage, or treatment
decisions. Read the following before relying on any number it produces.

**Input.** Monocular **RGB** video only. There is no RGB-D, infrared, or thermal capture
path; depth is *estimated* from colour frames, not sensed.

**Heart rate (rPPG).** Requires adequate, stable lighting, visible facial skin, and limited
subject motion. Accuracy degrades with motion, illumination change, video compression, and
low frame rate, and rPPG is known to be sensitive to skin tone. Always interpret a heart
rate alongside its `snr` — a value is reported whenever the analysis window is full,
regardless of signal quality.

**HRV is not validated.** Heart-rate variability is far more demanding than heart rate: the
heart rate is a spectral peak and tolerates a mis-detected beat, whereas every HRV index is
built from the intervals *between* beats, so one missed or spurious beat distorts it. The
frequency-domain indices need **at least 60 s** of clean pulse. PhysioTrack computes HRV and
warns when the beat-derived rate disagrees with the spectral rate, but **we make no accuracy
claim for HRV** — treat it as exploratory until validated against a contact reference on your
own data.

**Depth is relative, not metric.** Larger means nearer, on an arbitrary scale that is not
comparable between frames. There is no camera calibration or metric scale recovery.

**3D pose is monocular-lifted, not triangulated**, runs offline from a saved 2D-pose file
rather than streaming, and is single-subject: in a multi-person frame only the first subject
receives 3D keypoints.

**Kinematics are image-plane measurements.** Joint angles and range of motion come from pixel
coordinates and velocities are in pixels per second, so they are **not comparable across
subjects, camera distances, or resolutions**. Range of motion covers two geometrically
distinct measurements per hip (one per anatomical plane), not four independent clinical
movements — a single image-plane angle cannot separate flexion from extension without a
signed convention.

**Multi-person analysis needs a tracker.** Without one, instance ids are per-frame indices,
so any cross-frame quantity (velocity, acceleration) would mix subjects together. Attach a
`Tracker` to get persistent ids.

**Privacy.** The library performs no anonymization. It processes identifiable faces and
derives physiological signals from them; consent, retention, and data protection are the
responsibility of the deploying party.

---

## Project Layout

```
src/physiotrack/
├── capture/        # Video orchestrator (end-to-end pipeline)
├── detect/         # Detection
├── pose/           # Pose 2D, Pose3D, canonicalizer, evaluation
├── segment/        # Segmentation
├── depth/          # Depth estimation
├── face/           # Face detection + orientation
├── trackers/       # OC-SORT, ByteTrack, StrongSORT, BoostTrack
├── signals/        # rPPG (POS/CHROM/LGI/OMIT) + RR intervals/HRV/respiration, motion (joint angles, ROM), filters, plotting
├── core/           # inference loop, radar/floor view, ego view, depth view
├── modules/        # Neural backends (ViTPose, Sapiens, YOLO, DepthAnythingV2, ZipDepth,
│                   #   MotionBERT, DDHPose, 3DCPNet, 6DRepNet360, SegFace)
├── results.py      # Unified Result / DepthResult / TrackResult / Instance / Keypoints
└── models.py       # Models registry + Hugging Face auto-download
```

See [`examples/`](examples/) for runnable scripts covering each subsystem, and
[`docs/paper/API_REDESIGN.md`](https://github.com/tharindu326/physiotrack/blob/main/docs/paper/API_REDESIGN.md)
for the full public-API specification.

---

## Citations

If you use Physiotrack in your research, please cite the relevant papers:

```bibtex
@inproceedings{ekanayake2025evaluating,
  title={Evaluating the Accuracy and Reliability of Camera-Based Physiological and Motion Signal Extraction Techniques in Virtual Reality Training Environments},
  author={Ekanayake, Tharindu and {\'A}lvarez Casado, Constantino and Nguyen, Nhi and Sobocinski, Marta and Pramila-Savukoski, Sari and Wu, Xiaoting and Mikkonen, Kristina and Bordallo L{\'o}pez, Miguel},
  booktitle={Scandinavian Conference on Image Analysis},
  pages={442--456},
  year={2025},
  organization={Springer}
}

@inproceedings{ekanayake20263dpcnet,
  title={3DPCNet: Pose Canonicalization for Robust Viewpoint-Invariant 3D Kinematic Analysis from Monocular RGB Cameras},
  author={Ekanayake, Tharindu and Casado, Constantino {\'A}lvarez and L{\'o}pez, Miguel Bordallo},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={11007--11011},
  year={2026},
  organization={IEEE}
}

@article{casado2023face2ppg,
  title={Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces},
  author={Casado, Constantino Alvarez and L{\'o}pez, Miguel Bordallo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={27},
  number={11},
  pages={5530--5541},
  year={2023},
  publisher={IEEE}
}

@inproceedings{nguyen2025comparative,
  title={Comparative Analysis of rPPG and Motion-Based Approaches for Heart and Respiration Rate Estimation from Videos},
  author={Nguyen, Nhi and {\'A}lvarez Casado, Constantino and Nguyen, Le and Lage Ca{\~n}ellas, Manuel and Bordallo L{\'o}pez, Miguel},
  booktitle={Scandinavian Conference on Image Analysis},
  pages={32--46},
  year={2025},
  organization={Springer}
}
```

---

## Contributing

Physiotrack is an open-source project and welcomes contributions: new models, documentation,
bug fixes, or examples. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which covers the
development setup, the Conventional Commits our releases are generated from, the
Google-style docstring convention the documentation is built from, and the code standards.
Then open an issue or pull request on
[GitHub](https://github.com/tharindu326/physiotrack).

<div align="center">

[![Physiotrack contributors](https://contrib.rocks/image?repo=tharindu326/physiotrack)](https://github.com/tharindu326/physiotrack/graphs/contributors)

</div>

Every avatar is someone who has contributed code, documentation, or fixes — the grid updates
itself from the GitHub API. If your name or avatar looks wrong, a commit used a different git
identity; add yourself to [`.mailmap`](.mailmap) and it will be consolidated everywhere.

### Reproducing the published numbers

The scripts behind the figures and benchmarks live in
[`examples/evaluation/`](examples/evaluation/), so every reported value can be regenerated
rather than taken on trust:

```bash
python examples/evaluation/runtime_benchmark.py   # per-stage FPS / ms-per-frame, CPU and GPU
python examples/evaluation/rppg_figure.py         # rPPG figure + the vitals it reports
python examples/evaluation/pipeline_figure.py     # full-pipeline overlay figure
```

Each writes a JSON alongside its output recording the hardware, library version, and every
measured value, so a number can be traced to a specific run.

---

## License

Licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---

## Authors

**Developed by**
- M.Sc. Tharindu Ekanayake
- D.Sc. (Tech) Constantino Álvarez Casado *(PI)*

**Face-analysis thesis contribution**

- **M.Sc Mahdi Muhannad** — modular face-analysis pipeline development, component validation, integration, reproducibility evaluation, runtime characterization, and controlled robustness assessment.

**Affiliation**
Multimodal Sensing Lab (MMSLab) · Center for Machine Vision and Signal Processing (CMVS) ·
University of Oulu, Finland

### Acknowledgments

This research was supported by the University of Oulu and the Research Council of Finland
(former Academy of Finland) through the 6G Flagship Programme (Grant No. 346208), the Profi5 HiDyn
programme (326291), and the Profi7 Hybrid Intelligence programme (352788). The authors acknowledge
CSC (IT Center for Science, Finland) for computational resources.

---

<div align="center">

**Repository:** [github.com/tharindu326/physiotrack](https://github.com/tharindu326/physiotrack)

</div>
