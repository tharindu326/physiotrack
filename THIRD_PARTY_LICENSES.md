# Third-party components and licensing

PhysioTrack itself is licensed **GPL-3.0-or-later** (see `LICENSE`). It also
**vendors** upstream source code and **redistributes or fetches** third-party model
weights. Those carry their own terms, which are not superseded by this project's
licence.

Roughly 69% of `src/physiotrack` (about 35,600 of 51,300 lines, across `modules/`
and `trackers/`) is vendored upstream code. Only one vendored component currently
ships its upstream licence text (`src/physiotrack/modules/SegFace/LICENSE`).

> **Status: incomplete.** Rows marked **VERIFY** have not been confirmed against the
> upstream licence and must be resolved before any redistribution, PyPI release, or
> journal submission that claims a GPL-3.0 licence for the whole work. Do not treat
> a blank or VERIFY row as permission.

## Runtime dependency requiring attention

| Component | Licence | Why it matters |
|---|---|---|
| `mediapipe`, `onnxruntime`, `safetensors` (optional `face` extra) | Apache-2.0 / MIT / Apache-2.0 | Permissive and GPL-3.0-compatible; installed only with `physiotrack[face]`. |
| `ultralytics` (YOLO/RT-DETR backends) | **AGPL-3.0** | AGPL-3.0 is *stronger* copyleft than GPL-3.0. A hard dependency on it constrains how PhysioTrack may be distributed and, in particular, how it may be offered as a network service. **Reconcile the project licence with this, or make the ultralytics-backed backends optional.** |

## Vendored source code (`src/physiotrack/modules/`)

| Component | Upstream | Licence | Notes |
|---|---|---|---|
| `DepthAnythingV2/dinov2*` | Meta Platforms (DINOv2) | **Apache-2.0** (confirmed) | Copyright and licence header present in `dinov2.py:1-4`. Upstream `LICENSE` file is *not* vendored — add it. |
| `SegFace` | Narayan et al., 2024 | **MIT** (confirmed) | `LICENSE` present in the directory; header at `inference.py:3-4`. |
| `Sapiens` | Meta Platforms | **VERIFY** | No licence or copyright text vendored at all. Sapiens weights and code have their own terms; confirm before redistributing. |
| `ViTPose` | ViTPose authors / `JunkyByte/easy_ViTPose` | **VERIFY** | No vendored licence text. |
| `MotionBERT` | Zhu et al. | **VERIFY** | `utils/utils_smpl.py:2` states the SMPL body-model terms must be adhered to separately — SMPL is **not** free for commercial use. |
| `DDHPose` | DDHPose authors | **VERIFY** | `common/` contains VideoPose3D-derived files; VideoPose3D is CC-BY-NC-4.0 (non-commercial) upstream — confirm. |
| `_3DCPNet` | This project (ICASSP 2026) | Project licence | First-party. |
| `_6DRepNet360` | 6DRepNet authors | **VERIFY** | No vendored licence text. |
| `ZipDepth` | ZipDepth authors | **VERIFY** | No vendored licence text. |
| `EmotiEffLib` | Savchenko / `sb-ai-lab/EmotiEffLib` | **Apache-2.0** (confirmed) | Re-implements the library's ONNX preprocessing and class order; no upstream code copied verbatim. `LICENSE` vendored in the directory. |
| `Gaze` | hysts / `pytorch_mpiigaze_demo` (ptgaze) | **MIT** (confirmed) | `face_model.py` (3D face template) and the network definitions in `networks.py` are copied from ptgaze; `inference.py` follows its head-pose normalisation. `LICENSE` vendored in the directory. |
| `Yolo` (thin wrappers) | — | Project licence | Wrappers only; the model runtime is `ultralytics` (AGPL-3.0, above). |

## Vendored source code (`src/physiotrack/trackers/`)

| Component | Upstream | Licence | Notes |
|---|---|---|---|
| `ocsort/kalmanfilter.py` | Roger R. Labbe Jr. (`filterpy`) | **MIT** (confirmed) | Header at `kalmanfilter.py:92-94`. |
| `ocsort`, `boosttrack` | Derived from SORT (Alex Bewley) | **VERIFY** | Both state "adopted from the SORT script by Alex Bewley". SORT is GPL-3.0 upstream, which is compatible, but attribution should be explicit. |
| `bytetrack` | ByteTrack authors | **VERIFY** | MIT upstream; confirm and vendor the text. |
| `strongsort` | StrongSORT authors | **VERIFY** | Includes a bundled OSNet ReID model factory. |

## Data tables

| Component | Upstream | Licence | Notes |
|---|---|---|---|
| `pose/config.py` `FACEMESH_CONTOURS` | Google (MediaPipe `FaceLandmarksConnections`) | **Apache-2.0** (confirmed) | Landmark-id pairs for drawing the face mesh. |

## Model weights

Weights are **not** bundled in the wheel; they are downloaded on first use. Note
that *re-hosting* is itself the licence-sensitive act:

| Source | Notes |
|---|---|
| `tharindu326/physiotrack` (Hugging Face) | Project-hosted. Includes **re-hosted Depth-Anything-V2 weights**, i.e. this project redistributes them — the upstream licence must permit that. |
| `facebook/sapiens-*`, `noahcao/sapiens-pose-coco` | Upstream Sapiens weights. **VERIFY** terms. |
| `JunkyByte/easy_ViTPose` | Upstream ViTPose weights. **VERIFY**. |
| `walterzhu/MotionBERT` | Upstream MotionBERT weights. **VERIFY**; see SMPL note above. |
| Ultralytics (COCO YOLO) | Fetched by `ultralytics` itself; AGPL-3.0 model terms apply. |
| MediaPipe Face Landmarker (`Models.Face.Landmarks`) | Google model storage, pinned `float16/1`, SHA-256 verified. **Apache-2.0** (MediaPipe model card). |
| EmotiEffLib ONNX models (`Models.Face.Expression`) | `sb-ai-lab/EmotiEffLib` at a pinned commit, SHA-256 verified. Repository Apache-2.0, but the models are trained on **AffectNet**, whose licence is **non-commercial research only**. |
| ptgaze gaze models (`Models.Face.Gaze`) | `hysts/ptgaze-*` at pinned revisions, SHA-256 verified. Repositories MIT, but the models are trained on **ETH-XGaze** (CC BY-NC-SA 4.0), **MPIIGaze** and **MPIIFaceGaze** (both CC BY-NC-SA 4.0): non-commercial. |

## Required actions

1. Reconcile the **ultralytics AGPL-3.0** dependency with the project's GPL-3.0 claim.
2. Vendor the upstream `LICENSE` text for every component listed above.
3. Confirm redistribution rights for the **re-hosted Depth-Anything-V2** weights.
4. Resolve the **SMPL**, **VideoPose3D**, **AffectNet** (expression weights) and
   **ETH-XGaze / MPIIGaze / MPIIFaceGaze** (gaze weights) non-commercial terms, which are incompatible with an
   unqualified "free for any use" reading of GPL-3.0.
5. Add a per-checkpoint licence column to `docs/model-zoo.md`.
6. Add the licensor copyright line that GPLv3 requires at the top of `LICENSE`.
