# Model Zoo

Physiotrack ships a single, discoverable registry of every pretrained checkpoint
it can download and run — [`Models`][physiotrack.Models]. This page lists every
registered model, grouped by task, with the exact access path, weight filename,
and where the weights come from.

## The registry addressing scheme

Every checkpoint is addressed by a four-level path:

```text
Models.<Task>.<Backend>.<Enum>.<member>
```

- **Task** — what the model does: `Detection`, `Pose`, `Pose3D`, `Depth`,
  `Segmentation`, `Face`.
- **Backend** — the architecture/family: `YOLO`, `RTDETR`, `Sapiens`, `ViTPose`,
  `MotionBERT`, `DDH`, `Canonicalizer`, `DepthAnythingV2`, `ZipDepth`, `SegFace`; for
  `Face`, the stage (`Orientation`, `Landmarks`, `Expression`, `Gaze`).
- **Enum** — a group of interchangeable checkpoints, usually by dataset or size
  (e.g. `Detection.YOLO.PERSON`, `Pose.ViTPose.WholeBody`).
- **member** — one checkpoint. Its `.value` is the **weight filename** on disk;
  its `.name` is the short handle.

!!! info "`.value` is the weight filename"

    ```python
    from physiotrack import Models

    m = Models.Pose.ViTPose.WholeBody.s_wholebody
    print(m.name)    # 's_wholebody'
    print(m.value)   # 'vitpose-s-wholebody.pth'
    ```

A few groups differ from the strict four-level shape: the `Pose3D` backends
(`MotionBERT`, `DDH`) are `Enum`s **directly** under `Pose3D`;
`Pose3D.Canonicalizer` holds a nested `Models` enum (3DPCNet weights) plus a
`View` string enum; and the `Depth` backends (`DepthAnythingV2`, `ZipDepth`) and the
`Face` stage groups are `Enum`s directly under their task.

### Downloading weights

Selecting a member downloads nothing. Pass it to
[`Models.resolve`][physiotrack.Models.resolve] to get its local path, fetching it
only if it is not already cached (the high-level predictors do this for you):

```python
from physiotrack import Models

path = Models.resolve(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
```

`resolve` is the only place in the library that knows where weights live, so every
predictor and backend loader agrees on one location. Use
[`download_model`][physiotrack.Models.download_model] directly only to force a fetch
into a specific directory.

### Where weights are cached

Checkpoints are cached **outside** the installed package. Writing multi-gigabyte files
into `site-packages` breaks read-only and containerised installs, defeats Docker layer
caching, and stops two environments sharing one download.

| Condition | Cache location |
|---|---|
| `$PHYSIOTRACK_HOME` set | `$PHYSIOTRACK_HOME/weights` |
| `$XDG_CACHE_HOME` set | `$XDG_CACHE_HOME/physiotrack/weights` |
| Linux (default) | `~/.cache/physiotrack/weights` |
| macOS | `~/Library/Caches/physiotrack/weights` |
| Windows | `%LOCALAPPDATA%\physiotrack\weights` |

Set `PHYSIOTRACK_HOME` to share one cache between environments, images, or CI jobs:

```bash
export PHYSIOTRACK_HOME=/shared/physiotrack
```

!!! note "Upgrading from before 1.1"

    Older releases downloaded into the package's `modules/model_data` directory.
    [`migrate_weight_cache`][physiotrack.migrate_weight_cache] moves existing
    checkpoints into the cache so they are not fetched again:

    ```python
    import physiotrack
    physiotrack.migrate_weight_cache(dry_run=True)   # preview
    physiotrack.migrate_weight_cache()               # move them
    ```

Where a checkpoint comes from depends on the backend:

| Source | Which models | Notes |
|---|---|---|
| **Ultralytics** (auto, on demand) | All `Pose.YOLO`; every `PERSON` YOLO/RT-DETR variant (`Detection.YOLO.PERSON`, `Detection.RTDETR.PERSON`, `Segmentation.YOLO.PERSON`) | `download_model` returns `None` — ultralytics fetches stock weights itself |
| **`tharindu326/physiotrack`** (Hugging Face) | Detection FACE / VRFACE / VR / VRSTUDENT, RT-DETR VRSTUDENT, Segmentation VRHEAD, DDH, Face Orientation, Canonicalizer (3DPCNet), DepthAnythingV2, ZipDepth, SegFace | Project-hosted checkpoints |
| **Upstream publishers, pinned + SHA-256** | Face Landmarks → MediaPipe model storage; Face Expression → `sb-ai-lab/EmotiEffLib`; Face Gaze → `hysts/ptgaze-*` | Fetched at a fixed revision and refused if the checksum differs |
| **Upstream Hugging Face repos** | `Sapiens` pose → `noahcao/sapiens-pose-coco`; `Sapiens` seg → `facebook/sapiens-seg-{size}-torchscript`; `ViTPose` → `JunkyByte/easy_ViTPose`; `MotionBERT` → `walterzhu/MotionBERT` | Fetched from the original authors' repos |

!!! note "Weight-free members"

    `Canonicalizer.Models.GEOMETRIC` has an empty `.value` — it selects the
    algorithmic (weight-free) canonicalization path and is never downloaded.

## How to use a model

Hand a registry member to a predictor. The two most common patterns:

```python
import physiotrack as pt

# Detection — a specific person-detector variant
det = pt.Detection.Custom(pt.Models.Detection.YOLO.PERSON.l_person)
result = det.predict(frame)

# Pose — the huge whole-body ViTPose checkpoint
pose = pt.Pose.Custom(pt.Models.Pose.ViTPose.WholeBody.h_wholebody)
result = pose.predict(frame)
```

!!! tip "Presets vs. `Custom`"

    Every task exposes named presets (e.g. [`Detection.Person`][physiotrack.Detection],
    [`Pose.Person`][physiotrack.Pose]) that pin a sensible default model, plus a
    `Custom` preset that takes any validated registry member for that task. Passing
    a member from the wrong task raises a descriptive `ValueError` via the registry's
    `validate_*` guards.

---

## Detection

14 checkpoints across YOLO and RT-DETR. See the [Detection guide](guides/detection.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Detection.YOLO.PERSON.n_person` | YOLO | `yolo11n.pt` | YOLO11-n person (stock, ultralytics) |
| `Models.Detection.YOLO.PERSON.m_person` | YOLO | `yolo11m.pt` | YOLO11-m person (stock, ultralytics) |
| `Models.Detection.YOLO.PERSON.l_person` | YOLO | `yolo11l.pt` | YOLO11-l person (stock, ultralytics) |
| `Models.Detection.YOLO.FACE.n_face` | YOLO | `yolov11n-face.pt` | YOLO11-n face |
| `Models.Detection.YOLO.FACE.m_face` | YOLO | `yolov11m-face.pt` | YOLO11-m face |
| `Models.Detection.YOLO.FACE.l_face` | YOLO | `yolov11l-face.pt` | YOLO11-l face |
| `Models.Detection.YOLO.VRFACE.l_vrface` | YOLO | `yolov12l-face.pt` | YOLO12-l VR face |
| `Models.Detection.YOLO.VR.m_vr` | YOLO | `yolo11m_VR_head.pt` | Only published VR-head size; masks are discarded by the detection API |
| `Models.Detection.YOLO.VRSTUDENT.m_vrstudent` | YOLO | `yolo11m_VRstudent.pt` | YOLO11-m VR-student |
| `Models.Detection.YOLO.VRSTUDENT.l_vrstudent` | YOLO | `yolo11l_VRstudent.pt` | YOLO11-l VR-student |
| `Models.Detection.RTDETR.PERSON.l_person` | RT-DETR | `rtdetr-l.pt` | RT-DETR-l person (stock, ultralytics) |
| `Models.Detection.RTDETR.PERSON.x_person` | RT-DETR | `rtdetr-x.pt` | RT-DETR-x person (stock, ultralytics) |
| `Models.Detection.RTDETR.VRSTUDENT.l_person` | RT-DETR | `yolo11l_RLDETR_VRstudent.pt` | RT-DETR-l VR-student |
| `Models.Detection.RTDETR.VRSTUDENT.x_person` | RT-DETR | `yolo11x_RLDETR_VRstudent.pt` | RT-DETR-x VR-student |

!!! warning "RT-DETR VR-student member names"

    Under `RTDETR.VRSTUDENT`, the members are named `l_person` / `x_person` even
    though the weights are VR-student RT-DETR models. Use the member name as
    written above; the weight filename disambiguates them.

---

## Pose (2D)

13 checkpoints across YOLO-Pose, Sapiens, and ViTPose. See the [Pose guide](guides/pose.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Pose.YOLO.COCO.M11` | YOLO-Pose | `yolo11m-pose.pt` | YOLO11-m, COCO 17-kp (stock, ultralytics) |
| `Models.Pose.YOLO.COCO.L11` | YOLO-Pose | `yolo11l-pose.pt` | YOLO11-l, COCO 17-kp (stock, ultralytics) |
| `Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB` | Sapiens | `sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2` | Sapiens 0.3B, COCO-WholeBody, AP 62.0 |
| `Models.Pose.Sapiens.WholeBody.B06_TS_COCOHB` | Sapiens | `sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2` | Sapiens 0.6B, COCO-WholeBody, AP 69.5 |
| `Models.Pose.Sapiens.WholeBody.B1_TS_COCOHB` | Sapiens | `sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2` | Sapiens 1B, COCO-WholeBody, AP 72.7 |
| `Models.Pose.ViTPose.WholeBody.s_wholebody` | ViTPose | `vitpose-s-wholebody.pth` | ViT-S, whole-body (133+ kp) |
| `Models.Pose.ViTPose.WholeBody.b_wholebody` | ViTPose | `vitpose-b-wholebody.pth` | ViT-B, whole-body (133+ kp) |
| `Models.Pose.ViTPose.WholeBody.l_wholebody` | ViTPose | `vitpose-l-wholebody.pth` | ViT-L, whole-body (133+ kp) |
| `Models.Pose.ViTPose.WholeBody.h_wholebody` | ViTPose | `vitpose-h-wholebody.pth` | ViT-H, whole-body (133+ kp) |
| `Models.Pose.ViTPose.COCO.s_coco` | ViTPose | `vitpose-s-coco.pth` | ViT-S, COCO 17-kp |
| `Models.Pose.ViTPose.COCO.b_coco` | ViTPose | `vitpose-b-coco.pth` | ViT-B, COCO 17-kp |
| `Models.Pose.ViTPose.COCO.l_coco` | ViTPose | `vitpose-l-coco.pth` | ViT-L, COCO 17-kp |
| `Models.Pose.ViTPose.COCO.h_coco` | ViTPose | `vitpose-h-coco.pth` | ViT-H, COCO 17-kp |

!!! note "Top-down backends need person boxes"

    ViTPose and Sapiens are top-down: [`Pose`][physiotrack.Pose] runs a person
    detector first (or accepts boxes you pass). YOLO-Pose is single-stage. AP
    figures above are the upstream Sapiens COCO-WholeBody scores encoded in the
    filenames.

---

## Pose3D

3D lifting and pose canonicalization models. See the
[Pose3D guide](guides/pose3d.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite` | MotionBERT | `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` | Lite, fine-tuned on Human3.6M (global) |
| `Models.Pose3D.MotionBERT.mb_ft_h36m` | MotionBERT | `FT_MB_release_MB_ft_h36m/best_epoch.bin` | Release, fine-tuned on Human3.6M |
| `Models.Pose3D.MotionBERT.mb_train_h36m` | MotionBERT | `MB_train_h36m/best_epoch.bin` | Trained from scratch on Human3.6M |
| `Models.Pose3D.DDH.best` | DDH | `best_epoch_DDHPose.bin` | DDHPose 3D lifter |

### Canonicalizer (3DPCNet)

`Pose3D.Canonicalizer.Models` holds the pose-canonicalization checkpoints. See
[`PoseCanonicalizer`][physiotrack.PoseCanonicalizer] and the
[pose post-processing API](api/pose-postprocessing.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Pose3D.Canonicalizer.Models.GEOMETRIC` | Canonicalizer | _(empty)_ | Weight-free algorithmic canonicalization (no download) |
| `Models.Pose3D.Canonicalizer.Models._3DPCNetS2` | Canonicalizer | `best_model_3DPCNetS2.pth` | 3DPCNet, MMFi split S2 |
| `Models.Pose3D.Canonicalizer.Models._3DPCNetS3` | Canonicalizer | `best_model_3DPCNetS3.pth` | 3DPCNet, MMFi split S3 |
| `Models.Pose3D.Canonicalizer.Models._3DPCNetTC48_byCam` | Canonicalizer | `best_model_3DPCNetTC48_byCam.pth` | 3DPCNet, TotalCapture 48-cam, camera-disjoint split |
| `Models.Pose3D.Canonicalizer.Models._3DPCNetTC48_byAction` | Canonicalizer | `best_model_3DPCNetTC48_byAction.pth` | 3DPCNet, TotalCapture 48-cam, action-disjoint split |

`pt.CanonicalView` is a plain string enum of canonical viewpoints (not
weights): `FRONT` (`"front"`), `BACK` (`"back"`), `LEFT_SIDE` (`"left_side"`),
`RIGHT_SIDE` (`"right_side"`).

---

## Depth

Monocular depth via Depth Anything V2 (heavier, ViT-based) or ZipDepth
(lightweight, ~6M params). Both return a **relative** (affine-invariant) depth
map. See the [Depth guide](guides/depth.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Depth.DepthAnythingV2.vits` | DepthAnythingV2 | `depth_anything_v2_vits.pth` | ViT-S (small) encoder |
| `Models.Depth.DepthAnythingV2.vitb` | DepthAnythingV2 | `depth_anything_v2_vitb.pth` | ViT-B (base) encoder |
| `Models.Depth.DepthAnythingV2.vitl` | DepthAnythingV2 | `depth_anything_v2_vitl.pth` | ViT-L (large) encoder |
| `Models.Depth.ZipDepth.base` | ZipDepth | `zipdepth_base.pth` | Lightweight (~6M params), GPU/server upsampling head |
| `Models.Depth.ZipDepth.npu` | ZipDepth | `zipdepth_base_npu.pth` | Same weights, NPU/CPU/mobile-friendly upsampling head |

### Encoder configurations

`Models.Depth.MODEL_CONFIGS` (via
[`get_depth_config`][physiotrack.Models.get_depth_config]) supplies the
architecture parameters used to build each DepthAnythingV2 encoder:

| Encoder | `features` | `out_channels` |
|---|---|---|
| `vits` | 64 | `[48, 96, 192, 384]` |
| `vitb` | 128 | `[96, 192, 384, 768]` |
| `vitl` | 256 | `[256, 512, 1024, 1024]` |

`Models.Depth.ZIPDEPTH_CONFIGS` (also via
[`get_depth_config`][physiotrack.Models.get_depth_config]) supplies the ZipDepth
build settings. Both variants share the `base` architecture and differ only in
the upsampling head (`upsample_unfold`):

| Variant | `variant` | `global_mode` | `upsample_unfold` | `input_size` |
|---|---|---|---|---|
| `base` | `base` | `balanced` | `True` | 384 |
| `npu` | `base` | `balanced` | `False` | 384 |

---

## Segmentation

8 checkpoints across Sapiens body-part, YOLO, and SegFace. See the
[Segmentation guide](guides/segmentation.md).

| Access path | Backend | Weight file | Notes |
|---|---|---|---|
| `Models.Segmentation.Sapiens.BodyPart.B03_TS_SEG` | Sapiens | `sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2` | Sapiens 0.3B Goliath, mIoU 76.73 (epoch 194) |
| `Models.Segmentation.Sapiens.BodyPart.B06_TS_SEG` | Sapiens | `sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2` | Sapiens 0.6B Goliath, mIoU 77.77 (epoch 178) |
| `Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG` | Sapiens | `sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2` | Sapiens 1B Goliath, mIoU 79.94 (epoch 151) |
| `Models.Segmentation.YOLO.VRHEAD.M11` | YOLO | `yolo11m_VR_head.pt` | YOLO11-m VR head segmentation |
| `Models.Segmentation.YOLO.VRHEAD.M8_251029` | YOLO | `yolo8m_VR_head_251029.pt` | YOLO8-m VR head segmentation (2025-10-29) |
| `Models.Segmentation.YOLO.PERSON.m_person` | YOLO | `yolo11m-seg.pt` | YOLO11-m person seg (stock, ultralytics) |
| `Models.Segmentation.YOLO.PERSON.l_person` | YOLO | `yolo11l-seg.pt` | YOLO11-l person seg (stock, ultralytics) |
| `Models.Segmentation.SegFace.Face.swinb_celeba_512` | SegFace | `segface_swinb_celeba_512.pt` | SegFace Swin-B @ 512, CelebAMask-HQ (19 classes) |

---

## Face

The per-face analysis stages. Face *detection* checkpoints are under
[Detection](#detection). See the [Face analysis guide](guides/face.md).

| Access path | Stage | Weight file | Notes |
|---|---|---|---|
| `Models.Face.Orientation.default` | `FaceOrientation` | `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` | 6DRepNet360 full-rotation, 300W-LP + Panoptic |
| `Models.Face.Orientation.VR` | `FaceOrientation` | `CMVS-FO-VR_epoch80.pth` | VR-tuned head orientation, epoch 80 |
| `Models.Face.Landmarks.face_landmarker` | `FaceLandmarks` | `face_landmarker.task` | MediaPipe Face Landmarker (float16 v1), 478 points; Apache-2.0 |
| `Models.Face.Expression.enet_b0_8_best_afew` | `FaceExpression` | `enet_b0_8_best_afew.onnx` | EfficientNet-B0, 8 AffectNet classes, fine-tuned on AFEW (default) |
| `Models.Face.Expression.enet_b0_8_best_vgaf` | `FaceExpression` | `enet_b0_8_best_vgaf.onnx` | EfficientNet-B0, 8 classes, fine-tuned on VGAF |
| `Models.Face.Expression.enet_b0_8_va_mtl` | `FaceExpression` | `enet_b0_8_va_mtl.onnx` | EfficientNet-B0, 8 classes plus valence and arousal |
| `Models.Face.Expression.enet_b2_8` | `FaceExpression` | `enet_b2_8.onnx` | EfficientNet-B2 (260 px), 8 classes |
| `Models.Face.Expression.enet_b2_7` | `FaceExpression` | `enet_b2_7.onnx` | EfficientNet-B2 (260 px), 7 classes (no Contempt) |
| `Models.Face.Gaze.eth_xgaze_resnet18` | `GazeEstimator` | `eth-xgaze_resnet18.safetensors` | Full face, ResNet-18 trained on ETH-XGaze (default) |
| `Models.Face.Gaze.mpiifacegaze_resnet_simple` | `GazeEstimator` | `mpiifacegaze_resnet_simple.safetensors` | Full face, trained on MPIIFaceGaze |
| `Models.Face.Gaze.mpiigaze_resnet_preact` | `GazeEstimator` | `mpiigaze_resnet_preact.safetensors` | Per eye, trained on MPIIGaze |

The expression weights (AffectNet) and the gaze weights (ETH-XGaze, MPIIGaze,
MPIIFaceGaze) are licensed for non-commercial research only; see
`THIRD_PARTY_LICENSES.md`.

`FaceLandmarks`, `FaceExpression` and `GazeEstimator` need the face extra
(`pip install "physiotrack[face]"`).

---

## Model formats

| Extension | Format | Used by |
|---|---|---|
| `.pt` | Ultralytics native (weights + built-in config) | YOLO / RT-DETR detection, pose, segmentation; SegFace |
| `.pth` | PyTorch state dict (loaded with a matching config) | ViTPose, DepthAnythingV2, ZipDepth, FaceOrientation, 3DPCNet |
| `.bin` | PyTorch checkpoint | MotionBERT, DDH |
| `.pt2` | TorchScript (optimized inference) | Sapiens pose and segmentation |

---

## See also

- [`Models` API reference](api/models.md) — full docstrings for the registry and
  its `download_model` / `validate_*` / `get_depth_config` helpers.
- Task guides: [Detection](guides/detection.md) ·
  [Pose](guides/pose.md) · [Pose3D](guides/pose3d.md) ·
  [Depth](guides/depth.md) · [Segmentation](guides/segmentation.md) ·
  [Face](guides/face.md)
