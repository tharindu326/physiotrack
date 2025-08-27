# Physiotrack
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/release/python-380/)
![Static Badge](https://img.shields.io/badge/status%20-%20under%20construction%20-%20%23FF0000)


**Physiotrack** is a Python library for detection and pose estimation using a variety of state-of-the-art models like YOLO, RT-DETR, Sapiens, and ViTPose. The framework supports modular detection and pose pipelines with validated models for specialized tasks like VR student monitoring, face detection, and whole-body pose estimation.

---

## Features

- **Object Detection:**
  - YOLOv11 (person, face, VR, VRStudent)
  - RT-DETR (person, VRStudent)

- **Pose Estimation:**
  - YOLO-Pose
  - ViTPose (COCO & WholeBody)
  - Sapiens (COCO WholeBody)

---

## Installation

```bash
pip install -e .
```

> Note: Ensure PyTorch is installed for your system with CUDA if using GPU.

---

## Model Overview

### Detection Models

```python
from physiotrack import Models

# Accessing YOLO face detection model
face_model = Models.Detection.YOLO.FACE.m_face

# Accessing RT-DETR person model
rtdetr_model = Models.Detection.RLDETR.PERSON.x_person
```

### Pose Estimation Models

```python
# ViTPose WholeBody model
vitpose_model = Models.Pose.ViTPose.WholeBody.b_WHOLEBODY

# Sapiens Pose model
sapiens_model = Models.Pose.Sapiens.WholeBody.B1_TS_COCOHB
```

---

## Detection Usage

### Built-in Models

```python
from physiotrack import Detection

# Default Person Detector
detector = Detection.Person()
results, frame = detector.detect(image)
```

### Custom Model

```python
from physiotrack import Detection, Models

custom_model = Models.Detection.YOLO.VR.m_VR
custom_detector = Detection.Custom(model=custom_model)
results, frame = custom_detector.detect(image)
```

---

## Pose Estimation Usage

### Built-in VRStudent Pose Estimator

```python
from physiotrack import Pose

pose_estimator = Pose.VRStudent()
pose_image, data = pose_estimator.estimate(image)
```

### Using a Custom Pose Model

```python
from physiotrack import Pose, Models

model = Models.Pose.ViTPose.WholeBody.l_WHOLEBODY
pose_estimator = Pose.Custom(model=model)
pose_image, data = pose_estimator.estimate(image)
```

---

## Auto-Detection Integration for Pose Estimation

If no bounding boxes are provided, ViTPose and Sapiens estimators will auto-detect people using the default `Detection.Person()` detector:

```python
pose_image, data = pose_estimator.estimate(image)  # auto-detects person boxes
```

---

## Key Concepts

- `ValidatedDetector`: Ensures model matches the intended detection class.
- `PoseBase`: Abstracts pose estimation across frameworks (ViTPose, YOLO-Pose, Sapiens).
- All classes support `device`, `confidence`, `NMS`, and rendering options.

---

## Supported Pose Estimators

| Framework | Variants        | Keypoints      |
|----------|----------------|----------------|
| YOLO-Pose | COCO           | 17             |
| ViTPose   | COCO, WholeBody| Up to 133      |
| Sapiens   | WholeBody      | 133            |

---

## Notes

- Sapiens models are TorchScript `.pt2`
- ViTPose models are `.pth` and need compatible config
- Detection and Pose classes can be extended for new models

---
