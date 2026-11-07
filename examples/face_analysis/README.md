# Face analysis examples

Two runnable scripts show the face stages of
[`physiotrack.face`](../../docs/guides/face.md) and the face signals of
[`physiotrack.signals`](../../docs/guides/signals.md). Both need the face extra:

```bash
pip install "physiotrack[face]"
```

## One image

```bash
python examples/face_analysis/analyze_image.py
python examples/face_analysis/analyze_image.py --input path/to/photo.jpg --device cuda
```

Every face stage takes the faces of the previous step and adds one property:

```python
faces = pt.Face()(image)
faces = pt.FaceOrientation()(image, faces)   # head yaw / pitch / roll
faces = pt.FaceLandmarks()(image, faces)     # 478-point face mesh
faces = pt.FaceExpression()(image, faces)    # expression category
faces = pt.GazeEstimator()(image, faces)     # 3D gaze (needs the mesh)
faces = pt.FaceQuality()(image, faces)       # brightness, sharpness, size
```

The script prints each face's head pose, expression, image quality, eye aspect ratio,
mouth aspect ratio, iris position and gaze, and saves the annotated image and the
`Result` JSON (with the face meshes) to `results/`.

## A video

```bash
python examples/face_analysis/analyze_video.py
python examples/face_analysis/analyze_video.py --input path/to/clip.mp4 --device cuda --gaze
```

`Video` detects and tracks the faces and runs the stages on every frame
(`face_stages=[...]`). The script then uses the face signals on the returned
`VideoResults`:

| Function | Output |
| --- | --- |
| `face_feature_sequence(results)` | one row per face per frame: box, head pose, EAR, MAR, iris, expression, gaze, quality |
| `face_window_summary(results, window=5.0)` | per face per frame: mean / std / min / max over the last 5 s, blinks in the window, dominant expression |
| `detect_blinks(results, detection_id=i)` | blink events with start, end, duration |
| `blink_rate(results, detection_id=i)` | blinks per minute over the time the face was in view |
| `mouth_movement(results, detection_id=i)` | mouth aspect ratio, its frame-to-frame change and speed |

Outputs in `results/`: the annotated video, the per-frame JSON, the per-face CSV and
the 5-second window summaries.

Track ids are temporary associations within one video, not identities. Faces that are
small (roughly under 50 px) or strongly turned often get no face mesh, and so no eye or
mouth measures. The bundled clip is synthetic and has no ground truth, so the printed
numbers are a demonstration, not a validation; see the
[validation guide](../../docs/guides/face-validation.md).
