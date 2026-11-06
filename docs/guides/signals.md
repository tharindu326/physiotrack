# Signals

The `physiotrack.signals` subsystem turns pose, segmentation and video output into
**interpretable human-state signals**: a contactless heart rate from facial skin (rPPG),
keypoint motion features, anatomical joint angles and clinical range-of-motion (ROM), plus
the supporting DSP — filters, normalizers and signal-agreement metrics.

Everything is **plotter-free at the core**: the measurement functions return plain numbers,
NumPy arrays or DataFrames, and the live [`RPPGPlotter`][physiotrack.signals.RPPGPlotter],
[`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter] and
[`JointAnglePlotter`][physiotrack.signals.JointAnglePlotter] overlays are thin, optional
wrappers you can drop onto any video frame.

Import from the subsystem namespace:

```python
from physiotrack.signals import (
    POS, HeartRateEstimator, FaceSkinExtractor, bvp_to_hr,   # rPPG
    joint_angles, compute_rom_angles, extract_keypoints_sequence,  # motion / ROM
    bandpass_filter, z_score_normalize, compute_plv,          # DSP / metrics
)
```

## What the subsystem provides

| Area | Key API | Output | Guide section |
| --- | --- | --- | --- |
| rPPG extraction | [`POS`][physiotrack.signals.POS], `CHROM`, `LGI`, `OMIT` | BVP from an RGB skin trace `(3, N)` | [rPPG / Heart Rate](#rppg-heart-rate) |
| Skin ROI | [`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor] | SegFace skin mask + canvases | [rPPG / Heart Rate](#rppg-heart-rate) |
| HR / SNR | [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator], [`bvp_to_hr`][physiotrack.signals.bvp_to_hr], [`bvp_snr`][physiotrack.signals.bvp_snr] | bpm, dB | [rPPG / Heart Rate](#rppg-heart-rate) |
| HR / rPPG overlays | [`RPPGPlotter`][physiotrack.signals.RPPGPlotter], [`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter] | on-frame panels | [rPPG / Heart Rate](#rppg-heart-rate) |
| RR intervals | [`bvp_to_rri`][physiotrack.signals.bvp_to_rri], [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts] | inter-beat intervals (ms) | [Heart-Rate Variability](#heart-rate-variability-hrv) |
| HRV | [`compute_hrv`][physiotrack.signals.compute_hrv], [`hrv_time`][physiotrack.signals.hrv_time], `hrv_frequency`, `hrv_nonlinear` | RMSSD/SDNN/pNN50/SD1/SD2/LF-HF | [Heart-Rate Variability](#heart-rate-variability-hrv) |
| Respiration | [`respiration_from_pulse`][physiotrack.signals.respiration_from_pulse], [`respiration_from_motion`][physiotrack.signals.respiration_from_motion] | breaths/min | [Respiration](#respiration) |
| HRV / respiration overlays | [`HRVPlotter`][physiotrack.signals.HRVPlotter], [`RespirationPlotter`][physiotrack.signals.RespirationPlotter] | on-frame panels | [HRV](#heart-rate-variability-hrv) · [Respiration](#respiration) |
| Keypoint sequences | [`extract_keypoints_sequence`][physiotrack.signals.extract_keypoints_sequence], centroids | pandas DataFrames | [Motion, Joint Angles & ROM](#motion-features-joint-angles-rom) |
| Motion features | [`compute_all_motion_features`][physiotrack.signals.compute_all_motion_features] | velocity / accel / angles | [Motion, Joint Angles & ROM](#motion-features-joint-angles-rom) |
| Joint angles | [`joint_angles`][physiotrack.signals.joint_angles], [`compute_all_joint_angles`][physiotrack.signals.compute_all_joint_angles] | angles (see the [units note](#warning-degrees-vs-radians)) | [Motion, Joint Angles & ROM](#motion-features-joint-angles-rom) |
| Clinical ROM | [`compute_rom_angles`][physiotrack.signals.compute_rom_angles], [`JointAnglePlotter`][physiotrack.signals.JointAnglePlotter] | degrees + overlay | [Motion, Joint Angles & ROM](#motion-features-joint-angles-rom) |
| Face geometry | [`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio], [`mouth_aspect_ratio`][physiotrack.signals.mouth_aspect_ratio], [`iris_position`][physiotrack.signals.iris_position] | ratios per frame | [Face signals](#face-signals) |
| Face over time | [`detect_blinks`][physiotrack.signals.detect_blinks], [`blink_rate`][physiotrack.signals.blink_rate], [`mouth_movement`][physiotrack.signals.mouth_movement], [`face_feature_sequence`][physiotrack.signals.face_feature_sequence], [`face_window_summary`][physiotrack.signals.face_window_summary] | events, rates, DataFrames | [Face signals](#face-signals) |
| Filters | `bandpass_filter`, `highpass_filter`, `notch_filter`, … | filtered arrays | [Filters & Normalization](#filters-normalization) |
| Normalization | `z_score_normalize`, `min_max_normalize`, … | scaled series | [Filters & Normalization](#filters-normalization) |
| Signal metrics | `compute_plv`, `calculate_pearson_correlation`, `compute_rmse`, `calculate_dtw_distance` | agreement scores | [Signal metrics](#signal-metrics) |

---

## rPPG / Heart Rate

Remote photoplethysmography (rPPG) recovers the cardiac blood-volume pulse from tiny colour
changes in facial skin, giving a **contactless heart rate** from ordinary RGB video. The
pipeline is:

```
FaceSkinExtractor          RGB skin trace          POS / CHROM / LGI / OMIT      band-pass        bvp_to_hr
(SegFace skin mask)  ──▶   mean (3, N)      ──▶    BVP candidate           ──▶  0.75–4 Hz  ──▶   Welch-PSD peak  ──▶  HR (bpm)
```

![Contactless rPPG heart-rate overlay: SegFace parsing + skin ROI drive the live pulse and heart rate.](../images/rppg_heartrate_overlay.png)

### Extraction methods

Each method is a small class constructed with the frame rate and applied to an RGB skin trace
of shape `(3, N)` (rows ordered R, G, B). `.apply(trace)` returns the 1-D BVP.

| Method | Class | Reference / idea |
| --- | --- | --- |
| POS | [`POS`][physiotrack.signals.POS] | Plane-Orthogonal-to-Skin (Wang et al., 2017) — the default |
| CHROM | `CHROM` | Chrominance-based (de Haan & Jeanne) |
| LGI | `LGI` | Local Group Invariance |
| OMIT | `OMIT` | Orthogonal Matrix Image Transformation |

### Low-level: one method on a skin trace

When you already have an RGB skin trace, run a single method, band-pass it, and read the HR
off the Welch power-spectral-density peak:

```python
from physiotrack.signals import POS, bandpass_filter, bvp_to_hr

# rgb_trace: np.ndarray of shape (3, N) — rows R, G, B; one column per frame
bvp = POS(fps=30).apply(rgb_trace)                 # blood-volume-pulse candidate
clean = bandpass_filter(bvp, 0.75, 4.0, fs=30)     # keep the 45–240 bpm band
hr_bpm, times = bvp_to_hr(clean, fps=30)           # per-window HR (bpm) + window-centre times
print(hr_bpm[-1])                                  # latest estimate
```

!!! info "bvp_to_hr returns a per-window series"
    [`bvp_to_hr`][physiotrack.signals.bvp_to_hr] returns `(hr_bpm, times)` — arrays of the
    per-window heart rate and the window-centre timestamps. For a single number take
    `hr_bpm[-1]`. Score a full HR series against a reference with
    [`hr_errors`][physiotrack.signals.hr_errors] (MAE / RMSE / MAPE / Pearson) and quantify
    pulse quality with the de Haan SNR via [`bvp_snr`][physiotrack.signals.bvp_snr].

### High-level: streaming HeartRateEstimator

[`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator] wraps the whole flow in a
sliding window. Feed it frames plus a skin ROI (from
[`FaceSkinExtractor`][physiotrack.signals.FaceSkinExtractor]) and read the smoothed HR / SNR
after each call. `FaceSkinExtractor` runs SegFace, which **detects faces itself** — no
separate face detector or hard-coded ROI.

```python
import cv2
from physiotrack.signals import FaceSkinExtractor, HeartRateEstimator

fs = FaceSkinExtractor(device=0)                   # SegFace skin parsing
est = HeartRateEstimator("POS", fps=30,            # POS / CHROM / LGI / OMIT
                         hr_band=(0.75, 4.0), window_sec=10.0)

cap = cv2.VideoCapture("face.mp4")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    mask, skin_canvas = fs.extract(frame)          # skin ROI mask + skin-only canvas
    if mask.any():
        est.update(frame, roi_mask=mask)           # rPPG on the segmented skin
    print(est.hr, est.snr)                         # HR (bpm), de Haan SNR (dB); None until the window fills
cap.release()
```

!!! tip "Three ways to feed the estimator"
    - `est.update(frame, roi_mask=mask)` — segmentation-based (preferred; `roi_mask` wins over `box`).
    - `est.update(frame, box=(x1, y1, x2, y2))` — a face box; the built-in forehead/cheek skin ROI is sampled (a lightweight fallback when you do not run segmentation).
    - `est.push_rgb(r, g, b)` — feed one mean-RGB sample per frame when you do your own ROI.

??? note "One SegFace pass for both the ROI and a display canvas"
    [`FaceSkinExtractor.analyze`][physiotrack.signals.FaceSkinExtractor] runs the segmenter
    once and returns a [`FaceParsing`][physiotrack.signals.FaceParsing] named tuple with
    everything derived from that single inference:

    ```python
    fp = fs.analyze(frame)   # one SegFace pass
    fp.skin_mask       # bool (H, W)  — the skin ROI for rPPG
    fp.skin_canvas     # BGR (H, W, 3) — skin pixels only (display)
    fp.parsing_canvas  # BGR (H, W, 3) — all 19 CelebAMask-HQ classes colorized
    fp.seg_map         # int (H, W)   — raw class-index map
    est.update(frame, roi_mask=fp.skin_mask)
    ```

    Re-segmenting every few frames (the face region moves slowly) and reusing the mask in
    between keeps the loop fast — see `examples/rppg_vitals.py`.

### Live overlays

The heart rate is derived from the same BVP signal that the pulse trace shows, so **share one
estimator** across both plotters and the rPPG is computed once per frame.
[`RPPGPlotter`][physiotrack.signals.RPPGPlotter] draws the band-passed pulse trace;
[`HeartRatePlotter`][physiotrack.signals.HeartRatePlotter] draws the waveform + bpm + SNR.

=== "Shared estimator (compute once)"

    ```python
    import cv2
    from physiotrack.signals import (
        FaceSkinExtractor, HeartRateEstimator, RPPGPlotter, HeartRatePlotter,
    )

    fs = FaceSkinExtractor(device=0)
    est = HeartRateEstimator("POS", fps=30.0)
    sig = RPPGPlotter(estimator=est)          # rPPG / BVP pulse
    hrp = HeartRatePlotter(estimator=est)     # derived HR (bpm)

    cap = cv2.VideoCapture("face.mp4")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, _ = fs.extract(frame)
        if mask.any():
            est.update(frame, roi_mask=mask)  # compute rPPG once
        frame = sig.attach_to_frame(frame, position="top_right")
        frame = hrp.attach_to_frame(frame, position="top_right",
                                    above_element_height=sig.canvas_height)
        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    ```

=== "Standalone plotter (builds its own estimator)"

    ```python
    from physiotrack.signals import HeartRatePlotter

    hrp = HeartRatePlotter(method="POS", fps=30.0)   # owns a HeartRateEstimator
    for frame, skin_mask in stream:
        hrp.update(frame, roi_mask=skin_mask)        # delegates to the estimator
        frame = hrp.attach_to_frame(frame, position="bottom_right")
    ```

!!! note "All bands are configurable"
    The analysis band is `hr_band=(0.75, 4.0)` Hz (45–240 bpm) by default and drives both the
    band-pass and the HR search. `window_sec` sets the sliding-window length; `smooth_hr`
    reports the median HR over recent windows. A full standalone example (overlaying face
    parsing, skin ROI, the rPPG pulse, HR, HRV and respiration) lives in
    `examples/rppg_vitals.py`.

---

## Heart-Rate Variability (HRV)

HRV is the beat-to-beat variation of the heart rhythm, computed from the **RR-interval**
(inter-beat) series — not from the raw pulse. From the same rPPG BVP window, the pipeline is:

```
BVP  ──▶  detect_pulse_peaks  ──▶  bvp_to_rri  ──▶  correct_rr_artifacts  ──▶  compute_hrv
(pulse)   systolic peaks          RR intervals (ms)   Lipponen–Tarvainen        HRV indices
```

- [`bvp_to_rri`][physiotrack.signals.bvp_to_rri] turns the pulse into an RR-interval series;
  [`detect_pulse_peaks`][physiotrack.signals.detect_pulse_peaks] exposes the peak detector.
- [`find_rr_artifacts`][physiotrack.signals.find_rr_artifacts] /
  [`correct_rr_artifacts`][physiotrack.signals.correct_rr_artifacts] clean ectopic/missed beats
  (Lipponen & Tarvainen, 2019) before analysis.
- [`compute_hrv`][physiotrack.signals.compute_hrv] returns the indices across three domains —
  [`hrv_time`][physiotrack.signals.hrv_time] (RMSSD, SDNN, pNN50, …),
  [`hrv_frequency`][physiotrack.signals.hrv_frequency] (VLF/LF/HF power, LF/HF) and
  [`hrv_nonlinear`][physiotrack.signals.hrv_nonlinear] (Poincaré SD1/SD2,
  [`sample_entropy`][physiotrack.signals.sample_entropy] /
  [`approximate_entropy`][physiotrack.signals.approximate_entropy]).
- Validate a computed series against a reference with
  [`hrv_errors`][physiotrack.signals.hrv_errors].

The [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator] reads HRV off the **same
sliding window** it uses for HR — no extra buffering:

```python
from physiotrack.signals import FaceSkinExtractor, HeartRateEstimator

fs  = FaceSkinExtractor(device=0)
est = HeartRateEstimator("POS", fps=30, window_sec=60)   # HRV needs a LONG window

for frame in frames:
    mask, _ = fs.extract(frame)
    if mask.any():
        est.update(frame, roi_mask=mask)

rri_ms, t = est.rri(correct=True)     # artefact-corrected RR intervals (ms)
hrv = est.hrv()                       # {'RMSSD':…, 'SDNN':…, 'SD1':…, 'LFHF':…, …}
```

!!! warning "HRV needs a long, still window"
    Frequency-domain (LF/HF) and stable time-domain indices need **~60 s** of clean pulse.
    On short clips or with head motion the RR series is noisy and HRV values inflate wildly
    (RMSSD/SDNN in the hundreds of ms) — use `window_sec=60`, a steady front-facing subject,
    and good lighting. The math is numerically cross-checked against NeuroKit2.

**Live overlay** — [`HRVPlotter`][physiotrack.signals.HRVPlotter] shares the estimator with the
HR/rPPG panels (one rPPG computation per frame) and renders the HRV grid:

```python
from physiotrack.signals import HRVPlotter
hrv_panel = HRVPlotter(estimator=est)          # share the same estimator
frame = hrv_panel.attach_to_frame(frame, position="top_right")
```

In the [`Video`][physiotrack.Video] pipeline, just pass `hrv=True` (see the
[Video guide](video.md#vitals-rppg-heart-rate-hrv-respiration)).

---

## Respiration

Respiration rate (breaths/min) is recovered **contactlessly two ways** — pick per what your
footage shows:

| Source | Function | From | Needs |
| --- | --- | --- | --- |
| Pulse amplitude (RIAV) | [`respiration_from_pulse`][physiotrack.signals.respiration_from_pulse] | rPPG BVP modulation | face/skin ROI |
| RR variation (RSA) | [`respiration_from_rri`][physiotrack.signals.respiration_from_rri] | RR-interval series | face/skin ROI |
| Shoulder / torso motion | [`respiration_from_motion`][physiotrack.signals.respiration_from_motion] | pose keypoints (shoulders 5/6) | pose (no face) |

All three read the dominant in-band frequency via
[`respiration_rate_from_signal`][physiotrack.signals.respiration_rate_from_signal].

```python
# Pulse-derived (shares the rPPG estimator): RIAV amplitude or RSA
rr_pulse = est.respiration_rate("pulse")     # breaths/min
rr_rsa   = est.respiration_rate("rsa")

# Motion-derived: reuse the per-frame pose records (no face, no extra inference)
from physiotrack.signals import respiration_from_motion
records = pt.Video(source="in.mp4", pose=pt.Pose.Person()).run()   # VideoResults of frames
resp_wave, rr_motion = respiration_from_motion(records, original_fps=30.0)
```

!!! tip "Which source?"
    Face clearly visible → **pulse** (comes free with the rPPG estimator). Face occluded /
    far / masked (e.g. a **VR headset**) but the torso is visible → **motion**, which tracks
    the breathing rise-and-fall of the shoulders.

**Live overlay** — [`RespirationPlotter`][physiotrack.signals.RespirationPlotter] (labelled
`PULSE` or `MOTION`). In the [`Video`][physiotrack.Video] pipeline set `respiration=True` and
choose `respiration_source="pulse"` or `"motion"` — see the
[Video guide](video.md#vitals-rppg-heart-rate-hrv-respiration).

---

## Motion features, Joint Angles & ROM

From per-frame pose instances (the `instances` list in
[`Video`][physiotrack.Video] / `Result.to_dict()` output), the motion tools build keypoint
trajectories, derive velocity/acceleration, and measure anatomical joint angles and clinical
ROM.

### Keypoint sequences & centroids

[`extract_keypoints_sequence`][physiotrack.signals.extract_keypoints_sequence] produces a
**wide** DataFrame (one row per frame/person, columns `{k}_x`, `{k}_y`, `{k}_confidence` for
2D and `3d_{k}_x/_y/_z` for 3D). Before extracting, you can synthesize reference keypoints:

| Helper | Adds keypoint id | Definition |
| --- | --- | --- |
| [`add_head_centroid`][physiotrack.signals.add_head_centroid] | `133` | mean of face keypoints |
| [`add_body_centroid`][physiotrack.signals.add_body_centroid] | `134` | mean of the 17 COCO body joints |
| [`add_pelvic_centroid`][physiotrack.signals.add_pelvic_centroid] | `135` | midpoint of the hips (ids 11 & 12) — default motion reference |

```python
from physiotrack.signals import (
    add_pelvic_centroid, extract_keypoints_sequence,
    get_relative_coordinates, compute_all_motion_features,
    resample_dataframe_by_interpolation,
)

# `data` = frame/instance records from Video.run(...) or Result.to_dict()["instances"]
# `pose` is your Pose estimator
data = add_pelvic_centroid(data, pose.architecture)          # id 135 = hip midpoint
kp_df = extract_keypoints_sequence(data, candidate_key_points=list(range(17)) + [135])
rel = get_relative_coordinates(kp_df, reference_point_id=135)  # pelvis-centered (translation-invariant)
motion_df = compute_all_motion_features(rel)                   # + velocity, acceleration, joint angles
motion_df = resample_dataframe_by_interpolation(motion_df, input_fs=video_fps, output_fs=30)
```

!!! note "Single-keypoint extraction"
    For one keypoint's trajectory use
    [`extract_keypoint_sequence_2d`][physiotrack.signals.extract_keypoint_sequence_2d] /
    [`extract_keypoint_sequence_3d`][physiotrack.signals.extract_keypoint_sequence_3d]
    (tidy long-form). Slice one keypoint's feature columns out of a motion DataFrame with
    [`get_keypoint_features`][physiotrack.signals.get_keypoint_features] and pick the columns
    for a feature type (`coordinates` / `velocity` / `acceleration` / `angles`) with
    [`select_feature_data`][physiotrack.signals.select_feature_data]. See `examples/motion.py`.

### Joint angles (per frame)

[`joint_angles`][physiotrack.signals.joint_angles] measures the interior angle at each of the
eight major joints (left/right shoulder, elbow, hip, knee) directly on one frame's keypoints —
no plotter needed. [`compute_rom_angles`][physiotrack.signals.compute_rom_angles] measures
clinical range-of-motion movements (hip flexion / extension / abduction / adduction) against a
body reference axis.

```python
import physiotrack as pt
from physiotrack.signals import joint_angles, compute_rom_angles

result = pt.Pose.Person().predict(frame)
kps = result[0]            # an Instance; a Keypoints collection or a dict list also works

joint_angles(kps)                                       # {'leftElbow': 152.3, 'leftKnee': 174.1, ...}  degrees
joint_angles(kps, joints=["leftElbow", "rightElbow"])   # subset
compute_rom_angles(kps, movements=["leftHipFlexion"])   # {'leftHipFlexion': 12.4}  degrees
```

Both return `dict[str, float]` for only the confidently measured joints (all three involved
keypoints must clear `conf_threshold`, default `0.3`).

#### <a id="warning-degrees-vs-radians"></a>

!!! warning "Units: degrees vs radians"
    The per-frame functions and the whole-sequence function report angles in **different
    units** — a real quirk of the API, so be explicit about which you consume:

    | Function | Scope | Unit |
    | --- | --- | --- |
    | [`joint_angles`][physiotrack.signals.joint_angles] | one frame | **degrees** |
    | [`compute_rom_angles`][physiotrack.signals.compute_rom_angles] | one frame | **degrees** |
    | [`compute_all_joint_angles`][physiotrack.signals.compute_all_joint_angles] | whole DataFrame (`ang_2d_*` / `ang_3d_*` columns) | **radians** |

    So `compute_all_joint_angles` (and the `ang_*` columns produced inside
    [`compute_all_motion_features`][physiotrack.signals.compute_all_motion_features]) are in
    radians — apply `numpy.degrees(...)` if you want to compare them with the per-frame
    `joint_angles` output.

### Live joint-angle & ROM overlay

[`JointAnglePlotter`][physiotrack.signals.JointAnglePlotter] renders the interior angles and
(optionally) ROM as compact 2-column (left | right) grid panels — each cell shows a label,
the live value in degrees, a 0–180° gauge and a sparkline. The definitions are shared with
`physiotrack.signals.motion.features`, so the panel and any skeleton arcs agree.

=== "End-to-end via Video"

    ```python
    import physiotrack as pt

    pt.Video(
        source="in.mp4",
        detector=pt.Detection.Person(),
        pose=pt.Pose.Person(),          # whole-body pose feeds the angle panel
        plot_angles=True,               # interior joint-angle grid (left side)
        rom=True,                       # add clinical ROM grid + skeleton canvas
        # angle_joints=["leftElbow", "rightElbow", "leftKnee", "rightKnee"],  # optional subset
        # rom_render=False,             # keep ROM values in the grid but hide the skeleton canvas
    ).run(output_video="out_angles.mp4", output_json="poses.json")
    ```

=== "Standalone, frame by frame"

    ```python
    import cv2
    import physiotrack as pt
    from physiotrack.signals import JointAnglePlotter

    pose = pt.Pose.Person()
    plotter = JointAnglePlotter(rom=True, fps=30.0)   # 8 joints + clinical ROM

    cap = cv2.VideoCapture("in.mp4")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose_results = pose.predict(frame).to_dict()["instances"]
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        plotter.update(pose_results, frame_time=t)
        frame = plotter.attach_to_frame(frame, position="top_left")  # joint + ROM grids
        cv2.imshow("joint angles", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    ```

See `examples/joint_angle_overlay.py` for both paths.

---

## Face signals

Facial geometry measured from face landmarks, and the per-face signals built on it.
The landmarks come from [`FaceLandmarks`][physiotrack.FaceLandmarks] (the 478-point
`"FACEMESH"`) or from a COCO-WholeBody pose model, whose ids 23–90 are a 68-point face.
The two layouts number their points differently, so the layout is read from the
keypoints; plain dicts must name it (`layout="FACEMESH"` or `"WHOLEBODY"`).

### Per frame

| Function | Measures |
| --- | --- |
| [`eye_aspect_ratio`][physiotrack.signals.eye_aspect_ratio] | lid opening over eye width per eye (Soukupová & Čech 2016); ≈ 0.25–0.35 open, → 0 closed |
| [`mouth_aspect_ratio`][physiotrack.signals.mouth_aspect_ratio] | inner-lip gap at the midline over mouth-corner width, built like the EAR; 0 with the lips closed |
| [`iris_position`][physiotrack.signals.iris_position] | iris centre within each eye, in eye widths (FACEMESH only) |

```python
import physiotrack as pt

faces = pt.FaceLandmarks()(frame, pt.Face()(frame))
pt.signals.eye_aspect_ratio(faces[0])      # {'left': 0.29, 'right': 0.31, 'mean': 0.30}
pt.signals.mouth_aspect_ratio(faces[0])    # 0.08
pt.signals.iris_position(faces[0])         # {'x': 0.52, 'y': -0.03, ...}

# The same eye measure from a WholeBody pose, no face model needed:
person = pt.Pose.Custom(model=pt.Models.Pose.ViTPose.WholeBody.b_wholebody)(frame)[0]
pt.signals.eye_aspect_ratio(person)
```

### Over a video

These read the faces of a [`Video`][physiotrack.Video] run — `FrameResult.faces` — and
follow a face by its `id`, the track id of the subject it belongs to (see
[Faces in a video](face.md#faces-in-a-video)). Every processed frame counts: a frame in
which the face is absent, or has no face mesh, is a gap that interrupts a blink, a
movement estimate and a summary window. Frames skipped by `Video(fps=...)` are not
gaps. Times are the frame timestamps; a record without one is rejected.

| Function | Output |
| --- | --- |
| [`face_feature_sequence`][physiotrack.signals.face_feature_sequence] | DataFrame, one row per face per frame: box, head pose, EAR, MAR, iris, expression, gaze, quality, skin fraction |
| [`detect_blinks`][physiotrack.signals.detect_blinks] | DataFrame of blinks: `start`, `end`, `duration`, `min_ear` |
| [`blink_rate`][physiotrack.signals.blink_rate] | blinks per minute over the time the face was present |
| [`mouth_movement`][physiotrack.signals.mouth_movement] | DataFrame: `time`, `mar`, `mar_movement`, `mar_velocity` |
| [`face_window_summary`][physiotrack.signals.face_window_summary] | DataFrame, per face per frame: mean / std / min / max of each measure over the last `window` seconds, blinks in the window, dominant expression |

```python
results = pt.Video("clip.mp4", detector=pt.Face(), tracker=pt.Tracker(),
                   face_stages=[pt.FaceLandmarks(), pt.FaceExpression()]).run()

blinks = pt.signals.detect_blinks(results, detection_id=1)
rate = pt.signals.blink_rate(results, detection_id=1)
motion = pt.signals.mouth_movement(results, detection_id=1)
windows = pt.signals.face_window_summary(results, window=5.0)
```

- **Blinks.** At least `min_closed_frames` (3) consecutive frames with mean EAR below
  `threshold` (0.22), counted when the eye reopens; the duration runs from the first
  closed frame to the reopening frame. These are the settings validated on MPEBlink
  (see [Face validation](face-validation.md#results-of-the-project-validation)); they
  suit frontal faces at 25–30 fps. EAR depends on the viewing angle, so tune
  `threshold` per camera setup.
- **Blink rate.** Blinks divided by the time the face was in view — its frames times
  the mean processing interval — so frames without the face do not dilute it.
- **Mouth movement.** The absolute change of the MAR since the previous frame, and that
  change divided by the time between the frames; `0` on the first frame after a gap.
- **Window summary.** For each face, its most recent `round(window / interval)`
  frames in view, restarting when it leaves the view. Frames without a measure are
  skipped for that measure only.

!!! note "Validation"
    The geometry is checked against closed-form values on synthetic landmarks and, on
    two real face meshes committed as a test fixture, against the project's validation
    implementation. Blink counting, blink rate, mouth movement and the window summary
    reproduce that implementation on a scenario with gaps and edge cases, and blink
    detection is also cross-checked against an independent `scipy.signal.find_peaks`
    implementation (`tests/test_motion.py`).

---

## Filters & Normalization

DSP helpers for pre- and post-processing extracted signals. The band-pass used throughout the
rPPG pipeline is [`bandpass_filter`][physiotrack.signals.bandpass_filter]: a Butterworth IIR
design taking the two cutoffs in Hz, applied **zero-phase** (`sosfiltfilt`) so that pulse-peak
timing — and therefore every R-R interval and HRV index derived from it — is preserved. It
raises rather than returning a phase-distorted result if the segment is too short to filter
zero-phase.

Also available: [`highpass_filter`][physiotrack.signals.highpass_filter] /
[`lowpass_filter`][physiotrack.signals.lowpass_filter] for one-sided bands, and
[`notch_filter`][physiotrack.signals.notch_filter] to remove mains hum.

```python
from physiotrack.signals import bandpass_filter, z_score_normalize

clean = bandpass_filter(bvp, 0.75, 4.0, fs=30)      # keep the 45–240 bpm heart-rate band
z = z_score_normalize(motion_df["9_x"])             # standardize a motion series (mean 0, std 1)
```

The normalizers ([`min_max_normalize`][physiotrack.signals.min_max_normalize],
[`z_score_normalize`][physiotrack.signals.z_score_normalize],
[`robust_scale_normalize`][physiotrack.signals.robust_scale_normalize],
`max_abs_normalize`, and more) operate on a pandas `Series` and each handle constant / zero-
variance input gracefully. See the [Filters](../api/signals/filters.md) and
[Normalization](../api/signals/normalize.md) API pages for the full list and parameters.

---

## Signal metrics

Compare two 1-D signals (e.g. a video-estimated wrist trajectory against a reference). All
metrics first trim the inputs to a common length via
[`align_signals`][physiotrack.signals.align_signals].

| Metric | Function | Range / meaning |
| --- | --- | --- |
| Phase Locking Value | [`compute_plv`][physiotrack.signals.compute_plv] | `[0, 1]`, 1 = phase-locked |
| Phase synchrony | [`phase_synchrony`][physiotrack.signals.phase_synchrony] | `[0, 1]`, mean abs. phase agreement |
| Pearson correlation | [`calculate_pearson_correlation`][physiotrack.signals.calculate_pearson_correlation] | `[-1, 1]` (NaN for constant input) |
| Normalized cross-corr. | [`normalized_cross_correlation`][physiotrack.signals.normalized_cross_correlation] | `[-1, 1]` at zero lag |
| RMSE | [`compute_rmse`][physiotrack.signals.compute_rmse] | `>= 0`, same units as inputs |
| DTW distance | [`calculate_dtw_distance`][physiotrack.signals.calculate_dtw_distance] | `>= 0`, robust to time shifts |
| Event synchronization | [`event_synchronization`][physiotrack.signals.event_synchronization] | `[0, 1]`, matched-peak fraction |

```python
from physiotrack.signals import (
    calculate_pearson_correlation, calculate_dtw_distance,
    compute_rmse, compute_plv,
)

r   = calculate_pearson_correlation(reference, estimate)
dtw = calculate_dtw_distance(reference, estimate)
rmse = compute_rmse(reference, estimate)
plv = compute_plv(reference, estimate)
```

See `examples/motion.py` for a full 2D-vs-3D comparison and the
[Signal Metrics](../api/signals/evaluate.md) API page for details.

---

## See also

- [Signals API reference](../api/signals/index.md) — the full auto-generated docs, including:
    - [rPPG / heart rate](../api/signals/rppg.md)
    - [Motion features & angles](../api/signals/motion.md)
    - [Face signals](../api/signals/face.md)
    - [Filters](../api/signals/filters.md)
    - [Normalization](../api/signals/normalize.md)
    - [Signal metrics](../api/signals/evaluate.md)
    - [Plotting overlays](../api/signals/plotting.md)
- Related guides: [Pose](pose.md) · [Segmentation](segmentation.md) · [3D Pose](pose3d.md) · [Video pipeline](video.md)
- [Result objects](../api/results.md) — the `instances` / `keypoints` structures the motion tools consume.
