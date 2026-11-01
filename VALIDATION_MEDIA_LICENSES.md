# Validation Media: Provenance, Attribution and Redistribution Licenses

## 1. Purpose and Scope

This document records the provenance, licensing conditions, source attribution, transformations, and repository-distribution status of visual media associated with the PhysioTrack face-analysis validation framework.

The inventory covers benchmark-derived qualitative media, synthetic validation inputs, synthetic demonstration outputs, integration-test fixtures, robustness-test materials, and project-generated analytical figures.

Repository-relative paths are used throughout.

The validation framework distinguishes three categories of scientific evidence:

- **Benchmark evaluation:** Measurements obtained using external datasets and reference annotations.
- **Component and integration execution:** Evidence that the implemented components and pipelines execute and produce their expected outputs.
- **Synthetic demonstration:** Illustrations of component behavior under constructed conditions, distinct from independent benchmark accuracy evaluations.

Licensing classifications concern the visual material and its underlying sources. They do not alter the interpretation of numerical validation results.

### 1.1 Media Included in the Repository

The reviewed Git index contains **409 image and video files**, comprising 406 validation-media files and three existing upstream documentation images.

| Media category | Files |
|---|---:|
| MPIIFaceGaze qualitative figures | 9 |
| RAVDESS/FELT qualitative figures | 18 |
| MPEblink2.0 qualitative images and videos | 17 |
| Synthetic integration images | 5 |
| Synthetic validation input images and videos | 109 |
| Synthetic demonstration output images and videos | 211 |
| Project-generated quantitative and deterministic figures | 37 |
| Existing upstream documentation images | 3 |
| **Total** | **409** |

### 1.2 Media Excluded from the Repository

The local validation inventory contains **119 image and video files** excluded from Git tracking.

| Source or evaluation category | Excluded files |
|---|---:|
| WIDER FACE | 11 |
| 300-W — Face Landmarks | 9 |
| 300-W — Face Quality | 28 |
| AFLW | 9 |
| CelebAMask-HQ | 9 |
| CAER-S | 11 |
| ECCV 2016 Face Tracking | 8 |
| Pexels integration photographs and videos | 19 |
| Pexels-derived robustness images | 15 |
| **Total** | **119** |

Excluded media are retained in the local research environment. Their exclusion does not extend to independently generated numerical measurements, CSV/JSON records, statistical tables, or analytical plots that do not reproduce restricted source imagery.

Third-party software, model weights, vendored code, and other runtime dependencies are outside the scope of this visual-media inventory and are documented separately through the applicable project licensing records.

---

## 2. Licensing Principles

### 2.1 Creative Commons Attribution–NonCommercial–ShareAlike 4.0

The official records for MPIIFaceGaze, RAVDESS, and FELT identify the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International license (**CC BY-NC-SA 4.0**) as applicable to the relevant material.

Principal conditions include:

- **Attribution:** Appropriate credit to the original creators, identification of the source, a reference to the license, and indication of modifications.
- **NonCommercial:** Sharing and adaptation must comply with the license's non-commercial limitation.
- **ShareAlike:** Distributed adaptations must use the same license or an authorized compatible license.
- **No additional restrictions:** Additional legal or technological restrictions must not prevent recipients from exercising the rights granted by the license.
- **Other rights:** Privacy, publicity, personality, trademark, and other independently applicable rights may require separate consideration.

License:

https://creativecommons.org/licenses/by-nc-sa/4.0/

Legal code:

https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Figures incorporating selected source images or video frames, with overlays, annotations, measurements, or composite layouts, are identified as adapted visual material.

Placement alongside GPL-licensed PhysioTrack software does not replace the source media's CC BY-NC-SA 4.0 conditions.

### 2.2 Apache License 2.0

Apache License 2.0 grants rights to reproduce, modify, and distribute works covered by the license, subject to its terms.

Applicable redistribution conditions include providing a copy of the license, retaining required copyright and attribution notices, identifying modifications, and preserving applicable NOTICE-file information.

Official license:

https://www.apache.org/licenses/LICENSE-2.0

An Apache-2.0 designation on a dataset repository is distinct from the license of related implementation software. The dataset publisher's declared license and the scope of the published material must be identified independently.

A publisher's license declaration does not override rights held independently by third parties.

### 2.3 Pexels License

Pexels photographs and videos are governed by the Pexels License and its Terms of Service.

Official resources:

https://www.pexels.com/license/

https://www.pexels.com/terms-of-service/

The Pexels Terms restrict standalone redistribution of content that remains substantially in its original form without sufficient creative transformation.

The PhysioTrack repository provides source references and reproduction instructions rather than redistributing its selected original Pexels integration media or the controlled robustness images derived from them.

Photographer attribution is documented for provenance and research reproducibility, irrespective of whether attribution is required for a particular permitted use.

### 2.4 OpenAI-Generated Synthetic Media

The synthetic image inputs identified as OpenAI-generated were produced using OpenAI/ChatGPT.

Applicable terms:

https://openai.com/policies/terms-of-use/

These files are documented as AI-generated synthetic assets, not as photographs from external facial-image benchmarks.

Their provenance does not imply a Creative Commons or public-domain designation.

### 2.5 Blender Human Base Meshes

The Blender-based synthetic videos use the following recorded source model:

**Blender Studio & Community — Human Base Meshes v1.4.1, Body Male – Realistic**

Official source:

https://www.blender.org/download/demo-files/#assets

Original-model license:

**CC0 1.0 Universal**

https://creativecommons.org/publicdomain/zero/1.0/

The original 3D model was modified and animated to produce component-specific synthetic test sequences.

The CC0 designation applies to the identified source model. The project-generated animations and rendered outputs are documented separately from the original asset.

---

## 3. Benchmark-Derived Qualitative Media

### 3.1 MPIIFaceGaze — Nine Figures

**Dataset:** MPIIFaceGaze, Version 1.0.

**Dataset citation:**

Bulling, A. (2023). *MPIIFaceGaze* (Version 1.0) [Data set]. DaRUS. https://doi.org/10.18419/DARUS-3240

**Associated publication:**

Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2019). MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(1), 162–175. https://doi.org/10.1109/TPAMI.2017.2778103

**Dataset record:**

https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3240

**License:** CC BY-NC-SA 4.0, subject to the dataset's non-commercial scientific-use conditions.

**Original dataset location:**

`datasets/MPIIFaceGaze/Data/`

**Source-image selection manifest:**

`validation/gaze_estimation/results/qualitative/mpiifacegaze_qualitative_manifest.csv`

The eight individual qualitative figures incorporate selected facial images with gaze-estimation overlays and annotations. The contact sheet combines these examples into a single figure.

| Repository-relative file | Transformation |
|---|---|
| `validation/gaze_estimation/results/figures/mpiifacegaze_qualitative_contact_sheet.png` | Combined panel of annotated gaze examples |
| `validation/gaze_estimation/results/qualitative/case_01_p05_p00.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_02_p25_p01.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_03_p50_p09.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_04_p75_p12.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_05_p90_p02.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_06_p95_p07.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_07_p99_p08.png` | Selected facial image with gaze annotations |
| `validation/gaze_estimation/results/qualitative/case_08_maximum_p08.png` | Selected facial image with gaze annotations |

**Repository distribution:** Included under the applicable CC BY-NC-SA 4.0 conditions.

### 3.2 RAVDESS/FELT — Mouth Movement and Velocity

**Primary visual source:** RAVDESS speech videos.

**Dataset citation:**

Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1188976

**Associated publication:**

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

**Visual-media license:** CC BY-NC-SA 4.0.

**Reference annotation source:** FELT facial-landmark and tracking data.

**Source directories:**

`datasets/RAVDESS/Video_Speech/`

`datasets/FELT/raw_motion_speech/`

**Transformations:** Selected frame pairs, mouth-geometry annotations, numerical measurements, qualitative labels, and combined example panels.

| Repository-relative file | Original RAVDESS source |
|---|---|
| `validation/mouth_movement_velocity/results/figures/felt_ravdess_mouth_movement_velocity_qualitative_examples.png` | Combined panel of eight annotated transitions |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/01_low_movement_Actor_19_01-01-07-01-01-02-19_frames_0029_0030.png` | Actor 19; `01-01-07-01-01-02-19.mp4`; frames 29–30 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/02_medium_movement_Actor_18_01-01-07-02-02-01-18_frames_0061_0062.png` | Actor 18; `01-01-07-02-02-01-18.mp4`; frames 61–62 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/03_high_movement_Actor_13_01-01-03-02-02-01-13_frames_0058_0059.png` | Actor 13; `01-01-03-02-02-01-13.mp4`; frames 58–59 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/04_very_high_movement_Actor_07_01-01-05-01-01-02-07_frames_0068_0069.png` | Actor 07; `01-01-05-01-01-02-07.mp4`; frames 68–69 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/05_representative_error_Actor_09_01-01-06-01-02-01-09_frames_0072_0073.png` | Actor 09; `01-01-06-01-02-01-09.mp4`; frames 72–73 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/06_challenging_underestimate_Actor_23_01-01-08-01-02-02-23_frames_0048_0049.png` | Actor 23; `01-01-08-01-02-02-23.mp4`; frames 48–49 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/07_challenging_overestimate_Actor_24_01-01-05-02-01-01-24_frames_0042_0043.png` | Actor 24; `01-01-05-02-01-01-24.mp4`; frames 42–43 |
| `validation/mouth_movement_velocity/results/qualitative/annotated_transitions/08_high_error_actor_representative_Actor_24_01-01-03-02-01-02-24_frames_0096_0097.png` | Actor 24; `01-01-03-02-01-02-24.mp4`; frames 96–97 |

The displayed frames originate from RAVDESS, while reference tracking measurements are informed by FELT.

**Repository distribution:** Included as RAVDESS-derived adaptations under CC BY-NC-SA 4.0.

### 3.3 RAVDESS/FELT — Mouth Openness

**Primary visual source:** RAVDESS speech videos.

**Reference annotation source:** FELT.

**Visual-media license:** CC BY-NC-SA 4.0.

**Transformations:** Video-frame selection, mouth-geometry overlays, predicted and reference measurements, qualitative labels, and combined example figures.

| Repository-relative file | Original RAVDESS source |
|---|---|
| `validation/mouth_openness/results/figures/felt_ravdess_mouth_openness_qualitative_examples.png` | Combined panel of eight annotated examples |
| `validation/mouth_openness/results/qualitative/annotated_images/01_closed_representative_Actor_23_01-01-07-02-01-01-23_frame_0042.png` | Actor 23; `01-01-07-02-01-01-23.mp4`; frame 42 |
| `validation/mouth_openness/results/qualitative/annotated_images/02_low_opening_Actor_22_01-01-05-01-01-01-22_frame_0097.png` | Actor 22; `01-01-05-01-01-01-22.mp4`; frame 97 |
| `validation/mouth_openness/results/qualitative/annotated_images/03_medium_opening_Actor_05_01-01-02-02-01-01-05_frame_0023.png` | Actor 05; `01-01-02-02-01-01-05.mp4`; frame 23 |
| `validation/mouth_openness/results/qualitative/annotated_images/04_high_opening_Actor_17_01-01-06-02-02-02-17_frame_0015.png` | Actor 17; `01-01-06-02-02-02-17.mp4`; frame 15 |
| `validation/mouth_openness/results/qualitative/annotated_images/05_very_high_opening_Actor_06_01-01-04-02-02-01-06_frame_0033.png` | Actor 06; `01-01-04-02-02-01-06.mp4`; frame 33 |
| `validation/mouth_openness/results/qualitative/annotated_images/06_representative_error_Actor_19_01-01-03-02-02-01-19_frame_0043.png` | Actor 19; `01-01-03-02-02-01-19.mp4`; frame 43 |
| `validation/mouth_openness/results/qualitative/annotated_images/07_challenging_underestimate_Actor_11_01-01-07-02-01-02-11_frame_0026.png` | Actor 11; `01-01-07-02-01-02-11.mp4`; frame 26 |
| `validation/mouth_openness/results/qualitative/annotated_images/08_challenging_overestimate_Actor_08_01-01-02-01-01-02-08_frame_0044.png` | Actor 08; `01-01-02-01-01-02-08.mp4`; frame 44 |

The source mapping for the Actor 22 low-opening example is recorded in:

`validation/mouth_openness/results/qualitative/felt_ravdess_mouth_openness_qualitative_selection.csv`

The record identifies trial `01-01-05-01-01-01-22`, frame 97.

**Repository distribution:** Included as RAVDESS-derived adaptations under CC BY-NC-SA 4.0.

### 3.4 FELT — Reference Tracking Data

**Dataset:** Facial Expression and Landmark Tracking (FELT).

**Academic citation specified by the dataset provider:**

Liao, Z., Livingstone, S. R., & Russo, F. A. (2024). *RAVDESS Facial expression and landmark tracking* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13243600

**Official dataset record:**

https://zenodo.org/records/13243600

**License:** CC BY-NC-SA 4.0.

FELT supplies facial-landmark coordinates, expression tracking, head-pose measurements, and related reference data for RAVDESS trials.

The PhysioTrack mouth-related qualitative figures use original RAVDESS visual frames with project-generated measurements and annotations informed by FELT.

These figures are not direct redistributions of FELT's pre-rendered tracking videos.

The dataset provider requests attribution to both FELT and the underlying RAVDESS publication.

### 3.5 MPEblink2.0 — Eye Openness and Blink

**Dataset:** MPEblink2.0.

**Publisher:** Tao-HUST.

**Official dataset repository:**

https://huggingface.co/datasets/Tao-HUST/MPEblink2.0

**Dataset DOI:**

https://doi.org/10.57967/hf/9376

**Official research project:**

https://github.com/wenzhengzeng/MPEblink

**Associated publication:**

Zeng, W., Xiao, Y., Wei, S., Gan, J., Zhang, X., Cao, Z., Fang, Z., & Zhou, J. T. (2023). Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 13854–13863.

#### Dataset License

The official MPEblink2.0 dataset repository published by Tao-HUST on Hugging Face declares **Apache License 2.0 (`apache-2.0`)** in its dataset metadata.

The licensing designation is recorded at the dataset repository level and is separate from the software license displayed by the associated implementation repository.

The earlier MPEblink dataset release is a separately published resource:

https://doi.org/10.5281/zenodo.7754768

Its original Zenodo description specifies non-commercial research use. That earlier release statement is not represented here as the license metadata of the independently published MPEblink2.0 dataset repository.

The MPEblink2.0 license designation records the publisher's declared terms for the relevant dataset release. It does not override independently applicable rights in third-party source footage or other incorporated material.

#### PhysioTrack Qualitative Outputs

The qualitative validation uses the MPEblink2.0 test split identified in the project's evaluation protocol.

**Source dataset location:**

`datasets/MPEBlink2/mpeblink2.0/`

**Evaluation documentation:**

`validation/eye_openness_and_blink/README_MPEBLINK_EYE_OPENNESS_AND_BLINK.txt`

The qualitative inventory includes eight annotated images, eight annotated videos, and one combined figure.

**Base output directory:**

`validation/eye_openness_and_blink/results/qualitative/`

| Annotated image | Annotated video |
|---|---|
| `annotated_images/accurate_count_video_21_person0.png` | `annotated_videos/accurate_count_video_21_person0.mp4` |
| `annotated_images/challenging_false_negative_video_189_person0.png` | `annotated_videos/challenging_false_negative_video_189_person0.mp4` |
| `annotated_images/challenging_false_positive_video_190_person0.png` | `annotated_videos/challenging_false_positive_video_190_person0.mp4` |
| `annotated_images/high_blink_activity_video_191_person0.png` | `annotated_videos/high_blink_activity_video_191_person0.mp4` |
| `annotated_images/low_blink_activity_video_24_person1.png` | `annotated_videos/low_blink_activity_video_24_person1.mp4` |
| `annotated_images/mixed_detection_video_193_person0.png` | `annotated_videos/mixed_detection_video_193_person0.mp4` |
| `annotated_images/representative_video_43_person1.png` | `annotated_videos/representative_video_43_person1.mp4` |
| `annotated_images/strong_detection_video_167_person2.png` | `annotated_videos/strong_detection_video_167_person2.mp4` |

**Combined figure:**

`validation/eye_openness_and_blink/results/figures/mpeblink_qualitative_examples.png`

**Transformations:** Per-person identification, eye-region landmark overlays, blink-event annotations, qualitative labels, and selected representative frames.

**Attribution:** Tao-HUST MPEblink2.0 dataset, its official repository and DOI, and the associated Zeng et al. research publication.

**Repository distribution:** Included with identification of the publisher's declared Apache-2.0 dataset license, source attribution, and applicable modification notices.

---

## 4. Synthetic Integration Images

Five OpenAI-generated facial images are used as integration-test fixtures for gaze-direction processing.

**Source:** OpenAI/ChatGPT-generated synthetic images.

**Applicable terms:**

https://openai.com/policies/terms-of-use/

**Directory:**

`validation/integration/test_data/images/`

| File | Test condition |
|---|---|
| `gaze_center.png` | Central gaze |
| `gaze_down.png` | Downward gaze |
| `gaze_left.png` | Leftward gaze |
| `gaze_right.png` | Rightward gaze |
| `gaze_up.png` | Upward gaze |

These images are generated synthetic inputs rather than samples from MPIIFaceGaze or another photographic benchmark.

They support component and integration execution tests and are not identified as independent benchmark ground truth.

**Repository distribution:** Included.

---

## 5. Synthetic Validation Input Media

The synthetic validation input inventory contains **109 files**, comprising 91 PNG images and 18 MP4 videos.

**Base directory:**

`validation/synthetic_media/`

### 5.1 Input Inventory by Component

| Directory | File selection | Count | Source |
|---|---|---:|---|
| `face_detection_media/` | `01.png`–`10.png` | 10 | OpenAI-generated |
| `face_tracking_media/` | Five component-specific MP4 files | 5 | Blender-based |
| `face_landmarks_media/` | `01.png`–`10.png` | 10 | OpenAI-generated |
| `head_pose_media/` | `01.png`–`13.png` | 13 | OpenAI-generated |
| `eye_openness_and_blink_media/` | Five component-specific MP4 files | 5 | Blender-based |
| `gaze_estimation_media/` | `01.png`–`12.png` | 12 | OpenAI-generated |
| `face_regions_media/` | `01.png`–`10.png` | 10 | OpenAI-generated |
| `face_quality_media/` | `01.png`–`12.png` | 12 | OpenAI-generated |
| `mouth_openness_media/` | `01.png`–`12.png` | 12 | OpenAI-generated |
| `mouth_movement_velocity_media/` | Eight component-specific MP4 files | 8 | Blender-based |
| `emotion_recognition_media/` | `01.png`–`12.png` | 12 | OpenAI-generated |
| **Total** | | **109** | |

The PNG ranges are inclusive and denote zero-padded file sequences.

### 5.2 Face Tracking Synthetic Videos

**Directory:**

`validation/synthetic_media/face_tracking_media/`

| File | Condition |
|---|---|
| `01_single_face.mp4` | Single-face tracking |
| `02_multiple_faces.mp4` | Multiple-face tracking |
| `03_face_occlusion.mp4` | Temporary face occlusion |
| `04_enter_exit.mp4` | Entry and exit of tracked faces |
| `05_crossing_faces.mp4` | Crossing face trajectories |

Original source model:

Blender Studio & Community — Human Base Meshes v1.4.1, Body Male – Realistic.

Original-model license: CC0 1.0 Universal.

### 5.3 Eye Openness and Blink Synthetic Videos

**Directory:**

`validation/synthetic_media/eye_openness_and_blink_media/`

| File | Condition |
|---|---|
| `01_normal_blink.mp4` | Normal blinking |
| `02_fast_blink.mp4` | Fast blinking |
| `03_prolonged_closure.mp4` | Prolonged eye closure |
| `04_multi_person_blink.mp4` | Multiple-person blinking |
| `05_head_motion_blink.mp4` | Blinking with head movement |

Original source model:

Blender Studio & Community — Human Base Meshes v1.4.1, Body Male – Realistic.

Original-model license: CC0 1.0 Universal.

### 5.4 Mouth-Movement Velocity Synthetic Videos

**Directory:**

`validation/synthetic_media/mouth_movement_velocity_media/`

| File | Condition |
|---|---|
| `01_Static_Closed_Mouth.mp4` | Static closed mouth |
| `02_Slow_Opening_&_Closing.mp4` | Slow opening and closing |
| `03_Fast_Opening_&_Closing.mp4` | Fast opening and closing |
| `04_Repeated_Mouth_Movements.mp4` | Repeated mouth movements |
| `05_Opening_with_Pause.mp4` | Opening followed by a pause |
| `06_Speech_Like_Mouth_Motion.mp4` | Speech-like lip movement |
| `07_Temporary_Mouth_Occlusion.mp4` | Temporary mouth occlusion |
| `08_Multiple_Faces_with_Independent_Motion.mp4` | Independent mouth movement across multiple faces |

Original source model:

Blender Studio & Community — Human Base Meshes v1.4.1, Body Male – Realistic.

Original-model license: CC0 1.0 Universal.

### 5.5 Source Documentation

The input provenance register is:

`validation/synthetic_media/Documenting_synthetic_media_sources.xlsx`

It documents the source classification, media type, generation or modification procedure, and applicable source terms.

**Repository distribution:** All 109 synthetic input files are included.

---

## 6. Synthetic Demonstration Output Media

PhysioTrack generates annotated images, processed videos, segmentation maps, tracking visualizations, qualitative examples, and contact sheets from the synthetic validation inputs.

The reviewed Git inventory includes **211 synthetic image and video output files**.

| Component | Output directory | Image/video count |
|---|---|---:|
| Emotion Recognition | `validation/emotion_recognition/results/synthetic_demo/` | 13 |
| Eye Openness and Blink | `validation/eye_openness_and_blink/results/synthetic_demo/` | 11 |
| Eye Openness and Blink Accuracy | `validation/eye_openness_and_blink/results/synthetic_demo_accuracy/` | 11 |
| Face Detection | `validation/face_detection/results/synthetic_demo/` | 10 |
| Face Landmarks | `validation/face_landmarks/results/synthetic_demo/` | 11 |
| Face Quality | `validation/face_quality/results/synthetic_demo/` | 13 |
| Face Regions | `validation/face_regions/results/synthetic_demo/` | 41 |
| Face Tracking | `validation/face_tracking/results/synthetic_demo/` | 20 |
| Gaze Estimation | `validation/gaze_estimation/results/synthetic_demo/` | 13 |
| Head Pose | `validation/head_pose/results/synthetic_demo/` | 14 |
| Mouth Movement Velocity | `validation/mouth_movement_velocity/results/synthetic_demo/` | 41 |
| Mouth Openness | `validation/mouth_openness/results/synthetic_demo/` | 13 |
| **Total** | | **211** |

The counts refer to image and video outputs only. Associated numerical CSV, JSON, and TXT files are retained separately.

### 6.1 Output Provenance

The synthetic demonstration outputs retain the provenance classification of their source inputs:

- OpenAI-generated facial images produce project-annotated synthetic figures.
- Blender-based animated videos produce project-generated tracking, eye-blink, and mouth-motion visualizations.
- Measurement-based synthetic plots are project-generated analytical figures.

These outputs illustrate implemented component behavior rather than independent benchmark accuracy.

### 6.2 Synthetic Blink Accuracy

The `synthetic_demo_accuracy` directory contains supplementary synthetic event-matching reports, annotated examples, and accuracy summaries.

Its results describe performance under the specified synthetic conditions and remain distinct from MPEblink2.0 benchmark evaluation results.

**Repository distribution:** All 211 synthetic demonstration image and video outputs are included.

---

## 7. Pexels Integration Media

The integration validation uses 16 Pexels photographs and three Pexels videos.

The source photographs and videos are not distributed in the repository. Their original source pages, creator attribution, local filenames, and expected directories are provided for reproduction.

**Applicable license:** Pexels License.

https://www.pexels.com/license/

### 7.1 Integration Photographs

**Local directory:**

`validation/integration/test_data/images/`

| Local filename | Photographer | Original source |
|---|---|---|
| `emotion_angry.jpg` | Anna Tarazevich | https://www.pexels.com/photo/an-angry-woman-with-a-black-top-8066653/ |
| `emotion_disgust.jpg` | Pavel Danilyuk | https://www.pexels.com/photo/a-woman-in-red-long-sleeve-shirt-8638020/ |
| `emotion_fear.jpg` | Engin Akyurt | https://www.pexels.com/photo/close-up-photo-of-shocked-woman-6605132/ |
| `emotion_happy.jpg` | Tima Miroshnichenko | https://www.pexels.com/photo/smiling-elderly-woman-in-white-t-shirt-8727573/ |
| `emotion_neutral.jpg` | Liam Anderson | https://www.pexels.com/photo/shallow-focus-photography-of-woman-1458332/ |
| `emotion_sad.jpg` | Alena Darmel | https://www.pexels.com/photo/close-up-photo-of-a-woman-crying-6643007/ |
| `emotion_surprise.jpg` | Anna Tarazevich | https://www.pexels.com/photo/close-up-photo-of-a-woman-s-facial-expression-8066636/ |
| `eyes_1.jpg` | BYB BYB | https://www.pexels.com/photo/portrait-of-a-young-woman-smiling-with-her-eyes-shut-19999458/ |
| `eyes_2.jpg` | Andrea Piacquadio | https://www.pexels.com/photo/close-up-photo-of-shocked-woman-with-brown-curly-hair-3764398/ |
| `mouth_1.jpg` | Andrea Piacquadio | https://www.pexels.com/photo/close-up-photo-of-shocked-woman-with-brown-curly-hair-3764398/ |
| `mouth_2.jpg` | Pavel Danilyuk | https://www.pexels.com/photo/a-woman-in-red-long-sleeve-shirt-8637956/ |
| `frontal_1.jpg` | Christoph Sixt | https://www.pexels.com/photo/professional-portrait-of-a-smiling-man-in-business-attire-29856511/ |
| `frontal_2.jpg` | Andrea Piacquadio | https://www.pexels.com/photo/man-in-white-polo-shirt-smiling-3785110/ |
| `pose_1.jpg` | Laura Beauty | https://www.pexels.com/photo/ensaio-masculino-28293335/ |
| `multi_1.jpg` | Multi Motion Film Works | https://www.pexels.com/photo/smiling-group-in-white-shirts-17938967/ |
| `multi_2.jpg` | SHVETS production | https://www.pexels.com/photo/close-up-shot-of-two-men-9775652/ |

### 7.2 Integration Videos

| Repository-relative local filename | Creator | Original source |
|---|---|---|
| `validation/integration/test_data/multi_person/multi_person2.mp4` | Tima Miroshnichenko | https://www.pexels.com/video/women-wearing-suit-while-sitting-together-7203425/ |
| `validation/integration/test_data/single_person/ravdess_single_person.mp4` | KoolShooters | https://www.pexels.com/video/man-blinking-at-the-camera-6620642/ |
| `validation/integration/test_data/whole_project/ravdess_whole_project.mp4` | Tima Miroshnichenko | https://www.pexels.com/video/woman-recording-a-video-4962731/ |

The filenames `ravdess_single_person.mp4` and `ravdess_whole_project.mp4` are local integration-test identifiers.

Their recorded audiovisual sources are Pexels, not RAVDESS. These fixtures are therefore attributed to their identified Pexels creators and are not classified as RAVDESS-licensed material.

### 7.3 Source Register and Reproduction

The integration media register is:

`validation/integration/test_data/Media_Documentation_Table.xlsx`

It maps the original Pexels sources to the local filenames used by the validation framework.

For reproduction, researchers should obtain the relevant media directly from the original Pexels source pages and place the downloaded files under the documented local names and directory paths.

The integration validation scripts and recorded numerical results remain in the repository.

**Repository distribution:** The 19 original Pexels integration image and video files are excluded.

---

## 8. Pexels-Derived Robustness Media

The controlled robustness evaluation uses a single Pexels photograph as its fixed reference image.

### 8.1 Original Source

**Reference fixture:**

`validation/integration/test_data/images/frontal_1.jpg`

**Photographer:** Christoph Sixt.

**Source title:** *Professional Portrait of a Smiling Man in Business Attire.*

**Original source:**

https://www.pexels.com/photo/professional-portrait-of-a-smiling-man-in-business-attire-29856511/

**Applicable license:** Pexels License.

The source image has the recorded dimensions of 4160 × 6240 pixels.

The SHA-256 value recorded by the robustness generator is:

`31bb53b9a1b8c0d2ed1cf667651363e14e9e4071deb23fbd4f0697072e3896ff`

### 8.2 Controlled Transformations

The generator produces seven deterministic image conditions from the reference photograph.

| Condition | Transformation |
|---|---|
| Baseline | Unmodified reference image |
| Dim lighting | Global intensity factor 0.35 |
| Overexposed lighting | Linear intensity transformation, alpha 1.60 and beta 55.0 |
| Gaussian blur | Kernel size 31, sigma 7.0 |
| Motion blur | Horizontal blur kernel size 31 |
| Small face | Original scene scaled to 35% and centered |
| Partial occlusion | Normalized lower-face rectangle, x: 0.36–0.64 and y: 0.44–0.53 |

Generator:

`validation/robustness/robustness_dataset_generator.py`

### 8.3 Generated Controlled Inputs

**Directory:**

`validation/robustness/test_data/generated_controlled/`

| File |
|---|
| `01_baseline.png` |
| `02_dim_lighting.png` |
| `03_overexposed_lighting.png` |
| `04_gaussian_blur.png` |
| `05_motion_blur.png` |
| `06_small_face.png` |
| `07_partial_occlusion.png` |

### 8.4 Annotated Qualitative Outputs

**Directory:**

`validation/robustness/results/qualitative/annotated_images/`

| File |
|---|
| `01_baseline_annotated.png` |
| `02_dim_lighting_annotated.png` |
| `03_overexposed_lighting_annotated.png` |
| `04_gaussian_blur_annotated.png` |
| `05_motion_blur_annotated.png` |
| `06_small_face_annotated.png` |
| `07_partial_occlusion_annotated.png` |

### 8.5 Combined Figure

`validation/robustness/results/figures/robustness_qualitative_examples.png`

### 8.6 Distribution and Reproduction

The robustness photographic inventory contains 15 files: seven controlled inputs, seven annotated outputs, and one combined qualitative figure.

All are derived from the Pexels source photograph documented in §8.1.

They follow the same repository-distribution policy as the original Pexels integration media.

For reproduction, the source photograph should be downloaded from Pexels using its documented local filename. The robustness generator then constructs the seven controlled conditions from the accepted input.

Associated CSV/JSON measurements, quantitative figures, and evaluation scripts remain in the repository.

**Repository distribution:** All 15 Pexels-derived robustness image files are excluded.

---

## 9. Restricted Benchmark Visual Media

The following visual outputs are retained locally and excluded from repository distribution. The source terms listed below explain the provenance and licensing or access restrictions of the underlying benchmark data; exclusion of an output from this repository is a repository-distribution decision and does not by itself determine whether a limited example may be reproduced in a scholarly thesis.

### 9.1 WIDER FACE — 11 Files

- `validation/face_detection/results/figures/wider_face_qualitative_examples.png`
- `validation/face_detection/results/qualitative/annotated_images/`

**Dataset:** WIDER FACE, introduced by Yang et al. (2016).

**Official dataset and licensing sources:**

https://shuoyang1213.me/WIDERFACE/

https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html

https://huggingface.co/datasets/CUHK-CSE/wider_face

**Declared license:** Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**). Official license text: https://creativecommons.org/licenses/by-nc-nd/4.0/. The dataset publisher's record identifies the CC BY-NC-ND license; this does not grant permission to redistribute adapted images with newly drawn boxes, landmarks, or other overlays under that license. Repository distribution of selected original images, if any, would require adherence to the license and relevant source rights.

**Associated publication:** Yang, S., Luo, P., Loy, C.-C., & Tang, X. (2016). WIDER FACE: A Face Detection Benchmark. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 5525–5533.

**Repository distribution:** The 11 benchmark-derived qualitative visual files listed above are excluded. No original or annotated WIDER FACE photographs are included in the current validation-media selection.

### 9.2 300-W — Face Landmarks — 9 Files

- `validation/face_landmarks/results/figures/300w_landmark_qualitative_examples.png`
- `validation/face_landmarks/results/qualitative/annotated_images/`

**Dataset:** 300 Faces In-the-Wild (300-W).

**Official dataset source and stated usage conditions:**

https://ibug.doc.ic.ac.uk/resources/300-W/

**Declared restrictions:** The dataset provider states that the data are supplied **for research purposes only**, and that commercial use is not allowed. The public research-access statement is not treated here as permission to redistribute the underlying images or annotated derivatives through a public GitHub repository. The nine qualitative visual outputs are therefore excluded.

**Associated publications:** Sagonas et al. (2013), *300 Faces in-the-Wild Challenge: The First Facial Landmark Localization Challenge*; and Sagonas et al. (2016), *300 Faces In-The-Wild Challenge: Database and Results*.

### 9.3 300-W — Face Quality — 28 Files

- `validation/face_quality/results/figures/face_quality_qualitative_examples.png`
- `validation/face_quality/results/qualitative/sample_*.png`

**Dataset:** The same 300-W benchmark used in §9.2, with the same official source and research-only usage conditions:

https://ibug.doc.ic.ac.uk/resources/300-W/

**Declared restrictions:** Research use only; commercial use is not allowed. The original 300-W images and the visual outputs derived from them are excluded from the public repository under the selected redistribution policy.

**Associated publications:** Sagonas et al. (2013, 2016), cited in §9.2 and §14.

### 9.4 AFLW — Head Pose — 9 Files

- `validation/head_pose/results/figures/aflw_head_pose_qualitative_examples.png`
- `validation/head_pose/results/qualitative/annotated_images/`

**Dataset:** Annotated Facial Landmarks in the Wild (AFLW), Graz University of Technology.

**Official dataset and license agreement:**

https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw

**Declared restrictions:** Non-commercial research only. The dataset agreement expressly restricts further copying, publication, and distribution of the database, apart from specified internal organizational use. Images sourced from Flickr may also be subject to the original photographers' rights.

**Associated publication:** Köstinger, M., Wohlhart, P., Roth, P. M., & Bischof, H. (2011). Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. *Proceedings of the First IEEE International Workshop on Benchmarking Facial Image Analysis Technologies (BeFIT)*.

**Repository distribution:** The nine AFLW-derived qualitative visual files are excluded.

### 9.5 CelebAMask-HQ — Face Regions — 9 Files

- `validation/face_regions/results/figures/celebamaskhq_qualitative_examples.png`
- `validation/face_regions/results/qualitative/annotated_images/`

**Dataset:** CelebAMask-HQ.

**Official dataset page and terms:**

https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

https://github.com/switchablenorms/CelebAMask-HQ

**Declared restrictions:** The official dataset agreement permits **non-commercial research purposes only** and expressly prohibits further copying, publication, or distribution of portions of the dataset, other than specified internal organizational copies. These terms apply to the underlying dataset and cannot be bypassed by adding annotations to its images.

**Associated publication:** Lee, C.-H., Liu, Z., Wu, L., & Luo, P. (2020). MaskGAN: Towards Diverse and Interactive Facial Image Manipulation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 5549–5558.

**Repository distribution:** The nine CelebAMask-HQ-derived qualitative visual files are excluded.

### 9.6 CAER-S — Emotion Recognition — 11 Files

- `validation/emotion_recognition/results/figures/caers_emotion_qualitative_examples.png`
- `validation/emotion_recognition/results/qualitative/annotated_images/`

**Dataset:** Context-Aware Emotion Recognition – Static (CAER-S), a static-image subset associated with the CAER benchmark.

**Official dataset page and access statement:**

https://caer-dataset.github.io/

**Declared restrictions:** The dataset's official page makes CAER and CAER-S available for **research purposes** and states that copyright in the source videos remains with their original owners. The website's own publication terms are not treated as a redistribution license for the underlying video frames. No blanket authorization to republish those frames or annotated derivatives in the public repository is assumed.

**Associated publication:** Lee, J., Kim, S., Kim, S., Park, J., & Sohn, K. (2019). Context-Aware Emotion Recognition Networks. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10143–10152.

**Repository distribution:** The 11 CAER-S-derived qualitative visual files are excluded.

### 9.7 ECCV 2016 — Face Tracking — 8 Files

- `validation/face_tracking/results/figures/eccv16_tracking_qualitative_examples.png`
- `validation/face_tracking/results/qualitative/annotated_videos/`
- `validation/face_tracking/results/qualitative/representative_frames/`

**Benchmark:** Music-video multi-face tracking sequences associated with the ECCV 2016 study by Zhang et al.

**Official scholarly source:**

https://doi.org/10.1007/978-3-319-46454-1_26

**Source-media status:** The study describes a benchmark of eight music videos sourced from YouTube. A general dataset-wide license authorizing public redistribution of the underlying music videos, extracted frames, or annotated video copies was not established from the study's publication record. Publication of the paper and availability of a research benchmark do not themselves confer rights to redistribute third-party audiovisual footage.

**Associated publication:** Zhang, S., Gong, Y., Huang, J.-B., Lim, J., Wang, J., Ahuja, N., & Yang, M.-H. (2016). Tracking Persons-of-Interest via Adaptive Discriminative Features. *Computer Vision – ECCV 2016*, Lecture Notes in Computer Science, 9909, 415–433.

**Repository distribution:** The eight tracking benchmark-derived qualitative visual files are excluded pending appropriate permissions; numerical tracking results and evaluation code remain included.

### 9.8 Scope of the Exclusions

These categories account for 85 restricted benchmark-derived visual files.

The exclusion applies to original or modified photographic and video representations, annotated examples, selected frames, and composite figures containing the restricted media.

It does not apply to numerical measurements, benchmark prediction records, aggregate statistics, or independent analytical plots that do not reproduce the source photographs or videos.

Proprietary iStock media are likewise outside the selected validation-media release inventory.

---

## 10. Project-Generated Quantitative and Deterministic Figures

The repository contains 37 independently generated analytical or deterministic figures.

These figures present numerical evaluation results, error distributions, benchmark metrics, calibration measurements, or deterministic temporal sequences.

Their scientific provenance remains linked to the relevant validation datasets, but they are distinguished from reproductions of third-party photographic and video content.

| Repository-relative file | Source of depicted data |
|---|---|
| `validation/emotion_recognition/results/figures/caers_emotion_confusion_matrix.png` | CAER-S evaluation metrics |
| `validation/emotion_recognition/results/figures/caers_emotion_per_class_metrics.png` | CAER-S evaluation metrics |
| `validation/emotion_recognition/results/figures/caers_emotion_prediction_distribution.png` | CAER-S evaluation metrics |
| `validation/eye_openness_and_blink/results/figures/mpeblink_eye_blink_metrics.png` | MPEblink2.0 evaluation metrics |
| `validation/eye_openness_and_blink/results/figures/mpeblink_validation_f1_calibration.png` | MPEblink2.0 calibration metrics |
| `validation/face_detection/results/figures/wider_face_precision_recall.png` | WIDER FACE evaluation metrics |
| `validation/face_landmarks/results/figures/300w_landmark_ced.png` | 300-W evaluation metrics |
| `validation/face_quality/results/figures/face_quality_brightness_sensitivity.png` | 300-W-derived face-quality metrics |
| `validation/face_quality/results/figures/face_quality_face_area_ratio_sensitivity.png` | 300-W-derived face-quality metrics |
| `validation/face_quality/results/figures/face_quality_sharpness_sensitivity.png` | 300-W-derived face-quality metrics |
| `validation/face_regions/results/figures/celebamaskhq_normalized_confusion_matrix.png` | CelebAMask-HQ evaluation metrics |
| `validation/face_regions/results/figures/celebamaskhq_per_class_metrics.png` | CelebAMask-HQ evaluation metrics |
| `validation/face_tracking/results/figures/eccv16_tracking_metrics.png` | ECCV 2016 tracking metrics |
| `validation/gaze_estimation/results/figures/mpiifacegaze_ethxgaze_angular_error_distribution.png` | MPIIFaceGaze evaluation metrics |
| `validation/gaze_estimation/results/figures/mpiifacegaze_ethxgaze_per_person_mae.png` | MPIIFaceGaze evaluation metrics |
| `validation/head_pose/results/figures/aflw_head_pose_error_distribution.png` | AFLW evaluation metrics |
| `validation/head_pose/results/figures/aflw_head_pose_error_metrics.png` | AFLW evaluation metrics |
| `validation/mouth_movement_velocity/results/figures/felt_ravdess_mouth_movement_agreement.png` | FELT/RAVDESS mouth-movement metrics |
| `validation/mouth_movement_velocity/results/figures/felt_ravdess_mouth_movement_velocity_error_distribution.png` | FELT/RAVDESS mouth-movement metrics |
| `validation/mouth_movement_velocity/results/figures/felt_ravdess_mouth_movement_velocity_per_actor.png` | FELT/RAVDESS per-actor metrics |
| `validation/mouth_movement_velocity/results/figures/felt_ravdess_mouth_velocity_agreement.png` | FELT/RAVDESS mouth-velocity metrics |
| `validation/mouth_openness/results/figures/felt_ravdess_mouth_openness_agreement.png` | FELT/RAVDESS mouth-openness metrics |
| `validation/mouth_openness/results/figures/felt_ravdess_mouth_openness_error_distribution.png` | FELT/RAVDESS mouth-openness metrics |
| `validation/mouth_openness/results/figures/felt_ravdess_mouth_openness_per_actor.png` | FELT/RAVDESS per-actor metrics |
| `validation/robustness/results/figures/robustness_availability.png` | Pexels-derived controlled robustness metrics |
| `validation/robustness/results/figures/robustness_detector_confidence.png` | Pexels-derived controlled robustness metrics |
| `validation/robustness/results/figures/robustness_quality_sharpness.png` | Pexels-derived controlled robustness metrics |
| `validation/temporal_aggregation/results/figures/temporal_aggregation_numeric_error.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/figures/temporal_aggregation_qualitative_contact_sheet.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/figures/temporal_aggregation_validation_coverage.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_01_window_growth.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_02_sliding_window.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_03_person_isolation.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_04_reset_behavior.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_05_blink_events.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_06_dominant_emotion.png` | Deterministic temporal results |
| `validation/temporal_aggregation/results/qualitative/case_07_quality_brightness.png` | Deterministic temporal results |

No new license for the project-generated figures is assigned solely on the basis of the originating benchmark dataset.

**Repository distribution:** All 37 figures are included.

---

## 11. Existing Upstream Documentation Images

Three visual files are inherited from the original PhysioTrack documentation:

- `docs/images/full_inference_overlay.png`
- `docs/images/rppg_heartrate_overlay.png`
- `docs/images/vr_human_analysis.jpg`

These files are not newly introduced face-analysis validation assets.

Their provenance and applicable distribution conditions are governed by the original upstream documentation and associated licensing records.

The repository's `THIRD_PARTY_LICENSES.md` remains separate from this validation-media inventory.

---

## 12. Media Distribution Summary

| Category | Count | Repository status | Provenance or licensing basis |
|---|---:|---|---|
| MPIIFaceGaze qualitative figures | 9 | Included | CC BY-NC-SA 4.0 |
| RAVDESS/FELT qualitative figures | 18 | Included | RAVDESS CC BY-NC-SA 4.0 adaptations |
| MPEblink2.0 qualitative images and videos | 17 | Included | Publisher-declared Apache-2.0 dataset license and source attribution |
| Synthetic integration images | 5 | Included | OpenAI-generated |
| Synthetic validation input media | 109 | Included | OpenAI-generated images and Blender-derived animations |
| Synthetic demonstration output media | 211 | Included | Project-generated synthetic visualizations |
| Quantitative and deterministic figures | 37 | Included | Project-generated analytical visualizations |
| Existing upstream documentation images | 3 | Included | Upstream documentation and applicable source conditions |
| Restricted benchmark visual media | 85 | Excluded | Benchmark-specific source restrictions |
| Pexels integration photographs and videos | 19 | Excluded | Original Pexels media |
| Pexels-derived robustness images | 15 | Excluded | Controlled derivatives of a Pexels photograph |

The selected repository media inventory comprises **409 included image and video files**.

The excluded validation-media inventory comprises **119 image and video files**.

---

## 13. Reproducibility and Source Attribution

### 13.1 Benchmark Datasets

Benchmark data are acquired independently under their applicable source licenses and access conditions.

The local project layout places the relevant datasets in a sibling `datasets/` directory where required by the validation scripts.

Benchmark evaluations retain numerical measurements, per-sequence results, calibration outputs, statistical summaries, and analytical figures independently of restricted qualitative media.

### 13.2 Integration Tests

Integration-test photographs and videos are obtained from the original Pexels source pages identified in §7.

The documented local filenames and directories preserve the expected test fixture structure.

The source register provides the relationship between each original asset and its integration-test filename.

### 13.3 Robustness Tests

The controlled robustness protocol uses the documented Pexels reference photograph and deterministic transformation parameters.

The source hash provides a reproducibility check for the accepted reference image.

Generated photographic inputs and annotated qualitative outputs remain outside the repository, while the corresponding generation procedure, numerical results, and analytical figures are retained.

### 13.4 Synthetic Demonstrations

The synthetic source register identifies the generation and processing provenance of each synthetic media category.

Synthetic inputs and demonstration outputs are included to facilitate examination of the implemented component behavior.

Synthetic demonstrations are distinguished from independent benchmark accuracy evaluations.

### 13.5 Attribution and License Scope

External datasets, source photographs, video footage, generated media, and project-produced analytical figures are documented according to their individual provenance.

Academic citation and license attribution support scientific transparency but do not alter independently applicable rights.

Repository distribution, local research processing, and publication of material in a thesis or other scholarly work are distinct uses subject to their applicable source conditions.

---

## 14. References and Official Resources

1. Bulling, A. (2023). *MPIIFaceGaze* (Version 1.0) [Data set]. DaRUS. https://doi.org/10.18419/DARUS-3240
2. Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2019). MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(1), 162–175. https://doi.org/10.1109/TPAMI.2017.2778103
3. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1188976
4. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
5. Liao, Z., Livingstone, S. R., & Russo, F. A. (2024). *RAVDESS Facial expression and landmark tracking* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13243600
6. Zeng, W., Xiao, Y., Wei, S., Gan, J., Zhang, X., Cao, Z., Fang, Z., & Zhou, J. T. (2023). Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 13854–13863.
7. Tao-HUST. *MPEblink2.0* [Data set]. Hugging Face. https://doi.org/10.57967/hf/9376
8. Tao-HUST. *MPEblink2.0 Dataset Repository*. https://huggingface.co/datasets/Tao-HUST/MPEblink2.0
9. Zeng, W. (2023). *The MPEblink Dataset* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7754768
10. Creative Commons. *Attribution–NonCommercial–ShareAlike 4.0 International*. https://creativecommons.org/licenses/by-nc-sa/4.0/
11. Creative Commons. *Attribution–NonCommercial–ShareAlike 4.0 International: Legal Code*. https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
12. Apache Software Foundation. *Apache License, Version 2.0*. https://www.apache.org/licenses/LICENSE-2.0
13. Creative Commons. *CC0 1.0 Universal*. https://creativecommons.org/publicdomain/zero/1.0/
14. Blender Foundation. *Demo Files: Human Base Meshes*. https://www.blender.org/download/demo-files/#assets
15. OpenAI. *Terms of Use*. https://openai.com/policies/terms-of-use/
16. Pexels. *Pexels License*. https://www.pexels.com/license/
17. Pexels. *Terms of Service*. https://www.pexels.com/terms-of-service/
18. Yang, S., Luo, P., Loy, C.-C., & Tang, X. (2016). WIDER FACE: A Face Detection Benchmark. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 5525–5533. https://openaccess.thecvf.com/content_cvpr_2016/html/Yang_WIDER_FACE_A_CVPR_2016_paper.html
19. The Chinese University of Hong Kong. *WIDER FACE: A Face Detection Benchmark* [Dataset and terms]. https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html
20. CUHK-CSE. *WIDER FACE* [Dataset repository; CC BY-NC-ND 4.0]. https://huggingface.co/datasets/CUHK-CSE/wider_face
21. Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013). 300 Faces in-the-Wild Challenge: The First Facial Landmark Localization Challenge. *Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW)*. https://ibug.doc.ic.ac.uk/resources/300-W/
22. Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016). 300 Faces In-The-Wild Challenge: Database and Results. *Image and Vision Computing*, 47, 3–18. https://doi.org/10.1016/j.imavis.2016.01.002
23. Imperial College London, i·bug. *300 Faces In-the-Wild Challenge (300-W): Dataset and Research-Use Conditions*. https://ibug.doc.ic.ac.uk/resources/300-W/
24. Köstinger, M., Wohlhart, P., Roth, P. M., & Bischof, H. (2011). Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. *Proceedings of the First IEEE International Workshop on Benchmarking Facial Image Analysis Technologies (BeFIT)*. https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw
25. Graz University of Technology. *Annotated Facial Landmarks in the Wild (AFLW): Dataset and License Agreement*. https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw
26. Lee, C.-H., Liu, Z., Wu, L., & Luo, P. (2020). MaskGAN: Towards Diverse and Interactive Facial Image Manipulation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 5549–5558. https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_MaskGAN_Towards_Diverse_and_Interactive_Facial_Image_Manipulation_CVPR_2020_paper.html
27. The Chinese University of Hong Kong. *CelebAMask-HQ Dataset and Agreement*. https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
28. Lee, J., Kim, S., Kim, S., Park, J., & Sohn, K. (2019). Context-Aware Emotion Recognition Networks. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10143–10152. https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Context-Aware_Emotion_Recognition_Networks_ICCV_2019_paper.html
29. CAER Dataset Project. *Context-Aware Emotion Recognition Networks (CAER and CAER-S): Research Access and Source Copyright Notice*. https://caer-dataset.github.io/
30. Zhang, S., Gong, Y., Huang, J.-B., Lim, J., Wang, J., Ahuja, N., & Yang, M.-H. (2016). Tracking Persons-of-Interest via Adaptive Discriminative Features. *Computer Vision – ECCV 2016*, Lecture Notes in Computer Science, 9909, 415–433. https://doi.org/10.1007/978-3-319-46454-1_26
