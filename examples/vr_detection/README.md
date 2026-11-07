# VR detection example

This example compares three detection questions on the same synthetic VR training
scene:

| Detector | Output class | What its boxes mean |
| --- | --- | --- |
| `physiotrack.Detection.VR` | `VR-head` | the headset/head region |
| `physiotrack.Detection.VRStudent` | `VR-person` | the full body of a person using VR |
| `physiotrack.Detection.Person` | `person` | any person, with or without a headset |

From the repository root:

```bash
python examples/vr_detection/detect_vr_people.py
```

CPU is the default. Use CUDA, a custom input, or only selected views like this:

```bash
python examples/vr_detection/detect_vr_people.py --device cuda
python examples/vr_detection/detect_vr_people.py --input path/to/vr_scene.jpg
python examples/vr_detection/detect_vr_people.py --detectors vr_person person
```

### Select model sizes

Medium checkpoints are used by default. Large checkpoints are published for the
VR-person and generic-person detectors, but **not for VR-head**, so `largest` uses the
largest checkpoint published for each detector:

```bash
# medium VR-head, large VR-person, large generic person
python examples/vr_detection/detect_vr_people.py --model-size largest
```

The script writes one annotated PNG per detector, a vertically stacked
`comparison.png`, and `summary.json` under `examples/vr_detection/results/`. When both
person views are selected, it also writes the compact documentation preview
`comparison_person_vrperson.jpg`. The JSON records, per detector, the registry path
of the model, the class counts and the complete serialized `Result`; every image panel
shows the checkpoint filename and the detector's inference time.

![VR-person and generic-person comparison](../../docs/images/comparison_person_vrperson.jpg)

In this example run, the large VR-person model (top) retains two full people using
headsets, while the large generic-person model (bottom) retains twelve people. The
counts differ because the detectors answer different questions; they are not
directly comparable accuracy scores.

The displayed illustration is a copy stored at
[`docs/images/comparison_person_vrperson.jpg`](../../docs/images/comparison_person_vrperson.jpg).
Files under this example's `results/` directory remain ignored as reproducible run
artifacts. Recopy the generated JPEG deliberately when the documented result should
be updated.

The default image is reused from
[`examples/face_detection/data/vr/vr_training_lab.jpg`](../face_detection/data/vr/vr_training_lab.jpg),
so the project does not commit a duplicate. Its provenance, CC0 dedication, synthetic
person notice, and checksum are recorded in
[`examples/face_detection/data/MEDIA.yml`](../face_detection/data/MEDIA.yml).

## Interpretation

The counts answer different questions and should not be expected to match. For
example, a generic person detector may find people both with and without headsets,
while `VRStudent` returns only full VR-person boxes and `VR` returns smaller head
regions. The bundled scene has no ground-truth annotations, so it demonstrates API
composition and qualitative behavior—not precision, recall, or model accuracy.
