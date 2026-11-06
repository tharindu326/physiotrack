#!/usr/bin/env python3
"""Analyse every face in an image: orientation, face mesh, expression, gaze, quality.

Each face stage takes the faces of the previous step and adds one property, so the
pipeline is plain chaining. The eye / mouth / iris measures are computed from the face
mesh by the ``physiotrack.signals`` face functions.

Usage:
    python examples/face_analysis/analyze_image.py
    python examples/face_analysis/analyze_image.py --input photo.jpg --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import physiotrack as pt
from physiotrack.core.predictor import load_image

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR.parent / "face_detection" / "data" / "selfie" / "two_person_selfie.jpg"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or a CUDA index.")
    args = parser.parse_args()

    image = load_image(args.input)
    landmarks = pt.FaceLandmarks()
    faces = pt.Face(device=args.device)(image)
    for stage in (pt.FaceOrientation(device=args.device), landmarks,
                  pt.FaceExpression(device=args.device), pt.GazeEstimator(device=args.device),
                  pt.FaceQuality()):
        faces = stage(image, faces)
    landmarks.close()

    for index, face in enumerate(faces):
        print(f"face {index}  box {face.box.round().astype(int).tolist()}")
        o = face.orientation
        print(f"  head      yaw {o['yaw']:+.1f}  pitch {o['pitch']:+.1f}  roll {o['roll']:+.1f} deg")
        print(f"  expression {face.expression['label']} ({face.expression['confidence']:.2f})")
        print(f"  quality   brightness {face.quality['brightness']:.2f}  "
              f"sharpness {face.quality['sharpness']:.0f}")
        if face.keypoints is None:
            print("  no face mesh (face too small or turned away)")
            continue
        ear = pt.signals.eye_aspect_ratio(face)
        iris = pt.signals.iris_position(face)
        print(f"  eyes      EAR left {ear['left']:.3f}  right {ear['right']:.3f}")
        print(f"  mouth     MAR {pt.signals.mouth_aspect_ratio(face):.3f}")
        print(f"  iris      x {iris['x']:.2f}  y {iris['y']:+.2f} (eye widths)")
        print(f"  gaze      yaw {face.gaze['yaw']:+.1f}  pitch {face.gaze['pitch']:+.1f} deg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faces.save(args.output_dir / f"{args.input.stem}_faces.png")
    # Face meshes are 478 points per face; include_arrays keeps them in the JSON.
    faces.to_json(args.output_dir / f"{args.input.stem}_faces.json", include_arrays=True)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
