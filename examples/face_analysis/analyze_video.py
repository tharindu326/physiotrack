#!/usr/bin/env python3
"""Face analysis over a video: tracked faces, their mesh and expression, and blinks.

The core ``Video`` pipeline detects and tracks the faces and runs the face stages on
every frame; the ``physiotrack.signals`` face functions then turn the per-frame results
into a per-face table, blink events, blink rates and mouth movement.

Usage:
    python examples/face_analysis/analyze_video.py
    python examples/face_analysis/analyze_video.py --input clip.mp4 --device cuda --gaze

To follow the faces of tracked *people* (e.g. alongside pose), pass a person detector
as ``detector`` and the face detector as ``face``: each face then carries its person's
track id.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import physiotrack as pt

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR.parent / "face_tracking" / "data" / "students_face_tracking.mp4"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or a CUDA index.")
    parser.add_argument("--gaze", action="store_true", help="Also estimate 3D gaze.")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    landmarks = pt.FaceLandmarks()
    stages = [pt.FaceOrientation(device=args.device), landmarks, pt.FaceExpression(device=args.device)]
    if args.gaze:
        stages.append(pt.GazeEstimator(device=args.device))

    video = pt.Video(
        source=args.input,
        detector=pt.Face(device=args.device),                       # faces are the subjects
        tracker=pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0],
                                            enable_subject_lock=False)),
        face_stages=stages,
        batch_size=args.batch_size,
        verbose=True,
    )
    stem = args.input.stem
    results = video.run(args.output_dir / f"{stem}_faces.mp4",
                        args.output_dir / f"{stem}_faces.json")
    landmarks.close()

    table = pt.signals.face_feature_sequence(results)
    table.to_csv(args.output_dir / f"{stem}_face_features.csv", index=False)
    windows = pt.signals.face_window_summary(results, window=5.0)
    windows.to_csv(args.output_dir / f"{stem}_face_windows.csv", index=False)

    print(f"\n{'face':>4} {'frames':>6} {'blinks':>6} {'blinks/min':>10} "
          f"{'mouth speed':>11} {'expression':>11}")
    for face_id, rows in table.groupby("detection_id"):
        if rows.ear.notna().sum() < 10:
            continue  # too few meshes (small or turned-away face) to measure blinks
        blinks = pt.signals.detect_blinks(results, detection_id=face_id)
        rate = pt.signals.blink_rate(results, detection_id=face_id)
        mouth = pt.signals.mouth_movement(results, detection_id=face_id)
        expression = rows.expression.mode().iat[0] if rows.expression.notna().any() else "-"
        print(f"{face_id:>4} {len(rows):>6} {len(blinks):>6} {rate:>10.1f} "
              f"{mouth.mar_velocity.median():>11.3f} {expression:>11}")
    print(f"\nPer-face table:  {args.output_dir / f'{stem}_face_features.csv'}")
    print(f"5-s summaries:   {args.output_dir / f'{stem}_face_windows.csv'}")
    print("Track ids are temporary within-video associations, not identities.")


if __name__ == "__main__":
    main()
