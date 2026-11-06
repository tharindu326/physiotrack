#!/usr/bin/env python3
"""Detect and track faces in the bundled video with the core Video pipeline.

Track IDs are temporary associations within one video.  This example performs no face
recognition and does not determine anyone's identity.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import physiotrack as pt


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data" / "students_face_tracking.mp4"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
FACE_MODELS = pt.Models.Detection.YOLO.FACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the annotated video, JSON results, and track CSV.",
    )
    parser.add_argument(
        "--model",
        choices=[m.name for m in FACE_MODELS],
        default="m_face",
        help="YOLO face checkpoint (default: m_face).",
    )
    parser.add_argument("--device", default="cpu", help="Face detector device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--tracker",
        choices=("ocsort", "bytetrack", "boosttrack"),
        default="ocsort",
        help="Tracking-by-detection backend (default: ocsort).",
    )
    parser.add_argument("--show", action="store_true", help="Show a live window; press q to stop.")
    return parser.parse_args()


def write_tracks_csv(results, csv_path: Path) -> set[int]:
    """Write one row per tracked face per frame; return the observed track ids."""
    unique_ids: set[int] = set()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds", "track_id",
                         "x1", "y1", "x2", "y2", "confidence", "class_id"])
        for frame in results:
            for inst in frame:
                if inst.id is None or inst.box is None:
                    continue
                unique_ids.add(inst.id)
                x1, y1, x2, y2 = (round(float(v), 3) for v in inst.box)
                writer.writerow([
                    frame.meta.frame_index,
                    round(frame.meta.timestamp, 6),
                    inst.id, x1, y1, x2, y2,
                    "" if inst.confidence is None else round(float(inst.confidence), 6),
                    inst.cls,
                ])
    return unique_ids


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--conf and --iou must be between 0 and 1")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input video does not exist: {input_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Face detection + tracking is plain composition: the face detector supplies
    # the boxes and the tracker associates them over time. The Video pipeline
    # runs both, draws the overlays, and exports one Result per frame.
    detector = pt.Face(
        model=FACE_MODELS[args.model],
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    tracker = pt.Tracker(pt.TrackerConfig(
        tracker_type=args.tracker,
        classes=[0],
        show_original_tracks=True,
        show_all_trails=True,
        enable_subject_lock=False,
    ))
    video = pt.Video(
        source=input_path,
        detector=detector,
        tracker=tracker,
        output_dir=output_dir,
        verbose=True,
        show=args.show,
    )

    output_video = output_dir / f"{input_path.stem}_tracked.mp4"
    output_json = output_dir / f"{input_path.stem}_result.json"
    results = video.run(output_video, output_json)

    tracks_csv = output_dir / f"{input_path.stem}_tracks.csv"
    unique_ids = write_tracks_csv(results, tracks_csv)

    print(f"\nProcessed frames: {len(results)}")
    print(f"Temporary IDs observed: {sorted(unique_ids)}")
    print(f"Annotated video:  {output_video}")
    print(f"Per-frame JSON:   {output_json}")
    print(f"Track CSV:        {tracks_csv}")
    print(
        "Note: track IDs are temporary within-video associations, not recognized "
        "identities. The synthetic clip has no ground-truth tracks, so this is a "
        "qualitative example rather than a tracking-accuracy measurement."
    )


if __name__ == "__main__":
    main()
