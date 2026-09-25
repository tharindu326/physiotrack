#!/usr/bin/env python3
"""Detect faces in the bundled example scenes and save inspectable results.

Run this file without arguments from any directory. Use ``--input`` to process a
different image or a directory tree containing images. Per image it saves the annotated
PNG and the ``Result`` as JSON; for the run, a CSV summary and ``run.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import physiotrack as pt
from physiotrack.core.overlay import draw_info_panel
from physiotrack.core.predictor import load_image

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FACE_MODELS = pt.Models.Detection.YOLO.FACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="One image or a directory searched recursively (default: bundled data).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for annotated images, JSON, CSV and run metadata.")
    parser.add_argument("--model", choices=[m.name for m in FACE_MODELS], default="m_face",
                        help="YOLO face checkpoint (default: m_face).")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    return parser.parse_args()


def image_paths(input_path: Path) -> tuple[Path, list[Path]]:
    """Return a common input root and a sorted list of the images below it."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return input_path.parent, [input_path]
    paths = sorted(p for p in input_path.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found below: {input_path}")
    return input_path, paths


def main() -> None:
    args = parse_args()
    root, paths = image_paths(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    (output_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    model = FACE_MODELS[args.model]
    detector = pt.Face(model=model, conf=args.conf, iou=args.iou, device=args.device)

    rows = []
    for path in paths:
        relative = path.relative_to(root)
        stem = "__".join(relative.with_suffix("").parts)
        result = detector(load_image(path))
        confidences = [inst.confidence for inst in result]

        annotated = draw_info_panel(result.plot(conf=True), [
            f"Faces detected: {len(result)}",
            f"Detector: {model.value}",
            f"Device: {args.device}",
        ])
        cv2.imwrite(str(output_dir / "annotated" / f"{stem}.png"), annotated)
        result.to_json(output_dir / "predictions" / f"{stem}.json")

        rows.append({
            "image": relative.as_posix(),
            "width": result.orig_img.shape[1],
            "height": result.orig_img.shape[0],
            "faces_detected": len(result),
            "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else "",
            "minimum_confidence": round(float(min(confidences)), 4) if confidences else "",
        })
        print(f"{relative}: {len(result)} face(s)")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "run.json").write_text(json.dumps({
        "physiotrack": pt.__version__,
        "model": pt.Models.path_of(model),
        "device": args.device,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "images": len(rows),
        "faces": sum(row["faces_detected"] for row in rows),
        "mean_inference_ms": detector.get_avg_inference_time(),
        "note": ("The bundled synthetic scenes have no ground-truth boxes; the counts are "
                 "qualitative example outputs, not detector-accuracy measurements."),
    }, indent=2), encoding="utf-8")

    print(f"\nAnnotated images: {output_dir / 'annotated'}")
    print(f"Per-image JSON:   {output_dir / 'predictions'}")
    print(f"CSV summary:      {csv_path}")


if __name__ == "__main__":
    main()
