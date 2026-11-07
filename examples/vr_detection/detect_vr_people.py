#!/usr/bin/env python3
"""Compare VR-head, VR-person, and generic-person detection on one image.

The three detectors answer different questions, so their counts are not expected to
match: ``Detection.VR`` finds headsets, ``Detection.VRStudent`` finds whole people who
wear one, and ``Detection.Person`` finds everyone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import physiotrack as pt
from physiotrack.core.overlay import draw_info_panel
from physiotrack.core.predictor import load_image

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR.parent / "face_detection" / "data" / "vr" / "vr_training_lab.jpg"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
YOLO = pt.Models.Detection.YOLO
DETECTORS = {
    "vr_head": (pt.Detection.VR, "Where are the VR headsets?", (255, 170, 0)),
    "vr_person": (pt.Detection.VRStudent, "Which full people are using VR?", (190, 0, 255)),
    "person": (pt.Detection.Person, "Where are all people, with or without VR?", (0, 210, 0)),
}
# Checkpoints per size. Only a medium VR-head checkpoint is published, so "largest"
# means medium VR-head with large VR-person and generic-person models.
MODEL_SIZES = {
    "medium": {"vr_head": YOLO.VR.m_vr, "vr_person": YOLO.VRSTUDENT.m_vrstudent,
               "person": YOLO.PERSON.m_person},
    "largest": {"vr_head": YOLO.VR.m_vr, "vr_person": YOLO.VRSTUDENT.l_vrstudent,
                "person": YOLO.PERSON.l_person},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for annotated images and summary.json.")
    parser.add_argument("--detectors", nargs="+", choices=tuple(DETECTORS),
                        default=list(DETECTORS), help="Detector views to run (default: all).")
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES), default="medium",
                        help="Checkpoint size (default: medium).")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def stack_views(views: list[np.ndarray], width: int = 900) -> np.ndarray:
    """Resize annotated views to one width and stack them vertically."""
    width = min(width, min(v.shape[1] for v in views))
    return np.vstack([cv2.resize(v, (width, round(v.shape[0] * width / v.shape[1])))
                      for v in views])


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, views = [], {}
    for key in args.detectors:
        factory, question, color = DETECTORS[key]
        model = MODEL_SIZES[args.model_size][key]
        detector = factory(model=model, conf=args.conf, iou=args.iou, device=args.device)
        result = detector(image)
        counts = dict(Counter(inst.cls_name for inst in result))
        views[key] = draw_info_panel(result.plot(conf=True, color=color, thickness=3), [
            question,
            ", ".join(f"{label}: {n}" for label, n in counts.items()) or "No detections",
            f"Detector: {model.value}",
            f"Device: {args.device} | {detector.get_avg_inference_time():.1f} ms",
        ])
        cv2.imwrite(str(output_dir / f"{key}.png"), views[key])
        records.append({"detector": key, "model": pt.Models.path_of(model),
                        "class_counts": counts, "result": result.to_dict()})
        print(f"{key}: {len(result)} detection(s) {counts}")

    if len(views) > 1:
        cv2.imwrite(str(output_dir / "comparison.png"), stack_views(list(views.values())))
    if {"vr_person", "person"} <= views.keys():  # compact copy for docs/images
        cv2.imwrite(str(output_dir / "comparison_person_vrperson.jpg"),
                    stack_views([views["vr_person"], views["person"]]))

    (output_dir / "summary.json").write_text(json.dumps({
        "input": str(args.input), "model_size": args.model_size, "detectors": records,
        "note": ("The detectors find different regions or subject categories, and the "
                 "synthetic image has no ground truth: these are qualitative outputs."),
    }, indent=2), encoding="utf-8")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
