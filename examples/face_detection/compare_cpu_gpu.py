#!/usr/bin/env python3
"""Compare CPU and CUDA face detection on exactly the same image.

Warms up each device separately, times repeated end-to-end ``Face.predict`` calls
(synchronising CUDA so the timings include the GPU work), and checks whether the final
CPU and GPU boxes agree. Requires a CUDA-enabled PyTorch build and an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import physiotrack as pt
from physiotrack.core.boxes import box_iou
from physiotrack.core.overlay import draw_info_panel
from physiotrack.core.predictor import load_image

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data" / "pov" / "exercise_class_pov.jpg"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results" / "cpu_vs_gpu"
FACE_MODELS = pt.Models.Detection.YOLO.FACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for the CPU/GPU images and comparison.json.")
    parser.add_argument("--model", choices=[m.name for m in FACE_MODELS], default="m_face")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="Minimum box IoU for a CPU/GPU prediction match (default: 0.5).")
    return parser.parse_args()


def benchmark(detector, image, cuda: bool, warmup: int, repeats: int):
    """Return the last result and the per-call times in milliseconds."""
    times, result = [], None
    for i in range(warmup + repeats):
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = detector(image)
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - started) * 1000.0)
    return result, times


def agreement(cpu, gpu, min_iou: float) -> dict:
    """Match CPU and GPU boxes one-to-one by IoU and summarise the agreement."""
    iou = box_iou(cpu.boxes, gpu.boxes)
    pairs = []
    if iou.size:
        rows, cols = linear_sum_assignment(iou, maximize=True)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    return {
        "cpu_detections": len(cpu),
        "gpu_detections": len(gpu),
        "matched": len(pairs),
        "mean_matched_iou": float(np.mean([iou[r, c] for r, c in pairs])) if pairs else None,
        "max_confidence_difference": max(
            (abs(cpu[r].confidence - gpu[c].confidence) for r, c in pairs), default=None),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this Python process; install a "
                         "CUDA-enabled PyTorch build first.")
    image = load_image(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = FACE_MODELS[args.model]

    runs, views = {}, []
    for name, device in (("cpu", "cpu"), ("gpu", 0)):
        result, times = benchmark(pt.Face(model=model, device=device), image,
                                  device != "cpu", args.warmup, args.repeats)
        mean = statistics.fmean(times)
        runs[name] = (result, {"mean_ms": mean, "median_ms": statistics.median(times),
                               "stdev_ms": statistics.pstdev(times), "fps": 1000.0 / mean})
        views.append(draw_info_panel(result.plot(conf=True), [
            f"Device: {name.upper()} | Faces: {len(result)}",
            f"Mean: {mean:.1f} ms | {1000.0 / mean:.1f} FPS",
            f"Detector: {model.value}",
        ]))
        cv2.imwrite(str(output_dir / f"{name}.png"), views[-1])
    side_by_side = np.hstack(views)
    height = min(1000, side_by_side.shape[0])  # compact copy for docs/images
    cv2.imwrite(str(output_dir / "side_by_side.jpg"), cv2.resize(
        side_by_side, (round(side_by_side.shape[1] * height / side_by_side.shape[0]), height)))

    report = {
        "model": pt.Models.path_of(model),
        "timing": {name: stats for name, (_, stats) in runs.items()},
        "gpu_speedup": runs["cpu"][1]["mean_ms"] / runs["gpu"][1]["mean_ms"],
        "agreement": agreement(runs["cpu"][0], runs["gpu"][0], args.match_iou),
        "gpu": torch.cuda.get_device_name(0),
        "note": "Timings are specific to this hardware, software stack, model and image.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
