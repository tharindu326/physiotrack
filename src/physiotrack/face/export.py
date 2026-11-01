import csv
import json
from pathlib import Path


class FaceResultExporter:
    """Export face-analysis results to JSON or CSV."""

    @staticmethod
    def _flatten_dict(data, prefix=""):
        """Flatten nested dictionaries using dot-separated keys."""
        flat = {}

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(
                    FaceResultExporter._flatten_dict(
                        value,
                        prefix=full_key,
                    )
                )
            else:
                flat[full_key] = value

        return flat

    @staticmethod
    def frame_records(result, frame_index=None, timestamp=None):
        """Convert one face-analysis Result into per-person records."""
        records = []

        for instance in result:
            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "person_id": instance.id,
                "box": (
                    instance.box.tolist()
                    if instance.box is not None
                    else None
                ),
                "confidence": instance.confidence,
                "orientation": instance.orientation,
                "face_features": instance.face_features,
            }

            records.append(record)

        return records

    @staticmethod
    def window_records(result, frame_index=None, timestamp=None):
        """Convert temporal summaries into per-person window records."""
        records = []

        for instance in result:
            temporal = instance.face_features.get(
                "temporal",
                {},
            )

            summary = temporal.get("summary")

            if not temporal.get("available", False):
                continue

            if summary is None:
                continue

            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "person_id": instance.id,
                "window_frames": summary["window_frames"],
                "window_sec": summary["window_sec"],
                "head_pose": summary["head_pose"],
                "eyes": summary["eyes"],
                "gaze": summary["gaze"],
                "mouth": summary["mouth"],
                "quality": summary["quality"],
                "blink": summary["blink"],
                "emotion": summary["emotion"],
            }

            records.append(record)

        return records

    @staticmethod
    def save_json(records, path):
        """Save records to a JSON file."""
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                records,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def save_csv(records, path):
        """Save flattened records to a CSV file."""
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        flat_records = [
            FaceResultExporter._flatten_dict(record)
            for record in records
        ]

        if not flat_records:
            return

        fieldnames = sorted(
            {
                key
                for record in flat_records
                for key in record.keys()
            }
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(flat_records)