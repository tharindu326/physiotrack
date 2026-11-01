from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..core.predictor import PredictorMixin


class GazeEstimator(PredictorMixin):
    """Estimate a normalized 3D gaze direction using ptgaze."""

    SUPPORTED_MODES = {
        "eth-xgaze",
        "mpiigaze",
        "mpiifacegaze",
    }

    def __init__(
        self,
        mode: str = "eth-xgaze",
        device: str = "cpu",
        camera_path: Optional[Union[str, Path]] = None,
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported gaze estimation mode: {mode}"
            )

        if device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device must be 'cpu' or 'cuda'"
            )

        self.mode = mode
        self.device = device

        self.camera_path = (
            Path(camera_path).expanduser().resolve()
            if camera_path is not None
            else None
        )

        if (
            self.camera_path is not None
            and not self.camera_path.exists()
        ):
            raise FileNotFoundError(
                f"Camera calibration file not found: "
                f"{self.camera_path}"
            )

        self._estimator = None
        self._checkpoint_path = None

    @staticmethod
    def _import_ptgaze():
        """Import ptgaze lazily so it remains optional."""
        try:
            from ptgaze.gaze_estimator import (
                GazeEstimator as PTGazeEstimator,
            )

            from ptgaze.main import (
                download_ethxgaze_model,
                download_mpiifacegaze_model,
                download_mpiigaze_model,
                expanduser_all,
                load_mode_config,
            )

        except ImportError as error:
            raise ImportError(
                "GazeEstimator requires the optional "
                "'ptgaze' package. Install it in the "
                "environment used for gaze estimation."
            ) from error

        return {
            "estimator_class": PTGazeEstimator,
            "download_ethxgaze_model": (
                download_ethxgaze_model
            ),
            "download_mpiifacegaze_model": (
                download_mpiifacegaze_model
            ),
            "download_mpiigaze_model": (
                download_mpiigaze_model
            ),
            "expanduser_all": expanduser_all,
            "load_mode_config": load_mode_config,
        }

    def _resolve_checkpoint(
        self,
        ptgaze,
    ):
        """Resolve the pretrained checkpoint for the selected mode."""
        if self.mode == "eth-xgaze":
            checkpoint = ptgaze[
                "download_ethxgaze_model"
            ]()

        elif self.mode == "mpiifacegaze":
            checkpoint = ptgaze[
                "download_mpiifacegaze_model"
            ]()

        else:
            checkpoint = ptgaze[
                "download_mpiigaze_model"
            ]()

        return Path(
            checkpoint
        ).expanduser().resolve()

    def _build_estimator(
        self,
    ):
        """Create the underlying ptgaze estimator."""
        ptgaze = self._import_ptgaze()

        checkpoint_path = (
            self._resolve_checkpoint(
                ptgaze
            )
        )

        args = argparse.Namespace(
            mode=self.mode,
            face_detector="mediapipe",
            device=self.device,
            image=None,
            video=None,
            camera=(
                str(self.camera_path)
                if self.camera_path is not None
                else None
            ),
            output_dir=None,
            ext=None,
            no_screen=True,
        )

        config = ptgaze[
            "load_mode_config"
        ](
            args
        )

        config.gaze_estimator.checkpoint = (
            checkpoint_path.as_posix()
        )

        ptgaze[
            "expanduser_all"
        ](
            config
        )

        self._estimator = ptgaze[
            "estimator_class"
        ](
            config
        )

        self._checkpoint_path = (
            checkpoint_path
        )

    @staticmethod
    def _normalize_vector(
        vector,
    ):
        """Normalize and validate a 3D gaze vector."""
        vector = np.asarray(
            vector,
            dtype=np.float64,
        ).reshape(-1)

        if vector.size != 3:
            raise ValueError(
                "Expected a 3D gaze vector."
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):
            raise ValueError(
                "Gaze vector contains non-finite values."
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm <= 0:
            raise ValueError(
                "Gaze vector has zero length."
            )

        return (
            vector
            / norm
        )

    @staticmethod
    def _vector_to_angles(
        gaze_vector,
    ):
        """Convert a normalized gaze vector to pitch and yaw in degrees."""
        x, y, z = gaze_vector

        yaw = np.degrees(
            np.arctan2(
                x,
                -z,
            )
        )

        pitch = np.degrees(
            np.arctan2(
                y,
                np.sqrt(
                    x * x
                    + z * z
                ),
            )
        )

        return (
            float(pitch),
            float(yaw),
        )

    @staticmethod
    def _ptgaze_bbox_to_xyxy(
        bbox,
    ):
        """Convert a ptgaze bounding box to x1, y1, x2, y2."""
        bbox_array = np.asarray(
            bbox,
            dtype=np.float64,
        )

        if bbox_array.shape != (
            2,
            2,
        ):
            raise ValueError(
                "Expected ptgaze bounding box with shape (2, 2)."
            )

        x1 = float(
            bbox_array[
                0,
                0,
            ]
        )

        y1 = float(
            bbox_array[
                0,
                1,
            ]
        )

        x2 = float(
            bbox_array[
                1,
                0,
            ]
        )

        y2 = float(
            bbox_array[
                1,
                1,
            ]
        )

        if not np.isfinite(
            [
                x1,
                y1,
                x2,
                y2,
            ]
        ).all():
            raise ValueError(
                "Bounding box contains non-finite values."
            )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            raise ValueError(
                "Bounding box has invalid coordinates."
            )

        return np.array(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _validate_target_box(
        box,
    ):
        """Validate a PhysioTrack x1, y1, x2, y2 bounding box."""
        target_box = np.asarray(
            box,
            dtype=np.float64,
        ).reshape(-1)

        if target_box.size != 4:
            raise ValueError(
                "box must contain four coordinates."
            )

        if not np.all(
            np.isfinite(
                target_box
            )
        ):
            raise ValueError(
                "box contains non-finite values."
            )

        x1, y1, x2, y2 = (
            target_box
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            raise ValueError(
                "box has invalid coordinates."
            )

        return target_box

    @staticmethod
    def _box_iou(
        box_a,
        box_b,
    ):
        """Calculate IoU between two x1, y1, x2, y2 boxes."""
        box_a = np.asarray(
            box_a,
            dtype=np.float64,
        ).reshape(-1)

        box_b = np.asarray(
            box_b,
            dtype=np.float64,
        ).reshape(-1)

        if (
            box_a.size != 4
            or box_b.size != 4
        ):
            raise ValueError(
                "Bounding boxes must contain four coordinates."
            )

        ax1, ay1, ax2, ay2 = (
            box_a
        )

        bx1, by1, bx2, by2 = (
            box_b
        )

        intersection_x1 = max(
            ax1,
            bx1,
        )

        intersection_y1 = max(
            ay1,
            by1,
        )

        intersection_x2 = min(
            ax2,
            bx2,
        )

        intersection_y2 = min(
            ay2,
            by2,
        )

        intersection_width = max(
            0.0,
            intersection_x2
            - intersection_x1,
        )

        intersection_height = max(
            0.0,
            intersection_y2
            - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = (
            max(
                0.0,
                ax2 - ax1,
            )
            * max(
                0.0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0.0,
                bx2 - bx1,
            )
            * max(
                0.0,
                by2 - by1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection_area
        )

        if union <= 0:
            return 0.0

        return float(
            intersection_area
            / union
        )

    @staticmethod
    def _empty_result(
        association_iou=None,
    ):
        """Create an unavailable gaze-estimation result."""
        return {
            "available": False,
            "gaze_vector": None,
            "pitch": None,
            "yaw": None,
            "association_iou": association_iou,
        }

    def _estimate_face(
        self,
        image_array,
        face,
        association_iou,
    ):
        """Estimate gaze for one already matched ptgaze face."""
        self._estimator.estimate_gaze(
            image_array,
            face,
        )

        gaze_vector = (
            self._normalize_vector(
                face.gaze_vector
            )
        )

        pitch, yaw = (
            self._vector_to_angles(
                gaze_vector
            )
        )

        return {
            "available": True,
            "gaze_vector": (
                gaze_vector.tolist()
            ),
            "pitch": pitch,
            "yaw": yaw,
            "association_iou": float(
                association_iou
            ),
        }

    def _estimate_target_crop(
        self,
        image_array,
        target_box,
        min_iou,
    ):
        """Estimate gaze from one PhysioTrack face crop."""
        height, width = (
            image_array.shape[:2]
        )

        target_box = (
            self._validate_target_box(
                target_box
            )
        )

        (
            original_x1,
            original_y1,
            original_x2,
            original_y2,
        ) = target_box

        crop_x1 = max(
            0,
            min(
                int(
                    np.floor(
                        original_x1
                    )
                ),
                width,
            ),
        )

        crop_y1 = max(
            0,
            min(
                int(
                    np.floor(
                        original_y1
                    )
                ),
                height,
            ),
        )

        crop_x2 = max(
            0,
            min(
                int(
                    np.ceil(
                        original_x2
                    )
                ),
                width,
            ),
        )

        crop_y2 = max(
            0,
            min(
                int(
                    np.ceil(
                        original_y2
                    )
                ),
                height,
            ),
        )

        if (
            crop_x2 <= crop_x1
            or crop_y2 <= crop_y1
        ):
            return self._empty_result()

        crop = image_array[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        faces = (
            self._estimator.detect_faces(
                crop
            )
        )

        if not faces:
            return self._empty_result()

        best_face = None
        best_iou = 0.0

        for face in faces:
            crop_candidate_box = (
                self._ptgaze_bbox_to_xyxy(
                    face.bbox
                )
            )

            full_candidate_box = np.array(
                [
                    crop_candidate_box[0]
                    + crop_x1,
                    crop_candidate_box[1]
                    + crop_y1,
                    crop_candidate_box[2]
                    + crop_x1,
                    crop_candidate_box[3]
                    + crop_y1,
                ],
                dtype=np.float64,
            )

            iou = self._box_iou(
                target_box,
                full_candidate_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_face is None:
            return self._empty_result()

        if best_iou < min_iou:
            return self._empty_result(
                association_iou=float(
                    best_iou
                )
            )

        return self._estimate_face(
            crop,
            best_face,
            best_iou,
        )

    def predict_faces(
        self,
        image,
        boxes,
        min_iou: float = 0.10,
    ):
        """Estimate gaze for multiple PhysioTrack face boxes."""
        if image is None:
            raise ValueError(
                "image must not be None"
            )

        image_array = np.asarray(
            image
        )

        if image_array.ndim != 3:
            raise ValueError(
                "image must have shape (H, W, C)"
            )

        if not (
            0.0
            <= min_iou
            <= 1.0
        ):
            raise ValueError(
                "min_iou must be between 0 and 1."
            )

        if self._estimator is None:
            raise RuntimeError(
                "GazeEstimator is not initialized. "
                "Call initialize() before prediction."
            )

        boxes_array = np.asarray(
            boxes,
            dtype=np.float64,
        )

        if boxes_array.size == 0:
            return []

        if boxes_array.ndim == 1:
            boxes_array = (
                boxes_array.reshape(
                    1,
                    -1,
                )
            )

        if (
            boxes_array.ndim != 2
            or boxes_array.shape[1] != 4
        ):
            raise ValueError(
                "boxes must have shape (N, 4)."
            )

        target_boxes = [
            self._validate_target_box(
                box
            )
            for box in boxes_array
        ]

        results = [
            self._empty_result()
            for _ in target_boxes
        ]

        faces = (
            self._estimator.detect_faces(
                image_array
            )
        )

        if not faces:
            crop_results = []

            for target_box in target_boxes:
                crop_results.append(
                    self._estimate_target_crop(
                        image_array,
                        target_box,
                        min_iou,
                    )
                )

            return crop_results

        candidate_boxes = []

        for face in faces:
            candidate_boxes.append(
                self._ptgaze_bbox_to_xyxy(
                    face.bbox
                )
            )

        candidates = []

        for (
            target_index,
            target_box,
        ) in enumerate(
            target_boxes
        ):
            for (
                face_index,
                candidate_box,
            ) in enumerate(
                candidate_boxes
            ):
                iou = self._box_iou(
                    target_box,
                    candidate_box,
                )

                if iou >= min_iou:
                    candidates.append(
                        (
                            float(iou),
                            target_index,
                            face_index,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        used_targets = set()
        used_faces = set()

        for (
            iou,
            target_index,
            face_index,
        ) in candidates:
            if (
                target_index
                in used_targets
            ):
                continue

            if (
                face_index
                in used_faces
            ):
                continue

            results[
                target_index
            ] = self._estimate_face(
                image_array,
                faces[
                    face_index
                ],
                iou,
            )

            used_targets.add(
                target_index
            )

            used_faces.add(
                face_index
            )

        for (
            target_index,
            target_box,
        ) in enumerate(
            target_boxes
        ):
            if (
                target_index
                in used_targets
            ):
                continue

            best_iou = 0.0

            for candidate_box in (
                candidate_boxes
            ):
                iou = self._box_iou(
                    target_box,
                    candidate_box,
                )

                if iou > best_iou:
                    best_iou = iou

            results[
                target_index
            ] = self._empty_result(
                association_iou=float(
                    best_iou
                )
            )

        return results

    def predict_face(
        self,
        image,
        box,
        min_iou: float = 0.10,
    ):
        """Estimate gaze for one PhysioTrack face box."""
        results = self.predict_faces(
            image=image,
            boxes=[
                box
            ],
            min_iou=min_iou,
        )

        return results[0]

    def predict_image(
        self,
        image,
    ):
        """Estimate gaze for the first detected face in one image."""
        if image is None:
            raise ValueError(
                "image must not be None"
            )

        image_array = np.asarray(
            image
        )

        if image_array.ndim != 3:
            raise ValueError(
                "image must have shape (H, W, C)"
            )

        if self._estimator is None:
            raise RuntimeError(
                "GazeEstimator is not initialized. "
                "Call initialize() before prediction."
            )

        faces = (
            self._estimator.detect_faces(
                image_array
            )
        )

        if not faces:
            return {
                "available": False,
                "gaze_vector": None,
                "pitch": None,
                "yaw": None,
            }

        face = faces[0]

        self._estimator.estimate_gaze(
            image_array,
            face,
        )

        gaze_vector = (
            self._normalize_vector(
                face.gaze_vector
            )
        )

        pitch, yaw = (
            self._vector_to_angles(
                gaze_vector
            )
        )

        return {
            "available": True,
            "gaze_vector": (
                gaze_vector.tolist()
            ),
            "pitch": pitch,
            "yaw": yaw,
        }

    def initialize(
        self,
    ):
        """Initialize the pretrained gaze estimator."""
        if self._estimator is not None:
            self.close()

        self._build_estimator()

        return self

    def predict(
        self,
        source,
    ):
        """Estimate gaze for one image or a batch of images.

        Args:
            source (str | os.PathLike | np.ndarray | Sequence): A single BGR
                image, a path to an image file, or a sequence of either for
                batch inference.

        Returns:
            dict | list[dict]: One gaze-estimation result for a single image,
                or one result per input image for a batch.
        """
        frames, was_batch = self._as_frames(
            source
        )

        results = [
            self.predict_image(
                frame
            )
            for frame in frames
        ]

        return self._unwrap(
            results,
            was_batch,
        )

    @property
    def initialized(
        self,
    ):
        """Return whether the estimator is initialized."""
        return (
            self._estimator
            is not None
        )

    @property
    def checkpoint_path(
        self,
    ):
        """Return the resolved pretrained checkpoint path."""
        return self._checkpoint_path

    def close(
        self,
    ):
        """Release resources used by the estimator."""
        if (
            self._estimator is not None
            and hasattr(
                self._estimator,
                "close",
            )
        ):
            self._estimator.close()

        self._estimator = None