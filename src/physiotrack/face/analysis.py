import math
from numbers import Real
from pathlib import Path
from typing import Optional, Union

from ..core.predictor import PredictorMixin
from ..models import Models
from ..results import Instance, Result
from .blink import BlinkDetector
from .config import FaceAnalysisConfig
from .detect import Face
from .emotion import FaceEmotion
from .eyes import EyeOpenness
from .face_orientation import FaceOrientation
from .gaze import GazeDescriptor
from .gaze_estimation import GazeEstimator
from .landmarks import FaceLandmarks
from .mouth import MouthOpenness
from .mouth_motion import MouthMovement
from .quality import FaceQuality
from .regions import FaceRegions
from .temporal import FaceTemporalAggregator
from .tracking import FaceTracker


class FaceAnalysis(PredictorMixin):
    """Main modular face analysis pipeline."""

    def __init__(
        self,
        detector: Optional[object] = None,
        tracker: Optional[object] = None,
        orientation: Optional[object] = None,
        landmarks: Optional[object] = None,
        quality: Optional[object] = None,
        eyes: Optional[object] = None,
        blink: Optional[object] = None,
        gaze: Optional[object] = None,
        gaze_estimation: Optional[object] = None,
        mouth: Optional[object] = None,
        mouth_motion: Optional[object] = None,
        emotion: Optional[object] = None,
        regions: Optional[object] = None,
        temporal: Optional[object] = None,
        config: Optional[FaceAnalysisConfig] = None,
        fps: Optional[float] = None,
        temporal_window_sec: float = 5.0,
        blink_threshold: float = 0.22,
        min_closed_frames: int = 3,
        landmark_model_path: Optional[Union[str, Path]] = None,
        gaze_estimation_camera_path: Optional[
            Union[str, Path]
        ] = None,
        device: Union[str, int] = "cpu",
        verbose: bool = False,
    ):
        self.device = device
        self.verbose = verbose
        self.fps = float(fps) if fps is not None else None

        if self.fps is not None and self.fps <= 0:
            raise ValueError(
                "fps must be greater than zero"
            )

        if config is None:
            config = FaceAnalysisConfig(
                blink_threshold=blink_threshold,
                min_closed_frames=min_closed_frames,
                temporal_window_sec=temporal_window_sec,
            )

        if not isinstance(
            config,
            FaceAnalysisConfig,
        ):
            raise TypeError(
                "config must be a FaceAnalysisConfig instance"
            )

        config.validate()
        self.config = config

        if detector is None:
            detector = Face(
                device=device,
                verbose=verbose,
            )

        if self.config.tracking:
            if tracker is None:
                tracker = FaceTracker(
                    tracker_type=self.config.tracker_type,
                    device=device,
                )
        else:
            tracker = None

        if self.config.head_pose:
            if orientation is None:
                orientation = FaceOrientation(
                    device=device,
                    verbose=verbose,
                )
        else:
            orientation = None

        if self.config.landmarks:
            if landmarks is None:
                if landmark_model_path is None:
                    landmark_model_path = Models.resolve(
                        Models.Face.MediaPipe.Landmarks.face_landmarker
                    )

                landmarks = FaceLandmarks(
                    model_path=landmark_model_path,
                    num_faces=1,
                )
        else:
            landmarks = None

        if self.config.quality:
            if quality is None:
                quality = FaceQuality()
        else:
            quality = None

        if self.config.eyes:
            if eyes is None:
                eyes = EyeOpenness()
        else:
            eyes = None

        if (
            self.config.blink
            and self.fps is not None
        ):
            if blink is None:
                blink = BlinkDetector(
                    threshold=self.config.blink_threshold,
                    fps=self.fps,
                    min_closed_frames=self.config.min_closed_frames,
                )
        else:
            blink = None

        if self.config.gaze:
            if gaze is None:
                gaze = GazeDescriptor()
        else:
            gaze = None

        if self.config.gaze_estimation:
            if gaze_estimation is None:
                gaze_device = (
                    self._resolve_gaze_device(
                        device
                    )
                )

                gaze_estimation = GazeEstimator(
                    mode=self.config.gaze_estimation_mode,
                    device=gaze_device,
                    camera_path=gaze_estimation_camera_path,
                )

                gaze_estimation.initialize()
        else:
            gaze_estimation = None

        if self.config.mouth:
            if mouth is None:
                mouth = MouthOpenness()
        else:
            mouth = None

        if (
            self.config.mouth_motion
            and self.fps is not None
        ):
            if mouth_motion is None:
                mouth_motion = MouthMovement(
                    fps=self.fps,
                )
        else:
            mouth_motion = None

        if self.config.emotion:
            if emotion is None:
                emotion = FaceEmotion(
                    model_name=self.config.emotion_model,
                    engine=self.config.emotion_engine,
                )
        else:
            emotion = None

        if self.config.regions:
            if regions is None:
                regions = FaceRegions(
                    device=device,
                    verbose=verbose,
                )
        else:
            regions = None

        if (
            self.config.temporal
            and self.fps is not None
        ):
            if temporal is None:
                temporal = FaceTemporalAggregator(
                    fps=self.fps,
                    window_sec=self.config.temporal_window_sec,
                )
        else:
            temporal = None

        self.detector = detector
        self.tracker = tracker
        self.orientation = orientation
        self.landmarks = landmarks
        self.quality = quality
        self.eyes = eyes
        self.blink = blink
        self.gaze = gaze
        self.gaze_estimation = gaze_estimation
        self.mouth = mouth
        self.mouth_motion = mouth_motion
        self.emotion = emotion
        self.regions = regions
        self.temporal = temporal

        self._active_temporal_person_ids = set()
        self._frame_index = 0

    @staticmethod
    def _resolve_gaze_device(
        device,
    ):
        """Resolve a PhysioTrack device value for ptgaze."""
        if isinstance(
            device,
            int,
        ):
            if device < 0:
                return "cpu"

            return "cuda"

        device_string = str(
            device
        ).strip().lower()

        if device_string == "cpu":
            return "cpu"

        if (
            device_string == "cuda"
            or device_string.startswith(
                "cuda:"
            )
            or device_string.isdigit()
        ):
            return "cuda"

        raise ValueError(
            "Unsupported device for gaze estimation: "
            f"{device}"
        )

    @staticmethod
    def _finite_eye_measurement(value):
        """Require a finite real eye-openness measurement, including zero."""
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    @staticmethod
    def _crop_face(
        frame,
        box,
    ):
        """Return a valid face crop from a bounding box."""
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        x1 = max(
            0,
            min(
                x1,
                width,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                width,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                height,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                height,
            ),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def _box_iou(
        box_a,
        box_b,
    ):
        """Calculate intersection over union for two bounding boxes."""
        ax1, ay1, ax2, ay2 = map(
            float,
            box_a,
        )

        bx1, by1, bx2, by2 = map(
            float,
            box_b,
        )

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        intersection_width = max(
            0.0,
            ix2 - ix1,
        )

        intersection_height = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
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
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def _associate_regions(
        self,
        faces,
        region_faces,
    ):
        """Associate semantic-region outputs with face instances."""
        candidates = []

        for (
            face_index,
            face,
        ) in enumerate(
            faces
        ):
            if face.box is None:
                continue

            for (
                region_index,
                region_face,
            ) in enumerate(
                region_faces
            ):
                region_box = region_face.get(
                    "box"
                )

                if region_box is None:
                    continue

                iou = self._box_iou(
                    face.box,
                    region_box,
                )

                if iou > 0:
                    candidates.append(
                        (
                            iou,
                            face_index,
                            region_index,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        matches = {}
        used_faces = set()
        used_regions = set()

        for (
            iou,
            face_index,
            region_index,
        ) in candidates:
            if face_index in used_faces:
                continue

            if region_index in used_regions:
                continue

            matches[
                face_index
            ] = {
                "iou": iou,
                "face": region_faces[
                    region_index
                ],
            }

            used_faces.add(
                face_index
            )

            used_regions.add(
                region_index
            )

        return matches

    @staticmethod
    def _orientation_boxes_match(face_box, orientation_box):
        """Verify a pose box refers to the input face, allowing integer conversion."""
        try:
            face_coords = [float(value) for value in face_box]
            pose_coords = [float(value) for value in orientation_box]
        except (TypeError, ValueError, OverflowError):
            return False

        if len(face_coords) != 4 or len(pose_coords) != 4:
            return False
        if not all(math.isfinite(value) for value in face_coords + pose_coords):
            return False
        if (face_coords[2] <= face_coords[0]
                or face_coords[3] <= face_coords[1]
                or pose_coords[2] <= pose_coords[0]
                or pose_coords[3] <= pose_coords[1]):
            return False

        # 6DRepNet360 converts input boxes to integer coordinates before
        # returning them; each coordinate can change by less than one pixel.
        return all(
            abs(face_value - pose_value) < 1.0
            for face_value, pose_value in zip(face_coords, pose_coords)
        )

    def _associate_orientations(self, faces, orientation_result):
        """Map pose outputs to input face positions, never by compacted order."""
        if orientation_result is None:
            return {}

        matches = {}
        unindexed = []
        indexed_candidates = {}

        for pose_face in orientation_result:
            if pose_face.orientation is None or pose_face.box is None:
                continue

            # Backend ID is the index of the input bounding box, not a
            # persistent tracking/person ID. Skipped crops retain their gap.
            source_index = pose_face.id
            if isinstance(source_index, int) and not isinstance(source_index, bool):
                if (0 <= source_index < len(faces)
                        and faces[source_index].box is not None
                        and self._orientation_boxes_match(
                            faces[source_index].box, pose_face.box
                        )):
                    indexed_candidates.setdefault(source_index, []).append(pose_face)
                continue

            if source_index is None:
                unindexed.append(pose_face)

        for source_index, candidates in indexed_candidates.items():
            if len(candidates) == 1:
                matches[source_index] = candidates[0].orientation

        # Injected predictors may omit backend indices. Associate those
        # results only when their boxes uniquely identify an unmatched face.
        candidate_faces = {}
        for pose_index, pose_face in enumerate(unindexed):
            compatible = [
                index for index, face in enumerate(faces)
                if index not in matches
                and face.box is not None
                and self._orientation_boxes_match(face.box, pose_face.box)
            ]
            if len(compatible) == 1:
                candidate_faces.setdefault(compatible[0], []).append(pose_index)

        for source_index, pose_indices in candidate_faces.items():
            if len(pose_indices) == 1:
                matches[source_index] = unindexed[pose_indices[0]].orientation

        return matches

    def _handle_missing_temporal_person_ids(
        self,
        face_result,
        frame_index,
    ):
        """Break temporal continuity for tracked IDs missing from this frame."""
        current_person_ids = {
            face.id
            for face in face_result
            if face.id is not None
        }

        missing_person_ids = (
            self._active_temporal_person_ids
            - current_person_ids
        )

        for person_id in missing_person_ids:
            if self.blink is not None:
                if (
                    isinstance(self.blink, BlinkDetector)
                    and type(self.blink).update is BlinkDetector.update
                ):
                    self.blink.update(
                        None,
                        person_id=person_id,
                        frame_index=frame_index,
                    )
                else:
                    # Preserve the existing interface of injected components.
                    self.blink.update(
                        None,
                        person_id=person_id,
                    )

            if self.mouth_motion is not None:
                self.mouth_motion.update(
                    None,
                    person_id=person_id,
                )

            if self.temporal is not None:
                self.temporal.reset(
                    person_id=person_id
                )

        self._active_temporal_person_ids = (
            current_person_ids
        )

    def _predict_frame(
        self,
        frame,
    ) -> Result:
        """Process one frame and return face-analysis instances."""
        frame_index = self._frame_index
        self._frame_index += 1

        detection_result = self.detector.predict(
            frame
        )

        if self.tracker is not None:
            face_result = self.tracker.track(
                frame,
                detection_result,
            )
        else:
            face_result = detection_result

        self._handle_missing_temporal_person_ids(
            face_result,
            frame_index,
        )

        if len(face_result) == 0:
            return Result(
                orig_img=frame,
                instances=[],
                task="face",
            )

        orientation_matches = {}

        if self.orientation is not None:
            orientation_result = self.orientation.predict(
                frame,
                face_result.boxes,
            )
            orientation_matches = self._associate_orientations(
                face_result, orientation_result,
            )

        gaze_estimation_results = None

        if self.gaze_estimation is not None:
            gaze_estimation_results = (
                self.gaze_estimation.predict_faces(
                    image=frame,
                    boxes=face_result.boxes,
                    min_iou=(
                        self.config.gaze_estimation_min_iou
                    ),
                )
            )

            if (
                len(
                    gaze_estimation_results
                )
                != len(
                    face_result
                )
            ):
                raise RuntimeError(
                    "Gaze estimation result count "
                    "does not match face count."
                )

        region_matches = {}

        if self.regions is not None:
            regions_output = (
                self.regions.predict(
                    frame,
                    boxes=face_result.boxes,
                )
            )

            region_matches = (
                self._associate_regions(
                    face_result,
                    regions_output[
                        "faces"
                    ],
                )
            )

        instances = []

        for (
            face_index,
            face,
        ) in enumerate(
            face_result
        ):
            orientation_value = orientation_matches.get(face_index)

            face_landmarks = None

            if self.landmarks is not None:
                if (
                    isinstance(self.landmarks, FaceLandmarks)
                    and type(self.landmarks).predict_face
                    is FaceLandmarks.predict_face
                ):
                    # Other detected faces disambiguate a square crop that
                    # contains landmarks from more than one person.
                    other_boxes = [
                        other_face.box
                        for other_index, other_face in enumerate(face_result)
                        if other_index != face_index
                        and other_face.box is not None
                    ]
                    face_landmarks = self.landmarks.predict_face(
                        frame, face.box, other_boxes=other_boxes,
                    )
                else:
                    # Preserve the two-argument interface of injected tools.
                    face_landmarks = self.landmarks.predict_face(
                        frame, face.box,
                    )

            landmarks_available = (
                face_landmarks
                is not None
            )

            if self.quality is not None:
                quality_results = (
                    self.quality.predict(
                        frame,
                        [
                            face
                        ],
                    )
                )

                quality_result = (
                    quality_results[
                        0
                    ]
                    if quality_results
                    else None
                )
            else:
                quality_result = None

            if quality_result is not None:
                quality_features = {
                    "available": True,
                    **quality_result,
                }
            else:
                quality_features = {
                    "available": False,
                }

            if (
                self.eyes is not None
                and landmarks_available
            ):
                eye_result = (
                    self.eyes.predict(
                        face_landmarks,
                        image_size=(
                            frame.shape[1],
                            frame.shape[0],
                        ),
                    )
                )

                # A completed landmark prediction does not guarantee usable
                # eye geometry. Keep partial values, but never declare missing
                # or non-finite eye measurements available.
                eye_values = {
                    key: eye_result.get(key)
                    for key in (
                        "left_openness",
                        "right_openness",
                        "mean_openness",
                    )
                }
                for key, value in eye_values.items():
                    if not self._finite_eye_measurement(value):
                        eye_values[key] = None

                eyes_features = {
                    **eye_result,
                    **eye_values,
                    "available": all(
                        value is not None for value in eye_values.values()
                    ),
                }
            else:
                eyes_features = {
                    "available": False,
                    "left_openness": None,
                    "right_openness": None,
                    "mean_openness": None,
                }

            if self.blink is not None and face.id is not None:
                blink_arguments = {
                    "openness": (
                        eyes_features["mean_openness"]
                        if eyes_features["available"] else None
                    ),
                    "person_id": face.id,
                }
                if (
                    isinstance(self.blink, BlinkDetector)
                    and type(self.blink).update is BlinkDetector.update
                ):
                    blink_arguments["frame_index"] = frame_index
                blink_result = self.blink.update(**blink_arguments)

                blink_features = {
                    **blink_result,
                    "available": (
                        eyes_features["available"]
                        and blink_result.get("eye_state") in ("open", "closed")
                    ),
                }
                if not blink_features["available"]:
                    # The temporal component has consumed an unavailable sample
                    # without supplying a valid current blink classification.
                    blink_features.update(
                        eye_state="unknown", blink=False, blink_duration=None
                    )
            else:
                blink_features = {
                    "available": False,
                    "eye_state": "unknown",
                    "blink": False,
                    "blink_count": 0,
                    "blink_duration": None,
                    "blink_rate": None,
                }

            if (
                self.gaze is not None
                and landmarks_available
            ):
                gaze_result = (
                    self.gaze.predict(
                        face_landmarks,
                        image_size=(
                            frame.shape[1],
                            frame.shape[0],
                        ),
                    )
                )

                gaze_available = (
                    gaze_result[
                        "mean_iris_x"
                    ]
                    is not None
                    and gaze_result[
                        "mean_iris_y"
                    ]
                    is not None
                )

                gaze_features = {
                    "available": gaze_available,
                    **gaze_result,
                }
            else:
                gaze_features = {
                    "available": False,
                    "right_iris_x": None,
                    "right_iris_y": None,
                    "left_iris_x": None,
                    "left_iris_y": None,
                    "mean_iris_x": None,
                    "mean_iris_y": None,
                }

            if (
                self.mouth is not None
                and landmarks_available
            ):
                mouth_result = (
                    self.mouth.predict(
                        face_landmarks,
                        image_size=(
                            frame.shape[1],
                            frame.shape[0],
                        ),
                    )
                )

                mouth_available = (
                    mouth_result[
                        "mouth_openness"
                    ]
                    is not None
                )

                mouth_features = {
                    "available": mouth_available,
                    **mouth_result,
                }
            else:
                mouth_features = {
                    "available": False,
                    "mouth_openness": None,
                    "mouth_width": None,
                    "mouth_height": None,
                }

            if self.mouth_motion is not None and face.id is not None:
                motion_result = (
                    self.mouth_motion.update(
                        openness=mouth_features[
                            "mouth_openness"
                        ],
                        person_id=face.id,
                    )
                )

                mouth_motion_features = {
                    "available": (
                        mouth_features[
                            "available"
                        ]
                        and motion_result[
                            "mouth_movement"
                        ]
                        is not None
                    ),
                    **motion_result,
                }
            else:
                mouth_motion_features = {
                    "available": False,
                    "mouth_movement": None,
                    "mouth_velocity": None,
                }

            if self.emotion is not None:
                face_crop = (
                    self._crop_face(
                        frame,
                        face.box,
                    )
                )
            else:
                face_crop = None

            if (
                self.emotion is not None
                and face_crop is not None
            ):
                emotion_result = (
                    self.emotion.predict(
                        face_crop
                    )
                )

                emotion_features = {
                    "available": True,
                    **emotion_result,
                }
            else:
                emotion_features = {
                    "available": False,
                    "emotion": None,
                    "confidence": None,
                    "scores": None,
                }

            region_match = (
                region_matches.get(
                    face_index
                )
            )

            if region_match is not None:
                region_data = (
                    region_match[
                        "face"
                    ]
                )

                region_masks = (
                    region_data[
                        "regions"
                    ]
                )

                pixel_counts = {
                    name: int(
                        mask.sum()
                    )
                    for (
                        name,
                        mask,
                    ) in region_masks.items()
                }

                region_box = (
                    region_data[
                        "box"
                    ]
                )

                region_width = max(
                    0,
                    int(
                        region_box[
                            2
                        ]
                        - region_box[
                            0
                        ]
                    ),
                )

                region_height = max(
                    0,
                    int(
                        region_box[
                            3
                        ]
                        - region_box[
                            1
                        ]
                    ),
                )

                region_area = (
                    region_width
                    * region_height
                )

                skin_pixels = (
                    pixel_counts.get(
                        "skin",
                        0,
                    )
                )

                skin_fraction = (
                    float(
                        skin_pixels
                        / region_area
                    )
                    if region_area > 0
                    else None
                )

                regions_features = {
                    "available": True,
                    "classes": list(
                        pixel_counts.keys()
                    ),
                    "pixel_counts": pixel_counts,
                    "skin_pixel_count": (
                        skin_pixels
                    ),
                    "skin_fraction": (
                        skin_fraction
                    ),
                    "association_iou": float(
                        region_match[
                            "iou"
                        ]
                    ),
                }
            else:
                regions_features = {
                    "available": False,
                    "classes": [],
                    "pixel_counts": {},
                    "skin_pixel_count": 0,
                    "skin_fraction": None,
                    "association_iou": None,
                }

            face_features = {
                "landmarks": {
                    "available": (
                        landmarks_available
                    ),
                    "count": (
                        len(
                            face_landmarks
                        )
                        if landmarks_available
                        else 0
                    ),
                },
                "quality": quality_features,
                "eyes": eyes_features,
                "blink": blink_features,
                "gaze": gaze_features,
                "mouth": mouth_features,
                "mouth_motion": (
                    mouth_motion_features
                ),
                "emotion": emotion_features,
                "regions": regions_features,
            }

            if gaze_estimation_results is not None:
                face_features[
                    "gaze_estimation"
                ] = dict(
                    gaze_estimation_results[
                        face_index
                    ]
                )

            instance = Instance(
                id=face.id,
                box=face.box,
                confidence=face.confidence,
                cls=face.cls,
                cls_name=face.cls_name,
                orientation=orientation_value,
                face_features=face_features,
            )

            if self.temporal is not None:
                temporal_summary = (
                    self.temporal.update(
                        instance
                    )
                )

                instance.face_features = {
                    **face_features,
                    "temporal": {
                        "available": (
                            temporal_summary
                            is not None
                        ),
                        "summary": (
                            temporal_summary
                        ),
                    },
                }
            else:
                instance.face_features = {
                    **face_features,
                    "temporal": {
                        "available": False,
                        "summary": None,
                    },
                }

            instances.append(
                instance
            )

        return Result(
            orig_img=frame,
            instances=instances,
            task="face",
        )

    def predict(
        self,
        source,
    ):
        """Run face analysis on one image or a batch of images."""
        frames, was_batch = (
            self._as_frames(
                source
            )
        )

        results = []

        for frame in frames:
            result = (
                self._predict_frame(
                    frame
                )
            )

            results.append(
                result
            )

        return self._unwrap(
            results,
            was_batch,
        )

    def reset_temporal_state(self):
        """Reset temporal face-analysis state and video frame clock."""
        self._active_temporal_person_ids.clear()
        self._frame_index = 0

        if (
            self.blink is not None
            and hasattr(
                self.blink,
                "reset",
            )
        ):
            self.blink.reset()

        if (
            self.mouth_motion
            is not None
            and hasattr(
                self.mouth_motion,
                "reset",
            )
        ):
            self.mouth_motion.reset()

        if (
            self.temporal is not None
            and hasattr(
                self.temporal,
                "reset",
            )
        ):
            self.temporal.reset()

    def close(self):
        """Release resources used by face-analysis components."""
        if (
            self.landmarks is not None
            and hasattr(
                self.landmarks,
                "close",
            )
        ):
            self.landmarks.close()

        if (
            self.gaze_estimation
            is not None
            and hasattr(
                self.gaze_estimation,
                "close",
            )
        ):
            self.gaze_estimation.close()