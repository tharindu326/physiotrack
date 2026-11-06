import copy
import time
from collections import deque, defaultdict
import numpy as np
import cv2
from .config import TrackerConfig
from ..results import TrackResult, Instance
from ..core.boxes import box_iou
from ..core.overlay import draw_label

from .._logging import get_logger

logger = get_logger(__name__)


class Tracker:
    """Unified multi-object tracker that turns per-frame detections into tracks.

    Wraps four interchangeable tracking backends behind one API. You feed it a
    frame plus that frame's detection boxes and it returns a
    [`TrackResult`][physiotrack.TrackResult] whose instances carry persistent
    ``id``s across frames. The backend is selected by
    [`TrackerConfig.tracker_type`][physiotrack.TrackerConfig]:

    - ``"bytetrack"`` — [ByteTrack][physiotrack.Tracker], IOU + high/low score
      association (no appearance model).
    - ``"strongsort"`` — [StrongSORT][physiotrack.Tracker], Kalman motion plus an
      OSNet ReID appearance model (runs on :attr:`TrackerConfig.device`).
    - ``"ocsort"`` — [OC-SORT][physiotrack.Tracker], observation-centric SORT (the
      default backend).
    - ``"boosttrack"`` — [BoostTrack][physiotrack.Tracker], IOU/Mahalanobis/shape
      similarity with detection-confidence boosting.

    On top of raw tracking it can optionally isolate a single subject via a
    stability + IOU heuristic and draw a rich overlay (per-track
    boxes, the locked-subject box, and movement trails), all controlled by the config.

    Detections are expected as rows ``[x1, y1, x2, y2, conf, cls]``; only rows whose
    ``cls`` is in :attr:`TrackerConfig.classes` are tracked.

    Attributes:
        config (TrackerConfig): The active configuration.
        frame_ID (int): Frame counter (informational).
        locked_subject_id (int | None): Id of the currently locked subject, or
            ``None`` when no subject is locked.
        track_history (dict[int, collections.deque]): Recent center/box history per
            track id, used for trail drawing.

    Example:
        ```python
        import numpy as np
        import physiotrack as pt

        det = pt.Detection.Person()
        tracker = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))

        for frame in frames:                     # BGR frames, e.g. from OpenCV
            res = det.predict(frame)
            # Build the (N, 6) [x1, y1, x2, y2, conf, cls] array the tracker expects.
            detections = np.array(
                [[*i.box, i.confidence, i.cls] for i in res], dtype=np.float32
            ) if len(res) else np.empty((0, 6), np.float32)
            result = tracker.track(frame, detections)  # -> pt.TrackResult
            annotated = result.plot()                  # rich tracker overlay
            for inst in result:                        # persistent ids
                print(inst.id, inst.box)
        ```

    Note:
        The tracker is stateful: call [`track`][physiotrack.Tracker.track] once per
        frame in order, and use one ``Tracker`` instance per video stream.

    See Also:
        [`TrackerConfig`][physiotrack.TrackerConfig]: all tunable options.

        [`TrackResult`][physiotrack.TrackResult]: the returned per-frame result.

        [`Detection`][physiotrack.Detection]: produces the detection boxes fed in.
    """

    def __init__(self, config=None):
        """Initialize the tracker and instantiate the selected backend.

        Args:
            config (TrackerConfig, optional): Tracker configuration. Defaults to
                ``None``, in which case a default
                [`TrackerConfig`][physiotrack.TrackerConfig] is created (OC-SORT,
                tracking the person class). The backend named by
                ``config.tracker_type`` is constructed immediately.

        Raises:
            ValueError: If ``config.tracker_type`` is not one of ``"bytetrack"``,
                ``"strongsort"``, ``"ocsort"``, or ``"boosttrack"``.
        """
        # Copy rather than alias: a caller reusing one TrackerConfig across several
        # trackers (or mutating it afterwards) must not silently reconfigure a running
        # tracker.
        self.config = copy.deepcopy(config) if config is not None else TrackerConfig()
        self.frame_ID = 0
        self.id_list = []

        # FPS monitoring
        self.inference_times = deque(maxlen=100)
        
        # Track history storage
        self.locked_subject_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))
        self.track_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))
        # Seeded so overlay colours are reproducible across runs, matching the palette
        # in Result.plot(); unseeded colours made two renders of the same video differ.
        self.COLORS = np.random.default_rng(0).integers(0, 255, size=(256, 3)).tolist()
        
        # Initialize tracker
        tracker_type = self.config.tracker_type.lower()
        self.tracker = self._initialize_tracker(tracker_type)
        self.track_ids = []
        
        # Subject-lock state
        self.locked_subject_id = None
        self.consecutive_appearances = defaultdict(int)
        self.consecutive_misses = defaultdict(int)
        self.locked_subject_box = None
        self.consecutive_inconsistent_motion = defaultdict(int)
        self.locked_subject_trail = deque(maxlen=self.config.trail_length)
        self.debug_mode = self.config.debug_mode
    
    def _initialize_tracker(self, tracker_type):
        """Initialize the appropriate tracker based on configuration."""
        if tracker_type == 'bytetrack':
            from . import BYTETracker
            return BYTETracker(
                track_thresh=self.config.bytetrack_track_thresh,
                match_thresh=self.config.bytetrack_match_thresh,
                track_buffer=self.config.bytetrack_track_buffer,
                frame_rate=self.config.bytetrack_frame_rate
            )
        elif tracker_type == 'strongsort':
            from . import StrongSORT
            return StrongSORT(
                model_weights=self.config.strongsort_reid_weights,
                device=self.config.device,
                fp16=False,
                max_dist=self.config.strongsort_max_dist,
                max_iou_dist=self.config.strongsort_max_iou_dist,
                max_age=self.config.strongsort_max_age,
                max_unmatched_preds=self.config.strongsort_max_unmatched_preds,
                n_init=self.config.strongsort_n_init,
                nn_budget=self.config.strongsort_nn_budget,
                mc_lambda=self.config.strongsort_mc_lambda,
                ema_alpha=self.config.strongsort_ema_alpha,
            )
        elif tracker_type == 'ocsort':
            from . import OCSort
            return OCSort(
                det_thresh=self.config.ocsort_det_thresh,
                max_age=self.config.ocsort_max_age,
                min_hits=self.config.ocsort_min_hits,
                iou_threshold=self.config.ocsort_iou_thresh,
                delta_t=self.config.ocsort_delta_t,
                asso_func=self.config.ocsort_asso_func,
                inertia=self.config.ocsort_inertia,
                use_byte=self.config.ocsort_use_byte,
            )
        elif tracker_type == 'boosttrack':
            from . import BoostTrack
            return BoostTrack(
                det_thresh=self.config.boosttrack_det_thresh,
                lambda_iou=self.config.boosttrack_lambda_iou,
                lambda_mhd=self.config.boosttrack_lambda_mhd,
                lambda_shape=self.config.boosttrack_lambda_shape,
                dlo_boost_coef=self.config.boosttrack_dlo_boost_coef,
                use_dlo_boost=self.config.boosttrack_use_dlo_boost,
                use_duo_boost=self.config.boosttrack_use_duo_boost,
                max_age=self.config.boosttrack_max_age
            )
        # === Initialize a new locked subject ===
        else:
            supported_trackers = ['OCSort', 'BYTETrack', 'StrongSORT', 'BoostTrack']
            raise ValueError(f'Undefined Tracker. Please use one of: {", ".join(supported_trackers)}')
    
    # ===== Subject-lock logic =====
    def update_locked_subject(self, online_targets):
        """Update the locked-subject state from the current frame's tracks."""
        if not self.config.enable_subject_lock or len(online_targets) == 0:
            return
        
        targets_array = np.array(online_targets)
        track_ids = targets_array[:, 4].astype(int)
        bboxes = targets_array[:, :4]
        
        # === Check the existing locked subject ===
        if self.locked_subject_id is not None:
            locked_mask = track_ids == self.locked_subject_id
            
            if np.any(locked_mask):
                locked_idx = np.where(locked_mask)[0][0]
                current_bbox = bboxes[locked_idx]
                
                if self.locked_subject_box is not None:
                    iou = box_iou(current_bbox, self.locked_subject_box)[0, 0]
                    
                    if iou < self.config.subject_reinit_iou_threshold:
                        self.consecutive_inconsistent_motion[self.locked_subject_id] += 1
                        
                        if self.debug_mode:
                            logger.debug(f"Inconsistent motion for track {self.locked_subject_id} (IOU: {iou:.2f}) | CB: {current_bbox} LKB: {self.locked_subject_box}")
                        if self.consecutive_inconsistent_motion[self.locked_subject_id] >= self.config.inconsistent_motion_threshold:
                            if self.debug_mode:
                                logger.debug("Released the subject lock on %s after sustained inconsistent motion", self.locked_subject_id)
                            self.locked_subject_id = None
                            self.locked_subject_trail.clear()
                            return
                    else:
                        self.consecutive_inconsistent_motion[self.locked_subject_id] = 0
                        self.locked_subject_box = current_bbox.copy()
                        self.locked_subject_history[self.locked_subject_id].append(current_bbox)
                        
                        bottom_middle = ((current_bbox[0] + current_bbox[2]) / 2, current_bbox[3])
                        self.locked_subject_trail.append(bottom_middle)
                
                self.consecutive_misses[self.locked_subject_id] = 0
            else:
                self.consecutive_misses[self.locked_subject_id] += 1
                
                if self.consecutive_misses[self.locked_subject_id] >= self.config.required_consecutive_frames:
                    if self.debug_mode:
                        logger.debug("Released the subject lock on %s after too many missed frames", self.locked_subject_id)
                    self.locked_subject_id = None
                    self.locked_subject_trail.clear()
        
        else:
            current_track_set = set(track_ids)
            
            for track_id in current_track_set:
                self.consecutive_appearances[track_id] += 1
            
            missing_tracks = set(self.consecutive_appearances.keys()) - current_track_set
            for track_id in missing_tracks:
                del self.consecutive_appearances[track_id]
            
            stable_tracks = [tid for tid, count in self.consecutive_appearances.items() 
                           if count >= self.config.required_consecutive_frames]
            
            if stable_tracks:
                if self.locked_subject_box is not None:
                    stable_mask = np.isin(track_ids, stable_tracks)
                    stable_bboxes = bboxes[stable_mask]
                    stable_ids = track_ids[stable_mask]
                    
                    ious = box_iou(self.locked_subject_box, stable_bboxes)[0]
                    
                    # Find best matching candidate based on IOU
                    valid_matches = ious >= self.config.subject_reinit_iou_threshold
                    if np.any(valid_matches):
                        best_idx = np.argmax(ious)
                        self.locked_subject_id = stable_ids[best_idx]
                        self.locked_subject_box = stable_bboxes[best_idx].copy()
                        self.consecutive_misses[self.locked_subject_id] = 0
                        
                        if self.debug_mode:
                            logger.debug("Locked onto subject %s (IOU %.2f)", self.locked_subject_id, ious[best_idx])
                        self.consecutive_appearances.clear()
                else:
                    first_stable_idx = np.where(np.isin(track_ids, stable_tracks[0]))[0][0]
                    self.locked_subject_id = stable_tracks[0]
                    self.locked_subject_box = bboxes[first_stable_idx].copy()
                    self.consecutive_misses[self.locked_subject_id] = 0
                    
                    if self.debug_mode:
                        logger.debug("Locked onto subject %s (first stable track)", self.locked_subject_id)
                    self.consecutive_appearances.clear()
    
    def select_best_detection(self, detections, subject_box):
        if len(detections) <= 1:
            return detections
        best_detection = None
        max_iou = 0
        
        for detection in detections:
            iou = box_iou(detection[:4], subject_box)[0, 0]
            if iou > max_iou:
                max_iou = iou
                best_detection = detection
        filtered = np.array([best_detection]) if best_detection is not None else detections
        if self.debug_mode:
            logger.debug("Filtered candidate boxes for the locked subject: %d -> %d", len(detections), len(filtered))
        return filtered
    
    # ===== Drawing Methods =====
    def draw_tracks(self, frame, online_targets=None, raw_detections=None):
        """Draw various tracking overlays based on configuration."""
        # === 1. Draw raw detection boxes (RED) ===
        if self.config.show_detection_boxes and raw_detections is not None:
            for det in raw_detections:
                if len(det) >= 4:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['red'], 2)
                    draw_label(frame, (x1, y1 - 16), 'Det', size=14,
                               color=self.config.colors['red'])
        
        # === 2. Draw all original MOT tracks (GREEN) ===
        if self.config.show_original_tracks and online_targets is not None:
            for target in online_targets:
                x1, y1, x2, y2 = map(int, target[:4])
                track_id = int(target[4])
                
                if self.config.enable_subject_lock and track_id == self.locked_subject_id:
                    continue
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['green'], 2)
                draw_label(frame, (x1, y1 - 20), f'ID:{track_id}', size=18,
                           color=self.config.colors['green'], bold=True)
                
                if self.config.show_all_trails and track_id in self.track_history:
                    points = [(int((bbox[0] + bbox[2]) / 2), int(bbox[3])) 
                             for bbox in self.track_history[track_id]]
                    
                    if len(points) > 1:
                        color = self.COLORS[track_id % len(self.COLORS)]
                        cv2.polylines(frame, [np.array(points, dtype=np.int32)], 
                                    False, color, 2)
        
        # === 3. Draw the locked subject and its trail (BLUE) ===
        if self.config.enable_subject_lock and self.locked_subject_id is not None:
            if self.config.show_locked_subject and self.locked_subject_box is not None:
                x1, y1, x2, y2 = map(int, self.locked_subject_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['blue'], 3)
                draw_label(frame, (x1, y1 - 28), f'Subject:{self.locked_subject_id}', size=24,
                           color=self.config.colors['blue'], bold=True)
            
            if self.config.show_tracking_tail and len(self.locked_subject_trail) > 1:
                points = np.array(list(self.locked_subject_trail), dtype=np.int32)
                cv2.polylines(frame, [points], False, self.config.colors['blue'], 3)
        
        return frame

    # ===== Detection Processing =====
    def process_detections(self, dets):
        """Process and validate detection results."""
        if len(dets) == 0:
            return np.empty((0, 6))
        valid_class_mask = np.isin(dets[:, 5], self.config.classes)
        return dets[valid_class_mask]
        
    def update_track_history(self, online_targets):
        """Update tracking history for all tracked objects."""
        for target in online_targets:
            track_id = int(target[4])
            bbox = target[:4]
            self.track_history[track_id].append(bbox)
    
    # ===== Main Tracking Method =====
    def track(self, frame, detections) -> TrackResult:
        """Advance the tracker by one frame and return the current tracks.

        Filters ``detections`` to the configured classes, updates the selected
        backend, refreshes trail history, optionally updates the locked subject, and
        renders the overlay. Call this once per frame, in order, for the life of a
        video stream.

        Args:
            frame (np.ndarray): The current BGR frame ``(H, W, 3)``. Used by
                appearance-based backends and for rendering the overlay.
            detections (np.ndarray): Detection rows shaped ``(N, 6)`` as
                ``[x1, y1, x2, y2, conf, cls]``. Build this from a
                [`Result`][physiotrack.Result] with
                ``np.array([[*i.box, i.confidence, i.cls] for i in result])``,
                or pass a raw YOLO detection array directly. Rows whose ``cls`` is not
                in [`TrackerConfig.classes`][physiotrack.TrackerConfig] are dropped.

        Returns:
            TrackResult: A [`TrackResult`][physiotrack.TrackResult] whose
                ``.instances`` carry persistent ``id``s (each an
                [`Instance`][physiotrack.Instance] with ``box``/``cls``/
                ``confidence``), ``.rendered`` is the overlay frame, and
                ``result.plot()`` returns that overlay.

        Example:
            ```python
            import numpy as np
            import physiotrack as pt

            det = pt.Detection.Person()
            tracker = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))

            res = det.predict(frame)
            detections = np.array(
                [[*i.box, i.confidence, i.cls] for i in res], dtype=np.float32
            ) if len(res) else np.empty((0, 6), np.float32)
            result = tracker.track(frame, detections)
            print(result.ids)          # persistent track ids this frame
            annotated = result.plot()  # tracker overlay
            ```

        See Also:
            [`TrackResult`][physiotrack.TrackResult]: the returned result object.
        """
        start = time.time()

        detected_items = self.process_detections(detections)

        online_targets = self.tracker.update(detected_items, frame)

        self.update_track_history(online_targets)

        if self.config.enable_subject_lock:
            self.update_locked_subject(online_targets)

        rendered = self.draw_tracks(frame, online_targets, detected_items)

        inference_time = time.time() - start
        self.inference_times.append(inference_time)

        instances = []
        for t in online_targets:
            t = list(t)
            instances.append(Instance(
                id=int(t[4]) if len(t) > 4 else None,
                box=np.array(t[:4], dtype=np.float32),
                cls=int(t[5]) if len(t) > 5 else None,
                confidence=float(t[6]) if len(t) > 6 else None,
            ))
        return TrackResult(instances=instances, orig_img=frame,
                           rendered=rendered, raw=online_targets)

    def get_avg_inference_time(self):
        """Return the mean per-frame tracking time over a rolling window.

        Returns:
            float: Average time in milliseconds across the last (up to) 100 calls to
                [`track`][physiotrack.Tracker.track], or ``0.0`` if none yet.
        """
        if len(self.inference_times) == 0:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def get_avg_fps(self):
        """Return the mean tracking throughput over a rolling window.

        Returns:
            float: Average frames per second derived from the last (up to) 100 calls
                to [`track`][physiotrack.Tracker.track], or ``0.0`` if none yet.
        """
        if len(self.inference_times) == 0:
            return 0.0
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0