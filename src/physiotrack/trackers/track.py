import time
from collections import deque, defaultdict
import numpy as np
import cv2
from .config import Config


class Tracker:
    def __init__(self, config=None):
        """Initialize tracker with configuration and tracking variables."""
        self.config = config if config is not None else Config()
        self.frame_ID = 0
        self.id_list = []
        self.avg_fps = deque(maxlen=100)
        
        # Track history storage
        self.student_track_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))
        self.track_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))
        self.COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
        
        # Initialize tracker
        tracker_type = self.config.tracker_type.lower()
        self.tracker = self._initialize_tracker(tracker_type)
        self.track_ids = []
        
        # Student tracking state
        self.student_track_id = None
        self.consecutive_appearances = defaultdict(int)
        self.consecutive_misses = defaultdict(int)
        self.last_known_bbox = None
        self.consecutive_inconsistent_motion = defaultdict(int)
        self.student_trail_points = deque(maxlen=self.config.trail_length)
        self.debug_mode = False
    
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
        # === Initialize new student track ===
        else:
            supported_trackers = ['OCSort', 'BYTETrack', 'StrongSORT', 'BoostTrack']
            raise ValueError(f'Undefined Tracker. Please use one of: {", ".join(supported_trackers)}')
    
    # ===== IOU Calculation Methods =====
    @staticmethod
    def calculate_iou_vectorized(boxes1, boxes2):
        """Vectorized IOU calculation for better performance."""
        boxes1 = np.atleast_2d(boxes1)
        boxes2 = np.atleast_2d(boxes2)
        
        x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, np.newaxis] + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    # ===== Student Tracking Logic =====
    def update_student_track(self, online_targets):
        """Update student tracking state based on current detections."""
        if not self.config.enable_student_tracking or len(online_targets) == 0:
            return
        
        targets_array = np.array(online_targets)
        track_ids = targets_array[:, 4].astype(int)
        bboxes = targets_array[:, :4]
        
        # === Check existing student track ===
        if self.student_track_id is not None:
            student_mask = track_ids == self.student_track_id
            
            if np.any(student_mask):
                student_idx = np.where(student_mask)[0][0]
                current_bbox = bboxes[student_idx]
                
                if self.last_known_bbox is not None:
                    iou = self.calculate_iou(current_bbox, self.last_known_bbox)
                    
                    if iou < self.config.student_reinit_iou_threshold:
                        self.consecutive_inconsistent_motion[self.student_track_id] += 1
                        
                        if self.debug_mode:
                            print(f"Inconsistent motion for track {self.student_track_id} (IOU: {iou:.2f}) | CB: {current_bbox} LKB: {self.last_known_bbox}")
                        
                        if self.consecutive_inconsistent_motion[self.student_track_id] >= self.config.inconsistent_motion_threshold:
                            if self.debug_mode:
                                print(f"Lost student track {self.student_track_id} due to consistent inconsistent motion")
                            self.student_track_id = None
                            self.student_trail_points.clear()
                            return
                    else:
                        self.consecutive_inconsistent_motion[self.student_track_id] = 0
                        self.last_known_bbox = current_bbox.copy()
                        self.student_track_history[self.student_track_id].append(current_bbox)
                        
                        bottom_middle = ((current_bbox[0] + current_bbox[2]) / 2, current_bbox[3])
                        self.student_trail_points.append(bottom_middle)
                
                self.consecutive_misses[self.student_track_id] = 0
            else:
                self.consecutive_misses[self.student_track_id] += 1
                
                if self.consecutive_misses[self.student_track_id] >= self.config.required_consecutive_frames:
                    if self.debug_mode:
                        print(f"Lost student track {self.student_track_id} due to missing frames")
                    self.student_track_id = None
                    self.student_trail_points.clear()
        
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
                if self.last_known_bbox is not None:
                    stable_mask = np.isin(track_ids, stable_tracks)
                    stable_bboxes = bboxes[stable_mask]
                    stable_ids = track_ids[stable_mask]
                    
                    ious = self.calculate_iou_vectorized(
                        np.array([self.last_known_bbox]), 
                        stable_bboxes
                    ).flatten()
                    
                    # Find best matching candidate based on IOU
                    valid_matches = ious >= self.config.student_reinit_iou_threshold
                    if np.any(valid_matches):
                        best_idx = np.argmax(ious)
                        self.student_track_id = stable_ids[best_idx]
                        self.last_known_bbox = stable_bboxes[best_idx].copy()
                        self.consecutive_misses[self.student_track_id] = 0
                        
                        if self.debug_mode:
                            print(f"Initialized student track {self.student_track_id} with IOU {ious[best_idx]:.2f}")
                        
                        self.consecutive_appearances.clear()
                else:
                    first_stable_idx = np.where(np.isin(track_ids, stable_tracks[0]))[0][0]
                    self.student_track_id = stable_tracks[0]
                    self.last_known_bbox = bboxes[first_stable_idx].copy()
                    self.consecutive_misses[self.student_track_id] = 0
                    
                    if self.debug_mode:
                        print(f"Initialized first student track {self.student_track_id}")
                    
                    self.consecutive_appearances.clear()
    
    def select_best_detection(self, detections, student_bbox):
        if len(detections) <= 1:
            return detections
        best_detection = None
        max_iou = 0
        
        for detection in detections:
            iou = self.calculate_iou(detection[:4], student_bbox)
            if iou > max_iou:
                max_iou = iou
                best_detection = detection
        filtered = np.array([best_detection]) if best_detection is not None else detections
        if self.debug_mode:
            print(f"Multiple student boxes filtered. {len(detections)} --> {len(filtered)}")
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
                    cv2.putText(frame, 'Det', (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.config.colors['red'], 1)
        
        # === 2. Draw all original MOT tracks (GREEN) ===
        if self.config.show_original_tracks and online_targets is not None:
            for target in online_targets:
                x1, y1, x2, y2 = map(int, target[:4])
                track_id = int(target[4])
                
                if self.config.enable_student_tracking and track_id == self.student_track_id:
                    continue
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['green'], 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.colors['green'], 2)
                
                if self.config.show_all_trails and track_id in self.track_history:
                    points = [(int((bbox[0] + bbox[2]) / 2), int(bbox[3])) 
                             for bbox in self.track_history[track_id]]
                    
                    if len(points) > 1:
                        color = self.COLORS[track_id % len(self.COLORS)]
                        cv2.polylines(frame, [np.array(points, dtype=np.int32)], 
                                    False, color, 2)
        
        # === 3. Draw student track and tail (BLUE) ===
        if self.config.enable_student_tracking and self.student_track_id is not None:
            if self.config.show_student_track and self.last_known_bbox is not None:
                x1, y1, x2, y2 = map(int, self.last_known_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['blue'], 3)
                cv2.putText(frame, f'Student:{self.student_track_id}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.colors['blue'], 2)
            
            if self.config.show_tracking_tail and len(self.student_trail_points) > 1:
                points = np.array(list(self.student_trail_points), dtype=np.int32)
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
    def track(self, frame, detections):
        detected_items = self.process_detections(detections)
            
        online_targets = self.tracker.update(detected_items, frame)
        
        self.update_track_history(online_targets)
        
        if self.config.enable_student_tracking:
            self.update_student_track(online_targets)
        
        frame = self.draw_tracks(frame, online_targets, detected_items)
        
        return frame, online_targets