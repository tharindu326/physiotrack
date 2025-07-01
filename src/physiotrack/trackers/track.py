import time
from collections import deque, defaultdict
import numpy as np
import cv2
from .config import Config


class Tracker:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.frame_ID = 0
        self.id_list = []
        self.avg_fps = deque(maxlen=100)
        
        self.student_track_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))
        self.track_history = defaultdict(lambda: deque(maxlen=self.config.trail_length))  # Added back general track history
        self.COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
        
        tracker_type = self.config.tracker_type.lower()
        self.tracker = self._initialize_tracker(tracker_type)
        self.track_ids = []
        
        # Student tracking variables (only used if enable_student_tracking is True)
        self.student_track_id = None
        self.consecutive_appearances = defaultdict(int)
        self.consecutive_misses = defaultdict(int)
        self.last_known_bbox = None
        self.consecutive_inconsistent_motion = defaultdict(int)
    
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
        else:
            supported_trackers = ['OCSort', 'BYTETrack', 'StrongSORT', 'BoostTrack']
            raise ValueError(f'Undefined Tracker. Please use one of: {", ".join(supported_trackers)}')
        
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection)
    
    def update_student_track(self, online_targets):
        """Update student tracking state based on current detections."""
        if not self.config.enable_student_tracking:
            return
            
        current_track_ids = {int(target[4]) for target in online_targets}
        
        # If we have a student track, just check if it's still valid
        if self.student_track_id is not None:
            student_found = False
            
            for target in online_targets:
                track_id = int(target[4])
                if track_id == self.student_track_id:
                    current_bbox = target[:4]
                    
                    # Check if current detection is consistent with last known position
                    if self.last_known_bbox is not None:
                        iou = self.calculate_iou(current_bbox, self.last_known_bbox)
                        if iou < self.config.student_reinit_iou_threshold:
                            self.consecutive_inconsistent_motion[track_id] += 1
                            print(f"Lost student track {track_id} due to inconsistent motion (IOU: {iou:.2f}) | CB: {current_bbox} LKB: {self.last_known_bbox}")
                            if self.consecutive_inconsistent_motion[track_id] >= self.config.inconsistent_motion_threshold:
                                print(f"Lost student track {track_id} due to consistent inconsistent motion")
                                self.student_track_id = None
                                break
                        else:
                            # Reset inconsistent motion counter if motion is consistent
                            self.consecutive_inconsistent_motion[track_id] = 0
                            # Update state for valid student track
                            self.last_known_bbox = current_bbox
                            self.student_track_history[track_id].append(current_bbox)
                    student_found = True
                    break
            
            # If student track wasn't found in current frame
            if not student_found:
                self.consecutive_misses[self.student_track_id] += 1
                if self.consecutive_misses[self.student_track_id] >= self.config.required_consecutive_frames:
                    print(f"Lost student track {self.student_track_id} due to missing frames")
                    self.student_track_id = None
        
        # Only track appearances when looking for a new student track
        else:
            # Update appearance counts for potential new student tracks
            for track_id in current_track_ids:
                self.consecutive_appearances[track_id] += 1
            
            # Clear appearance counts for missing tracks
            for track_id in list(self.consecutive_appearances.keys()):
                if track_id not in current_track_ids:
                    self.consecutive_appearances[track_id] = 0
            
            # Try to initialize new student track
            best_candidate = None
            best_iou = 0
            
            for target in online_targets:
                track_id = int(target[4])
                current_bbox = target[:4]
                
                # Check if track has been stable
                if self.consecutive_appearances[track_id] >= self.config.required_consecutive_frames:
                    # For reinitialization, check overlap with last known position
                    if self.last_known_bbox is not None:
                        iou = self.calculate_iou(current_bbox, self.last_known_bbox)
                        print(iou, current_bbox, self.last_known_bbox)
                        # Keep track of best matching candidate
                        if iou > best_iou and iou >= self.config.student_reinit_iou_threshold:
                            best_iou = iou
                            best_candidate = track_id
                            self.last_known_bbox = current_bbox
                    else:
                        # For initial tracking, take the first stable track
                        best_candidate = track_id
                        self.last_known_bbox = current_bbox
                        break
            
            if best_candidate is not None:
                self.student_track_id = best_candidate
                self.consecutive_misses[best_candidate] = 0  # Reset miss counter for new track
                print(f"Initialized student track {best_candidate}" + 
                    (f" with IOU {best_iou:.2f}" if best_iou > -1 else ""))
                
                # Clear all appearance counters since we don't need them anymore
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
        print(f"Multiple student boxes have filtered. {detections} --> {filtered}")
        return filtered
    
    def draw_tracks(self, frame, online_targets=None, raw_detections=None):
        """Draw various tracking overlays based on configuration.
        
        Args:
            frame: Current video frame
            online_targets: Tracked objects from MOT
            raw_detections: Raw detection boxes before tracking
        """
        # 1. Draw raw detection boxes (RED)
        if self.config.show_detection_boxes and raw_detections is not None:
            for det in raw_detections:
                if len(det) >= 4:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['red'], 2)
                    cv2.putText(frame, 'Det', (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.config.colors['red'], 1)
        
        # 2. Draw all original MOT tracks (GREEN)
        if self.config.show_original_tracks and online_targets is not None:
            for target in online_targets:
                x1, y1, x2, y2 = map(int, target[:4])
                track_id = int(target[4])
                
                # Skip drawing the student track in green if student tracking is enabled
                if self.config.enable_student_tracking and track_id == self.student_track_id:
                    continue
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['green'], 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.colors['green'], 2)
                
                # Draw trails for all tracks if enabled
                if self.config.show_all_trails and track_id in self.track_history:
                    points = []
                    for bbox in self.track_history[track_id]:
                        bottom_middle = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                        points.append(bottom_middle)
                    
                    if len(points) > 1:
                        # Use a different color for each track
                        color = self.COLORS[track_id % len(self.COLORS)]
                        for i in range(len(points) - 1):
                            alpha = (i + 1) / len(points) * 0.5  # Less opacity for general trails
                            thickness = max(1, int(3 * alpha))
                            cv2.line(frame, points[i], points[i + 1], color, thickness)
        
        # 3. Draw student track and tail (BLUE) - only if student tracking is enabled
        if self.config.enable_student_tracking and self.student_track_id is not None:
            # Draw student box
            if self.config.show_student_track and self.last_known_bbox is not None:
                x1, y1, x2, y2 = map(int, self.last_known_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.colors['blue'], 3)
                cv2.putText(frame, f'Student:{self.student_track_id}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.colors['blue'], 2)
            
            # Draw tracking tail
            if self.config.show_tracking_tail and self.student_track_id in self.student_track_history:
                points = []
                for bbox in self.student_track_history[self.student_track_id]:
                    bottom_middle = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                    points.append(bottom_middle)
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        # Progressive opacity and thickness
                        alpha = (i + 1) / len(points) * self.config.tail_opacity
                        thickness = max(1, int(5 * alpha))
                        
                        # Create overlay for transparency effect
                        overlay = frame.copy()
                        cv2.line(overlay, points[i], points[i + 1], self.config.colors['blue'], thickness)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

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
    
    def track(self, frame, detections):
        detected_items = self.process_detections(detections)
            
        online_targets = self.tracker.update(detected_items, frame)
        
        # Update general track history
        self.update_track_history(online_targets)
        
        # Only update student tracking if enabled
        if self.config.enable_student_tracking:
            self.update_student_track(online_targets)
        
        # Draw overlays based on configuration
        frame = self.draw_tracks(frame, online_targets, detected_items)
        
        return frame, online_targets