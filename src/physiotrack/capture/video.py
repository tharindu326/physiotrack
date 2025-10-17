import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from physiotrack.modules.Yolo.classes_and_palettes import COLORS


class Video:
    """
    A video processor class for handling different processing types on video files.
    """
    
    def __init__(self,
                 video_path: Union[str, Path, int],
                 detector=None,
                 pose_estimator=None,
                 segmentator=None,
                 tracker=None,
                 output_path: Optional[Union[str, Path]] = None,
                 required_fps: Optional[int] = None,
                 frame_resize: Optional[Tuple[int, int]] = None,
                 frame_rotate: bool = False,
                 floor_map: Optional[List[Tuple[int, int]]] = None,
                 verbose: bool = False):

        self.video_path = video_path
        # Support both single instance and list of instances for detector and segmentator
        self.detectors = detector if isinstance(detector, list) else ([detector] if detector is not None else [])
        self.segmentators = segmentator if isinstance(segmentator, list) else ([segmentator] if segmentator is not None else [])
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.verbose = verbose
        self.required_fps = required_fps
        self.frame_resize = frame_resize
        self.frame_rotate = frame_rotate
        self.floor_map = floor_map

        # Initialize radar view components
        self.radar_view_enabled = floor_map is not None and len(floor_map) == 4
        self.person_trajectories = {}  # {track_id: [(x, y), ...]}
        self.person_colors = {}  # {track_id: color}
        self.homography_matrix = None
        self.radar_canvas_size = (300, 300)  # Will be updated based on floor_map aspect ratio

        if self.radar_view_enabled:
            self._setup_homography()
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self._setup_source_info()
        
        if output_path:
            self.output_path = Path(output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = Path.cwd()
        
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.video_fps > 0:
            self.video_fps = 30  # Default FPS
            if self.verbose:
                print(f'Using default FPS: {self.video_fps}')
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"Video properties: {self.width}x{self.height}, {self.video_fps} FPS")
            print(f"Source: {self.source_identifier}")
    
    def _setup_source_info(self):
        """Setup source identifier and total frames based on video path type."""
        if isinstance(self.video_path, str):
            path_prefix = ''.join(letter for letter in str(self.video_path).split(':')[0] if letter.isalnum())
            if path_prefix == 'rtsp':
                if self.verbose:
                    print(f'Start processing RTSP stream {self.video_path}')
                source_name = ".".join(self.video_path.split('@')[-1].split('.')[:-1]).replace(':', '-').replace('/', '_')
                self.source_identifier = f'{source_name}'
                self.total_frames = None
            else:
                if self.verbose:
                    print(f'Start processing video {self.video_path}')
                source_name = Path(self.video_path).stem
                self.source_identifier = f'{source_name}'
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames = frame_count if frame_count > 0 else None
        elif isinstance(self.video_path, int):
            if self.verbose:
                print(f'Start processing camera device {self.video_path}')
            self.source_identifier = f'CAM_device_{self.video_path}'
            self.total_frames = None
        else:
            self.source_identifier = 'unknown_source'
            self.total_frames = None
    
    @staticmethod
    def select_frames(camera_fps: int, required_fps: Optional[int]) -> List[int]:
        """
        Select frame indices based on required FPS.
        """
        if required_fps == camera_fps or required_fps is None:
            return [int(i) for i in np.arange(1, camera_fps+1, dtype=float)]
        
        delta = camera_fps - 1
        step = delta / required_fps
        y = np.arange(0, required_fps, dtype=float) * step + 1
        return [int(i) for i in y]
    
    def _setup_homography(self):
        """
        Setup homography matrix for perspective transformation from image coordinates to floor map.
        floor_map should contain 4 points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if not self.floor_map or len(self.floor_map) != 4:
            return

        # Source points from floor_map (in video coordinates)
        src_pts = np.array(self.floor_map, dtype=np.float32)

        # Calculate the bounding box of floor_map to determine aspect ratio
        xs = [pt[0] for pt in self.floor_map]
        ys = [pt[1] for pt in self.floor_map]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Scale to fit within max dimension of 400 while maintaining aspect ratio
        max_dim = 400
        if width > height:
            canvas_width = max_dim
            canvas_height = int(max_dim * height / width)
        else:
            canvas_height = max_dim
            canvas_width = int(max_dim * width / height)

        self.radar_canvas_size = (canvas_width, canvas_height)

        # Destination points (normalized 2D floor space scaled to canvas size)
        dst_pts = np.array([
            [0, 0],                              # top-left
            [canvas_width, 0],                   # top-right
            [canvas_width, canvas_height],       # bottom-right
            [0, canvas_height]                   # bottom-left
        ], dtype=np.float32)

        # Compute homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if self.verbose:
            print(f"Homography matrix computed for floor map: {self.floor_map}")
            print(f"Radar canvas size: {canvas_width}x{canvas_height}")

    def _get_foot_center(self, pose_keypoints):
        """
        Calculate foot center from pose keypoints.
        Returns the midpoint between left and right ankle positions.
        Keypoints format: [{'id': int, 'x': float, 'y': float, 'confidence': float}, ...]
        COCO format: 15=left ankle, 16=right ankle
        """
        if pose_keypoints is None or len(pose_keypoints) == 0:
            return None

        # Create a dictionary for quick lookup by id
        keypoints_dict = {kp['id']: kp for kp in pose_keypoints}

        # COCO keypoint indices: 15=left ankle, 16=right ankle
        left_ankle_id = 15
        right_ankle_id = 16

        left_ankle = keypoints_dict.get(left_ankle_id)
        right_ankle = keypoints_dict.get(right_ankle_id)

        # Check if both ankles are visible (confidence > 0.3)
        if left_ankle and right_ankle:
            if left_ankle['confidence'] > 0.3 and right_ankle['confidence'] > 0.3:
                foot_x = (left_ankle['x'] + right_ankle['x']) / 2
                foot_y = (left_ankle['y'] + right_ankle['y']) / 2
                return (foot_x, foot_y)
            elif left_ankle['confidence'] > 0.3:
                return (left_ankle['x'], left_ankle['y'])
            elif right_ankle['confidence'] > 0.3:
                return (right_ankle['x'], right_ankle['y'])

        return None

    def _transform_to_floor_coords(self, point):
        """
        Transform a point from video coordinates to floor map coordinates using homography.
        """
        if self.homography_matrix is None or point is None:
            return None

        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), self.homography_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def _get_color_for_track_id(self, track_id):
        """
        Get a unique color for a track ID.
        """
        if track_id not in self.person_colors:
            # Use COLORS palette to assign unique colors
            color_names = list(COLORS.keys())
            color_idx = len(self.person_colors) % len(color_names)
            self.person_colors[track_id] = tuple(COLORS[color_names[color_idx]])

        return self.person_colors[track_id]

    def _draw_radar_view(self, radar_canvas):
        """
        Draw the radar view with all person trajectories.
        """
        canvas_width, canvas_height = self.radar_canvas_size

        # Fill background
        radar_canvas.fill(40)

        # Draw floor boundary
        cv2.rectangle(radar_canvas, (0, 0), (canvas_width-1, canvas_height-1), (100, 100, 100), 2)

        # Draw trajectories for each person
        for track_id, trajectory in self.person_trajectories.items():
            if len(trajectory) > 0:
                color = self._get_color_for_track_id(track_id)

                # Draw trajectory path
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i-1]
                    pt2 = trajectory[i]
                    # Ensure points are within bounds
                    if (0 <= pt1[0] < canvas_width and 0 <= pt1[1] < canvas_height and
                        0 <= pt2[0] < canvas_width and 0 <= pt2[1] < canvas_height):
                        cv2.line(radar_canvas, pt1, pt2, color, 2)

                # Draw current position (larger circle)
                if len(trajectory) > 0:
                    current_pos = trajectory[-1]
                    if 0 <= current_pos[0] < canvas_width and 0 <= current_pos[1] < canvas_height:
                        cv2.circle(radar_canvas, current_pos, 6, color, -1)
                        cv2.circle(radar_canvas, current_pos, 8, (255, 255, 255), 1)
                        # Add track ID label
                        cv2.putText(radar_canvas, f"ID:{track_id}",
                                   (current_pos[0] + 10, current_pos[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add title
        cv2.putText(radar_canvas, "Floor Map", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return radar_canvas

    def preprocess_frame(self, frame):
        """
        Preprocess frame with resize and rotation if specified.
        """
        if self.frame_resize:
            frame = cv2.resize(frame, self.frame_resize)
        if self.frame_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
        
    def run(self, 
            output_video_path: Optional[Union[str, Path]] = None,
            output_json_path: Optional[Union[str, Path]] = None,
            progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process the video for pose estimation, segmentation, or detection.
        """
        
        pbar = None
        if self.total_frames and self.verbose:
            pbar = tqdm(total=self.total_frames, desc=f'Processing {self.source_identifier}')
        
        selected_frame_ids = self.select_frames(self.video_fps, self.required_fps)
        out_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if self.frame_resize:
                output_width, output_height = self.frame_resize
            else:
                output_width, output_height = self.width, self.height
            if self.frame_rotate:
                output_width, output_height = output_height, output_width
            
            effective_fps = self.required_fps if self.required_fps else self.video_fps
            out_writer = cv2.VideoWriter(str(output_video_path), fourcc, effective_fps, 
                                       (output_width, output_height))
        
        all_detection_data = []
        frame_count = 0
        frame_filter_count = 1
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if pbar:
                pbar.update(1)
            
            video_timestamp = round(frame_count / self.video_fps, 3)
            frame = self.preprocess_frame(frame)
            
            if frame_filter_count in selected_frame_ids:
                boxes = None
                result_frame = frame.copy()
                segmentation_img = None
                online_targets = []  # Initialize online_targets for each frame

                # Run all detectors and combine results
                if len(self.detectors) > 0:
                    all_detections = []
                    combined_frame = frame.copy()

                    # Use predefined colors from COLORS palette
                    color_names = list(COLORS.keys())

                    for idx, detector in enumerate(self.detectors):
                        results, detected_frame = detector.detect(frame)
                        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

                        # Get color for this detector
                        color_name = color_names[idx % len(color_names)]
                        color = tuple(COLORS[color_name])

                        # Draw boxes with unique color for this detector
                        for det in detections:
                            x1, y1, x2, y2, conf, cls = det
                            cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            # Add label with detector index
                            label = f"D{idx}-C{int(cls)}: {conf:.2f}"
                            cv2.putText(combined_frame, label, (int(x1), int(y1)-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        all_detections.append(detections)

                    # Combine all detections into one array
                    if len(all_detections) > 0:
                        detections = np.vstack(all_detections)
                        boxes = detections[:, :-2].astype(int)
                        frame = combined_frame

                        if self.pose_estimator and self.pose_estimator.__class__.__name__ != "Custom":
                            raise ValueError("Please use Pose.Custom class if you want to use a custom detector with the Video class. Alternatively, you can: 1) Not provide a detector to use the default detector (Models.Detection.YOLO.PERSON.m_person) of the Pose estimator, or 2) Pass a detector to the Pose estimator from the Models zoo if you don't wish to use the Tracker class.")

                        if self.tracker is not None:
                            frame, online_targets = self.tracker.track(frame, detections)
                            # filter boxes based on student track to obtain only the student box
                            if self.tracker.student_track_id is not None and self.tracker.last_known_bbox is not None:
                                boxes = np.array(self.tracker.last_known_bbox, dtype=int).reshape(1, -1)

                if self.pose_estimator is not None:
                    result_frame, results = self.pose_estimator.estimate(frame, boxes)
                    pose_results = results.to_json()['detections']

                # Run all segmentators and combine results
                if len(self.segmentators) > 0:
                    h, w = frame.shape[:2]
                    combined_segmentation_map = np.zeros((h, w), dtype=np.uint8)
                    combined_segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

                    # Use predefined colors from COLORS palette
                    color_names = list(COLORS.keys())
                    color_list = [COLORS[name] for name in color_names]

                    for seg_idx, segmentator in enumerate(self.segmentators):
                        # Determine which bboxes to use for filtering
                        filter_bboxes = None
                        if segmentator.bbox_filter and len(all_detections) > 0:
                            # Use specific detector if detector_index is set
                            if segmentator.detector_index is not None and segmentator.detector_index < len(all_detections):
                                detections = all_detections[segmentator.detector_index]

                                # Filter by class if detector_class_filter is specified
                                if segmentator.detector_class_filter is not None:
                                    class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                                    class_mask = np.isin(detections[:, 5], class_filter)
                                    filter_bboxes = detections[class_mask][:, :4] if np.any(class_mask) else None
                                else:
                                    filter_bboxes = detections[:, :4]
                            else:
                                # Use all detections if no specific detector is specified
                                all_dets = np.vstack(all_detections) if len(all_detections) > 0 else None
                                if all_dets is not None:
                                    # Filter by class if detector_class_filter is specified
                                    if segmentator.detector_class_filter is not None:
                                        class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                                        class_mask = np.isin(all_dets[:, 5], class_filter)
                                        filter_bboxes = all_dets[class_mask][:, :4] if np.any(class_mask) else None
                                    else:
                                        filter_bboxes = all_dets[:, :4]

                        # Segment with optional bbox filtering
                        seg_img, seg_map = segmentator.segment(frame, bboxes=filter_bboxes)

                        # Remap class IDs to make them unique across segmentators
                        # Offset by segmentator index * 100 to avoid conflicts
                        unique_classes = np.unique(seg_map)
                        for cls_id in unique_classes:
                            if cls_id > 0:  # Skip background
                                new_cls_id = seg_idx * 100 + cls_id
                                mask = seg_map == cls_id

                                # Assign color from palette based on segmentator and class
                                # Use different color for each segmentator-class combination
                                color_idx = (seg_idx * 10 + cls_id) % len(color_list)
                                color = color_list[color_idx]

                                # Apply color to segmentation
                                combined_segmentation_map[mask] = new_cls_id
                                combined_segmentation_img[mask] = color

                    # Overlay combined segmentation on result_frame
                    result_frame = cv2.addWeighted(result_frame, 0.7, combined_segmentation_img, 0.3, 0)

                # Track foot centers and update radar view
                if self.radar_view_enabled and self.tracker is not None and self.pose_estimator is not None:
                    # Extract track IDs and their corresponding poses
                    if len(online_targets) > 0:
                        # online_targets format: [[x1, y1, x2, y2, track_id, cls, conf], ...]
                        for target in online_targets:
                            track_id = int(target[4])  # track_id is at index 4
                            bbox = target[:4]  # [x1, y1, x2, y2]

                            # Find the pose result that matches this bbox (by proximity)
                            for pose_result in pose_results:
                                if 'keypoints' in pose_result and pose_result['keypoints'] is not None:
                                    # Get foot center from keypoints
                                    keypoints = pose_result['keypoints']
                                    foot_center = self._get_foot_center(keypoints)

                                    if foot_center is not None:
                                        # Check if foot is inside bbox (simple matching)
                                        x1, y1, x2, y2 = bbox
                                        if x1 <= foot_center[0] <= x2 and y1 <= foot_center[1] <= y2:
                                            # Transform to floor coordinates
                                            floor_coords = self._transform_to_floor_coords(foot_center)

                                            if floor_coords is not None:
                                                # Initialize trajectory for new track ID
                                                if track_id not in self.person_trajectories:
                                                    self.person_trajectories[track_id] = []

                                                # Add to trajectory (keep last 100 points)
                                                self.person_trajectories[track_id].append(floor_coords)
                                                if len(self.person_trajectories[track_id]) > 100:
                                                    self.person_trajectories[track_id].pop(0)

                                                break  # Found matching pose for this track

                    # Create and overlay radar view
                    canvas_width, canvas_height = self.radar_canvas_size
                    radar_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    radar_canvas = self._draw_radar_view(radar_canvas)

                    # Overlay radar view on bottom right corner
                    h, w = result_frame.shape[:2]
                    margin = 10
                    radar_h, radar_w = canvas_height, canvas_width

                    # Position: bottom right with margin
                    y1 = h - radar_h - margin
                    y2 = h - margin
                    x1 = w - radar_w - margin
                    x2 = w - margin

                    # Ensure coordinates are valid
                    if y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
                        # Add semi-transparent background
                        overlay = result_frame.copy()
                        cv2.rectangle(overlay, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 0), -1)
                        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)

                        # Overlay radar view
                        result_frame[y1:y2, x1:x2] = radar_canvas

                # Store frame data
                frame_data = {
                    'frame_id': frame_count,
                    'timestamp': video_timestamp,
                }
                
                # Add tracking box if tracker is available
                if self.tracker is not None and hasattr(self.tracker, 'last_known_bbox') and self.tracker.last_known_bbox is not None:
                    frame_data['track_box'] = self.tracker.last_known_bbox.tolist()
                
                # Add pose results if available
                if self.pose_estimator is not None:
                    frame_data['detections'] = pose_results
                
                all_detection_data.append(frame_data)

                if out_writer:
                    out_writer.write(result_frame)
                    
                if progress_callback:
                    progress_callback(frame_count, self.total_frames, results)
                
            frame_count += 1
            frame_filter_count = frame_filter_count + 1 if frame_filter_count < self.video_fps else 1
            
        if pbar:
            pbar.close()
            
        self.cap.release()
        if out_writer:
            out_writer.release()
            
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        if self.verbose:
            print(f"Processing complete: {frame_count} frames, {avg_fps:.2f} FPS, "
                  f"{len(all_detection_data)} total detections")
                  
        if output_json_path:
            self._save_json_data(all_detection_data, output_json_path)
        
        return all_detection_data
    
    def batch_run(self, 
                  input_paths: List[Union[str, Path]],
                  output_dir: Union[str, Path],
                  save_videos: bool = True,
                  save_json: bool = True):
        """
        Process multiple videos in batch.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for input_path in input_paths:
            input_path = Path(input_path)
            video_name = input_path.stem
            
            if self.verbose:
                print(f"Processing video: {input_path}")
            
            video_output_path = output_dir / f"{video_name}_processed.mp4" if save_videos else None
            json_output_path = output_dir / f"{video_name}_result.json" if save_json else None
            
            detection_data = self.run(video_output_path, json_output_path)
            results[video_name] = detection_data
            
        return results
    
    def _save_json_data(self, detection_data: List[Dict[str, Any]], output_path: Union[str, Path]):
        """Save detection data to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        if self.verbose:
            print(f"JSON data saved to: {output_path}")