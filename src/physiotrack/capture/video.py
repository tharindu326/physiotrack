import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from physiotrack.modules.Yolo.classes_and_palettes import COLORS
from physiotrack.core.radar_view import RadarView


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
                 verbose: bool = False,
                 show_fps: bool = False):

        self.video_path = video_path
        # Support both single instance and list of instances for detector and segmentator
        self.detectors = detector if isinstance(detector, list) else ([detector] if detector is not None else [])
        self.segmentators = segmentator if isinstance(segmentator, list) else ([segmentator] if segmentator is not None else [])
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.verbose = verbose
        self.show_fps = show_fps
        self.required_fps = required_fps
        self.frame_resize = frame_resize
        self.frame_rotate = frame_rotate
        self.floor_map = floor_map

        # Initialize radar view
        self.radar_view = RadarView(floor_map) if floor_map else None

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
            if self.frame_resize:
                output_width, output_height = self.frame_resize
            else:
                output_width, output_height = self.width, self.height
            if self.frame_rotate:
                output_width, output_height = output_height, output_width

            effective_fps = self.required_fps if self.required_fps else self.video_fps

            # Use avc1 codec for MP4 output
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out_writer = cv2.VideoWriter(str(output_video_path), fourcc, effective_fps,
                                       (output_width, output_height))

            if self.verbose:
                print(f"Using H.264 (avc1) codec for video encoding")
        
        all_detection_data = []
        frame_count = 0
        frame_filter_count = 1
        start_time = time.time()
        last_fps_print_time = start_time
        fps_print_interval = 2.0  # Print FPS every 2 seconds for real-time monitoring

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
                result_frame = frame
                segmentation_img = None
                online_targets = []  # Initialize online_targets for each frame

                # Run all detectors and combine results
                if len(self.detectors) > 0:
                    all_detections = []
                    combined_frame = frame.copy() if len(self.detectors) > 0 else frame

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
                elif result_frame is frame:
                    result_frame = frame.copy()

                if len(self.segmentators) > 0:
                    h, w = frame.shape[:2]
                    # Pre-allocate once outside loop
                    combined_segmentation_map = np.zeros((h, w), dtype=np.uint8)
                    combined_segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

                    # Use predefined colors from COLORS palette
                    color_names = list(COLORS.keys())
                    color_list = [COLORS[name] for name in color_names]

                    # Collect all segmentation results first
                    seg_results = []
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
                        seg_results.append((seg_idx, seg_img, seg_map))

                    # Merge all segmentations using palette colors
                    # Later segmentators overwrite earlier ones (higher priority for specific segments)
                    for seg_idx, seg_img, seg_map in seg_results:
                        # Get mask for all non-background pixels at once
                        seg_mask = seg_map > 0
                        if not np.any(seg_mask):
                            continue

                        # Remap class IDs (vectorized)
                        remapped_map = seg_map.copy()
                        remapped_map[seg_mask] = (seg_idx * 100) + seg_map[seg_mask]

                        # Create color image for this segmentation (vectorized)
                        # Get unique classes to build color mapping
                        unique_classes = np.unique(seg_map[seg_mask])
                        seg_colored = np.zeros_like(combined_segmentation_img)

                        for cls_id in unique_classes:
                            if cls_id > 0:  # Skip background
                                cls_mask = seg_map == cls_id
                                color_idx = (seg_idx * 10 + cls_id) % len(color_list)
                                seg_colored[cls_mask] = color_list[color_idx]

                        # Apply to combined maps (vectorized assignment)
                        combined_segmentation_map[seg_mask] = remapped_map[seg_mask]
                        combined_segmentation_img[seg_mask] = seg_colored[seg_mask]

                    # Overlay combined segmentation on result_frame
                    result_frame = cv2.addWeighted(result_frame, 0.7, combined_segmentation_img, 0.3, 0)

                # Update and attach radar view
                if self.radar_view and self.tracker is not None and self.pose_estimator is not None:
                    self.radar_view.update(online_targets, pose_results)
                    result_frame = self.radar_view.attach_to_frame(result_frame)

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

            # Real-time FPS monitoring
            if self.show_fps:
                current_time = time.time()
                if current_time - last_fps_print_time >= fps_print_interval:
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0

                    # Build component-wise FPS dict
                    fps_dict = {"Pipeline": f"{current_fps:.2f}"}

                    if len(self.detectors) > 0:
                        for idx, detector in enumerate(self.detectors):
                            if hasattr(detector, 'get_avg_fps'):
                                det_fps = detector.get_avg_fps()
                                fps_dict[f"Det[{idx}]"] = f"{det_fps:.2f}"

                    if self.pose_estimator and hasattr(self.pose_estimator, 'pose_estimator'):
                        if hasattr(self.pose_estimator.pose_estimator, 'get_avg_fps'):
                            pose_fps = self.pose_estimator.pose_estimator.get_avg_fps()
                            fps_dict["Pose"] = f"{pose_fps:.2f}"

                    if self.tracker and hasattr(self.tracker, 'get_avg_fps'):
                        tracker_fps = self.tracker.get_avg_fps()
                        fps_dict["Track"] = f"{tracker_fps:.2f}"

                    if len(self.segmentators) > 0:
                        for idx, segmentor in enumerate(self.segmentators):
                            if hasattr(segmentor, 'get_avg_fps'):
                                seg_fps = segmentor.get_avg_fps()
                                fps_dict[f"Seg[{idx}]"] = f"{seg_fps:.2f}"

                    if pbar:
                        # Update progress bar with FPS info
                        pbar.set_postfix(fps_dict)
                    else:
                        # Print FPS on same line if no progress bar
                        fps_parts = [f"{k}: {v}" for k, v in fps_dict.items()]
                        fps_display = " | ".join(fps_parts)
                        print(f"\r{fps_display}", end='', flush=True)

                    last_fps_print_time = current_time

            frame_count += 1
            frame_filter_count = frame_filter_count + 1 if frame_filter_count < self.video_fps else 1
            
        if pbar:
            pbar.close()
            
        self.cap.release()
        if out_writer:
            out_writer.release()
            
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        # Print detailed FPS statistics
        if self.show_fps:
            print("\n\n" + "="*60)  # Extra newline to clear real-time FPS line
            print("PERFORMANCE METRICS")
            print("="*60)
            print(f"Overall Pipeline FPS: {avg_fps:.2f}")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print("-"*60)

            # Component-level FPS
            if len(self.detectors) > 0:
                for idx, detector in enumerate(self.detectors):
                    if hasattr(detector, 'get_avg_fps'):
                        det_fps = detector.get_avg_fps()
                        det_time = detector.get_avg_inference_time()
                        print(f"Detector[{idx}] FPS: {det_fps:.2f} ({det_time:.2f}ms per frame)")

            if self.pose_estimator and hasattr(self.pose_estimator, 'pose_estimator'):
                if hasattr(self.pose_estimator.pose_estimator, 'get_avg_fps'):
                    pose_fps = self.pose_estimator.pose_estimator.get_avg_fps()
                    pose_time = self.pose_estimator.pose_estimator.get_avg_inference_time()
                    print(f"Pose Estimator FPS: {pose_fps:.2f} ({pose_time:.2f}ms per frame)")

            if self.tracker and hasattr(self.tracker, 'get_avg_fps'):
                tracker_fps = self.tracker.get_avg_fps()
                tracker_time = self.tracker.get_avg_inference_time()
                print(f"Tracker FPS: {tracker_fps:.2f} ({tracker_time:.2f}ms per frame)")

            if len(self.segmentators) > 0:
                for idx, segmentor in enumerate(self.segmentators):
                    if hasattr(segmentor, 'get_avg_fps'):
                        seg_fps = segmentor.get_avg_fps()
                        seg_time = segmentor.get_avg_inference_time()
                        print(f"Segmentor[{idx}] FPS: {seg_fps:.2f} ({seg_time:.2f}ms per frame)")

            print("="*60)
                  
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