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
    A video processor class for handling different processing types on video files with batch processing support.
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
                 floor_map_background: Optional[Union[str, np.ndarray]] = None,
                 verbose: bool = False,
                 show_fps: bool = False,
                 batch_size: int = 1):  # New parameter for batch processing

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
        self.batch_size = max(1, batch_size)  # Ensure batch size is at least 1

        # Initialize radar view with background mode
        self.radar_view = RadarView(
            floor_map=floor_map,
            background=floor_map_background
        ) if floor_map else None

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
            print(f"Batch size: {self.batch_size}")
    
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
    
    def process_batch_detections(self, frames_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process a batch of frames through detectors.
        
        Returns:
            List of tuples (combined_frame, all_detections) for each frame in batch
        """
        batch_results = []
        
        if len(self.detectors) == 0:
            # No detectors, return empty results for each frame
            for frame in frames_batch:
                batch_results.append((frame, []))
            return batch_results
        
        # Prefer batched detection if available
        if len(self.detectors) > 0 and hasattr(self.detectors[0], 'detect_batch'):
            det = self.detectors[0]
            det_outputs = det.detect_batch(frames_batch)
            batch_results = []
            color_names = list(COLORS.keys())
            for frame, (results, detected_frame) in zip(frames_batch, det_outputs):
                detections = results[0].boxes.data.cpu().numpy()  # match single-frame schema
                combined_frame = frame.copy()
                color = tuple(COLORS[color_names[0]])
                for det_box in (detections[:, :4].astype(int) if detections.size else []):
                    x1, y1, x2, y2 = det_box
                    cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                batch_results.append((combined_frame, [detections]))
            return batch_results

        # Process each frame in batch (fallback)
        for frame in frames_batch:
            all_detections = []
            combined_frame = frame.copy()
            
            # Use predefined colors from COLORS palette
            color_names = list(COLORS.keys())
            
            for idx, detector in enumerate(self.detectors):
                # YOLO can handle single frame or batch - we pass single for now
                # TODO: Update to pass batch directly when YOLO batch inference is confirmed
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
            
            batch_results.append((combined_frame, all_detections))
        
        return batch_results
    
    def process_batch_pose(self, frames_batch: List[np.ndarray], boxes_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, Any]]:
        """
        Process a batch of frames through pose estimator.
        
        Returns:
            List of tuples (result_frame, pose_results) for each frame in batch
        """
        batch_results = []
        
        if self.pose_estimator is None:
            for frame in frames_batch:
                batch_results.append((frame, None))
            return batch_results
        
        # If pose supports batch, call once
        if hasattr(self.pose_estimator, 'estimate_batch'):
            batched = self.pose_estimator.estimate_batch(frames_batch, boxes_batch)
            for (result_frame, results) in batched:
                pose_results = results.to_json()['detections']
                batch_results.append((result_frame, pose_results))
            return batch_results

        # Fallback per-frame
        for frame, boxes in zip(frames_batch, boxes_batch):
            if boxes is not None and len(boxes) > 0:
                result_frame, results = self.pose_estimator.estimate(frame, boxes)
                pose_results = results.to_json()['detections']
            else:
                result_frame = frame
                pose_results = []
            batch_results.append((result_frame, pose_results))
        
        return batch_results
    
    def process_batch_segmentation(self, frames_batch: List[np.ndarray], 
                                  all_detections_batch: List[List[np.ndarray]],
                                  overlay_frames: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Process a batch of frames through segmentators.
        
        Returns:
            List of result frames with segmentation overlay
        """
        batch_results = []

        if len(self.segmentators) == 0:
            return frames_batch

        # Prepare colors for merging
        color_names = list(COLORS.keys())
        color_list = [COLORS[name] for name in color_names]

        # First, gather segmentation maps per segmentator, using segment_batch if available
        base_frames_for_overlay = overlay_frames if overlay_frames is not None else frames_batch
        seg_maps_per_segmentator = []  # List[List[np.ndarray]] indexed [seg_idx][frame_idx]
        for seg_idx, segmentator in enumerate(self.segmentators):
            # Try wrapper's batch API, then inner segmentor batch API, else per-frame
            batched_outputs = None

            if hasattr(segmentator, 'segmentor') and hasattr(segmentator.segmentor, 'segment_batch'):
                batched_outputs = segmentator.segmentor.segment_batch(frames_batch)

            seg_maps_for_this_segmentator = []
            if batched_outputs is not None:
                # Expect list of (seg_img, seg_map)
                for (_, seg_map) in batched_outputs:
                    seg_maps_for_this_segmentator.append(seg_map)
            else:
                # Fallback: per-frame segmentation without filtering; we'll filter below
                for frame in frames_batch:
                    seg_img, seg_map = segmentator.segment(frame)
                    seg_maps_for_this_segmentator.append(seg_map)

            # Apply optional bbox filtering per frame, matching single-frame logic
            for frame_idx, seg_map in enumerate(seg_maps_for_this_segmentator):
                filter_bboxes = None
                all_detections = all_detections_batch[frame_idx]
                if getattr(segmentator, 'bbox_filter', False) and len(all_detections) > 0:
                    if segmentator.detector_index is not None and segmentator.detector_index < len(all_detections):
                        detections = all_detections[segmentator.detector_index]
                        if segmentator.detector_class_filter is not None:
                            class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                            class_mask = np.isin(detections[:, 5], class_filter)
                            filter_bboxes = detections[class_mask][:, :4] if np.any(class_mask) else None
                        else:
                            filter_bboxes = detections[:, :4]
                    else:
                        all_dets = np.vstack(all_detections) if len(all_detections) > 0 else None
                        if all_dets is not None:
                            if segmentator.detector_class_filter is not None:
                                class_filter = segmentator.detector_class_filter if isinstance(segmentator.detector_class_filter, list) else [segmentator.detector_class_filter]
                                class_mask = np.isin(all_dets[:, 5], class_filter)
                                filter_bboxes = all_dets[class_mask][:, :4] if np.any(class_mask) else None
                            else:
                                filter_bboxes = all_dets[:, :4]

                if filter_bboxes is not None and len(filter_bboxes) > 0:
                    h, w = seg_map.shape[:2]
                    filtered_map = np.zeros((h, w), dtype=seg_map.dtype)
                    for bbox in filter_bboxes:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        filtered_map[y1:y2, x1:x2] = seg_map[y1:y2, x1:x2]
                    seg_maps_for_this_segmentator[frame_idx] = filtered_map

            seg_maps_per_segmentator.append(seg_maps_for_this_segmentator)

        # Now merge per-frame across segmentators and overlay
        for frame_idx, frame in enumerate(frames_batch):
            h, w = frame.shape[:2]
            combined_segmentation_map = np.zeros((h, w), dtype=np.uint8)
            combined_segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

            for seg_idx, seg_maps in enumerate(seg_maps_per_segmentator):
                seg_map = seg_maps[frame_idx]
                seg_mask = seg_map > 0
                if not np.any(seg_mask):
                    continue

                # Remap class IDs and colorize
                remapped_map = seg_map.copy()
                remapped_map[seg_mask] = (seg_idx * 100) + seg_map[seg_mask]

                unique_classes = np.unique(seg_map[seg_mask])
                seg_colored = np.zeros_like(combined_segmentation_img)
                for cls_id in unique_classes:
                    if cls_id > 0:
                        cls_mask = seg_map == cls_id
                        color_idx = (seg_idx * 10 + int(cls_id)) % len(color_list)
                        seg_colored[cls_mask] = color_list[color_idx]

                combined_segmentation_map[seg_mask] = remapped_map[seg_mask]
                combined_segmentation_img[seg_mask] = seg_colored[seg_mask]

            base_frame = base_frames_for_overlay[frame_idx]
            result_frame = cv2.addWeighted(base_frame, 0.7, combined_segmentation_img, 0.3, 0)
            batch_results.append(result_frame)

        return batch_results
        
    def run(self, 
            output_video_path: Optional[Union[str, Path]] = None,
            output_json_path: Optional[Union[str, Path]] = None,
            progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process the video with batch processing support.
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
        
        # Batch collection variables
        frame_batch = []
        frame_batch_metadata = []
        
        # Flag to track if we need to extract floor background from first frame
        extract_floor_from_first_frame = (
            self.radar_view is not None and 
            self.radar_view.background_mode in ["auto", "extract"]
        )

        while True:
            # Collect frames into batch
            batch_ready = False
            
            while len(frame_batch) < self.batch_size:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video, process remaining batch if any
                    if len(frame_batch) > 0:
                        batch_ready = True
                    break
                
                if pbar:
                    pbar.update(1)
                
                video_timestamp = round(frame_count / self.video_fps, 3)
                frame = self.preprocess_frame(frame)
                
                # Extract floor area from first frame if needed
                if extract_floor_from_first_frame and frame_count == 0:
                    if self.verbose:
                        print("Extracting floor area from first frame and transforming to top-down view...")
                    self.radar_view.set_background_from_frame(frame)
                    extract_floor_from_first_frame = False  # Only do this once
                
                # Check if this frame should be processed
                if frame_filter_count in selected_frame_ids:
                    frame_batch.append(frame)
                    frame_batch_metadata.append({
                        'frame_id': frame_count,
                        'timestamp': video_timestamp,
                        'frame_filter_count': frame_filter_count
                    })
                
                frame_count += 1
                frame_filter_count = frame_filter_count + 1 if frame_filter_count < self.video_fps else 1
                
                # Check if batch is full
                if len(frame_batch) == self.batch_size:
                    batch_ready = True
                    break
            
            # Process batch if ready
            if batch_ready and len(frame_batch) > 0:
                # Step 1: Batch detection
                detection_results = self.process_batch_detections(frame_batch)
                
                # Extract frames and detections from results
                frames_with_detections = [r[0] for r in detection_results]
                all_detections_batch = [r[1] for r in detection_results]
                
                # Step 2: Extract boxes for pose estimation
                boxes_batch = []
                for all_detections in all_detections_batch:
                    if len(all_detections) > 0:
                        detections = np.vstack(all_detections)
                        boxes = detections[:, :-2].astype(int)
                        boxes_batch.append(boxes)
                    else:
                        boxes_batch.append(None)
                
                # Step 3: Batch pose estimation (if applicable)
                pose_results_batch = []
                if self.pose_estimator:
                    if self.pose_estimator.__class__.__name__ != "Custom" and len(self.detectors) > 0:
                        raise ValueError("Please use Pose.Custom class if you want to use a custom detector with the Video class.")
                    
                    pose_batch_results = self.process_batch_pose(frames_with_detections, boxes_batch)
                    frames_with_pose = [r[0] for r in pose_batch_results]
                    pose_results_batch = [r[1] for r in pose_batch_results]
                else:
                    frames_with_pose = frames_with_detections
                    pose_results_batch = [[] for _ in frame_batch]
                
                # Step 4: Process tracking frame-by-frame (tracker doesn't support batch)
                frames_with_tracking = []
                online_targets_batch = []
                
                for idx, (frame, all_detections, metadata) in enumerate(zip(frames_with_pose, all_detections_batch, frame_batch_metadata)):
                    if self.tracker is not None and len(all_detections) > 0:
                        detections = np.vstack(all_detections)
                        frame, online_targets = self.tracker.track(frame, detections)
                        
                        # Filter boxes based on student track
                        if self.tracker.student_track_id is not None and self.tracker.last_known_bbox is not None:
                            # Update boxes for this frame
                            boxes_batch[idx] = np.array(self.tracker.last_known_bbox, dtype=int).reshape(1, -1)
                    else:
                        online_targets = []
                    
                    frames_with_tracking.append(frame)
                    online_targets_batch.append(online_targets)
                
                # Step 5: Batch segmentation
                # Inference uses clean frames; overlay uses frames_with_tracking to keep pose drawings
                if len(self.segmentators) > 0:
                    result_frames = self.process_batch_segmentation(frame_batch, all_detections_batch, overlay_frames=frames_with_tracking)
                else:
                    result_frames = frames_with_tracking
                
                # Step 6: Process radar view and save results
                for idx, (result_frame, metadata, pose_results, online_targets) in enumerate(
                        zip(result_frames, frame_batch_metadata, pose_results_batch, online_targets_batch)):
                    
                    # Update and attach radar view
                    if self.radar_view and self.tracker is not None and self.pose_estimator is not None:
                        self.radar_view.update(online_targets, pose_results)
                        result_frame = self.radar_view.attach_to_frame(result_frame)
                    
                    # Store frame data
                    frame_data = {
                        'frame_id': metadata['frame_id'],
                        'timestamp': metadata['timestamp'],
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
                        progress_callback(metadata['frame_id'], self.total_frames, pose_results)
                
                # Clear batch
                frame_batch = []
                frame_batch_metadata = []
                
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
                        
                        fps_dict["Batch"] = f"{self.batch_size}"
                        
                        if pbar:
                            # Update progress bar with FPS info
                            pbar.set_postfix(fps_dict)
                        else:
                            # Print FPS on same line if no progress bar
                            fps_parts = [f"{k}: {v}" for k, v in fps_dict.items()]
                            fps_display = " | ".join(fps_parts)
                            print(f"\r{fps_display}", end='', flush=True)
                        
                        last_fps_print_time = current_time
            
            # Check if we're done
            if not ret and len(frame_batch) == 0:
                break
            
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
            print(f"Batch Size: {self.batch_size}")
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
