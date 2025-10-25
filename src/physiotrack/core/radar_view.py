"""
Radar view for visualizing person trajectories on a floor map.
Maps tracked person positions from video coordinates to a 2D floor plan view.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from collections import deque
from pathlib import Path
from physiotrack.modules.Yolo.classes_and_palettes import COLORS
from physiotrack.utils.spatial_transforms import compute_homography, transform_point, get_foot_position


class RadarView:
    """
    Radar view visualization for tracking person movements on a floor map.
    
    Supports three modes:
    1. Default black canvas with floor_map for homography
    2. Direct canvas: Use a pre-made floor map image directly
    3. Extracted canvas: Extract and transform a floor area from source image using homography
    """

    def __init__(self, floor_map: Optional[List[Tuple[int, int]]] = None,
                 max_trajectory_length: int = 100,
                 max_canvas_dim: int = 400,
                 background: Optional[Union[str, np.ndarray]] = None):
        """
        Initialize radar view.

        Args:
            floor_map: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] defining floor area
            max_trajectory_length: Maximum number of points to keep in trajectory
            max_canvas_dim: Maximum dimension for radar canvas
            background: Canvas background mode/image:
                       - None or "default": Black canvas with gray background (default)
                       - "auto" or "extract": Extract floor area from first video frame and transform 
                         to top-down view using homography (set later via set_background_from_frame)
                       - str (path): Load pre-made floor plan image from file path
                       - np.ndarray: Use provided image array as canvas
                       
        Note:
            Canvas size is always determined by floor_map and max_canvas_dim, regardless of background mode.
            Custom images (path or array) will be automatically resized to match the computed canvas size
            for consistency with default mode. This ensures the radar view overlay has the same dimensions
            on the video frame regardless of which background is used.
        """
        self.enabled = floor_map is not None and len(floor_map) == 4
        self.max_trajectory_length = max_trajectory_length
        self.custom_canvas = None
        self.use_custom_canvas = False
        self.background_mode = background
        self.floor_map = floor_map
        self.max_canvas_dim = max_canvas_dim

        if self.enabled:
            # Compute canvas size based on floor_map (consistent across all modes)
            self.homography_matrix, self.canvas_size = compute_homography(floor_map, max_canvas_dim)
            
            # Handle background modes
            if background is None or background == "default":
                # Mode 1: Default black canvas
                pass
            elif background == "auto" or background == "extract":
                # Mode 2: Extract from video frame (will be set later via set_background_from_frame)
                pass
            elif isinstance(background, str):
                # Mode 3: Load from file path and resize to match canvas_size
                self.custom_canvas = self._load_canvas(background)
                if self.custom_canvas is not None:
                    # Resize to match the computed canvas size for consistency
                    self.custom_canvas = cv2.resize(self.custom_canvas, self.canvas_size)
                    self.use_custom_canvas = True
                    print(f"RadarView: Using floor plan from file (resized to {self.canvas_size[0]}x{self.canvas_size[1]})")
            elif isinstance(background, np.ndarray):
                # Mode 4: Use provided numpy array and resize to match canvas_size
                self.custom_canvas = self._load_canvas(background)
                if self.custom_canvas is not None:
                    # Resize to match the computed canvas size for consistency
                    self.custom_canvas = cv2.resize(self.custom_canvas, self.canvas_size)
                    self.use_custom_canvas = True
                    print(f"RadarView: Using provided canvas array (resized to {self.canvas_size[0]}x{self.canvas_size[1]})")
        else:
            self.homography_matrix = None
            self.canvas_size = (max_canvas_dim, max_canvas_dim)

        self.trajectories: Dict[int, deque] = {}
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}

    def _load_canvas(self, canvas_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Load canvas from file path or numpy array.

        Args:
            canvas_input: Path to image file or numpy array

        Returns:
            Loaded canvas as BGR numpy array or None if failed
        """
        try:
            if isinstance(canvas_input, str):
                canvas_path = Path(canvas_input)
                if not canvas_path.exists():
                    print(f"Warning: Floor map canvas file not found: {canvas_input}")
                    return None
                canvas = cv2.imread(str(canvas_path))
                if canvas is None:
                    print(f"Warning: Failed to load floor map canvas from: {canvas_input}")
                    return None
                return canvas
            elif isinstance(canvas_input, np.ndarray):
                # Ensure it's 3-channel BGR
                if len(canvas_input.shape) == 2:
                    return cv2.cvtColor(canvas_input, cv2.COLOR_GRAY2BGR)
                elif canvas_input.shape[2] == 3:
                    return canvas_input.copy()
                elif canvas_input.shape[2] == 4:
                    return cv2.cvtColor(canvas_input, cv2.COLOR_BGRA2BGR)
                else:
                    print(f"Warning: Unsupported canvas array format with {canvas_input.shape[2]} channels")
                    return None
            else:
                print(f"Warning: Unsupported canvas input type: {type(canvas_input)}")
                return None
        except Exception as e:
            print(f"Error loading floor map canvas: {e}")
            return None

    def _extract_floor_canvas(self, source_image: Union[str, np.ndarray],
                              floor_map_points: List[Tuple[int, int]],
                              max_canvas_dim: int) -> Optional[np.ndarray]:
        """
        Extract floor area from source image and transform to top-down view using homography.

        Args:
            source_image: Source image (path or numpy array) containing the floor area
            floor_map_points: List of 4 corner points defining the floor area in the source image
            max_canvas_dim: Maximum dimension for the output canvas

        Returns:
            Transformed floor canvas as BGR numpy array or None if failed
        """
        try:
            # Load source image
            if isinstance(source_image, str):
                source_path = Path(source_image)
                if not source_path.exists():
                    print(f"Warning: Floor map source image not found: {source_image}")
                    return None
                img = cv2.imread(str(source_path))
                if img is None:
                    print(f"Warning: Failed to load floor map source image from: {source_image}")
                    return None
            elif isinstance(source_image, np.ndarray):
                # Ensure it's 3-channel BGR
                if len(source_image.shape) == 2:
                    img = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
                elif source_image.shape[2] == 3:
                    img = source_image.copy()
                elif source_image.shape[2] == 4:
                    img = cv2.cvtColor(source_image, cv2.COLOR_BGRA2BGR)
                else:
                    print(f"Warning: Unsupported source image format with {source_image.shape[2]} channels")
                    return None
            else:
                print(f"Warning: Unsupported source image type: {type(source_image)}")
                return None

            # Compute homography matrix and canvas size
            homography_matrix, canvas_size = compute_homography(floor_map_points, max_canvas_dim)

            # Apply perspective transformation to extract and transform the floor area to top-down view
            canvas = cv2.warpPerspective(img, homography_matrix, canvas_size)

            return canvas

        except Exception as e:
            print(f"Error extracting floor canvas: {e}")
            return None

    def set_background_from_frame(self, frame: np.ndarray) -> bool:
        """
        Extract floor area from a video frame and set it as the canvas background.
        The floor area will be transformed to top-down view using homography.
        
        This should be called with the first frame when background mode is "auto" or "extract".

        Args:
            frame: Video frame (numpy array) from which to extract the floor area

        Returns:
            True if background was successfully set, False otherwise
        """
        if not self.enabled or self.floor_map is None:
            print("Warning: Cannot extract floor from frame - floor_map not defined")
            return False

        try:
            # Extract and transform floor area to top-down view
            self.custom_canvas = self._extract_floor_canvas(
                frame, 
                self.floor_map, 
                self.max_canvas_dim
            )
            
            if self.custom_canvas is not None:
                self.use_custom_canvas = True
                self.canvas_size = (self.custom_canvas.shape[1], self.custom_canvas.shape[0])
                print(f"RadarView: Extracted floor area from frame and transformed to top-down view "
                      f"({self.canvas_size[0]}x{self.canvas_size[1]})")
                return True
            else:
                print("Warning: Failed to extract floor area from frame")
                return False
                
        except Exception as e:
            print(f"Error setting background from frame: {e}")
            return False

    def update(self, online_targets: List, pose_results: List[dict]) -> None:
        """
        Update trajectories based on current tracking and pose data.

        Args:
            online_targets: List of tracked targets [[x1, y1, x2, y2, track_id, cls, conf], ...]
            pose_results: List of pose estimation results with keypoints
        """
        if not self.enabled or len(online_targets) == 0:
            return

        for target in online_targets:
            track_id = int(target[4])
            bbox = target[:4]

            # Find the pose result that matches this bbox
            for pose_result in pose_results:
                if 'keypoints' not in pose_result or pose_result['keypoints'] is None:
                    continue

                keypoints = pose_result['keypoints']
                foot_position = get_foot_position(keypoints)

                if foot_position is None:
                    continue

                # Check if foot is inside bbox (simple matching)
                x1, y1, x2, y2 = bbox
                if x1 <= foot_position[0] <= x2 and y1 <= foot_position[1] <= y2:
                    # Transform to floor coordinates
                    floor_coords = transform_point(foot_position, self.homography_matrix)

                    if floor_coords is not None:
                        # Initialize trajectory for new track ID
                        if track_id not in self.trajectories:
                            self.trajectories[track_id] = deque(maxlen=self.max_trajectory_length)

                        # Add to trajectory
                        self.trajectories[track_id].append(floor_coords)
                        break

    def render(self) -> np.ndarray:
        """
        Render the radar view canvas with all trajectories.

        Returns:
            Radar canvas as numpy array
        """
        canvas_width, canvas_height = self.canvas_size

        if not self.enabled:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            return canvas

        # Use custom canvas if available, otherwise create default black canvas
        if self.use_custom_canvas and self.custom_canvas is not None:
            canvas = self.custom_canvas.copy()
        else:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            # Fill background
            canvas.fill(40)
            # Draw floor boundary
            cv2.rectangle(canvas, (0, 0), (canvas_width - 1, canvas_height - 1), (100, 100, 100), 2)

        # Draw trajectories for each person
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) == 0:
                continue

            color = self._get_track_color(track_id)

            # Draw trajectory path
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i - 1]
                pt2 = trajectory[i]

                # Ensure points are within bounds
                if (0 <= pt1[0] < canvas_width and 0 <= pt1[1] < canvas_height and
                        0 <= pt2[0] < canvas_width and 0 <= pt2[1] < canvas_height):
                    cv2.line(canvas, pt1, pt2, color, 2)

            # Draw current position (larger circle)
            current_pos = trajectory[-1]
            if 0 <= current_pos[0] < canvas_width and 0 <= current_pos[1] < canvas_height:
                cv2.circle(canvas, current_pos, 6, color, -1)
                cv2.circle(canvas, current_pos, 8, (255, 255, 255), 1)

                # Add track ID label
                cv2.putText(canvas, f"ID:{track_id}",
                            (current_pos[0] + 10, current_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add title
        cv2.putText(canvas, "Floor Map", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return canvas

    def attach_to_frame(self, frame: np.ndarray, position: str = 'bottom_right',
                        margin: int = 10) -> np.ndarray:
        """
        Attach radar view to a video frame.

        Args:
            frame: Video frame to attach radar view to
            position: Position on frame ('bottom_right', 'bottom_left', 'top_right', 'top_left')
            margin: Margin from frame edge in pixels

        Returns:
            Frame with radar view attached
        """
        if not self.enabled:
            return frame

        radar_canvas = self.render()
        h, w = frame.shape[:2]
        radar_h, radar_w = self.canvas_size[1], self.canvas_size[0]

        # Calculate position based on specified location
        if position == 'bottom_right':
            y1 = h - radar_h - margin
            y2 = h - margin
            x1 = w - radar_w - margin
            x2 = w - margin
        elif position == 'bottom_left':
            y1 = h - radar_h - margin
            y2 = h - margin
            x1 = margin
            x2 = margin + radar_w
        elif position == 'top_right':
            y1 = margin
            y2 = margin + radar_h
            x1 = w - radar_w - margin
            x2 = w - margin
        elif position == 'top_left':
            y1 = margin
            y2 = margin + radar_h
            x1 = margin
            x2 = margin + radar_w
        else:
            raise ValueError(f"Invalid position: {position}")

        # Ensure coordinates are valid
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return frame

        result_frame = frame.copy()

        # Add semi-transparent background
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 0), -1)
        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)

        # Overlay radar view
        result_frame[y1:y2, x1:x2] = radar_canvas

        return result_frame

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get a unique color for a track ID.

        Args:
            track_id: Tracking ID

        Returns:
            RGB color tuple
        """
        if track_id not in self.track_colors:
            color_names = list(COLORS.keys())
            color_idx = len(self.track_colors) % len(color_names)
            self.track_colors[track_id] = tuple(COLORS[color_names[color_idx]])

        return self.track_colors[track_id]

    def clear_trajectories(self) -> None:
        """Clear all stored trajectories."""
        self.trajectories.clear()
        self.track_colors.clear()
