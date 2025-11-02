"""
Real-time keypoint motion plotter for tracking keypoint movements during video processing.
Plots keypoint positions relative to pelvis (normalized) with optional filtering.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, Tuple, List, Dict
from scipy.signal import butter, lfilter


class KeypointMotionPlotter:
    """
    Real-time plotter for tracking keypoint motion relative to pelvis.
    
    Maintains a sliding window of keypoint positions and renders them as a plot overlay.
    """
    
    def __init__(self,
                 keypoint_id: int = 9,
                 keypoint_name: str = "left_wrist",
                 window_size: int = 300,
                 canvas_width: int = 450,
                 canvas_height: int = 180,
                 filter_signal: bool = True,
                 filter_bandpass: Tuple[float, float] = (0.5, 5.0),
                 fps: float = 30.0):
        """
        Initialize keypoint motion plotter.
        
        Args:
            keypoint_id: COCO keypoint ID to track (9=left_wrist, 10=right_wrist, etc.)
            keypoint_name: Name of the keypoint for display
            window_size: Number of frames to display in the plot window
            canvas_width: Width of the plot canvas in pixels
            canvas_height: Height of the plot canvas in pixels
            filter_signal: Whether to apply band-pass filtering
            filter_bandpass: Bandpass filter frequencies [low, high] in Hz
            fps: Video frame rate for time axis
        """
        self.keypoint_id = keypoint_id
        self.keypoint_name = keypoint_name
        self.window_size = window_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.filter_signal = filter_signal
        self.filter_bandpass = filter_bandpass
        self.fps = fps
        
        # Data buffers (relative to pelvis)
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # Track colors for each tracked person
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # Cache for matplotlib figure/axes (reuse for performance)
        self._fig = None
        self._ax = None
        self._line_x = None
        self._line_y = None
        
        # Initialize filter coefficients if filtering is enabled
        if self.filter_signal and fps > 0:
            self._init_filter()
        
    def _init_filter(self):
        """Initialize Butterworth bandpass filter."""
        try:
            order = 3
            self.filter_b, self.filter_a = butter(
                order, 
                self.filter_bandpass, 
                btype='bandpass', 
                fs=self.fps
            )
        except Exception as e:
            print(f"Warning: Could not initialize filter: {e}")
            self.filter_signal = False
    
    def _get_pelvis_position(self, keypoints: List[dict]) -> Optional[Tuple[float, float]]:
        """
        Calculate pelvis position as the midpoint between left and right hips.
        
        Args:
            keypoints: List of keypoints [{'id': int, 'x': float, 'y': float, 'confidence': float}, ...]
        
        Returns:
            Pelvis position (x, y) or None if hips not visible
        """
        if not keypoints:
            return None
        
        # Create lookup dict
        kp_dict = {kp['id']: kp for kp in keypoints}
        
        # COCO keypoint IDs: 11=left_hip, 12=right_hip
        left_hip = kp_dict.get(11)
        right_hip = kp_dict.get(12)
        
        confidence_threshold = 0.3
        
        if left_hip and right_hip:
            if left_hip['confidence'] > confidence_threshold and right_hip['confidence'] > confidence_threshold:
                pelvis_x = (left_hip['x'] + right_hip['x']) / 2.0
                pelvis_y = (left_hip['y'] + right_hip['y']) / 2.0
                return (pelvis_x, pelvis_y)
        
        return None
    
    def _get_keypoint_position(self, keypoints: List[dict], keypoint_id: int) -> Optional[Tuple[float, float, float]]:
        """
        Get position and confidence of a specific keypoint.
        
        Returns:
            (x, y, confidence) or None if not found
        """
        if not keypoints:
            return None
        
        for kp in keypoints:
            if kp['id'] == keypoint_id:
                if kp['confidence'] > 0.3:
                    return (kp['x'], kp['y'], kp['confidence'])
        
        return None
    
    def update(self, pose_results: List[dict], frame_time: float):
        """
        Update motion data with new pose information.
        
        Args:
            pose_results: List of pose estimation results with keypoints
            frame_time: Current frame timestamp in seconds
        """
        # For now, track the first person with valid keypoints
        # TODO: Extend to multi-person tracking
        
        for pose_result in pose_results:
            if 'keypoints' not in pose_result or pose_result['keypoints'] is None:
                continue
            
            keypoints = pose_result['keypoints']
            
            # Get pelvis position (reference point)
            pelvis = self._get_pelvis_position(keypoints)
            if pelvis is None:
                continue
            
            # Get tracked keypoint position
            kp_data = self._get_keypoint_position(keypoints, self.keypoint_id)
            if kp_data is None:
                continue
            
            kp_x, kp_y, kp_conf = kp_data
            
            # Calculate position relative to pelvis (normalized)
            rel_x = kp_x - pelvis[0]
            rel_y = kp_y - pelvis[1]
            
            # Add to buffers
            self.x_buffer.append(rel_x)
            self.y_buffer.append(rel_y)
            self.time_buffer.append(frame_time)
            
            # Only track first valid person for now
            break
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the motion plot as an image.
        
        Returns:
            Plot canvas as BGR numpy array or None if not enough data
        """
        if len(self.x_buffer) < 10:
            # Not enough data to plot
            return self._render_empty_canvas()
        
        # Convert buffers to arrays
        x_data = np.array(self.x_buffer)
        y_data = np.array(self.y_buffer)
        time_data = np.array(self.time_buffer)
        
        # Apply filtering if enabled
        if self.filter_signal and len(x_data) > 20:
            try:
                x_filtered = lfilter(self.filter_b, self.filter_a, x_data)
                y_filtered = lfilter(self.filter_b, self.filter_a, y_data)
            except Exception as e:
                # If filtering fails, use unfiltered data
                x_filtered = x_data
                y_filtered = y_data
        else:
            x_filtered = x_data
            y_filtered = y_data
        
        # Create or reuse cached figure for performance
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(self.canvas_width / 80, self.canvas_height / 80), dpi=80)
            
            # Initial plot setup
            self._line_x, = self._ax.plot([], [], label='X', linewidth=1.2, color='#2E86AB', antialiased=True)
            self._line_y, = self._ax.plot([], [], label='Y', linewidth=1.2, color='#A23B72', antialiased=True)
            
            # Styling (only set once)
            self._ax.set_xlabel('Time (s)', fontsize=8)
            self._ax.set_ylabel('Rel. to pelvis (px)', fontsize=8)
            self._ax.set_title(f'{self.keypoint_name}', fontsize=9, fontweight='bold', pad=5)
            self._ax.legend(loc='upper right', fontsize=7, frameon=False)
            self._ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            self._ax.tick_params(labelsize=7)
            self._fig.tight_layout()
        
        # Update data (much faster than recreating plot)
        self._line_x.set_data(time_data, x_filtered)
        self._line_y.set_data(time_data, y_filtered)
        
        # Update axis limits
        if len(time_data) > 0:
            time_window = self.window_size / self.fps
            self._ax.set_xlim(time_data[-1] - time_window, time_data[-1])
            
            # Auto-scale y-axis based on visible data
            y_min = min(np.min(x_filtered), np.min(y_filtered))
            y_max = max(np.max(x_filtered), np.max(y_filtered))
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 10
            self._ax.set_ylim(y_min - margin, y_max + margin)
        
        # Render to numpy array
        self._fig.canvas.draw()
        
        # Get buffer data (compatible with newer matplotlib versions)
        try:
            # Try newer API first
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to BGR for OpenCV
            canvas = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # Fallback for older matplotlib versions
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            # Convert RGB to BGR for OpenCV
            canvas = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        # Resize to exact canvas dimensions
        if canvas.shape[:2] != (self.canvas_height, self.canvas_width):
            canvas = cv2.resize(canvas, (self.canvas_width, self.canvas_height))
        
        return canvas
    
    def _render_empty_canvas(self) -> np.ndarray:
        """Render an empty canvas with a waiting message."""
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 245
        
        text = "Collecting motion data..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for centering
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (self.canvas_width - text_width) // 2
        y = (self.canvas_height + text_height) // 2
        
        cv2.putText(canvas, text, (x, y), font, font_scale, (100, 100, 100), thickness)
        
        return canvas
    
    def attach_to_frame(self, frame: np.ndarray, position: str = 'top_right',
                        margin: int = 10) -> np.ndarray:
        """
        Attach motion plot to a video frame.
        
        Args:
            frame: Video frame to attach plot to
            position: Position on frame ('top', 'top_right', 'top_left', 'bottom', 'bottom_right', 'bottom_left')
            margin: Margin from frame edge in pixels
        
        Returns:
            Frame with motion plot attached
        """
        plot_canvas = self.render()
        if plot_canvas is None:
            return frame
        
        h, w = frame.shape[:2]
        plot_h, plot_w = plot_canvas.shape[:2]
        
        # Calculate position
        if position == 'top' or position == 'top_left':
            y1 = margin
            y2 = margin + plot_h
            x1 = margin
            x2 = margin + plot_w
        elif position == 'top_right':
            y1 = margin
            y2 = margin + plot_h
            x1 = w - plot_w - margin
            x2 = w - margin
        elif position == 'bottom' or position == 'bottom_left':
            y1 = h - plot_h - margin
            y2 = h - margin
            x1 = margin
            x2 = margin + plot_w
        elif position == 'bottom_right':
            y1 = h - plot_h - margin
            y2 = h - margin
            x1 = w - plot_w - margin
            x2 = w - margin
        else:
            raise ValueError(f"Invalid position: {position}")
        
        # Ensure coordinates are valid
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return frame
        
        result_frame = frame.copy()
        
        # Add semi-transparent background
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 0), -1)
        result_frame = cv2.addWeighted(result_frame, 0.85, overlay, 0.15, 0)
        
        # Overlay plot
        result_frame[y1:y2, x1:x2] = plot_canvas
        
        return result_frame
    
    def clear(self):
        """Clear all buffered data and cached figure."""
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.time_buffer.clear()
        
        # Clean up matplotlib figure
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._line_x = None
            self._line_y = None

