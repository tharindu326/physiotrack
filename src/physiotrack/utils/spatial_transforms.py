"""
Spatial transformation utilities for coordinate mapping and homography.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


def compute_homography(floor_map_points: List[Tuple[int, int]],
                       max_canvas_dim: int = 400,
                       margin: int = 20) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute homography matrix for perspective transformation from image coordinates to floor map.

    Args:
        floor_map_points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        max_canvas_dim: Maximum dimension for the output canvas
        margin: Margin to add around the canvas to prevent points going out of bounds

    Returns:
        Tuple of (homography_matrix, canvas_size)
    """
    if not floor_map_points or len(floor_map_points) != 4:
        raise ValueError("floor_map_points must contain exactly 4 points")

    # Source points from floor_map (in video coordinates)
    src_pts = np.array(floor_map_points, dtype=np.float32)

    # Calculate the bounding box of floor_map to determine aspect ratio
    xs = [pt[0] for pt in floor_map_points]
    ys = [pt[1] for pt in floor_map_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Scale to fit within max dimension while maintaining aspect ratio
    if width > height:
        canvas_width = max_canvas_dim
        canvas_height = int(max_canvas_dim * height / width)
    else:
        canvas_height = max_canvas_dim
        canvas_width = int(max_canvas_dim * width / height)

    # Add margin to prevent points going out of bounds
    canvas_width += margin * 2
    canvas_height += margin * 2

    canvas_size = (canvas_width, canvas_height)

    # Destination points (normalized 2D floor space scaled to canvas size with margin)
    dst_pts = np.array([
        [margin, margin],                                    # top-left
        [canvas_width - margin, margin],                     # top-right
        [canvas_width - margin, canvas_height - margin],     # bottom-right
        [margin, canvas_height - margin]                     # bottom-left
    ], dtype=np.float32)

    # Compute homography matrix
    homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return homography_matrix, canvas_size


def transform_point(point: Tuple[float, float],
                    homography_matrix: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Transform a point from video coordinates to floor map coordinates using homography.

    Args:
        point: Point coordinates (x, y)
        homography_matrix: Homography transformation matrix

    Returns:
        Transformed point (x, y) or None if transformation fails
    """
    if homography_matrix is None or point is None:
        return None

    pt = np.array([[point[0], point[1]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), homography_matrix)
    return (int(transformed[0][0][0]), int(transformed[0][0][1]))


def get_foot_position(pose_keypoints: List[dict]) -> Optional[Tuple[float, float]]:
    """
    Calculate foot center position from pose keypoints.
    Returns the midpoint between left and right ankle positions.

    Args:
        pose_keypoints: List of keypoints [{'id': int, 'x': float, 'y': float, 'confidence': float}, ...]

    Returns:
        Foot center position (x, y) or None if ankles not visible
    """
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return None

    # Create a dictionary for quick lookup by id
    keypoints_dict = {kp['id']: kp for kp in pose_keypoints}

    # COCO keypoint indices: 15=left ankle, 16=right ankle
    left_ankle_id = 15
    right_ankle_id = 16
    confidence_threshold = 0.3

    left_ankle = keypoints_dict.get(left_ankle_id)
    right_ankle = keypoints_dict.get(right_ankle_id)

    # Check if both ankles are visible
    if left_ankle and right_ankle:
        if left_ankle['confidence'] > confidence_threshold and right_ankle['confidence'] > confidence_threshold:
            foot_x = (left_ankle['x'] + right_ankle['x']) / 2
            foot_y = (left_ankle['y'] + right_ankle['y']) / 2
            return (foot_x, foot_y)
        elif left_ankle['confidence'] > confidence_threshold:
            return (left_ankle['x'], left_ankle['y'])
        elif right_ankle['confidence'] > confidence_threshold:
            return (right_ankle['x'], right_ankle['y'])

    return None
