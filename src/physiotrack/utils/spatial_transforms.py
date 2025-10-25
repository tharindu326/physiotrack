"""
Spatial transformation utilities for coordinate mapping and homography.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


def compute_homography(floor_map_points: List[Tuple[int, int]],
                       max_canvas_dim: int = 400,
                       margin: int = 20,
                       rotation: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute homography matrix for perspective transformation from image coordinates to floor map.

    Args:
        floor_map_points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        max_canvas_dim: Maximum dimension for the output canvas
        margin: Margin to add around the canvas to prevent points going out of bounds
        rotation: Rotation angle in degrees (0, 90, 180, 270) to orient the top-down view.
                 Use this to align movement directions with the desired orientation.
                 - 0°: No rotation (default)
                 - 90°: Rotate 90° clockwise
                 - 180°: Rotate 180°
                 - 270° (or -90°): Rotate 90° counter-clockwise

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

    # For 90° and 270° rotations, swap width and height
    if rotation in [90, 270, -90]:
        width, height = height, width

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

    # Base destination points (normalized 2D floor space)
    base_dst_pts = [
        [margin, margin],                                    # top-left
        [canvas_width - margin, margin],                     # top-right
        [canvas_width - margin, canvas_height - margin],     # bottom-right
        [margin, canvas_height - margin]                     # bottom-left
    ]

    # Apply rotation by rearranging destination point order
    # This rotates the coordinate system to align with desired orientation
    rotation = rotation % 360  # Normalize to 0-359
    
    if rotation == 0:
        # No rotation: TL, TR, BR, BL
        dst_pts = base_dst_pts
    elif rotation == 90:
        # 90° clockwise: map src [TL,TR,BR,BL] to dst [BL,TL,TR,BR]
        dst_pts = [base_dst_pts[3], base_dst_pts[0], base_dst_pts[1], base_dst_pts[2]]
    elif rotation == 180:
        # 180° rotation: map src [TL,TR,BR,BL] to dst [BR,BL,TL,TR]
        dst_pts = [base_dst_pts[2], base_dst_pts[3], base_dst_pts[0], base_dst_pts[1]]
    elif rotation == 270:
        # 270° clockwise (90° counter-clockwise): map src [TL,TR,BR,BL] to dst [TR,BR,BL,TL]
        dst_pts = [base_dst_pts[1], base_dst_pts[2], base_dst_pts[3], base_dst_pts[0]]
    else:
        # Default to no rotation for unsupported angles
        dst_pts = base_dst_pts

    dst_pts = np.array(dst_pts, dtype=np.float32)

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
