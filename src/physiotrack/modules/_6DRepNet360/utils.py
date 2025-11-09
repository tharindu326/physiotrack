"""
Utility functions for 6DRepNet360 head pose estimation.

This module provides:
- Visualization functions for drawing pose axes and cubes
- Rotation matrix computation from 6D representation
- Euler angle conversion from rotation matrices
- Helper functions for pose annotation processing
"""
import os
import math
from math import cos, sin

import numpy as np
import torch
import cv2


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    """
    Draw a 3D pose cube on the image to visualize head orientation.
    
    Args:
        img: Input image (cv2 format)
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        tdx: X coordinate of face center (optional)
        tdy: Y coordinate of face center (optional)
        size: Size of the cube
    
    Returns:
        Image with pose cube drawn
    """
    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx is not None and tdy is not None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x), int(y2+y1-face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x), int(y1+y2-face_y)), (0, 0, 255), 3)
    
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x), int(y1+y3-face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x), int(y2+y3-face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2+x1-face_x), int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (255, 0, 0), 2)
    
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x), int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2+x3-face_x), int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x), int(y3+y1-face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x), int(y3+y2-face_y)), (0, 255, 0), 2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    """
    Draw 3D axes on the image to visualize head orientation.
    
    Args:
        img: Input image (cv2 format)
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        tdx: X coordinate of face center (optional)
        tdy: Y coordinate of face center (optional)
        size: Length of the axes
    
    Returns:
        Image with axes drawn
        - Red axis: X-axis (pointing right)
        - Green axis: Y-axis (pointing down)
        - Blue axis: Z-axis (pointing forward/out of screen)
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right, drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing down, drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

    return img


def normalize_vector(v):
    """
    Normalize a batch of vectors.
    
    Args:
        v: Tensor of shape (batch, n)
    
    Returns:
        Normalized tensor
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    """
    Compute cross product of two batches of 3D vectors.
    
    Args:
        u: Tensor of shape (batch, 3)
        v: Tensor of shape (batch, 3)
    
    Returns:
        Cross product tensor of shape (batch, 3)
    """
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    
    return out


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Compute rotation matrix from 6D orthogonal representation.
    
    The 6D representation is more stable for learning than direct rotation matrices
    or Euler angles, as it avoids discontinuities.
    
    Args:
        poses: Tensor of shape (batch, 6) representing two 3D vectors
    
    Returns:
        Rotation matrices of shape (batch, 3, 3)
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
    
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    """
    Compute Euler angles from rotation matrices.
    
    The rotation is in the sequence of x, y, z (pitch, yaw, roll).
    
    Args:
        rotation_matrices: Tensor of shape (batch, 3, 3) or (batch, 4, 4)
    
    Returns:
        Euler angles tensor of shape (batch, 3) with [pitch, yaw, roll] in radians
    """
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()
    
    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    
    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0
    
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular
    
    return out_euler


def get_R(x, y, z):
    """
    Get rotation matrix from three rotation angles (radians). Right-handed coordinate system.
    
    Args:
        x: Rotation around x-axis (pitch) in radians
        y: Rotation around y-axis (yaw) in radians
        z: Rotation around z-axis (roll) in radians
    
    Returns:
        R: 3x3 rotation matrix
    """
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


__all__ = [
    'plot_pose_cube',
    'draw_axis',
    'normalize_vector',
    'cross_product',
    'compute_rotation_matrix_from_ortho6d',
    'compute_euler_angles_from_rotation_matrices',
    'get_R'
]
