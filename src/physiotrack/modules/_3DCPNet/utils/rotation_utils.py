import torch
import torch.nn.functional as F
import math
import numpy as np


def normalize_vector(v):
    """
    Normalize vector batch-wise (adapted from 6DRepNet)
    Args:
        v: (batch, n) vectors
    Returns:
        normalized vectors
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    # Device-agnostic epsilon
    eps = torch.tensor(1e-8, device=v.device, dtype=v.dtype)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    
    return v / v_mag


def cross_product(u, v):
    """
    Cross product for batch of 3D vectors (adapted from 6DRepNet)
    Args:
        u, v: (batch, 3) vectors
    Returns:
        cross product (batch, 3)
    """
    batch = u.shape[0]
    
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out


def sixd_to_rotation_matrix(poses):
    """
    Convert 6D rotation representation to rotation matrix (from 6DRepNet)
    Args:
        poses: (batch, 6) 6D rotation representation
    Returns:
        rotation matrices (batch, 3, 3)
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


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    Args:
        q: (batch, 4) normalized quaternions [w, x, y, z]
    Returns:
        R: (batch, 3, 3) rotation matrices
    """
    batch_size = q.shape[0]
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Rotation matrix elements
    R = torch.zeros(batch_size, 3, 3, device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def apply_rotation_to_pose(pose_3d: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrix to 3D pose.
    Args:
        pose_3d: (batch, joints, 3) 3D pose
        rotation_matrix: (batch, 3, 3) rotation matrices
    Returns:
        rotated_pose: (batch, joints, 3) rotated 3D pose
    """
    batch_size, num_joints, _ = pose_3d.shape
    
    # Reshape for batch matrix multiplication
    # Use reshape to support non-contiguous tensors
    pose_flat = pose_3d.reshape(batch_size, num_joints * 3, 1)  # (batch, joints*3, 1)
    # Apply rotation: R @ pose (need to handle joint dimension)
    rotated_poses = []
    for i in range(num_joints):
        joint_pos = pose_3d[:, i:i+1, :].transpose(-1, -2)  # (batch, 3, 1)
        rotated_joint = torch.bmm(rotation_matrix, joint_pos)  # (batch, 3, 1)
        rotated_poses.append(rotated_joint.transpose(-1, -2))  # (batch, 1, 3)
    rotated_pose = torch.cat(rotated_poses, dim=1)  # (batch, joints, 3)
    return rotated_pose


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    """
    Convert rotation matrices to Euler angles (from 6DRepNet)
    Useful for visualization and analysis
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
    
    out_euler = torch.zeros(batch, 3, device=rotation_matrices.device)
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular
    return out_euler


def create_rotation_matrix_from_euler(angles):
    """
    Create rotation matrix from Euler angles (pitch, yaw, roll)
    Useful for data generation
    """
    if isinstance(angles, torch.Tensor):
        angles = angles.cpu().numpy()
    
    x, y, z = angles[0], angles[1], angles[2]
    # X rotation (pitch)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # Y rotation (yaw) 
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # Z rotation (roll)
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry.dot(Rx))
    return torch.tensor(R, dtype=torch.float32)