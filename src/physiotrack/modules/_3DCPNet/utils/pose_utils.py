import torch


def normalize_pose_scale(pose_3d, root_joint_idx=0):
    """
    Normalize pose by centering at root and scaling by bone length
    Args:
        pose_3d: (batch, joints, 3) 3D poses
        root_joint_idx: index of root joint (usually pelvis/hip)
    Returns:
        normalized pose and scale factor
    """
    batch_size, num_joints, _ = pose_3d.shape
    # Center at root joint
    root_pos = pose_3d[:, root_joint_idx:root_joint_idx+1, :]  # (batch, 1, 3)
    pose_centered = pose_3d - root_pos
    # Compute characteristic scale (torso length or similar)
    # Assuming COCO-17: left_shoulder=5, right_hip=11
    if num_joints >= 12:
        scale = torch.norm(pose_centered[:, 5] - pose_centered[:, 11], dim=-1, keepdim=True)
    else:
        # Fallback: use overall pose scale
        scale = torch.norm(pose_centered.view(batch_size, -1), dim=-1, keepdim=True)
    scale = scale.unsqueeze(-1)  # (batch, 1, 1)
    scale = torch.clamp(scale, min=1e-6)  # Avoid division by zero
    pose_normalized = pose_centered / scale
    return pose_normalized, scale


def center_pose_at(pose_3d, center_spec):
    """
    Center pose around a specified joint or the midpoint of a joint pair.
    Args:
        pose_3d: (batch, joints, 3) tensor
        center_spec: int for single joint index, or iterable of two indices for midpoint
    Returns:
        centered_pose: (batch, joints, 3)
        center_point: (batch, 1, 3)
    """
    if not isinstance(pose_3d, torch.Tensor):
        raise ValueError("pose_3d must be a torch.Tensor")

    if isinstance(center_spec, int):
        center_point = pose_3d[:, center_spec:center_spec+1, :]
    else:
        try:
            i, j = int(center_spec[0]), int(center_spec[1])
        except Exception:
            raise ValueError("center_spec must be an int or an iterable of two ints")
        center_point = 0.5 * (pose_3d[:, i:i+1, :] + pose_3d[:, j:j+1, :])

    centered_pose = pose_3d - center_point
    return centered_pose, center_point


    


    


def denormalize_pose(normalized_pose, scale, root_pos):
    """
    Denormalize pose back to original scale and position
    Args:
        normalized_pose: (batch, joints, 3) normalized poses
        scale: (batch, 1, 1) scale factors used for normalization
        root_pos: (batch, 1, 3) original root positions
    Returns:
        denormalized_pose: (batch, joints, 3) poses in original scale
    """
    # Restore scale
    pose_scaled = normalized_pose * scale
    # Restore root position
    pose_denormalized = pose_scaled + root_pos
    return pose_denormalized


def compute_bone_lengths(pose_3d, bone_connections=None):
    """
    Compute bone lengths for pose analysis
    Args:
        pose_3d: (batch, joints, 3) 3D poses
        bone_connections: List of (joint1_idx, joint2_idx) tuples
                         If None, uses COCO-17 default connections
    Returns:
        bone_lengths: (batch, num_bones) bone lengths
    """
    if bone_connections is None:
        # Custom 17-joint skeleton
        bone_connections = [
            # Spine and head
            (0, 7), (7, 8), (8, 9), (9, 10),
            # Left leg
            (0, 1), (1, 2), (2, 3),
            # Right leg
            (0, 4), (4, 5), (5, 6),
            # Right arm from neck
            (8, 11), (11, 12), (12, 13),
            # Left arm from neck
            (8, 14), (14, 15), (15, 16)
        ]
    batch_size = pose_3d.shape[0]
    bone_lengths = []
    for joint1_idx, joint2_idx in bone_connections:
        joint1 = pose_3d[:, joint1_idx, :]  # (batch, 3)
        joint2 = pose_3d[:, joint2_idx, :]  # (batch, 3)
        length = torch.norm(joint2 - joint1, dim=-1)  # (batch,)
        bone_lengths.append(length)
    
    bone_lengths = torch.stack(bone_lengths, dim=1)  # (batch, num_bones)
    return bone_lengths


def validate_pose_format(pose_3d, expected_joints=17):
    """
    Validate that pose tensor has the expected format
    Args:
        pose_3d: Input pose tensor
        expected_joints: Expected number of joints
    Returns:
        bool: True if valid format
    """
    if not isinstance(pose_3d, torch.Tensor):
        raise ValueError("pose_3d must be a torch.Tensor")
    
    if pose_3d.dim() not in [2, 3]:
        raise ValueError(f"pose_3d must be 2D or 3D tensor, got {pose_3d.dim()}D")
    
    if pose_3d.dim() == 2:
        # Flattened format (batch, joints*3)
        if pose_3d.shape[1] != expected_joints * 3:
            raise ValueError(f"Expected {expected_joints * 3} features for flattened pose, got {pose_3d.shape[1]}")
    else:
        # Standard format (batch, joints, 3)
        if pose_3d.shape[1] != expected_joints:
            raise ValueError(f"Expected {expected_joints} joints, got {pose_3d.shape[1]}")
        if pose_3d.shape[2] != 3:
            raise ValueError(f"Expected 3D coordinates, got {pose_3d.shape[2]}D")
    return True


def convert_pose_format(pose_3d, target_format='3d'):
    """
    Convert between different pose formats
    Args:
        pose_3d: Input pose tensor
        target_format: 'flat' for (batch, joints*3) or '3d' for (batch, joints, 3)
    Returns:
        converted pose tensor
    """
    if target_format == 'flat':
        if pose_3d.dim() == 3:
            return pose_3d.view(pose_3d.shape[0], -1)
        else:
            return pose_3d
    elif target_format == '3d':
        if pose_3d.dim() == 2:
            num_joints = pose_3d.shape[1] // 3
            return pose_3d.view(pose_3d.shape[0], num_joints, 3)
        else:
            return pose_3d
    else:
        raise ValueError(f"Unknown target_format: {target_format}")


def flip_pose_horizontally(pose_3d, flip_pairs=None):
    """
    Flip pose horizontally (left-right mirror)
    Args:
        pose_3d: (batch, joints, 3) 3D poses
        flip_pairs: List of (left_joint_idx, right_joint_idx) pairs to swap
                   If None, uses COCO-17 default pairs
    Returns:
        flipped_pose: (batch, joints, 3) horizontally flipped poses
    """
    if flip_pairs is None:
        # COCO-17 left-right joint pairs
        flip_pairs = [
            (1, 2),   # left_eye, right_eye
            (3, 4),   # left_ear, right_ear  
            (5, 6),   # left_shoulder, right_shoulder
            (7, 8),   # left_elbow, right_elbow
            (9, 10),  # left_wrist, right_wrist
            (11, 12), # left_hip, right_hip
            (13, 14), # left_knee, right_knee
            (15, 16)  # left_ankle, right_ankle
        ]
    
    flipped_pose = pose_3d.clone()
    # Flip x-coordinate
    flipped_pose[:, :, 0] = -flipped_pose[:, :, 0]
    # Swap left-right joints
    for left_idx, right_idx in flip_pairs:
        temp = flipped_pose[:, left_idx, :].clone()
        flipped_pose[:, left_idx, :] = flipped_pose[:, right_idx, :]
        flipped_pose[:, right_idx, :] = temp
    return flipped_pose