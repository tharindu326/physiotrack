import numpy as np
from enum import Enum
from typing import Tuple, Union

class CanonicalView(Enum):
    FRONT = "front"
    BACK = "back" 
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"

class PoseCanonicalizer:
    """
    Transform 3D poses to canonical orientations for IMU comparison
    """
    
    # H36M joint indices
    JOINT_INDICES = {
        'left_shoulder': 11,
        'right_shoulder': 14,
        'left_hip': 4,
        'right_hip': 1
    }
    
    @staticmethod
    def extract_torso_plane(poses_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract torso plane from 4 key points: shoulders and hips
        Args:
            poses_3d: Shape (N, 17, 3) - N frames, 17 joints, 3 coords
        Returns:
            Tuple of (plane_center, normal_vector, plane_points)
        """
        left_shoulder = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['left_shoulder'], :]
        right_shoulder = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['right_shoulder'], :]
        left_hip = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['left_hip'], :]
        right_hip = poses_3d[:, PoseCanonicalizer.JOINT_INDICES['right_hip'], :]
        
        # Stack for easier processing: (N, 4, 3)
        torso_points = np.stack([left_shoulder, right_shoulder, left_hip, right_hip], axis=1)
        
        # plane center (centroid of 4 points)
        plane_center = np.mean(torso_points, axis=1)  # (N, 3)
        
        shoulder_vector = right_shoulder - left_shoulder  # (N, 3)
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_vector = shoulder_center - hip_center  # (N, 3)
        # Normal to the torso plane
        normal_vector = np.cross(shoulder_vector, torso_vector)  # (N, 3)
        # Normalize
        normal_vector = normal_vector / (np.linalg.norm(normal_vector, axis=1, keepdims=True) + 1e-8)
        
        return plane_center, normal_vector, torso_points
    
    @staticmethod
    def compute_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix to align from_vec with to_vec
        Args:
            from_vec: Source vector (N, 3)
            to_vec: Target vector (N, 3) or (3,)
        Returns:
            Rotation matrices (N, 3, 3)
        """
        # Ensure inputs are normalized
        from_vec = from_vec / (np.linalg.norm(from_vec, axis=-1, keepdims=True) + 1e-8)
        if to_vec.ndim == 1:
            to_vec = np.tile(to_vec, (from_vec.shape[0], 1))
        to_vec = to_vec / (np.linalg.norm(to_vec, axis=-1, keepdims=True) + 1e-8)
        # Cross product for rotation axis
        cross = np.cross(from_vec, to_vec)
        cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)
        # Dot product for rotation angle
        dot = np.sum(from_vec * to_vec, axis=-1, keepdims=True)
        # Handle parallel vectors
        parallel_mask = cross_norm.squeeze(-1) < 1e-6
        # Rodrigues' rotation formula
        N = from_vec.shape[0]
        R = np.eye(3)[None, :, :].repeat(N, axis=0)
        # For non-parallel vectors
        non_parallel = ~parallel_mask
        if np.any(non_parallel):
            k = cross[non_parallel] / (cross_norm[non_parallel] + 1e-8)
            theta = np.arccos(np.clip(dot[non_parallel], -1, 1))
            # Skew-symmetric matrix
            K = np.zeros((np.sum(non_parallel), 3, 3))
            K[:, 0, 1] = -k[:, 2]
            K[:, 0, 2] = k[:, 1]
            K[:, 1, 0] = k[:, 2]
            K[:, 1, 2] = -k[:, 0]
            K[:, 2, 0] = -k[:, 1]
            K[:, 2, 1] = k[:, 0]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R[non_parallel] = (np.eye(3)[None, :, :] + 
                              sin_theta[:, :, None] * K + 
                              (1 - cos_theta)[:, :, None] * np.matmul(K, K))
        return R
    
    @staticmethod
    def transform_to_front_view(poses_3d: np.ndarray) -> np.ndarray:
        """
        Transform poses so torso plane is parallel to XY plane (front view)
        AND shoulders are parallel to X-axis (using minimal rotation)
        Args:
            poses_3d: Shape (N, 17, 3)
        Returns:
            Transformed poses (N, 17, 3)
        """
        plane_center, normal_vector, _ = PoseCanonicalizer.extract_torso_plane(poses_3d)
        # Step 1: Align torso plane normal with Z-axis
        target_normal = np.array([0, 0, 1])
        R1 = PoseCanonicalizer.compute_rotation_matrix(normal_vector, target_normal)
        # Apply first rotation
        centered_poses = poses_3d - plane_center[:, None, :]
        rotated_once = np.matmul(centered_poses, R1.transpose(0, 2, 1))
        # Step 2: Align shoulders with X-axis using minimal rotation
        # Get shoulder positions after first rotation
        left_shoulder = rotated_once[:, PoseCanonicalizer.JOINT_INDICES['left_shoulder'], :]
        right_shoulder = rotated_once[:, PoseCanonicalizer.JOINT_INDICES['right_shoulder'], :]
        shoulder_vector = right_shoulder - left_shoulder
        # Project shoulder vector onto XY plane (remove Z component)
        shoulder_vector[:, 2] = 0
        shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate current angle of shoulder vector in XY plane
        current_angle = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])  # (N,)
        # Determine minimal rotation to align with X-axis
        # Two options: rotate to 0° or to 180° (±π)
        angle_to_0 = -current_angle
        angle_to_pi = np.where(current_angle > 0, 
                            np.pi - current_angle,
                            -np.pi - current_angle)
        # Choose the rotation with smaller absolute angle
        use_angle_to_0 = np.abs(angle_to_0) <= np.abs(angle_to_pi)
        rotation_angle = np.where(use_angle_to_0, angle_to_0, angle_to_pi)
        # Create rotation matrices for Z-axis rotation
        N = poses_3d.shape[0]
        R2 = np.zeros((N, 3, 3))
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        R2[:, 0, 0] = cos_angle
        R2[:, 0, 1] = -sin_angle
        R2[:, 1, 0] = sin_angle
        R2[:, 1, 1] = cos_angle
        R2[:, 2, 2] = 1
        # Apply second rotation
        final_poses = np.matmul(rotated_once, R2.transpose(0, 2, 1))
        # Translate back (optional)
        final_poses = final_poses + plane_center[:, None, :]
        return final_poses
    
    @staticmethod
    def transform_to_back_view(poses_3d: np.ndarray) -> np.ndarray:
        """
        Transform poses for back view (front view + 180° rotation around Y)
        """
        front_poses = PoseCanonicalizer.transform_to_front_view(poses_3d)
        # 180 degree rotation around Y axis
        R_flip = np.array([[-1, 0, 0],
                          [0, 1, 0], 
                          [0, 0, -1]])
        # Apply to all frames
        plane_center, _, _ = PoseCanonicalizer.extract_torso_plane(front_poses)
        centered = front_poses - plane_center[:, None, :]
        rotated = np.matmul(centered, R_flip.T)
        back_poses = rotated + plane_center[:, None, :]
        return back_poses
    
    @staticmethod
    def transform_to_left_side_view(poses_3d: np.ndarray) -> np.ndarray:
        """
        Transform poses for left side view (front view + 90° left rotation around Y)
        """
        front_poses = PoseCanonicalizer.transform_to_front_view(poses_3d)
        # 90 degree rotation around Y axis (counterclockwise)
        R_left = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
        plane_center, _, _ = PoseCanonicalizer.extract_torso_plane(front_poses)
        centered = front_poses - plane_center[:, None, :]
        rotated = np.matmul(centered, R_left.T)
        left_poses = rotated + plane_center[:, None, :]
        return left_poses
    
    @staticmethod
    def transform_to_right_side_view(poses_3d: np.ndarray) -> np.ndarray:
        """
        Transform poses for right side view (front view + 90° right rotation around Y)
        """
        front_poses = PoseCanonicalizer.transform_to_front_view(poses_3d)
        # 90 degree rotation around Y axis (clockwise)
        R_right = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 0]])
        plane_center, _, _ = PoseCanonicalizer.extract_torso_plane(front_poses)
        centered = front_poses - plane_center[:, None, :]
        rotated = np.matmul(centered, R_right.T)
        right_poses = rotated + plane_center[:, None, :]
        return right_poses
    
    @staticmethod
    def get_canonical_form(poses_3d: np.ndarray, view: Union[CanonicalView, str]) -> np.ndarray:
        """
        Transform poses to specified canonical orientation
        Args:
            poses_3d: Input poses with shape (N, 17, 3)
            view: Desired canonical view
        Returns:
            Transformed poses (N, 17, 3)
        """
        if isinstance(view, str):
            view = CanonicalView(view.lower())
        print(f'Pose is canonicalized to direction of {view.value}')
        transform_map = {
            CanonicalView.FRONT: PoseCanonicalizer.transform_to_front_view,
            CanonicalView.BACK: PoseCanonicalizer.transform_to_back_view,
            CanonicalView.LEFT_SIDE: PoseCanonicalizer.transform_to_left_side_view,
            CanonicalView.RIGHT_SIDE: PoseCanonicalizer.transform_to_right_side_view
        }
        if view not in transform_map:
            raise ValueError(f"Unsupported view: {view}. Choose from {list(CanonicalView)}")
        return transform_map[view](poses_3d)


def canonicalize_pose(poses_3d: np.ndarray, view: Union[CanonicalView, str] = "front") -> np.ndarray:
    """
    Main function to canonicalize poses
    """
    return PoseCanonicalizer.get_canonical_form(poses_3d, view)


if __name__ == "__main__":
    poses = np.random.randn(100, 17, 3)  # 100 frames, 17 joints, 3D coords
    front_poses = canonicalize_pose(poses, "front")
    back_poses = canonicalize_pose(poses, CanonicalView.BACK)
    left_poses = canonicalize_pose(poses, "left_side")
    right_poses = canonicalize_pose(poses, "right_side")
    print(f"Original poses shape: {poses.shape}")
    print(f"Canonical poses shape: {front_poses.shape}")
    print(f"Available views: {[v.value for v in CanonicalView]}")