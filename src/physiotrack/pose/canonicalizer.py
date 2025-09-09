import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
import json
from pathlib import Path
from ..models import Models

# Import the enums from Models
CanonicalView = Models.Pose3DCanonicalizer.View
CanonicalMethod = Models.Pose3DCanonicalizer.Method

class PoseCanonicalizer:
    """
    Transform 3D poses to canonical orientations using different methods.
    Supports geometric transformation and future 3DPCNet-based canonicalization.
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
    def canonicalize_3dcpnet(poses_3d: np.ndarray, view: Union[CanonicalView, str]) -> np.ndarray:
        """
        Future 3DPCNet-based canonicalization method.
        
        Args:
            poses_3d: Input poses with shape (N, 17, 3)
            view: Desired canonical view
            
        Returns:
            Transformed poses (N, 17, 3)
        """
        # Placeholder for future 3DPCNet implementation
        raise NotImplementedError(
            "3DPCNet canonicalization is not yet implemented. "
            "This will be added when the 3DPCNet model is integrated."
        )
    
    @staticmethod
    def get_canonical_form_geometric(poses_3d: np.ndarray, view: Union[CanonicalView, str]) -> np.ndarray:
        """
        Transform poses to specified canonical orientation using geometric method.
        Args:
            poses_3d: Input poses with shape (N, 17, 3)
            view: Desired canonical view
        Returns:
            Transformed poses (N, 17, 3)
        """
        if isinstance(view, str):
            view = CanonicalView(view.lower())
        print(f'Pose is canonicalized to {view.value} view using geometric method')
        transform_map = {
            CanonicalView.FRONT: PoseCanonicalizer.transform_to_front_view,
            CanonicalView.BACK: PoseCanonicalizer.transform_to_back_view,
            CanonicalView.LEFT_SIDE: PoseCanonicalizer.transform_to_left_side_view,
            CanonicalView.RIGHT_SIDE: PoseCanonicalizer.transform_to_right_side_view
        }
        if view not in transform_map:
            raise ValueError(f"Unsupported view: {view}. Choose from {list(CanonicalView)}")
        return transform_map[view](poses_3d)
    
    @staticmethod
    def process_json_file(json_path: str, output_path: Optional[str] = None, 
                         view: Union[CanonicalView, str] = "front",
                         method: Union[CanonicalMethod, str] = CanonicalMethod.GEOMETRIC) -> Dict[str, Any]:
        """
        Process 3D poses from a JSON file containing detection data.
        
        Args:
            json_path: Path to input JSON file with 3D keypoints
            output_path: Optional path to save processed JSON
            view: Target canonical view
            method: Canonicalization method to use
            
        Returns:
            Updated detection data with canonicalized 3D poses
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract 3D keypoints from detection data
        poses_3d = PoseCanonicalizer._extract_3d_from_json(data)
        
        if poses_3d is not None:
            # Apply canonical transformation
            canonical_poses = canonicalize_pose(poses_3d, view, method)
            
            # Update the data with canonical poses
            data = PoseCanonicalizer._update_json_with_3d(data, canonical_poses, view, method)
            
            # Save if output path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
        
        return data
    
    @staticmethod
    def process_npy_file(npy_path: str, output_path: Optional[str] = None,
                        view: Union[CanonicalView, str] = "front",
                        method: Union[CanonicalMethod, str] = CanonicalMethod.GEOMETRIC) -> np.ndarray:
        """
        Process 3D poses from a numpy file.
        
        Args:
            npy_path: Path to input numpy file with 3D poses
            output_path: Optional path to save processed numpy array
            view: Target canonical view
            method: Canonicalization method to use
            
        Returns:
            Canonicalized poses array
        """
        poses_3d = np.load(npy_path)
        canonical_poses = canonicalize_pose(poses_3d, view, method)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, canonical_poses)
        
        return canonical_poses
    
    @staticmethod
    def _extract_3d_from_json(data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract 3D keypoints from detection JSON data.
        
        Args:
            data: Detection data dictionary
            
        Returns:
            3D poses array or None if not found
        """
        # Check if data contains 3D keypoints
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        # Collect all 3D keypoints
        poses_3d_list = []
        for frame_data in data:
            if 'keypoints_3d' in frame_data:
                poses_3d_list.append(frame_data['keypoints_3d'])
        
        if len(poses_3d_list) > 0:
            poses_3d = np.array(poses_3d_list)
            # Reshape if necessary (assuming 17 joints with 3 coordinates each)
            if poses_3d.ndim == 2:
                poses_3d = poses_3d.reshape(-1, 17, 3)
            return poses_3d
        
        return None
    
    @staticmethod
    def _update_json_with_3d(data: Dict[str, Any], poses_3d: np.ndarray, 
                            view: Union[CanonicalView, str], method: Union[CanonicalMethod, str]) -> Dict[str, Any]:
        """
        Update detection JSON data with canonicalized 3D poses.
        
        Args:
            data: Original detection data
            poses_3d: Canonicalized 3D poses
            view: Applied canonical view
            method: Applied canonicalization method
            
        Returns:
            Updated detection data
        """
        if isinstance(view, str):
            view = CanonicalView(view.lower())
        if isinstance(method, str):
            method = CanonicalMethod(method.lower())
            
        if isinstance(data, list) and len(data) == len(poses_3d):
            for i, frame_data in enumerate(data):
                if 'keypoints_3d' in frame_data:
                    # Update with canonicalized poses
                    frame_data['keypoints_3d'] = poses_3d[i].flatten().tolist()
                    frame_data['canonical_view_applied'] = view.value
                    frame_data['canonical_method'] = method.value
        
        return data


def canonicalize_pose(poses_3d: np.ndarray, 
                     view: Union[CanonicalView, str] = "front",
                     method: Union[CanonicalMethod, str] = CanonicalMethod.GEOMETRIC) -> np.ndarray:
    """
    Main function to canonicalize poses using specified method.
    
    Args:
        poses_3d: Input poses with shape (N, 17, 3)
        view: Desired canonical view (front, back, left_side, right_side)
        method: Canonicalization method (geometric or 3dcpnet)
        
    Returns:
        Canonicalized poses (N, 17, 3)
    """
    if isinstance(method, str):
        method = CanonicalMethod(method.lower())
    
    if method == CanonicalMethod.GEOMETRIC:
        return PoseCanonicalizer.get_canonical_form_geometric(poses_3d, view)
    elif method == CanonicalMethod.THREECPNET:
        return PoseCanonicalizer.canonicalize_3dcpnet(poses_3d, view)
    else:
        raise ValueError(f"Unsupported canonicalization method: {method}. Choose from {list(CanonicalMethod)}")


if __name__ == "__main__":
    poses = np.random.randn(100, 17, 3)  # 100 frames, 17 joints, 3D coords
    
    # Test geometric method
    print("Testing geometric canonicalization method:")
    front_poses = canonicalize_pose(poses, "front", method="geometric")
    back_poses = canonicalize_pose(poses, CanonicalView.BACK, method=CanonicalMethod.GEOMETRIC)
    left_poses = canonicalize_pose(poses, "left_side")
    right_poses = canonicalize_pose(poses, "right_side")
    print(f"Original poses shape: {poses.shape}")
    print(f"Canonical poses shape: {front_poses.shape}")
    print(f"Available views: {[v.value for v in CanonicalView]}")
    print(f"Available methods: {[m.value for m in CanonicalMethod]}")
    
    # Test future 3DPCNet method (will raise NotImplementedError)
    print("\nTesting 3DPCNet method (not yet implemented):")
    try:
        future_poses = canonicalize_pose(poses, "front", method="3dcpnet")
    except NotImplementedError as e:
        print(f"Expected error: {e}")