"""
Evaluation metrics for 3D pose estimation and canonicalization.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union


def compute_similarity_transform(X: np.ndarray, Y: np.ndarray, 
                                compute_optimal_scale: bool = False) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Computes the similarity transform (rotation, translation, scale) that best aligns Y to X.
    
    Args:
        X: Target poses, shape (N, 3) where N is number of points
        Y: Input poses to be transformed, shape (N, 3)
        compute_optimal_scale: If True, compute optimal scale; if False, scale = 1
        
    Returns:
        d: Squared error after transformation
        Z: Transformed Y
        T: Rotation matrix (3, 3)
        b: Scale factor
        c: Translation vector (3,)
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # Centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # Scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # Optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def calculate_mpjpe(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Calculate Mean Per Joint Position Error (MPJPE).
    
    Args:
        preds: Predicted poses, shape (N, J, 3) where N is batch size, J is number of joints
        gts: Ground truth poses, shape (N, J, 3)
        
    Returns:
        mpjpe: Mean per joint position error in the same units as input
    """
    assert preds.shape == gts.shape, f"Shape mismatch: preds {preds.shape} vs gts {gts.shape}"
    
    # Compute Euclidean distance for each joint
    distances = np.linalg.norm(preds - gts, axis=-1)  # (N, J)
    
    # Mean over all joints and samples
    mpjpe = np.mean(distances)
    
    return mpjpe


def calculate_pampjpe(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Calculate Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE).
    This metric aligns each predicted pose to the ground truth using similarity transform.
    
    Args:
        preds: Predicted poses, shape (N, J, 3)
        gts: Ground truth poses, shape (N, J, 3)
        
    Returns:
        pampjpe: Procrustes aligned MPJPE
    """
    assert preds.shape == gts.shape, f"Shape mismatch: preds {preds.shape} vs gts {gts.shape}"
    
    N = preds.shape[0]
    num_joints = preds.shape[1]
    
    pampjpe_per_sample = []
    
    for n in range(N):
        frame_pred = preds[n]  # (J, 3)
        frame_gt = gts[n]      # (J, 3)
        
        # Apply similarity transform to align prediction with ground truth
        _, _, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred_aligned = (b * frame_pred.dot(T)) + c
        
        # Compute error after alignment
        joint_errors = np.linalg.norm(frame_pred_aligned - frame_gt, axis=-1)  # (J,)
        pampjpe_per_sample.append(np.mean(joint_errors))
    
    pampjpe = np.mean(pampjpe_per_sample)
    
    return pampjpe


def calculate_rotation_error(pred_rotation: np.ndarray, gt_rotation: np.ndarray, 
                            return_degrees: bool = True,
                            method: str = 'frobenius') -> float:
    """
    Calculate rotation error between predicted and ground truth rotation matrices.
    
    Args:
        pred_rotation: Predicted rotation matrices, shape (N, 3, 3)
        gt_rotation: Ground truth rotation matrices, shape (N, 3, 3)
        return_degrees: If True, return error in degrees; if False, in radians
        method: 'geodesic' for proper rotation angle, 'frobenius' to match 3DPCNet (default)
        
    Returns:
        rotation_error: Mean rotation error
    """
    assert pred_rotation.shape == gt_rotation.shape, f"Shape mismatch: {pred_rotation.shape} vs {gt_rotation.shape}"
    
    if method == 'frobenius':
        # Match 3DPCNet's approach: Frobenius norm treated as radians
        rotation_error = np.mean(np.linalg.norm(pred_rotation - gt_rotation, axis=(1, 2)))
        if return_degrees:
            rotation_error = rotation_error * 180.0 / np.pi
    else:  # geodesic
        N = pred_rotation.shape[0]
        errors = []
        
        for i in range(N):
            # Compute relative rotation: R_rel = R_gt @ R_pred^T
            R_rel = np.dot(gt_rotation[i], pred_rotation[i].T)
            
            # Extract angle from rotation matrix using trace
            # angle = arccos((trace(R) - 1) / 2)
            trace = np.trace(R_rel)
            trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
            angle = np.arccos((trace - 1.0) / 2.0)
            
            errors.append(angle)
        
        rotation_error = np.mean(errors)
        
        if return_degrees:
            rotation_error = np.degrees(rotation_error)
    
    return rotation_error


def evaluate_pose_predictions(preds: np.ndarray, gts: np.ndarray, 
                             scale: float = 1000.0) -> Dict[str, float]:
    """
    Evaluate 3D pose predictions with multiple metrics.
    
    Args:
        preds: Predicted poses, shape (N, J, 3)
        gts: Ground truth poses, shape (N, J, 3)
        scale: Scale factor for output (e.g., 1000 to convert meters to millimeters)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()
    
    # Calculate metrics
    mpjpe = calculate_mpjpe(preds, gts) * scale
    pampjpe = calculate_pampjpe(preds, gts) * scale
    
    # Additional metrics
    # Per-joint errors
    joint_errors = np.mean(np.linalg.norm(preds - gts, axis=-1), axis=0) * scale  # (J,)
    
    return {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'mpjpe_per_joint': joint_errors.tolist(),
        'max_joint_error': np.max(joint_errors),
        'min_joint_error': np.min(joint_errors)
    }


def evaluate_canonicalization(pred_canonical: np.ndarray, gt_canonical: np.ndarray,
                             pred_rotation: Optional[np.ndarray] = None, 
                             gt_rotation: Optional[np.ndarray] = None,
                             scale: float = 1000.0) -> Dict[str, float]:
    """
    Evaluate pose canonicalization results.
    
    Args:
        pred_canonical: Predicted canonical poses, shape (N, J, 3)
        gt_canonical: Ground truth canonical poses, shape (N, J, 3)
        pred_rotation: Optional predicted rotation matrices, shape (N, 3, 3)
        gt_rotation: Optional ground truth rotation matrices, shape (N, 3, 3)
        scale: Scale factor for pose errors (e.g., 1000 for mm)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(pred_canonical, torch.Tensor):
        pred_canonical = pred_canonical.detach().cpu().numpy()
    if isinstance(gt_canonical, torch.Tensor):
        gt_canonical = gt_canonical.detach().cpu().numpy()
    
    # Pose metrics
    metrics = evaluate_pose_predictions(pred_canonical, gt_canonical, scale=scale)
    
    # Add pose error (direct L2 distance)
    pose_error = np.mean(np.linalg.norm(pred_canonical - gt_canonical, axis=-1)) * scale
    metrics['pose_error_mm'] = pose_error
    
    # Rotation metrics if provided
    if pred_rotation is not None and gt_rotation is not None:
        if isinstance(pred_rotation, torch.Tensor):
            pred_rotation = pred_rotation.detach().cpu().numpy()
        if isinstance(gt_rotation, torch.Tensor):
            gt_rotation = gt_rotation.detach().cpu().numpy()
        
        rotation_error_deg = calculate_rotation_error(pred_rotation, gt_rotation, return_degrees=True)
        rotation_error_rad = calculate_rotation_error(pred_rotation, gt_rotation, return_degrees=False)
        
        metrics.update({
            'rotation_error_deg': rotation_error_deg,
            'rotation_error_rad': rotation_error_rad
        })
    
    return metrics


def compare_canonicalization_methods(poses_3d: np.ndarray, 
                                    gt_canonical: Optional[np.ndarray] = None,
                                    gt_rotation: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare different canonicalization methods on the same input poses.
    
    Args:
        poses_3d: Input poses, shape (N, J, 3)
        gt_canonical: Optional ground truth canonical poses
        gt_rotation: Optional ground truth rotation matrices
        
    Returns:
        Dictionary with metrics for each method
    """
    from .canonicalizer import canonicalize_pose
    from ..models import Models
    
    results = {}
    
    # Geometric method
    try:
        geometric_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['geometric'] = evaluate_pose_predictions(
                geometric_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['geometric'] = {'status': 'computed', 'shape': geometric_canonical.shape}
    except Exception as e:
        results['geometric'] = {'error': str(e)}
    
    # 3DPCNet S2 method
    try:
        s2_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS2,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['3dpcnet_s2'] = evaluate_pose_predictions(
                s2_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['3dpcnet_s2'] = {'status': 'computed', 'shape': s2_canonical.shape}
    except Exception as e:
        results['3dpcnet_s2'] = {'error': str(e)}
    
    # 3DPCNet S3 method (if available)
    try:
        s3_canonical = canonicalize_pose(
            poses_3d,
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS3,
            view=Models.Pose3D.Canonicalizer.View.FRONT
        )
        
        if gt_canonical is not None:
            results['3dpcnet_s3'] = evaluate_pose_predictions(
                s3_canonical, gt_canonical, scale=1000.0
            )
        else:
            results['3dpcnet_s3'] = {'status': 'computed', 'shape': s3_canonical.shape}
    except Exception as e:
        results['3dpcnet_s3'] = {'error': str(e)}
    
    return results