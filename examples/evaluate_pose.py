"""
Evaluation script for comparing pose canonicalization methods.
Evaluates GEOMETRIC, 3DPCNet S2, and 3DPCNet S3 methods on test.npz data.

IMPORTANT COORDINATE SYSTEM NOTES:
- test.npz contains data in 3DPCNet format (axis-remapped: x=old_z, y=-old_x, z=-old_y, and centered at pelvis)
- 3DPCNet models expect and work directly with this format (NO transformation needed)
- GEOMETRIC method expects standard format (like results_3d from pose estimator) so we reverse transform
"""

import numpy as np
import sys
import os
from tqdm import tqdm

# # Add parent directory to path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physiotrack import Models, canonicalize_pose
from physiotrack.pose.evaluate import evaluate_canonicalization
from physiotrack.modules._3DCPNet.inference import reverse_3dpcnet_transform


# Load test data
print("Loading test data from: test.npz")
data = np.load('S3test.npz', allow_pickle=True)

# Get ALL data - in 3DPCNET FORMAT (transformed and centered)
input_poses_3dpcnet = data['input_pose']  # All samples
canonical_gt_3dpcnet = data['canonical_pose']  # GT in 3DPCNet format
rotation_gt = data['rotation_matrix']

print(f"Total samples in dataset: {input_poses_3dpcnet.shape[0]}")
print(f"Input shape: {input_poses_3dpcnet.shape}")
print(f"Ground truth canonical shape: {canonical_gt_3dpcnet.shape}")
print(f"Ground truth rotation shape: {rotation_gt.shape}")
print("\nNote: All data in test.npz is in 3DPCNet format (axis-remapped and centered)")

# Batch processing configuration
BATCH_SIZE = 1000  # Process in batches to avoid memory issues
num_samples = input_poses_3dpcnet.shape[0]
num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nProcessing {num_samples} samples in {num_batches} batches of size {BATCH_SIZE}")

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# ==============================================================================
# Method 1: GEOMETRIC (needs standard format input)
# ==============================================================================
print("\n--- GEOMETRIC Method ---")
print("Converting from 3DPCNet format to standard format for geometric method...")
try:
    # Process in batches
    all_geometric_canonical = []
    all_geometric_rotation = []
    
    for batch_idx in tqdm(range(num_batches), desc="  Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        
        # Get batch
        batch_input_3dpcnet = input_poses_3dpcnet[start_idx:end_idx]
        
        # GEOMETRIC needs standard format (like results_3d from pose estimator)
        batch_input_standard = reverse_3dpcnet_transform(batch_input_3dpcnet)
        
        # Run geometric canonicalization on batch
        batch_canonical, batch_rotation = canonicalize_pose(
            batch_input_standard,  # Standard format input
            model=Models.Pose3D.Canonicalizer.Models.GEOMETRIC,
            view=Models.Pose3D.Canonicalizer.View.FRONT,
            return_rotation=True
        )
        
        all_geometric_canonical.append(batch_canonical)
        all_geometric_rotation.append(batch_rotation)
    
    # Concatenate all batches
    geometric_canonical = np.concatenate(all_geometric_canonical, axis=0)
    geometric_rotation = np.concatenate(all_geometric_rotation, axis=0)
    
    # For fair comparison, convert GT to standard format too
    canonical_gt_standard = reverse_3dpcnet_transform(canonical_gt_3dpcnet)
    
    # Evaluate in standard format
    metrics = evaluate_canonicalization(
        geometric_canonical,  # Standard format output
        canonical_gt_standard,  # Standard format GT
        pred_rotation=geometric_rotation,
        gt_rotation=rotation_gt,
        scale=1000.0
    )
    
    print(f"\n  Results for {num_samples} samples:")
    print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
    print(f"  PA-MPJPE: {metrics['pampjpe']:.2f} mm")
    print(f"  Pose Error: {metrics['pose_error_mm']:.2f} mm")
    if 'rotation_error_deg' in metrics:
        print(f"  Rotation Error: {metrics['rotation_error_deg']:.2f}°")
    
except Exception as e:
    print(f"  Error: {e}")

# ==============================================================================
# Method 2: 3DPCNet S2 (via physiotrack wrapper with auto-transformation)
# ==============================================================================
print("\n--- 3DPCNet S2 Method (physiotrack wrapper) ---")
print("Using physiotrack wrapper which handles transformations internally...")
try:
    # Process in batches
    all_s2_canonical = []
    all_s2_rotation = []
    
    for batch_idx in tqdm(range(num_batches), desc="  Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        
        # Get batch
        batch_input_3dpcnet = input_poses_3dpcnet[start_idx:end_idx]
        
        # Since data is already in 3DPCNet format, use apply_transform=False
        batch_canonical, batch_rotation = canonicalize_pose(
            batch_input_3dpcnet,  # Already in 3DPCNet format
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS2,
            view=Models.Pose3D.Canonicalizer.View.FRONT,
            apply_transform=False,  # Don't transform since already in 3DPCNet format
            verbose=False,  # Suppress print messages
            return_rotation=True  # Get rotation matrices
        )
        
        all_s2_canonical.append(batch_canonical)
        all_s2_rotation.append(batch_rotation)
    
    # Concatenate all batches
    s2_canonical = np.concatenate(all_s2_canonical, axis=0)
    s2_rotation = np.concatenate(all_s2_rotation, axis=0)
    
    metrics = evaluate_canonicalization(
        s2_canonical,  # 3DPCNet format output
        canonical_gt_3dpcnet,  # 3DPCNet format GT
        pred_rotation=s2_rotation,  # Now we have rotation matrices!
        gt_rotation=rotation_gt,
        scale=1000.0
    )
    
    print(f"\n  Results for {num_samples} samples:")
    print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
    print(f"  PA-MPJPE: {metrics['pampjpe']:.2f} mm")
    print(f"  Pose Error: {metrics['pose_error_mm']:.2f} mm")
    if 'rotation_error_deg' in metrics:
        print(f"  Rotation Error: {metrics['rotation_error_deg']:.2f}°")
    
except Exception as e:
    print(f"  Error: {e}")

# ==============================================================================
# Method 3: 3DPCNet S3 (via physiotrack wrapper with auto-transformation)
# ==============================================================================
print("\n--- 3DPCNet S3 Method (physiotrack wrapper) ---")
print("Using physiotrack wrapper which handles transformations internally...")
try:
    # Process in batches
    all_s3_canonical = []
    all_s3_rotation = []
    
    for batch_idx in tqdm(range(num_batches), desc="  Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        
        # Get batch
        batch_input_3dpcnet = input_poses_3dpcnet[start_idx:end_idx]
        
        # Since data is already in 3DPCNet format, use apply_transform=False
        batch_canonical, batch_rotation = canonicalize_pose(
            batch_input_3dpcnet,  # Already in 3DPCNet format
            model=Models.Pose3D.Canonicalizer.Models._3DPCNetS3,
            view=Models.Pose3D.Canonicalizer.View.FRONT,
            apply_transform=False,  # Don't transform since already in 3DPCNet format
            verbose=False,  # Suppress print messages
            return_rotation=True  # Get rotation matrices
        )
        
        all_s3_canonical.append(batch_canonical)
        all_s3_rotation.append(batch_rotation)
    
    # Concatenate all batches
    s3_canonical = np.concatenate(all_s3_canonical, axis=0)
    s3_rotation = np.concatenate(all_s3_rotation, axis=0)
    
    metrics = evaluate_canonicalization(
        s3_canonical,  # 3DPCNet format output
        canonical_gt_3dpcnet,  # 3DPCNet format GT
        pred_rotation=s3_rotation,  # Now we have rotation matrices!
        gt_rotation=rotation_gt,
        scale=1000.0
    )
    
    print(f"\n  Results for {num_samples} samples:")
    print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
    print(f"  PA-MPJPE: {metrics['pampjpe']:.2f} mm")
    print(f"  Pose Error: {metrics['pose_error_mm']:.2f} mm")
    if 'rotation_error_deg' in metrics:
        print(f"  Rotation Error: {metrics['rotation_error_deg']:.2f}°")
    
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*60)


print("\n" + "="*60)
print("Evaluation complete!")