"""
Example demonstrating three methods to apply canonical view to 3D poses
"""

from physiotrack.pose.pose3D import Pose3D
from physiotrack import Models, PoseCanonicalizer, canonicalize_pose
import numpy as np
from pathlib import Path

# Configuration
json_path = 'output/BV_S17_cut1_result.json'
video_path = 'BV_S17_cut1.mp4'
output_dir = 'output/'

# Initialize 3D pose estimator
pose3D = Pose3D(
    model=Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite,
    device='cuda',
    clip_len=243,
    render_video=True,
    save_npy=True
)

# =============================================================================
# Method 1: Integrated - Apply canonical view during 3D pose estimation
# =============================================================================
frames_data, results_3d = pose3D.estimate(
    json_path=json_path,
    vid_path=video_path,
    out_path=output_dir,
    canonical_view=Models.Pose3DCanonicalizer.View.FRONT,
    canonical_method=Models.Pose3DCanonicalizer.Method.GEOMETRIC
)

# =============================================================================
# Method 2: Direct - Apply canonical view to results_3d array
# =============================================================================
# First estimate without canonical view
frames_data_raw, results_3d_raw = pose3D.estimate(
    json_path=json_path,
    vid_path=video_path,
    out_path=output_dir,
    canonical_view=None
)

# Apply canonical view directly to the array
canonical_poses = canonicalize_pose(
    results_3d_raw,
    view=Models.Pose3DCanonicalizer.View.FRONT,
    method=Models.Pose3DCanonicalizer.Method.GEOMETRIC
)

# =============================================================================
# Method 3: File-based - Process from saved .npy file
# =============================================================================
# Assume you have a saved 3D poses file
npy_file = 'output/X3D_20250109_120000.npy'

if Path(npy_file).exists():
    canonical_from_file = PoseCanonicalizer.process_npy_file(
        npy_file,
        output_path='output/X3D_canonical.npy',
        view=Models.Pose3DCanonicalizer.View.FRONT,
        method=Models.Pose3DCanonicalizer.Method.GEOMETRIC
    )