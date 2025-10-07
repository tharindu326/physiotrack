"""
3DPCNet Inference Module
Provides functions for loading and running 3DPCNet model for pose canonicalization.
"""

import os
import torch
import yaml
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
from .models.model import create_pose_canonicalization_model


# Cache for loaded model
_MODEL_CACHE = {
    'model': None,
    'device': None,
    'config': None
}


def load_3dpcnet_model(checkpoint_path: Optional[str] = None, 
                       config_path: Optional[str] = None, 
                       device: str = 'cuda',
                       force_reload: bool = False) -> torch.nn.Module:
    """
    Load 3DPCNet model with caching support.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config YAML
        device: Device to load model on ('cuda' or 'cpu')
        force_reload: Force reload even if model is cached
        
    Returns:
        Loaded model in eval mode
    """
    global _MODEL_CACHE
    
    # Convert checkpoint_path to string for cache key
    cache_key = str(checkpoint_path) if checkpoint_path else 'default'
    
    # Check cache with checkpoint-specific key
    if not force_reload and _MODEL_CACHE.get('model') is not None and _MODEL_CACHE.get('checkpoint_path') == cache_key:
        return _MODEL_CACHE['model']
    
    # Import model creation function
    import sys
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    # Default paths
    if checkpoint_path is None:
        checkpoint_path = module_dir / 'checkpoints' / 'best_model.pth'
    if config_path is None:
        config_path = module_dir / 'checkpoints' / 'config.yaml'
    
    # Load configuration
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_cfg = config.get('model', {})
        hybrid_config = model_cfg.pop('hybrid', None)
    else:
        # Default configuration
        model_cfg = {
            'input_joints': 17,
            'encoder_type': 'hybrid',
            'rotation_type': '6d',
            'hidden_dim': 512,
            'encoder_output_dim': 256,
            'dropout': 0.1,
            'predict_mode': 'rotation_only'
        }
        hybrid_config = {
            'gcn_hidden_dim': 128,
            'transformer_hidden_dim': 256,
            'num_gcn_layers': 3,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'use_learnable_graph': True,
            'use_multi_scale': True
        }
    
    # Create model
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if model_cfg.get('encoder_type') in ['hybrid', 'hybrid_simple']:
        model = create_pose_canonicalization_model(**model_cfg, hybrid_config=hybrid_config)
    else:
        model = create_pose_canonicalization_model(**model_cfg)
    
    model = model.to(device_obj)
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded 3DPCNet model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.eval()
    
    # Update cache
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['device'] = device_obj
    _MODEL_CACHE['config'] = model_cfg
    _MODEL_CACHE['checkpoint_path'] = cache_key
    
    return model


def apply_3dpcnet_transform(poses: np.ndarray) -> np.ndarray:
    """
    Apply coordinate transformation required for 3DPCNet model.
    Transforms from standard 3D pose format to 3DPCNet expected format.
    
    Args:
        poses: Input poses in standard format (N, 17, 3) or (17, 3)
        
    Returns:
        Transformed poses ready for 3DPCNet
    """
    # Apply axis remapping: new_x = old_z, new_y = -old_x, new_z = -old_y
    transformed = np.zeros_like(poses)
    transformed[..., 0] = poses[..., 2]    # new_x = old_z
    transformed[..., 1] = -poses[..., 0]   # new_y = -old_x
    transformed[..., 2] = -poses[..., 1]   # new_z = -old_y
    
    # Center at pelvis (joint 0)
    centered = transformed - transformed[..., 0:1, :]
    
    return centered


def reverse_3dpcnet_transform(poses: np.ndarray) -> np.ndarray:
    """
    Reverse the coordinate transformation to get back to standard format.
    
    Args:
        poses: Canonicalized poses from 3DPCNet (N, 17, 3) or (17, 3)
        
    Returns:
        Poses in standard 3D format
    """
    # Since poses are centered, we can't recover the original pelvis position
    # but we can reverse the axis remapping
    reversed = np.zeros_like(poses)
    reversed[..., 0] = -poses[..., 1]  # old_x = -new_y
    reversed[..., 1] = -poses[..., 2]  # old_y = -new_z
    reversed[..., 2] = poses[..., 0]   # old_z = new_x
    
    return reversed


def canonicalize_poses_3dpcnet(poses_3d: Union[np.ndarray, torch.Tensor],
                               checkpoint_path: Optional[str] = None,
                               config_path: Optional[str] = None,
                               device: str = 'cuda',
                               batch_size: int = 256,
                               apply_transform: bool = True,
                               verbose: bool = True,
                               return_rotation: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Canonicalize 3D poses using 3DPCNet model.
    
    Args:
        poses_3d: Input poses with shape (N, 17, 3) or (17, 3)
        checkpoint_path: Optional path to model checkpoint
        config_path: Optional path to model config
        device: Device to run inference on
        batch_size: Batch size for processing multiple poses
        apply_transform: If True, apply coordinate transformation (for standard format input).
                        If False, assume input is already in 3DPCNet format.
        verbose: If True, print progress messages
        return_rotation: If True, also return rotation matrices
        
    Returns:
        If return_rotation is False: Canonicalized poses as numpy array
        If return_rotation is True: Tuple of (canonicalized poses, rotation matrices)
    """
    # Load model
    model = load_3dpcnet_model(checkpoint_path, config_path, device)
    device_obj = _MODEL_CACHE['device']
    
    # Convert to numpy for transformation
    if isinstance(poses_3d, torch.Tensor):
        poses_np = poses_3d.cpu().numpy()
    else:
        poses_np = poses_3d.copy()
    
    # Apply coordinate transformation if needed
    if apply_transform:
        transformed_poses = apply_3dpcnet_transform(poses_np)
    else:
        transformed_poses = poses_np  # Already in 3DPCNet format
    
    # Convert to tensor
    poses_tensor = torch.from_numpy(transformed_poses).float()
    
    # Handle single pose
    single_pose = False
    if poses_tensor.dim() == 2:
        poses_tensor = poses_tensor.unsqueeze(0)
        single_pose = True
    
    # Run inference
    num_samples = poses_tensor.shape[0]
    all_canonical = []
    all_rotations = [] if return_rotation else None
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = poses_tensor[start_idx:end_idx].to(device_obj)
            
            # Run model
            output = model(batch)
            
            # Extract canonical poses and rotation representation
            if isinstance(output, tuple):
                canonical = output[0]
                rotation_repr = output[1] if len(output) > 1 else None
            else:
                canonical = output
                rotation_repr = None
            
            all_canonical.append(canonical.cpu())
            
            # Convert rotation representation to rotation matrix if needed
            if return_rotation and rotation_repr is not None:
                rotation_matrices = model.get_rotation_matrix(rotation_repr)
                all_rotations.append(rotation_matrices.cpu())
    
    # Concatenate results
    canonical_poses = torch.cat(all_canonical, dim=0)
    rotation_matrices = None
    if return_rotation and all_rotations:
        rotation_matrices = torch.cat(all_rotations, dim=0)
    
    # Convert to numpy
    canonical_poses = canonical_poses.numpy()
    if rotation_matrices is not None:
        rotation_matrices = rotation_matrices.numpy()
    
    # Remove batch dimension if single pose
    if single_pose:
        canonical_poses = canonical_poses.squeeze(0)
        if rotation_matrices is not None:
            rotation_matrices = rotation_matrices.squeeze(0)
    
    # Reverse the coordinate transformation if we applied it initially
    if apply_transform:
        canonical_poses_output = reverse_3dpcnet_transform(canonical_poses)
    else:
        canonical_poses_output = canonical_poses  # Keep in 3DPCNet format
    
    if return_rotation:
        return canonical_poses_output, rotation_matrices
    return canonical_poses_output


def process_npz_file(npz_path: str,
                    output_path: Optional[str] = None,
                    checkpoint_path: Optional[str] = None,
                    config_path: Optional[str] = None,
                    device: str = 'cuda',
                    batch_size: int = 256) -> Dict[str, np.ndarray]:
    """
    Process poses from NPZ file.
    
    Args:
        npz_path: Path to input NPZ file
        output_path: Optional path to save output NPZ
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        device: Device for inference
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with canonicalized poses and other data
    """
    # Load data
    data = np.load(npz_path)
    
    # Extract poses (handle different formats)
    if 'poses' in data:
        poses = data['poses']
    elif 'poses_3d' in data:
        poses = data['poses_3d']
    elif 'keypoints_3d' in data:
        poses = data['keypoints_3d']
    else:
        raise ValueError(f"No pose data found in {npz_path}")
    
    # Canonicalize
    canonical_poses = canonicalize_poses_3dpcnet(
        poses, checkpoint_path, config_path, device, batch_size
    )
    
    # Prepare output
    output_data = {
        'canonical_poses': canonical_poses,
        'original_poses': poses
    }
    
    # Add other data from input file
    for key in data.keys():
        if key not in ['poses', 'poses_3d', 'keypoints_3d']:
            output_data[key] = data[key]
    
    # Save if output path provided
    if output_path:
        np.savez(output_path, **output_data)
        print(f"Saved canonicalized poses to {output_path}")
    
    return output_data


# For backward compatibility with command-line usage
if __name__ == "__main__":
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description="3DPCNet Inference")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--npz', type=str, required=True, help='Path to input data NPZ file')
    parser.add_argument('--save', type=str, default=None, help='Path to save outputs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('3dpcnet_inference')
    
    # Process file
    logger.info(f"Processing {args.npz}")
    results = process_npz_file(
        args.npz,
        args.save,
        args.checkpoint,
        args.config,
        args.device,
        args.batch_size
    )
    
    logger.info(f"Canonicalized poses shape: {results['canonical_poses'].shape}")
    if args.save:
        logger.info(f"Results saved to {args.save}")