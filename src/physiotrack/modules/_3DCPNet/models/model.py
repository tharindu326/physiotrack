import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .encoders import MLPEncoder, GCNEncoder, TransformerEncoder
from .hybrid_encoder import HybridGCNTransformerEncoder, SimplifiedHybridEncoder
from ..utils.rotation_utils import sixd_to_rotation_matrix, quaternion_to_rotation_matrix, apply_rotation_to_pose


class PoseCanonicalizationNet(nn.Module):
    """
    Main model for pose canonicalization: 3D pose -> Canonical 3D pose + rotation
    Adapted from 6DRepNet architecture for pose canonicalization task
    """
    def __init__(self, 
                 input_joints: int = 17,
                 encoder_type: str = 'mlp',
                 rotation_type: str = '6d',
                 hidden_dim: int = 512,
                 encoder_output_dim: int = 256,
                 dropout: float = 0.1,
                 predict_mode: str = 'rotation_only',
                 num_layers: int = None,
                 hybrid_config: dict = None):
        """
        Args:
            input_joints: Number of input joints
            encoder_type: 'mlp', 'gcn', or 'transformer'
            rotation_type: '6d', 'quaternion', or 'matrix'
            hidden_dim: Hidden dimension for encoder
            encoder_output_dim: Output dimension of encoder
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_joints = input_joints
        self.encoder_type = encoder_type
        self.rotation_type = rotation_type
        self.predict_mode = predict_mode  # 'rotation_only' | 'rotation_plus_residual'
        self.has_aux_outputs = False  # Flag for hybrid encoder auxiliary outputs
        
        # Set default num_layers based on encoder type if not specified
        if num_layers is None:
            if encoder_type == 'mlp':
                num_layers = 4
            elif encoder_type == 'gcn':
                num_layers = 3
            elif encoder_type == 'transformer':
                num_layers = 4
        
        self.num_layers = num_layers
        
        # Initialize encoder based on type
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(
                input_dim=input_joints * 3,  # 3D coordinates
                hidden_dim=hidden_dim,
                output_dim=encoder_output_dim,
                dropout=dropout,
                num_layers=self.num_layers
            )
        elif encoder_type == 'gcn':
            self.encoder = GCNEncoder(
                input_dim=3,  # 3D coordinates per joint
                hidden_dim=hidden_dim // 4,  # Smaller per-node features
                output_dim=encoder_output_dim,
                num_joints=input_joints,
                dropout=dropout,
                num_layers=self.num_layers
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_dim=3,  # 3D coordinates per joint
                hidden_dim=min(hidden_dim, 256),
                num_heads=8,
                num_layers=4,
                num_joints=input_joints,
                dropout=dropout,
                output_dim=encoder_output_dim
            )
        elif encoder_type == 'hybrid':
            # Get hybrid-specific config or use defaults
            hybrid_cfg = hybrid_config or {}
            self.encoder = HybridGCNTransformerEncoder(
                input_dim=3,
                hidden_dim=hidden_dim,
                gcn_hidden_dim=hybrid_cfg.get('gcn_hidden_dim', hidden_dim // 4),
                transformer_hidden_dim=hybrid_cfg.get('transformer_hidden_dim', min(hidden_dim, 256)),
                output_dim=encoder_output_dim,
                num_joints=input_joints,
                num_gcn_layers=hybrid_cfg.get('num_gcn_layers', 3),
                num_transformer_layers=hybrid_cfg.get('num_transformer_layers', 2),
                num_heads=hybrid_cfg.get('num_heads', 8),
                dropout=dropout,
                use_learnable_graph=hybrid_cfg.get('use_learnable_graph', True),
                use_multi_scale=hybrid_cfg.get('use_multi_scale', True)
            )
            self.has_aux_outputs = True
        elif encoder_type == 'hybrid_simple':
            # Get hybrid-specific config or use defaults
            hybrid_cfg = hybrid_config or {}
            self.encoder = SimplifiedHybridEncoder(
                input_dim=3,
                hidden_dim=hidden_dim,
                output_dim=encoder_output_dim,
                num_joints=input_joints,
                num_gcn_layers=hybrid_cfg.get('num_gcn_layers', 2),
                num_transformer_layers=hybrid_cfg.get('num_transformer_layers', 2),
                num_heads=hybrid_cfg.get('num_heads', 4),
                dropout=dropout
            )
            self.has_aux_outputs = False
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Output heads (similar to 6DRepNet's linear_reg)
        if rotation_type == '6d':
            rotation_dim = 6
        elif rotation_type == 'quaternion':
            rotation_dim = 4
        elif rotation_type == 'matrix':
            rotation_dim = 9
        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}")
        
        # Prediction heads (rotation + optional residual)
        self.rotation_head = nn.Linear(encoder_output_dim, rotation_dim)
        self.residual_head = nn.Linear(encoder_output_dim, input_joints * 3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights (following 6DRepNet pattern)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, pose_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            pose_3d: (batch, joints, 3) or (batch, joints*3) input 3D poses
        Returns:
            canonical_pose: (batch, joints, 3) canonical 3D pose
            rotation_repr: (batch, rotation_dim) rotation representation
        """
        batch_size = pose_3d.shape[0]
        
        # Ensure correct input format for encoder
        if self.encoder_type == 'mlp':
            if pose_3d.dim() == 3:
                pose_3d_flat = pose_3d.view(batch_size, -1)
            else:
                pose_3d_flat = pose_3d
            features = self.encoder(pose_3d_flat)
        else:  # gcn, transformer, hybrid
            if pose_3d.dim() == 2:
                pose_3d = pose_3d.view(batch_size, self.input_joints, 3)
            
            # Handle hybrid encoder with auxiliary outputs
            if self.encoder_type == 'hybrid' and self.has_aux_outputs:
                features, self.aux_outputs = self.encoder(pose_3d)
            else:
                features = self.encoder(pose_3d)
                self.aux_outputs = None
        
        # Generate outputs (following 6DRepNet pattern)
        rotation_repr = self.rotation_head(features)
        
        # Post-process rotation if needed
        if self.rotation_type == 'quaternion':
            rotation_repr = F.normalize(rotation_repr, dim=1)
        
        # Compute rotation-inverted canonical and optional residual
        R = self.get_rotation_matrix(rotation_repr)
        R_inv = R.transpose(1, 2)
        pose_in = pose_3d if pose_3d.dim() == 3 else pose_3d.view(batch_size, self.input_joints, 3)
        # R^{-1} * pose_in
        canon_by_rot = torch.bmm(pose_in.reshape(batch_size, -1, 3), R_inv).reshape(batch_size, self.input_joints, 3)
        
        if self.predict_mode == 'rotation_only':
            canonical_pose = canon_by_rot
        else:  # rotation_plus_residual
            residual_flat = self.residual_head(features)
            residual = residual_flat.view(batch_size, self.input_joints, 3)
            canonical_pose = canon_by_rot + residual
        
        return canonical_pose, rotation_repr
    
    def get_rotation_matrix(self, rotation_repr: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation representation to rotation matrix
        (adapted from 6DRepNet's compute_rotation_matrix_from_ortho6d)
        Args:
            rotation_repr: (batch, rotation_dim) rotation representation
        Returns:
            R: (batch, 3, 3) rotation matrices
        """
        if self.rotation_type == '6d':
            return sixd_to_rotation_matrix(rotation_repr)
        elif self.rotation_type == 'quaternion':
            return quaternion_to_rotation_matrix(rotation_repr)
        elif self.rotation_type == 'matrix':
            return rotation_repr.view(-1, 3, 3)
        else:
            raise ValueError(f"Unknown rotation_type: {self.rotation_type}")
    
    def apply_rotation(self, pose_3d: torch.Tensor, rotation_repr: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to 3D pose
        Args:
            pose_3d: (batch, joints, 3) 3D poses
            rotation_repr: (batch, rotation_dim) rotation representation
        Returns:
            rotated_pose: (batch, joints, 3) rotated poses
        """
        rotation_matrix = self.get_rotation_matrix(rotation_repr)
        return apply_rotation_to_pose(pose_3d, rotation_matrix)
    
    def canonicalize_pose(self, pose_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete canonicalization pipeline
        Args:
            pose_3d: (batch, joints, 3) input 3D poses
        Returns:
            Dictionary with canonical pose, rotation matrix, and intermediate results
        """
        # Forward pass
        canonical_pose, rotation_repr = self.forward(pose_3d)
        
        # Get rotation matrix
        rotation_matrix = self.get_rotation_matrix(rotation_repr)
        
        # Verify by applying rotation to canonical pose
        reconstructed_pose = self.apply_rotation(canonical_pose, rotation_repr)
        
        return {
            'canonical_pose': canonical_pose,
            'rotation_matrix': rotation_matrix,
            'rotation_repr': rotation_repr,
            'reconstructed_pose': reconstructed_pose
        }


def create_pose_canonicalization_model(
    input_joints: int = 17,
    encoder_type: str = 'mlp',
    rotation_type: str = '6d',
    hidden_dim: int = 512,
    encoder_output_dim: int = 256,
    dropout: float = 0.1,
    predict_mode: str = 'rotation_only',
    num_layers: int = None,
    hybrid_config: dict = None
) -> PoseCanonicalizationNet:
    """
    Factory function to create pose canonicalization model
    Args:
        input_joints: Number of input joints (default: 17 for COCO)
        encoder_type: 'mlp', 'gcn', or 'transformer'
        rotation_type: '6d', 'quaternion', or 'matrix'
        hidden_dim: Hidden dimension for encoder
        dropout: Dropout rate
        
    Returns:
        Configured PoseCanonicalizationNet model
    """
    return PoseCanonicalizationNet(
        input_joints=input_joints,
        encoder_type=encoder_type,
        rotation_type=rotation_type,
        hidden_dim=hidden_dim,
        encoder_output_dim=encoder_output_dim,
        dropout=dropout,
        predict_mode=predict_mode,
        num_layers=num_layers,
        hybrid_config=hybrid_config
    )