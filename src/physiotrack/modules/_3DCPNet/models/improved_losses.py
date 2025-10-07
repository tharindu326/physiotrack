"""
Improved rotation losses with multiple formulations for better training stability and accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedGeodesicLoss(nn.Module):
    """
    Improved geodesic loss with multiple formulations and better numerical stability
    """
    def __init__(self, loss_type='geodesic', eps=1e-7, reduction='mean', use_weighted=False):
        """
        Args:
            loss_type: 'geodesic', 'chordal', 'quaternion', or 'combined'
            eps: Small value for numerical stability
            reduction: 'mean', 'sum', or 'none'
            use_weighted: Weight loss by rotation magnitude
        """
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.reduction = reduction
        self.use_weighted = use_weighted
    
    def geodesic_distance(self, R1, R2):
        """
        Compute geodesic distance between rotation matrices (most accurate)
        This is the arc length on SO(3) manifold
        """
        # Relative rotation: R_rel = R1^T @ R2
        R_rel = torch.bmm(R1.transpose(1, 2), R2)
        
        # Trace of relative rotation
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        
        # Improved numerical stability
        # cos(theta) = (trace - 1) / 2
        cos_theta = (trace - 1.0) / 2.0
        
        # More robust clamping
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        
        # Geodesic distance in radians
        theta = torch.acos(cos_theta)
        
        return theta
    
    def chordal_distance(self, R1, R2):
        """
        Compute chordal distance (Frobenius norm of difference)
        Faster but less accurate for large rotations
        """
        # ||R1 - R2||_F
        diff = R1 - R2
        
        # Frobenius norm
        distance = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=1)
        
        # Normalize by sqrt(8) to have similar scale as geodesic
        # Max chordal distance between rotations is 2*sqrt(2)
        distance = distance / (2.0 * np.sqrt(2.0))
        
        return distance
    
    def quaternion_distance(self, R1, R2):
        """
        Convert to quaternions and compute distance
        More stable for certain rotation ranges
        """
        q1 = self.matrix_to_quaternion(R1)
        q2 = self.matrix_to_quaternion(R2)
        
        # Quaternion dot product
        dot = torch.sum(q1 * q2, dim=1)
        
        # Handle double cover (q and -q represent same rotation)
        dot = torch.abs(dot)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        
        # Angular distance
        theta = 2.0 * torch.acos(dot)
        
        return theta
    
    def matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion
        Using Shepperd's method for numerical stability
        """
        batch = R.shape[0]
        
        # Allocate quaternion
        q = torch.zeros(batch, 4, device=R.device, dtype=R.dtype)
        
        # Trace
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Case 1: trace > 0 (most common)
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
        
        # Case 2: R[0,0] is max diagonal
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
        
        # Case 3: R[1,1] is max diagonal
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        
        # Case 4: R[2,2] is max diagonal
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
        
        # Normalize quaternion
        q = F.normalize(q, p=2, dim=1)
        
        return q
    
    def combined_distance(self, R1, R2):
        """
        Combine multiple distance metrics for robustness
        """
        geo_dist = self.geodesic_distance(R1, R2)
        chord_dist = self.chordal_distance(R1, R2)
        
        # Use geodesic for small rotations, chordal for numerical stability
        # Smooth transition based on rotation magnitude
        alpha = torch.sigmoid(10.0 * (geo_dist - np.pi/2))
        
        combined = (1 - alpha) * geo_dist + alpha * chord_dist
        
        return combined
    
    def forward(self, R1, R2, weights=None):
        """
        Compute rotation loss between two sets of rotation matrices
        Args:
            R1, R2: (batch, 3, 3) rotation matrices
            weights: (batch,) optional per-sample weights
        Returns:
            loss value
        """
        # Select distance metric
        if self.loss_type == 'geodesic':
            distances = self.geodesic_distance(R1, R2)
        elif self.loss_type == 'chordal':
            distances = self.chordal_distance(R1, R2)
        elif self.loss_type == 'quaternion':
            distances = self.quaternion_distance(R1, R2)
        elif self.loss_type == 'combined':
            distances = self.combined_distance(R1, R2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Convert to degrees for interpretability (optional)
        distances_deg = distances * 180.0 / np.pi
        
        # Apply sample weights if provided
        if weights is not None:
            distances = distances * weights
        
        # Apply weighting based on rotation magnitude
        if self.use_weighted:
            # Larger rotations get higher weight
            rotation_weights = 1.0 + torch.tanh(distances / (np.pi/4))
            distances = distances * rotation_weights
        
        # Reduction
        if self.reduction == 'mean':
            return torch.mean(distances), torch.mean(distances_deg)
        elif self.reduction == 'sum':
            return torch.sum(distances), torch.sum(distances_deg)
        else:
            return distances, distances_deg


class RotationConsistencyLoss(nn.Module):
    """
    Additional consistency checks for rotation predictions
    """
    def __init__(self):
        super().__init__()
    
    def orthogonality_loss(self, R):
        """
        Ensure R is orthogonal: R @ R^T = I
        """
        batch_size = R.shape[0]
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        RRT = torch.bmm(R, R.transpose(1, 2))
        diff = RRT - I
        
        # Frobenius norm
        loss = torch.mean(torch.norm(diff.view(batch_size, -1), p=2, dim=1))
        
        return loss
    
    def determinant_loss(self, R):
        """
        Ensure det(R) = 1 (proper rotation, not reflection)
        """
        det = torch.det(R)
        loss = torch.mean((det - 1.0) ** 2)
        
        return loss
    
    def symmetry_loss(self, R):
        """
        For certain applications, enforce symmetry constraints
        """
        # Example: penalize large rotations around certain axes
        # Extract rotation angles from matrix
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # Penalize large rotations
        loss = torch.mean(F.relu(angle - np.pi/2))  # Penalize rotations > 90 degrees
        
        return loss
    
    def forward(self, R, ortho_weight=1.0, det_weight=1.0, sym_weight=0.0):
        """
        Combined consistency losses
        """
        total_loss = 0.0
        
        if ortho_weight > 0:
            total_loss += ortho_weight * self.orthogonality_loss(R)
        
        if det_weight > 0:
            total_loss += det_weight * self.determinant_loss(R)
        
        if sym_weight > 0:
            total_loss += sym_weight * self.symmetry_loss(R)
        
        return total_loss


class AdaptiveRotationLoss(nn.Module):
    """
    Adaptive loss that switches between different formulations based on rotation magnitude
    """
    def __init__(self):
        super().__init__()
        self.geodesic_loss = ImprovedGeodesicLoss(loss_type='geodesic')
        self.chordal_loss = ImprovedGeodesicLoss(loss_type='chordal')
        self.consistency_loss = RotationConsistencyLoss()
    
    def forward(self, R_pred, R_gt, consistency_weight=0.1):
        """
        Adaptive combination of losses
        """
        # Compute both losses
        geo_loss, geo_deg = self.geodesic_loss(R_pred, R_gt)
        chord_loss, _ = self.chordal_loss(R_pred, R_gt)
        
        # Adaptive weighting based on average rotation magnitude
        avg_rotation = geo_deg.detach()
        
        if avg_rotation < 30:  # Small rotations
            # Geodesic is more accurate for small rotations
            main_loss = geo_loss
        elif avg_rotation > 120:  # Large rotations
            # Chordal is more stable for large rotations
            main_loss = chord_loss
        else:  # Medium rotations
            # Blend both
            alpha = (avg_rotation - 30) / 90
            main_loss = (1 - alpha) * geo_loss + alpha * chord_loss
        
        # Add consistency regularization
        if consistency_weight > 0:
            cons_loss = self.consistency_loss(R_pred, ortho_weight=1.0, det_weight=0.1)
            total_loss = main_loss + consistency_weight * cons_loss
        else:
            total_loss = main_loss
        
        return total_loss, geo_deg


def test_improved_losses():
    """Test the improved rotation losses"""
    
    print("Testing Improved Rotation Losses")
    print("=" * 60)
    
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample rotation matrices
    # Random rotations
    from utils.rotation_utils import euler_to_rotation_matrix
    
    angles1 = torch.randn(batch_size, 3) * 0.5  # Small rotations
    angles2 = angles1 + torch.randn(batch_size, 3) * 0.1  # Slightly different
    
    R1 = euler_to_rotation_matrix(angles1)
    R2 = euler_to_rotation_matrix(angles2)
    
    R1 = R1.to(device)
    R2 = R2.to(device)
    
    # Test different loss formulations
    print("\n1. Testing Different Loss Formulations:")
    print("-" * 40)
    
    losses = {
        'geodesic': ImprovedGeodesicLoss(loss_type='geodesic'),
        'chordal': ImprovedGeodesicLoss(loss_type='chordal'),
        'quaternion': ImprovedGeodesicLoss(loss_type='quaternion'),
        'combined': ImprovedGeodesicLoss(loss_type='combined'),
    }
    
    for name, loss_fn in losses.items():
        loss_val, loss_deg = loss_fn(R1, R2)
        print(f"{name:12s}: {loss_val.item():.6f} rad, {loss_deg.item():.2f} deg")
    
    # Test consistency loss
    print("\n2. Testing Consistency Loss:")
    print("-" * 40)
    
    consistency = RotationConsistencyLoss()
    ortho_loss = consistency.orthogonality_loss(R1)
    det_loss = consistency.determinant_loss(R1)
    
    print(f"Orthogonality loss: {ortho_loss.item():.6f}")
    print(f"Determinant loss:   {det_loss.item():.6f}")
    
    # Test adaptive loss
    print("\n3. Testing Adaptive Loss:")
    print("-" * 40)
    
    adaptive = AdaptiveRotationLoss()
    total_loss, rotation_deg = adaptive(R1, R2)
    
    print(f"Adaptive total loss: {total_loss.item():.6f}")
    print(f"Average rotation:    {rotation_deg.item():.2f} degrees")
    
    # Test with large rotations
    print("\n4. Testing with Large Rotations:")
    print("-" * 40)
    
    angles_large = torch.randn(batch_size, 3) * 2.0  # Large rotations
    R_large = euler_to_rotation_matrix(angles_large).to(device)
    
    for name, loss_fn in losses.items():
        loss_val, loss_deg = loss_fn(R1, R_large)
        print(f"{name:12s}: {loss_val.item():.6f} rad, {loss_deg.item():.2f} deg")
    
    print("\n" + "=" * 60)
    print("Summary of Improvements:")
    print("1. Multiple loss formulations for different rotation ranges")
    print("2. Better numerical stability with improved clamping")
    print("3. Adaptive loss selection based on rotation magnitude")
    print("4. Consistency checks for valid rotations")
    print("5. Option to weight by rotation magnitude")


if __name__ == "__main__":
    test_improved_losses()