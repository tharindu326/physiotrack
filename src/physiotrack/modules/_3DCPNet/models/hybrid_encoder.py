import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HybridGCNTransformerEncoder(nn.Module):
    """
    Hybrid encoder combining Graph Convolutional Networks for local spatial relationships
    with Transformer attention for global context modeling.
    
    Architecture:
    1. GCN layers process local joint relationships
    2. Transformer layers capture global dependencies
    3. Cross-attention fusion combines both representations
    4. Adaptive weighting learns optimal feature combination
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        gcn_hidden_dim: int = 128,
        transformer_hidden_dim: int = 256,
        output_dim: int = 384,
        num_joints: int = 17,
        num_gcn_layers: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_learnable_graph: bool = True,
        use_multi_scale: bool = True
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.use_learnable_graph = use_learnable_graph
        self.use_multi_scale = use_multi_scale
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # ============= GCN Branch =============
        # Skeleton connectivity
        self.register_buffer('base_adj_matrix', self._compute_base_adjacency())
        
        # Learnable graph topology (if enabled)
        if use_learnable_graph:
            self.graph_weight = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
            self.graph_alpha = nn.Parameter(torch.tensor(0.5))  # Balance between fixed and learned
        
        # GCN layers with residual connections
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_gcn_layers):
            self.gcn_layers.append(nn.Linear(current_dim, gcn_hidden_dim))
            self.gcn_norms.append(nn.LayerNorm(gcn_hidden_dim))
            current_dim = gcn_hidden_dim
        
        # Multi-scale graph pooling (if enabled)
        if use_multi_scale:
            self.body_part_pool = self._create_body_part_pooling()
            self.global_pool_gcn = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        
        # ============= Transformer Branch =============
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, transformer_hidden_dim) * 0.02)
        
        # Joint type embeddings (different joints have different semantic meanings)
        self.joint_type_embed = nn.Embedding(num_joints, transformer_hidden_dim)
        
        # Transformer encoder with pre-norm
        self.transformer_proj = nn.Linear(hidden_dim, transformer_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=num_heads,
            dim_feedforward=transformer_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # ============= Cross-Attention Fusion =============
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Feature projection for fusion
        self.gcn_proj_to_hidden = nn.Linear(gcn_hidden_dim, hidden_dim)
        self.transformer_proj_to_hidden = nn.Linear(transformer_hidden_dim, hidden_dim)
        
        # ============= Adaptive Feature Fusion =============
        # Learn task-specific weights for combining features
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # ============= Output Processing =============
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_joints, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _compute_base_adjacency(self):
        """Compute base skeleton adjacency matrix"""
        # Define skeleton connections (17-joint format)
        connections = [
            # Spine and head
            (0, 7), (7, 8), (8, 9), (9, 10),
            # Left leg
            (0, 1), (1, 2), (2, 3),
            # Right leg
            (0, 4), (4, 5), (5, 6),
            # Right arm
            (8, 11), (11, 12), (12, 13),
            # Left arm
            (8, 14), (14, 15), (15, 16)
        ]
        
        adj = torch.zeros(self.num_joints, self.num_joints)
        for i, j in connections:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        
        # Add self-loops
        adj += torch.eye(self.num_joints)
        
        # Normalize
        degree = adj.sum(dim=1, keepdim=True)
        degree_sqrt_inv = 1.0 / torch.sqrt(degree + 1e-8)
        adj_normalized = adj * degree_sqrt_inv * degree_sqrt_inv.t()
        
        return adj_normalized
    
    def _create_body_part_pooling(self):
        """Create pooling indices for body parts"""
        # Define body part groups
        body_parts = {
            'head': [9, 10],
            'torso': [0, 7, 8],
            'left_leg': [1, 2, 3],
            'right_leg': [4, 5, 6],
            'left_arm': [14, 15, 16],
            'right_arm': [11, 12, 13]
        }
        
        # Create pooling matrix
        num_parts = len(body_parts)
        pool_matrix = torch.zeros(num_parts, self.num_joints)
        
        for i, (part, joints) in enumerate(body_parts.items()):
            for j in joints:
                pool_matrix[i, j] = 1.0 / len(joints)
        
        return nn.Parameter(pool_matrix, requires_grad=False)
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_adaptive_adjacency(self):
        """Compute adaptive adjacency matrix combining fixed and learned components"""
        if self.use_learnable_graph:
            # Make learned component symmetric
            learned_adj = (self.graph_weight + self.graph_weight.t()) / 2
            # Apply sigmoid to ensure positive weights
            learned_adj = torch.sigmoid(learned_adj)
            # Add self-loops
            learned_adj = learned_adj + torch.eye(self.num_joints, device=learned_adj.device)
            # Normalize
            degree = learned_adj.sum(dim=1, keepdim=True)
            degree_sqrt_inv = 1.0 / torch.sqrt(degree + 1e-8)
            learned_adj_norm = learned_adj * degree_sqrt_inv * degree_sqrt_inv.t()
            
            # Combine fixed and learned with learnable weighting
            alpha = torch.sigmoid(self.graph_alpha)
            adj_matrix = alpha * self.base_adj_matrix + (1 - alpha) * learned_adj_norm
        else:
            adj_matrix = self.base_adj_matrix
        
        return adj_matrix
    
    def gcn_forward(self, x, adj_matrix):
        """Forward pass through GCN layers with residual connections"""
        batch_size = x.shape[0]
        
        for i, (layer, norm) in enumerate(zip(self.gcn_layers, self.gcn_norms)):
            # Graph convolution: aggregate neighbor features
            x_agg = torch.matmul(adj_matrix, x)
            # Apply linear transformation
            x_new = layer(x_agg)
            # Layer norm and activation
            x_new = norm(x_new)
            
            # Residual connection (if dimensions match)
            if x.shape[-1] == x_new.shape[-1]:
                x = x + F.relu(x_new)
            else:
                x = F.relu(x_new)
        
        # Multi-scale aggregation
        if self.use_multi_scale:
            # Body part level features
            x_parts = torch.matmul(self.body_part_pool.unsqueeze(0), x)  # (batch, num_parts, hidden)
            # Global features
            x_global = x.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
            x_global = self.global_pool_gcn(x_global)
            
            # Broadcast back and combine
            x_parts_broadcast = torch.matmul(self.body_part_pool.t().unsqueeze(0), x_parts)
            x_global_broadcast = x_global.expand(-1, self.num_joints, -1)
            
            # Weighted combination
            x = x + 0.3 * x_parts_broadcast + 0.2 * x_global_broadcast
        
        return x
    
    def transformer_forward(self, x):
        """Forward pass through Transformer with positional and type embeddings"""
        batch_size = x.shape[0]
        
        # Project to transformer dimension
        x = self.transformer_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Add joint type embeddings
        joint_indices = torch.arange(self.num_joints, device=x.device)
        joint_type_emb = self.joint_type_embed(joint_indices).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + joint_type_emb
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x
    
    def cross_attention_fusion(self, gcn_features, transformer_features):
        """Fuse GCN and Transformer features using cross-attention"""
        # Project to common dimension
        gcn_hidden = self.gcn_proj_to_hidden(gcn_features)
        trans_hidden = self.transformer_proj_to_hidden(transformer_features)
        
        # Cross-attention: GCN queries attend to Transformer keys/values
        attended_features, _ = self.cross_attention(
            query=gcn_hidden,
            key=trans_hidden,
            value=trans_hidden
        )
        
        # Residual connection and normalization
        fused_features = self.cross_norm(gcn_hidden + attended_features)
        
        return fused_features, gcn_hidden, trans_hidden
    
    def adaptive_fusion(self, fused_features, gcn_features, trans_features):
        """Adaptively combine features based on learned task-specific weights"""
        batch_size = fused_features.shape[0]
        
        # Compute global features for gating
        fused_global = fused_features.mean(dim=1)  # (batch, hidden)
        gcn_global = gcn_features.mean(dim=1)
        trans_global = trans_features.mean(dim=1)
        
        # Concatenate for gating decision
        gate_input = torch.cat([fused_global, gcn_global], dim=-1)
        
        # Get fusion weights
        fusion_weights = self.fusion_gate(gate_input)  # (batch, 2)
        
        # Apply weights
        gcn_weight = fusion_weights[:, 0:1].unsqueeze(1)  # (batch, 1, 1)
        trans_weight = fusion_weights[:, 1:2].unsqueeze(1)  # (batch, 1, 1)
        
        # Weighted combination
        final_features = (fused_features + 
                         gcn_weight * gcn_features + 
                         trans_weight * trans_features)
        
        return final_features, fusion_weights
    
    def forward(self, x):
        """
        Forward pass through hybrid encoder
        Args:
            x: (batch, joints, 3) 3D pose coordinates
        Returns:
            features: (batch, output_dim) encoded features
            aux_outputs: dict with intermediate outputs for analysis
        """
        batch_size, num_joints, input_dim = x.shape
        
        # Initial projection
        x = self.input_proj(x)  # (batch, joints, hidden_dim)
        
        # Get adaptive adjacency matrix
        adj_matrix = self.get_adaptive_adjacency()
        
        # GCN branch
        gcn_features = self.gcn_forward(x, adj_matrix)  # (batch, joints, gcn_hidden)
        
        # Transformer branch
        transformer_features = self.transformer_forward(x)  # (batch, joints, trans_hidden)
        
        # Cross-attention fusion
        fused_features, gcn_hidden, trans_hidden = self.cross_attention_fusion(
            gcn_features, transformer_features
        )
        
        # Adaptive fusion
        final_features, fusion_weights = self.adaptive_fusion(
            fused_features, gcn_hidden, trans_hidden
        )
        
        # Flatten and project to output
        features_flat = final_features.reshape(batch_size, -1)
        output = self.output_proj(features_flat)
        
        # Store auxiliary outputs for analysis
        aux_outputs = {
            'gcn_features': gcn_features,
            'transformer_features': transformer_features,
            'fusion_weights': fusion_weights,
            'adjacency_matrix': adj_matrix if self.use_learnable_graph else None
        }
        
        return output, aux_outputs


class SimplifiedHybridEncoder(nn.Module):
    """
    A simplified version of the hybrid encoder for faster training and inference
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_joints: int = 17,
        num_gcn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_joints = num_joints
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Fixed adjacency matrix
        self.register_buffer('adj_matrix', self._compute_adjacency())
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Transformer
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _compute_adjacency(self):
        """Compute skeleton adjacency matrix"""
        connections = [
            (0, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
            (8, 11), (11, 12), (12, 13),
            (8, 14), (14, 15), (15, 16)
        ]
        
        adj = torch.zeros(self.num_joints, self.num_joints)
        for i, j in connections:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj += torch.eye(self.num_joints)
        
        degree = adj.sum(dim=1, keepdim=True)
        adj_normalized = adj / torch.sqrt(degree * degree.t() + 1e-8)
        
        return adj_normalized
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, joints, 3) 3D pose coordinates
        Returns:
            features: (batch, output_dim) encoded features
        """
        batch_size = x.shape[0]
        
        # Input projection
        x = self.input_proj(x)
        
        # GCN branch
        gcn_out = x
        for layer, norm in zip(self.gcn_layers, self.gcn_norms):
            gcn_out = torch.matmul(self.adj_matrix, gcn_out)
            gcn_out = layer(gcn_out)
            gcn_out = norm(gcn_out)
            gcn_out = F.relu(gcn_out)
        
        # Transformer branch
        trans_out = x + self.pos_embed
        trans_out = self.transformer(trans_out)
        
        # Fusion
        fused = torch.cat([gcn_out, trans_out], dim=-1)
        fused = self.fusion(fused)
        
        # Output
        features = fused.reshape(batch_size, -1)
        output = self.output_proj(features)
        
        return output