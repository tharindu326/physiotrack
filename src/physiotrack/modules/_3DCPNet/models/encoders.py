import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for 3D pose features
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1, num_layers=4):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            hidden_dim = hidden_dim // 2  # Gradually reduce dimension
        
        # Final layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights like 6DRepNet"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) flattened 3D pose coordinates
        Returns:
            features: (batch, output_dim)
        """
        return self.encoder(x)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for 3D poses
    Vectorized implementation for efficient batch processing
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=256, 
                 num_joints=17, dropout=0.1, num_layers=3):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_layers = num_layers
        
        # Define skeleton connectivity (custom 17-joint format)
        self.register_buffer('edge_index', self._get_skeleton_edges())
        
        # Pre-compute adjacency matrix for efficient operations
        self.register_buffer('adj_matrix', self._compute_adjacency_matrix())
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            self.gcn_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _get_skeleton_edges(self):
        """Define custom 17-joint skeleton connections"""
        connections = [
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
        
        edge_index = torch.tensor(connections).t().contiguous()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def _compute_adjacency_matrix(self):
        """Pre-compute adjacency matrix for efficient batch operations"""
        # Create adjacency matrix
        adj = torch.zeros(self.num_joints, self.num_joints, dtype=torch.float32)
        row, col = self.edge_index
        
        # Set connections to 1
        adj[row, col] = 1.0
        
        # Add self-loops
        adj += torch.eye(self.num_joints)
        
        # Normalize by degree (symmetric normalization)
        degree = adj.sum(dim=1, keepdim=True)
        degree_sqrt = torch.sqrt(degree + 1e-8)  # Add small epsilon to avoid division by zero
        adj_normalized = adj / (degree_sqrt * degree_sqrt.t())
        
        return adj_normalized
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _gcn_forward_vectorized(self, x, adj_matrix, layer):
        """
        Vectorized GCN forward pass using matrix multiplication
        Args:
            x: (batch, joints, features) input features
            adj_matrix: (joints, joints) normalized adjacency matrix
            layer: Linear layer to apply
        Returns:
            (batch, joints, features) output features
        """
        # Graph convolution: H = σ(D^(-1/2) * A * D^(-1/2) * X * W)
        # Since we pre-computed D^(-1/2) * A * D^(-1/2), we just need: H = σ(A * X * W)
        
        # Apply adjacency matrix: (batch, joints, features) = (joints, joints) @ (batch, joints, features)
        # Use torch.matmul for efficient batch matrix multiplication
        x_aggregated = torch.matmul(adj_matrix, x)  # (batch, joints, features)
        
        # Apply linear transformation
        x_transformed = layer(x_aggregated)  # (batch, joints, hidden_dim)
        
        return x_transformed
    
    def forward(self, x):
        """
        Vectorized forward pass for entire batch
        Args:
            x: (batch, joints, 3) 3D pose coordinates
        Returns:
            features: (batch, output_dim)
        """
        batch_size, num_joints, input_dim = x.shape
        
        # Process entire batch at once using vectorized operations
        current_x = x  # (batch, joints, input_dim)
        
        # Apply GCN layers with vectorized operations
        for i, layer in enumerate(self.gcn_layers):
            current_x = self._gcn_forward_vectorized(current_x, self.adj_matrix, layer)
            
            if i < len(self.gcn_layers) - 1:  # No activation on last layer
                current_x = F.relu(current_x)
                current_x = self.dropout(current_x)
        
        # Global pooling: flatten and apply MLP
        x = current_x.reshape(batch_size, -1)  # (batch, joints * hidden_dim)
        x = self.global_pool(x)  # (batch, output_dim)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for 3D pose sequences
    """
    def __init__(self, input_dim=3, hidden_dim=256, num_heads=8, 
                 num_layers=4, num_joints=17, dropout=0.1, output_dim=256):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        # Joint embedding
        self.joint_embed = nn.Linear(input_dim, hidden_dim)
        # Positional encoding for joints (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()
    
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
            features: (batch, output_dim)
        """
        batch_size, num_joints, _ = x.shape
        
        # Joint embedding
        x = self.joint_embed(x)  # (batch, joints, hidden_dim)
        # Add positional encoding
        x = x + self.pos_embed
        # Transformer encoding
        x = self.transformer(x)  # (batch, joints, hidden_dim)
        # Global pooling
        x = x.view(batch_size, -1)
        x = self.global_pool(x)
        return x