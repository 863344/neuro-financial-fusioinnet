import torch
import torch.nn as nn
from transformer_module import TemporalTransformer
from graph_module import SpatialFusionGAT

class NeuroFinancialFusionNet(nn.Module):
    def __init__(self, temporal_features=13, spatial_features=773, hidden_dim=64):
        super(NeuroFinancialFusionNet, self).__init__()
        
        # FIXED: Explicit keyword arguments mapping to TemporalTransformer signature
        self.temporal_encoder = TemporalTransformer(
            in_features=temporal_features, 
            d_model=hidden_dim, 
            nhead=4, 
            num_layers=2, 
            out_features=hidden_dim  # Forces 64-dim output for MultiheadAttention math
        )
        
        self.spatial_encoder = SpatialFusionGAT(
            in_channels=spatial_features, 
            hidden_channels=hidden_dim, 
            out_channels=hidden_dim, 
            heads=4
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_spatial, edge_index, batch_idx):
        # 1. Temporal Stream (Query)
        temp_out = self.temporal_encoder(x_seq)
        if temp_out.dim() == 2:
            q = temp_out.unsqueeze(1) # Shape: [Batch, 1, Hidden]
        
        # 2. Spatial Stream (Keys & Values)
        spatial_nodes = self.spatial_encoder(x_spatial, edge_index, batch_idx) 
        
        # Reshape spatial nodes back to batched format for attention
        batch_size = q.size(0)
        num_nodes = spatial_nodes.size(0) // batch_size
        kv = spatial_nodes.view(batch_size, num_nodes, -1) # Shape: [Batch, Num_Nodes, Hidden]
        
        # 3. Cross-Attention Fusion
        attn_out, _ = self.cross_attention(q, kv, kv)
        
        # 4. Final Regression
        out = self.regressor(attn_out.squeeze(1))
        return out.squeeze()