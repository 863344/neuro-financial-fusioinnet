import torch
import torch.nn as nn
from transformer_module import TemporalTransformer
from graph_module import SpatialFusionGAT
from torch_geometric.nn import global_mean_pool

class TemporalBaseline(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super(TemporalBaseline, self).__init__()
        
        self.encoder = TemporalTransformer(
            in_features=input_dim, 
            d_model=hidden_dim, 
            nhead=4, 
            num_layers=2, 
            out_features=hidden_dim
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq):
        out = self.encoder(x_seq)
        return self.regressor(out).squeeze()

class SpatialBaseline(nn.Module):
    def __init__(self, in_channels=773, hidden_channels=64):
        super(SpatialBaseline, self).__init__()
        self.encoder = SpatialFusionGAT(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=hidden_channels, 
            heads=4
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_spatial, edge_index, batch_idx):
        # Encode to node-level embeddings
        node_embeddings = self.encoder(x_spatial, edge_index, batch_idx)
        
        # Collapse multi-node graph topology into single vector for isolated regression
        graph_embedding = global_mean_pool(node_embeddings, batch_idx)
        
        return self.regressor(graph_embedding).squeeze()