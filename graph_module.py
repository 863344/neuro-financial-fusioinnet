import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpatialFusionGAT(nn.Module):
    def __init__(self, in_channels=773, hidden_channels=64, out_channels=64, heads=4):
        super(SpatialFusionGAT, self).__init__()
        # PyG Graph Attention Layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        # We omit global pooling here to preserve node-level embeddings for the Cross-Attention Transformer
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

def build_dynamic_sparse_edge_index(node_features: torch.Tensor, num_nodes: int, threshold: float = 0.4) -> torch.Tensor:
    """
    Dynamically generates a sparse edge index based on the cosine similarity 
    of the node embeddings to prevent topological over-smoothing.
    """
    device = node_features.device
    batch_size = node_features.size(0) // num_nodes
    
    edge_indices = []
    # Reshape flattened features back to [Batch, Num_Nodes, Features]
    features_reshaped = node_features.view(batch_size, num_nodes, -1)
    
    for i in range(batch_size):
        # Compute Cosine Similarity matrix for the macroeconomic nodes
        feat_norm = F.normalize(features_reshaped[i], p=2, dim=1)
        sim_matrix = torch.mm(feat_norm, feat_norm.transpose(0, 1))
        
        # Isolate statistically significant connections (severing weak edges)
        edges = (sim_matrix > threshold).nonzero(as_tuple=False).t()
        
        # Shift node indices to align with PyG's batched graph format
        edges = edges + (i * num_nodes)
        edge_indices.append(edges)
        
    if not edge_indices:
        # Failsafe: Add self-loops if all edges are severed to prevent crashing
        batch_idx = torch.arange(batch_size * num_nodes, device=device)
        return torch.stack([batch_idx, batch_idx], dim=0)
        
    return torch.cat(edge_indices, dim=1)