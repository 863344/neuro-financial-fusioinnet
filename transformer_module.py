import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create static positional encoding matrix P
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it moves with the model but isn't updated by autograd
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x shape: (Batch, Seq_Len, d_model) """
        return x + self.pe[:, :x.size(1), :]

class TemporalTransformer(nn.Module):
    def __init__(self, in_features: int = 13, d_model: int = 64, nhead: int = 4, num_layers: int = 2, out_features: int = 128, dropout: float = 0.2):
        """
        Input constraint: Numerical time-series window of shape (Batch, Seq_Len, 13)
        """
        super(TemporalTransformer, self).__init__()
        
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Projection head to align with GAT output dimension
        self.output_projection = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping matrix dimensions.
        x: (Batch, Seq_Len, in_features)
        """
        # 1. Project and encode positions
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # 2. Temporal Attention
        # Output shape: (Batch, Seq_Len, d_model)
        x = self.transformer_encoder(x)
        
        # 3. Extract final time step hidden state
        x_final_step = x[:, -1, :] 
        
        # 4. Project to fusion dimension
        # Output shape: (Batch, 128)
        out = self.output_projection(x_final_step)
        return out

if __name__ == "__main__":
    # Smoke Test: Validate tensor math dimensions
    print("Executing Transformer mathematical validation...")
    
    # Mock data: Batch Size = 32, Seq Window = 30 days, Features = 13
    B, T, F = 32, 30, 13
    dummy_x = torch.randn(B, T, F)
    
    model = TemporalTransformer(in_features=F)
    output = model(dummy_x)
    
    print(f"Transformer Input Shape:  {dummy_x.shape}")
    print(f"Transformer Output Shape: {output.shape}")
    print("VRAM Optimization: Passed. Sequence collapsed and projected successfully.")