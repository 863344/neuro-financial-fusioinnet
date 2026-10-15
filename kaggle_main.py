import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
from data_loader import get_dataloaders
from graph_module import build_fully_connected_edge_index
from train import NeuroFinancialFusionNet
from ablation_study import TemporalBaseline, SpatialBaseline

def train_and_evaluate(model: nn.Module, model_name: str, train_dl, test_dl, device: torch.device):
    print(f"\n{'='*50}\nTraining {model_name}...\n{'='*50}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') # AMP Gradient Scaler
    
    EPOCHS = 50
    PATIENCE = 5
    best_rmse = float('inf')
    patience_counter = 0
    best_weights = None
    
    base_edge = build_fully_connected_edge_index(5).to(device)
    
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        for x_seq, x_spatial, y in train_dl:
            x_seq, y = x_seq.to(device), y.to(device)
            bs = x_spatial.size(0)
            x_spatial_flat = x_spatial.view(bs * 5, -1).to(device)
            batched_edge = torch.cat([base_edge + i * 5 for i in range(bs)], dim=1)
            batch_idx = torch.arange(bs).repeat_interleave(5).to(device)
            
            optimizer.zero_grad()
            
            # AMP Forward Pass
            with torch.amp.autocast('cuda'):
                if model_name == "FusionNet":
                    preds = model(x_seq, x_spatial_flat, batched_edge, batch_idx)
                elif model_name == "Transformer-Only":
                    preds = model(x_seq)
                else: # GNN-Only
                    preds = model(x_spatial_flat, batched_edge, batch_idx)
                
                loss = criterion(preds, y)
            
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # --- EVALUATION ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_seq, x_spatial, y in test_dl:
                x_seq, y = x_seq.to(device), y.to(device)
                bs = x_spatial.size(0)
                x_spatial_flat = x_spatial.view(bs * 5, -1).to(device)
                batched_edge = torch.cat([base_edge + i * 5 for i in range(bs)], dim=1)
                batch_idx = torch.arange(bs).repeat_interleave(5).to(device)
                
                with torch.amp.autocast('cuda'):
                    if model_name == "FusionNet":
                        preds = model(x_seq, x_spatial_flat, batched_edge, batch_idx)
                    elif model_name == "Transformer-Only":
                        preds = model(x_seq)
                    else:
                        preds = model(x_spatial_flat, batched_edge, batch_idx)
                    
                    test_loss += criterion(preds, y).item()
                    
        val_rmse = math.sqrt(test_loss / len(test_dl))
        print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss/len(train_dl):.6f} | Test RMSE: {val_rmse:.6f}")
        
        # Early Stopping Logic
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered. Best Test RMSE: {best_rmse:.6f}")
                break
                
    return best_rmse

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("WARNING: Executing on CPU. Cloud migration highly recommended.")
        
    train_dl, test_dl = get_dataloaders(batch_size=64)
    
    models_to_test = {
        "Transformer-Only": TemporalBaseline().to(device),
        "GNN-Only": SpatialBaseline().to(device),
        "FusionNet": NeuroFinancialFusionNet().to(device)
    }
    
    results = {}
    for name, model in models_to_test.items():
        results[name] = train_and_evaluate(model, name, train_dl, test_dl, device)
        
    print(f"\n{'='*50}\nFINAL Q1 ABLATION RESULTS (TEST RMSE)\n{'='*50}")
    for name, rmse in results.items():
        print(f"{name+':':<20} {rmse:.6f}")