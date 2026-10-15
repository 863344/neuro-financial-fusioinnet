import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class FusionNetDataset(Dataset):
    def __init__(self, parquet_path: str, pt_path: str, seq_len: int = 30, horizon: int = 7):
        # 1. Load Data
        df = pd.read_parquet(parquet_path)
        nlp_dict = torch.load(pt_path)
        
        # 2. Strict Temporal Alignment (Inner Join)
        valid_dates = sorted(list(set(df.index.strftime('%Y-%m-%d')).intersection(nlp_dict.keys())))
        df = df[df.index.strftime('%Y-%m-%d').isin(valid_dates)]
        self.nlp_tensors = {date: nlp_dict[date] for date in valid_dates}
        self.dates = valid_dates
        
        # 3. Scaling
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.num_data = self.scaler.fit_transform(df.values)
        
        # Feature Index Maps based on data_extraction.py output
        self.tickers = ['SPY', 'XLK', 'XLF', 'XLV', 'XLE']
        # Patched: Replaced 'DFF' with 'Fed_Funds' to match extracted schema
        self.macro_idx = [df.columns.get_loc(c) for c in ['Fed_Funds', 'CPI', '10Y_Yield']]
        self.node_idx = {t: [df.columns.get_loc(f"{t}_Close"), df.columns.get_loc(f"{t}_Volume")] for t in self.tickers}
        self.spy_close_idx = df.columns.get_loc("SPY_Close")
        
        self.seq_len = seq_len
        self.horizon = horizon
        self.raw_spy = df['SPY_Close'].values # Unscaled for volatility target

    def __len__(self):
        return len(self.num_data) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        # Time indices
        t_start = idx
        t_end = idx + self.seq_len
        t_target_end = t_end + self.horizon
        current_date = self.dates[t_end - 1]

        # 1. Temporal Tensor (30, 13)
        x_seq = torch.tensor(self.num_data[t_start:t_end], dtype=torch.float32)

        # 2. Spatial Tensor at t_end - 1 (5, 773)
        macro_features = torch.tensor(self.num_data[t_end - 1, self.macro_idx], dtype=torch.float32)
        nlp_feature = self.nlp_tensors[current_date]
        
        node_features = []
        for ticker in self.tickers:
            price_features = torch.tensor(self.num_data[t_end - 1, self.node_idx[ticker]], dtype=torch.float32)
            # Concatenate: [2] + [3] + [768] = [773]
            f_concat = torch.cat([price_features, macro_features, nlp_feature])
            node_features.append(f_concat)
            
        x_spatial = torch.stack(node_features)

        # 3. Target Volatility (Standard Deviation of log returns over horizon)
        future_prices = self.raw_spy[t_end:t_target_end]
        log_returns = np.diff(np.log(future_prices))
        target_vol = np.std(log_returns) if len(log_returns) > 1 else 0.0
        y = torch.tensor([target_vol], dtype=torch.float32)

        return x_seq, x_spatial, y

def get_dataloaders(batch_size: int = 32):
    dataset = FusionNetDataset("macro_graph_data.parquet", "daily_nlp_embeddings.pt")
    
    # 80/20 Chronological Split (No random shuffling to prevent data leakage)
    split_idx = int(len(dataset) * 0.8)
    
    train_ds = torch.utils.data.Subset(dataset, range(0, split_idx))
    test_ds = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Initializing DataLoader mapping...")
    train_dl, test_dl = get_dataloaders(batch_size=32)
    
    x_seq, x_spatial, y = next(iter(train_dl))
    print(f"Batched Sequence Shape: {x_seq.shape}")
    print(f"Batched Spatial Shape:  {x_spatial.shape}")
    print(f"Batched Target Shape:   {y.shape}")