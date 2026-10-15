# Neuro Financial FusionNet

Financial volatility forecasting project combining a temporal Transformer, a graph attention network, and FinBERT news embeddings.

## Project files

- `data_extraction.py`: fetches equity and macroeconomic data.
- `nlp_embedder.py`: generates daily FinBERT embeddings from `Combined_News_DJIA.csv`.
- `data_loader.py`: aligns numerical data and news embeddings.
- `transformer_module.py`, `graph_module.py`, and `train.py`: model components.
- `ablation_study.py` and `kaggle_main.py`: baseline comparisons and training entry point.

The CSV and Parquet input files are included. The large `fusionnet_v8.zip` Kaggle upload bundle is kept locally and excluded from Git, together with Python caches. Generate `daily_nlp_embeddings.pt` with `python nlp_embedder.py` before loading the dataset.

Python dependencies used by the source include torch, torch-geometric, transformers, pandas, numpy, scikit-learn, yfinance, pandas-datareader, tqdm, and a Parquet engine such as pyarrow.

## Current status

This repository preserves the current project snapshot; training results have not been verified. `kaggle_main.py` currently imports `build_fully_connected_edge_index`, while `graph_module.py` provides `build_dynamic_sparse_edge_index`; this mismatch must be resolved before running that entry point.
