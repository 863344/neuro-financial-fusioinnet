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

The training entry point uses `build_dynamic_sparse_edge_index` for both training and evaluation. Loss calculation aligns prediction and target shapes, and mixed precision is enabled only on CUDA. Training results have not yet been verified by an end-to-end run.
