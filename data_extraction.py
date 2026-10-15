import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime

def fetch_and_align_data(start_date: str, end_date: str) -> pd.DataFrame:
    tickers = ['SPY', 'XLK', 'XLF', 'XLV', 'XLE']
    # Patched: Replaced deprecated DFEDTAR with DFF
    macro_series = {'DFF': 'Fed_Funds', 'CPIAUCSL': 'CPI', 'DGS10': '10Y_Yield'}
    
    raw_equity = yf.download(tickers, start=start_date, end=end_date)
    equity_data = raw_equity[['Close', 'Volume']].copy()
    
    if 'SPY' in equity_data.columns.get_level_values(0):
        equity_data.columns = [f"{t}_{m}" for t, m in equity_data.columns]
    else:
        equity_data.columns = [f"{t}_{m}" for m, t in equity_data.columns]
        
    macro_data = web.DataReader(list(macro_series.keys()), 'fred', start_date, end_date)
    macro_data.rename(columns=macro_series, inplace=True)
    
    merged_df = equity_data.join(macro_data, how='outer')
    merged_df.ffill(inplace=True)
    merged_df = merged_df.loc[equity_data.index] 
    merged_df.dropna(inplace=True)
    
    return merged_df

if __name__ == "__main__":
    START = "2010-01-01"
    END = datetime.date.today().strftime("%Y-%m-%d")
    
    final_dataset = fetch_and_align_data(START, END)
    final_dataset.to_parquet("macro_graph_data.parquet")
    print(f"Extraction complete. Shape: {final_dataset.shape}")