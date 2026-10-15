import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    return re.sub(r'^b[\'"]|[\'"]$', '', text).strip()

def generate_daily_embeddings(csv_path: str, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    headline_cols = [f'Top{i}' for i in range(1, 26)]
    
    daily_embeddings = {}
    print("Generating FinBERT embeddings...")
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            date = row['Date']
            daily_text = " ".join([clean_text(row[col]) for col in headline_cols])
            
            if not daily_text.strip():
                daily_embeddings[date] = torch.zeros(768)
                continue

            inputs = tokenizer(
                daily_text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)

            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0, :]
            daily_embeddings[date] = cls_embedding.cpu()

    torch.save(daily_embeddings, output_path)
    print(f"Successfully saved {len(daily_embeddings)} dense vectors to {output_path}.")

if __name__ == "__main__":
    CSV_FILE = "Combined_News_DJIA.csv"
    OUTPUT_TENSOR_FILE = "daily_nlp_embeddings.pt"
    generate_daily_embeddings(CSV_FILE, OUTPUT_TENSOR_FILE)