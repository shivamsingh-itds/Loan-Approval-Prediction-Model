import pandas as pd
from pathlib import Path

def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "loan_data.csv"

    df = pd.read_csv(data_path)
    return df
