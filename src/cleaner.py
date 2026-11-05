import pandas as pd
import numpy as np

def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def handle_missing_values(df: pd.DataFrame, strategy: str = "mean", fill_value = None) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns

    if strategy == "drop":
        return data.dropna().reset_index(drop=True)

    if strategy == "mean":
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == "median":
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == "constant" and fill_value is not None:
        data = data.fillna(fill_value)

    for col in categorical_cols:
        mode_val = data[col].mode()
        if not mode_val.empty:
            data[col] = data[col].fillna(mode_val[0])

    return data
