import pandas as pd
import numpy as np

def min_max_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for col in columns:
        min_val = data[col].min()
        max_val = data[col].max()
        if max_val > min_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0.0
    return data

def z_score_standardize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for col in columns:
        mean_val = data[col].mean()
        std_val = data[col].std()
        if std_val > 0:
            data[col] = (data[col] - mean_val) / std_val
        else:
            data[col] = 0.0
    return data
