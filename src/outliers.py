import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return pd.Series(False, index=df.index)
    z_scores = np.abs((df[column] - mean_val) / std_val)
    return z_scores > threshold

def cap_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    data = df.copy()
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    data[column] = np.clip(data[column], lower_bound, upper_bound)
    return data
