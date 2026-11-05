import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_summary_stats(df: pd.DataFrame) -> dict:
    return df.describe().round(3).to_dict()

def export_processed_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
