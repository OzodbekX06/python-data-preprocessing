import os
import json
import numpy as np
import pandas as pd
from src.cleaner import load_csv, remove_duplicates, handle_missing_values
from src.outliers import detect_outliers_iqr, cap_outliers_iqr
from src.scaler import min_max_normalize, z_score_standardize
from src.visualizer import generate_summary_stats, export_processed_data

def run_pipeline():
    np.random.seed(42)
    sample_size = 200

    age = np.random.randint(18, 65, sample_size).astype(float)
    income = np.random.normal(50000, 15000, sample_size)
    spending = np.random.uniform(100, 2000, sample_size)

    age[10] = np.nan
    age[45] = np.nan
    income[22] = np.nan
    income[100] = 350000.0
    spending[150] = 12000.0

    raw_df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "Spending": spending,
        "Segment": np.random.choice(["Retail", "Corporate", "Small Business"], sample_size)
    })

    raw_df = pd.concat([raw_df, raw_df.iloc[[5, 12, 30]]], ignore_index=True)

    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    raw_path = os.path.join(raw_dir, "sample_raw_data.csv")
    raw_df.to_csv(raw_path, index=False)

    df_loaded = load_csv(raw_path)
    initial_rows = len(df_loaded)

    df_deduped = remove_duplicates(df_loaded)
    deduped_rows = len(df_deduped)

    df_imputed = handle_missing_values(df_deduped, strategy="median")

    income_outliers = detect_outliers_iqr(df_imputed, "Income")
    spending_outliers = detect_outliers_iqr(df_imputed, "Spending")

    df_capped = cap_outliers_iqr(df_imputed, "Income")
    df_capped = cap_outliers_iqr(df_capped, "Spending")

    numeric_cols = ["Age", "Income", "Spending"]
    df_standardized = z_score_standardize(df_capped, numeric_cols)
    df_normalized = min_max_normalize(df_capped, numeric_cols)

    summary_stats = generate_summary_stats(df_capped)

    proc_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    processed_path = os.path.join(proc_dir, "cleaned_features.csv")
    export_processed_data(df_standardized, processed_path)

    print("Data Preprocessing Pipeline Executed:")
    print("-" * 55)
    print(f"Initial Rows:             {initial_rows}")
    print(f"Post Deduplication Rows:  {deduped_rows} (Removed: {initial_rows - deduped_rows})")
    print(f"Missing Values Imputed:   {int(df_loaded.isnull().sum().sum())}")
    print(f"Income Outliers Capped:   {int(income_outliers.sum())}")
    print(f"Spending Outliers Capped: {int(spending_outliers.sum())}")
    print(f"Exported Processed Data:  {processed_path}")
    print("-" * 55)

if __name__ == "__main__":
    run_pipeline()
