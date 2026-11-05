import unittest
import pandas as pd
import numpy as np
from src.cleaner import remove_duplicates, handle_missing_values
from src.outliers import detect_outliers_iqr, cap_outliers_iqr
from src.scaler import min_max_normalize, z_score_standardize

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "val": [10.0, 20.0, np.nan, 20.0, 1000.0],
            "category": ["A", "B", "A", "B", "A"]
        })

    def test_duplicates(self):
        duplicate_free = remove_duplicates(self.df)
        self.assertEqual(len(duplicate_free), 4)

    def test_missing_values(self):
        imputed = handle_missing_values(self.df, strategy="median")
        self.assertFalse(imputed["val"].isnull().any())
        self.assertEqual(imputed["val"].iloc[2], 20.0)

    def test_outlier_detection(self):
        outliers = detect_outliers_iqr(self.df.dropna(), "val")
        self.assertTrue(outliers.iloc[-1])

    def test_min_max_scaling(self):
        clean_df = pd.DataFrame({"num": [0.0, 50.0, 100.0]})
        scaled = min_max_normalize(clean_df, ["num"])
        self.assertEqual(scaled["num"].min(), 0.0)
        self.assertEqual(scaled["num"].max(), 1.0)

if __name__ == "__main__":
    unittest.main()
