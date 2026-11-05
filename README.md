# Python Data Preprocessing

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Overview
Python Data Preprocessing is a modular, production-ready data preparation library and pipeline designed to transform raw, noisy tabular data into clean, normalized feature matrices ready for machine learning algorithms. It provides standalone components for deduplication, statistical imputation, outlier bounding, and dual-mode feature scaling.

## Objectives
- Automate repetitive data cleaning tasks across tabular datasets.
- Provide multiple missing value imputation strategies (mean, median, mode, constant, drop).
- Detect and treat statistical anomalies via Interquartile Range (IQR) and Z-score metrics.
- Support both Min-Max normalization ($[0, 1]$) and Z-score standardization ($\mu=0, \sigma=1$).
- Generate clean, reproducible CSV exports for downstream training pipelines.

## Features
- **Deterministic Missing Value Imputation:** Flexible strategies accommodating skewed continuous data (median) and categorical data (mode).
- **Outlier Detection & Capping:** Flags extreme records without throwing away valuable training rows via Tukey's IQR bounding.
- **Deduplication Engine:** Identifies and purges redundant row records while preserving index integrity.
- **Vectorized Scaling Transforms:** Pure vectorized NumPy and Pandas implementations for scaling continuous vectors.
- **Statistical Profiling:** Automated summaries of central tendency and dispersion.

## Technologies Used
- **Python 3.10+**
- **Pandas:** Tabular schema management, missing value masking, and data transformations.
- **NumPy:** Mathematical calculations, array clipping, and vector operations.
- **Matplotlib:** Distribution chart generation.

## Project Structure
```text
python-data-preprocessing/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/
│   │   └── sample_raw_data.csv
│   └── processed/
│       └── cleaned_features.csv
├── src/
│   ├── __init__.py
│   ├── cleaner.py
│   ├── outliers.py
│   ├── scaler.py
│   └── visualizer.py
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py
└── main.py
```

## Dataset
The built-in sample dataset models consumer finance profiles with introduced noise (missing values, duplicated records, and synthetic outliers):
- `Age`: Customer age with missing values.
- `Income`: Annual earnings with severe high-end anomalies.
- `Spending`: Transaction spending with distribution outliers.
- `Segment`: Categorical group indicator.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/OzodbekX06/python-data-preprocessing.git
cd python-data-preprocessing

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage
Run the unit test suite:

```bash
python -m unittest discover tests
```

Execute the end-to-end preprocessing pipeline:

```bash
python main.py
```

Cleaned features are exported to `data/processed/cleaned_features.csv`.

## Methodology
1. **Deduplication:** Hash-based exact duplicate detection removing duplicated records.
2. **Missing Value Treatment:** Median imputation applied to continuous columns to prevent bias from skewed distributions:
   $$\tilde{x} = \operatorname{median}(\{x_i \mid x_i \neq \text{NaN}\})$$
3. **Outlier Mitigation:** IQR outlier detection calculates boundaries:
   $$\text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$
   Outliers exceeding bounds are smoothly capped via `np.clip` rather than discarded.
4. **Standardization:** Features are transformed according to:
   $$z = \frac{x - \mu}{\sigma}$$

## Results
The pipeline executed on the raw sample dataset with the following verified outcomes:

| Processing Stage | Before | After | Delta / Action |
| :--- | :--- | :--- | :--- |
| **Record Deduplication** | 203 records | 200 records | 3 duplicate rows eliminated |
| **Missing Imputation** | 3 missing entries | 0 missing entries | Imputed using column medians |
| **Income Outlier Capping** | 1 severe anomaly | 0 anomalies | Capped to $Q_3 + 1.5 \times \text{IQR}$ upper bound |
| **Spending Outlier Capping** | 1 severe anomaly | 0 anomalies | Capped to upper bound |
| **Output Standardization** | Raw units ($\$, \text{years}$) | $\mu = 0, \sigma = 1$ | Successfully standardized |

## Screenshots
Processed datasets are saved to `data/processed/cleaned_features.csv`, and transformation summaries are output directly to stdout.

## Future Improvements
- Implement Scikit-Learn custom `BaseEstimator` and `TransformerMixin` compatibility.
- Add support for Power Transforms (Box-Cox and Yeo-Johnson) for heavily non-Gaussian attributes.
- Add automated missingness mechanism diagnostics (MCAR, MAR, MNAR).

## Author
**Ozodbek Dalaboyev**  
Undergraduate in Artificial Intelligence, Tashkent University of Information Technologies (TUIT)  
GitHub: [@OzodbekX06](https://github.com/OzodbekX06) &bull; Email: ozodbekd06@gmail.com

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
