# Salary Prediction - Linear Regression

A gradient descent implementation of linear regression applied to data science salary data.

## Dataset

**Source:** [Kaggle - Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)

**Setup:**
1. Download `ds_salaries.csv` from Kaggle
2. Place in `data/` directory

## Problem

Predict data science salaries based on years of experience.

## Approach

**Algorithm:** Linear Regression with Gradient Descent
- Custom implementation (no sklearn for core algorithm)
- Batch gradient descent optimizer
- Feature standardization for convergence

## Performance

| Metric | Value |
|--------|-------|
| Test R² | 0.24 |
| Test MAE | $41,080 |
| Test RMSE | $55,079 |

**Note:** Low R² reflects data limitations - experience alone explains only 24% of salary variance. Additional features (job title, location, company size) needed for higher accuracy.

## Key Features

- Gradient descent from scratch
- Validated against sklearn (exact match)

## Tech Stack

- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (validation only)

## Structure
```
linear-regression/
├── notebooks/
│   └── linear_regression.ipynb    # Learning & experiments
├── src/
│   ├── __init__.py
│   └── linear_regression.py       # Production implementation
├── data/
│   └── .gitkeep                   # Download dataset here
├── visualizations/                # Generated plots
├── train_model.py                 # Main script
├── requirements.txt
└── README.md
```

## Usage

**Setup:**
```bash
pip install -r requirements.txt
```

**Jupyter (Learning):**
```bash
jupyter notebook
# Open notebooks/linear_regression.ipynb
```

**Production Script:**
```bash
python train_model.py
```

## Implementation Details

**Gradient Descent:**
```
Initialize: θ = 0
For each iteration:
  1. ŷ = Xθ + b
  2. ∂J/∂θ = (1/m)X^T(ŷ - y)
  3. θ := θ - α∇J
```

Convergence: 99% cost reduction in ~200 iterations

## Validation

Custom implementation matches sklearn exactly:
- R² difference: 0.000000
- Proves gradient descent correctness

## Insights

- Experience alone explains 24% of salary variance
- Each year → ~$12k salary increase (average)
- Low R² due to single feature, not implementation error
- Need job title, location, company for better predictions

## Visualizations

- Distribution analysis
- Relationship plots
- Segmented analysis by experience level
- Learning curve (gradient descent convergence)
- Predictions vs actual
- Learning rate comparison

