"""
Salary Prediction - Production Training Script.

Usage: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.linear_regression import LinearRegression


def main():
    """Run training pipeline."""
    
    print("="*60)
    print("SALARY PREDICTION - LINEAR REGRESSION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/salary_data_clean.csv')
    print(f"Loaded {len(df)} records")
    
    # Prepare
    X = df[['years_experience']].values
    y = df['salary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train custom model
    print("\n🔧 Training custom model...")
    model = LinearRegression(
        learning_rate=0.01,
        iterations=1000,
        verbose=True
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluation:")
    test_r2 = model.score(X_test_scaled, y_test)
    y_test_pred = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Test MAE:  ${test_mae:,.0f}")
    print(f"  Test RMSE: ${test_rmse:,.0f}")
    
    # Compare with sklearn
    print("\nValidation against sklearn...")
    sklearn_model = SklearnLR()
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_r2 = sklearn_model.score(X_test_scaled, y_test)
    
    print(f"  Our R²:     {test_r2:.6f}")
    print(f"  Sklearn R²: {sklearn_r2:.6f}")
    print(f"  Difference: {abs(test_r2 - sklearn_r2):.6f}")
    
    if abs(test_r2 - sklearn_r2) < 0.001:
        print("  ✅ Models match perfectly!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()