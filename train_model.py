# train_model.py
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from features import build_training_dataset_from_similar_companies

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rev_cagr_3y","rev_cagr_5y",
    "fcf_cagr_3y","fcf_cagr_5y",
    "gross_margin_last","net_margin_last",
    "fcf_last","revenue_last","net_income_last",
    "leverage_ratio",
]

def build_dynamic_training_data(target_ticker: str = "AAPL", n_similar: int = 30):
    """
    Build training dataset dynamically by finding similar companies and calculating
    their optimal DCF forecast periods based on actual market cap accuracy.
    """
    print(f"🚀 Building dynamic training dataset for {target_ticker}...")
    
    # Build training dataset from similar companies
    df = build_training_dataset_from_similar_companies(target_ticker, n_similar)
    
    if df.empty:
        raise ValueError("Could not build training dataset")
    
    # Prepare features and target
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["optimal_forecast_period"].astype(float)
    
    print(f"📊 Training dataset built:")
    print(f"   Sample count: {len(y)}")
    print(f"   Forecast period range: {y.min():.1f} to {y.max():.1f} years")
    print(f"   Mean forecast period: {y.mean():.1f} years")
    print(f"   Forecast period distribution: {y.value_counts().sort_index().to_dict()}")
    
    return X, y, df

def train_model_on_dynamic_data(X, y):
    """Train XGBoost model on the dynamic dataset."""
    # Split data for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Use XGBoost Regressor
    clf = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Regression metrics
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"📈 Model Performance:")
    print(f"   Mean Squared Error: {mse:.2f}")
    print(f"   Root Mean Squared Error: {rmse:.2f}")
    print(f"   Mean Absolute Error: {mae:.2f}")
    print(f"   R² Score: {r2:.3f}")
    
    # Show some predictions vs actual
    print(f"\n🔍 Sample predictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = preds[i]
        print(f"   Actual: {actual:.1f}, Predicted: {predicted:.1f}, Diff: {predicted-actual:+.1f}")

    return clf

if __name__ == "__main__":
    # Build dynamic training dataset
    X, y, df = build_dynamic_training_data("AAPL", n_similar=30)
    
    # Train model
    clf = train_model_on_dynamic_data(X, y)
    
    # Save model
    joblib.dump(clf, MODEL_DIR / "horizon_xgb.pkl")
    print(f"✅ Saved dynamic model -> {MODEL_DIR/'horizon_xgb.pkl'}")
    
    # Save training dataset for reference
    df.to_csv("data/training/dynamic_training_dataset.csv", index=False)
    print(f"✅ Saved training dataset -> data/training/dynamic_training_dataset.csv")
