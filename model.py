import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ----- Load Labeled Data -----
df = pd.read_csv("data/labeled_company_financials.csv")
df = df[df["optimal_dcf_years"].notna()]

# ----- Features and Label -----
X = df[["sector", "market_cap", "gross_margin", "roe", "debt_to_equity", 
        "fcf"]].copy()
y = df["optimal_dcf_years"]

# Encode categorical feature
le = LabelEncoder()
X.loc[:, "sector"] = le.fit_transform(X["sector"].astype(str))

# Fill missing values with median for each column and ensure numeric types
for col in X.columns:
    if col != "sector":  # Skip sector as it's already encoded
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

# Ensure all data is numeric
X = X.astype(float)

# ----- Train/Test Split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Train Regression Model -----
model = XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# ----- Evaluate -----
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
accuracy = (y_pred_rounded == y_test).mean()

print(f"✅ Model Trained (XGBRegressor)")
print(f"📉 Mean Absolute Error: {mae:.2f} years")
print(f"📉 RMSE: {rmse:.2f} years")
print(f"🎯 Exact Match Accuracy (Rounded): {accuracy:.2%}")

# ----- Additional Analysis -----
print(f"\n📊 Model Insights:")
print(f"   • Model predicts DCF horizons with {mae:.1f} years average error")
print(f"   • {accuracy:.1%} of predictions match exactly when rounded")
print(f"   • RMSE of {rmse:.1f} years indicates prediction spread")

# Show some example predictions
print(f"\n🔮 Sample Predictions:")
sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
for i, idx in enumerate(sample_indices):
    actual = y_test.iloc[idx]
    predicted = y_pred[idx]
    predicted_rounded = round(predicted)
    print(f"   Company {i+1}: Actual {actual:.0f} years → Predicted {predicted:.1f} years ({predicted_rounded:.0f} rounded)")

# ----- Feature Importance -----
plt.figure(figsize=(8, 5))
plt.title("Feature Importance (XGBoost Regression)")
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top features
print(f"\n🔝 Top 3 Most Important Features:")
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_df.head(3))

# ----- Save Model and Encoder -----
print(f"\n💾 Saving model and encoder...")
model.save_model("xgb_model.json")
joblib.dump(le, "sector_encoder.pkl")
print(f"✅ Model saved as 'xgb_model.json'")
print(f"✅ Encoder saved as 'sector_encoder.pkl'") 