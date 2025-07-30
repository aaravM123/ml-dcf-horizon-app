# ML-DCF Horizon: Regression Model Results

## 🎯 Model Overview
- **Model Type**: XGBoost Regression (XGBRegressor)
- **Task**: Predict optimal DCF forecast horizon (1-50 years) as continuous values
- **Dataset**: 50 companies with labeled optimal DCF horizons

## 📊 Performance Metrics
- **Mean Absolute Error**: 7.68 years
- **Root Mean Square Error**: 11.46 years  
- **Exact Match Accuracy (Rounded)**: 30.00%

## 🔝 Feature Importance (Top 3)
1. **FCF Growth (3Y)**: 38.5% - Most critical factor
2. **Market Cap**: 22.3% - Company size matters
3. **FCF Volatility**: 10.8% - Cash flow stability

## 💡 Key Insights
- **FCF Growth Rate** is the dominant factor in determining DCF horizon
- **Market Capitalization** plays a significant role in horizon selection
- Model achieves 30% exact match accuracy when predictions are rounded
- Average prediction error is ~7.7 years, indicating reasonable precision

## 🔮 Sample Predictions
- Company 1: Actual 1 year → Predicted 1.0 years ✅
- Company 2: Actual 50 years → Predicted 50.3 years ✅  
- Company 3: Actual 50 years → Predicted 50.7 years ✅
- Company 4: Actual 50 years → Predicted 38.3 years ❌

## 🚀 Next Steps
- Deploy model for real-time predictions
- Build web interface for interactive DCF horizon recommendations
- Expand dataset for improved accuracy
- Fine-tune hyperparameters for better performance

## 📁 Files Generated
- `model.py` - Regression model implementation
- `feature_importance.png` - Feature importance visualization
- `data/labeled_company_financials.csv` - Training dataset 