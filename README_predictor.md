# ML-DCF Horizon Predictor

## 🎯 Overview
The `predictor.py` script uses a trained XGBoost regression model to predict optimal DCF (Discounted Cash Flow) forecast horizons for any publicly traded company and provides valuation analysis.

## 🚀 How to Use

### Prerequisites
- Python 3.7+
- Required packages: `yfinance`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `joblib`
- Trained model files: `xgb_model.json` and `sector_encoder.pkl`

### Running the Predictor
```bash
python predictor.py
```

When prompted, enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA, XOM).

## 📊 What It Does

1. **Fetches Financial Data**: Automatically retrieves company financial data from Yahoo Finance
2. **Predicts DCF Horizon**: Uses ML model to predict optimal forecast period (1-50 years)
3. **Calculates Intrinsic Value**: Performs DCF valuation using predicted horizon
4. **Compares to Market**: Evaluates if stock is undervalued, overvalued, or fairly priced

## 📈 Sample Output

```
Enter ticker symbol: AAPL
🔍 Fetching data for AAPL...

📈 Predicted DCF Horizon for AAPL: 45 years
💰 Estimated Intrinsic Value: $2,003,287,919,626.60
🏛️ Market Cap: $3,194,468,958,208.00

Verdict: ❌ Overvalued
📊 Difference: -37.3% (-1,191,181,038,581)
```

## 🔧 Technical Details

### Model Features
- **Sector**: Company business sector
- **Market Cap**: Market capitalization
- **Gross Margin**: Gross profit margin
- **ROE**: Return on equity
- **Debt/Equity**: Debt-to-equity ratio
- **FCF Average**: Average free cash flow
- **FCF Volatility**: FCF standard deviation
- **FCF Growth 3Y**: 3-year FCF growth rate

### Valuation Parameters
- **Discount Rate**: 10%
- **Growth Rate**: 5%
- **Horizon**: ML-predicted optimal period

### Valuation Logic
- **Undervalued**: Intrinsic value > Market cap by >10%
- **Overvalued**: Intrinsic value < Market cap by >10%
- **Fairly Priced**: Within ±10% of market cap

## ⚠️ Limitations

- Requires sufficient financial data availability
- Assumes 5% growth rate (simplified)
- Uses 10% discount rate (standard)
- Negative FCF companies use conservative valuation
- Model accuracy: ~7.7 years average error

## 🎯 Next Steps

- Build web interface (Streamlit)
- Add more sophisticated growth modeling
- Implement sector-specific discount rates
- Add sensitivity analysis
- Include terminal value calculations 