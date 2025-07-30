# ML-Powered DCF Valuation Tool

A Streamlit web application that uses machine learning to predict optimal DCF horizons and calculate intrinsic values for stocks.

## Features

- **ML-Powered DCF Horizon Prediction**: Uses trained XGBoost model to predict optimal DCF horizons
- **Real-time Data**: Fetches live financial data using Yahoo Finance
- **Intrinsic Value Calculation**: Calculates intrinsic value using DCF methodology
- **Valuation Analysis**: Compares intrinsic value with market cap to determine valuation status
- **Investment Insights**: Provides buy/sell recommendations and maximum undervalued price

## Prerequisites

Make sure you have the following files in your project directory:
- `xgb_model.json` - Trained XGBoost model
- `sector_encoder.pkl` - Sector label encoder

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

4. Click "Run Valuation" to get the analysis

## Output

The app provides:

- **Predicted DCF Horizon**: ML-predicted optimal number of years for DCF calculation
- **Intrinsic Value**: Calculated intrinsic value using DCF method
- **Market Cap**: Current market capitalization
- **Valuation Status**: Undervalued, Fairly Priced, or Overvalued
- **Investment Insights**: Maximum undervalued price and recommendations
- **Key Metrics**: P/E ratio, ROE, debt/equity, gross margin

## Example

For a stock like AAPL, the app might show:
- Predicted DCF Horizon: 8 years
- Intrinsic Value: $2.5 trillion
- Market Cap: $2.8 trillion
- Status: Fairly Priced
- Recommendation: Hold

## Technical Details

- **Model**: XGBoost regression model trained on financial data
- **Features**: Sector, market cap, gross margin, ROE, debt/equity, free cash flow
- **DCF Parameters**: 5% growth rate, 10% discount rate
- **Valuation Thresholds**: 15% margin for undervaluation/overvaluation

## Troubleshooting

- If you get an error loading the model, ensure `xgb_model.json` and `sector_encoder.pkl` are in the same directory
- If data fetching fails, check your internet connection and verify the ticker symbol
- For negative FCF companies, the app uses conservative valuation estimates 