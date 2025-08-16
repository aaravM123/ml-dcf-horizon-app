# ML-DCF Horizon: Enhanced DCF Analysis with Similar Company Context

## Overview

This project provides an enhanced Discounted Cash Flow (DCF) analysis system that:

1. **Finds similar companies** for any given ticker based on sector, industry, and financial metrics
2. **Uses XGBoost regression** to predict optimal DCF horizons (not limited to discrete values)
3. **Builds dynamic features** using patterns from similar companies when company data is missing
4. **Calculates intrinsic value per share** using the predicted horizon and company financials

## Key Features

### 🎯 **Similar Company Analysis**
- Automatically finds 20 similar companies for any ticker entered
- Filters by sector/industry first, then by financial similarity
- Uses semiconductor companies as analogs (ADI, AMD, ASML, etc.)

### 🧠 **ML-Powered Horizon Prediction**
- **XGBoost Regression Model** (not classification)
- Can predict **any horizon value** (3.2, 5.7, 8.9 years, etc.)
- No artificial limits on horizon values
- Trained on historical data with actual horizon labels

### 📊 **Dynamic Feature Engineering**
- Falls back to similar company patterns when company data is missing
- Provides robust feature sets even with incomplete financial data
- Uses median/mean values from similar companies as fallbacks

### 💰 **Enhanced DCF Calculation**
- Calculates total intrinsic value and per-share value
- Uses predicted optimal horizon from ML model
- Provides detailed breakdown of DCF components
- Includes 15% margin of safety for buy recommendations

## How It Works

### 1. **Input & Similar Company Discovery**
```
Enter ticker (e.g., AAPL) → Find 20 similar companies → Analyze sector/industry patterns
```

### 2. **Feature Building with Fallbacks**
```
Company financials + Similar company patterns → Robust feature set → ML model input
```

### 3. **Horizon Prediction**
```
XGBoost regression → Continuous horizon value (e.g., 7.3 years) → DCF calculation
```

### 4. **DCF Analysis**
```
Projected FCFs + Terminal value → Total intrinsic value → Per-share value → Valuation gap
```

## Usage

### Training the Model
```bash
python train_model.py
```

### Running DCF Analysis
```bash
python predict_and_run_dcf.py
```

## Example Output

```
🔍 Analyzing AAPL...
✅ Found 19 similar companies
   Similar companies: MU, STM, KLAC, ON, AMD, TXN, ASML, QCOM, INFU, ADI

🧠 Predicting optimal DCF horizon...
   Model predicted horizon: 10.0 years

💰 Running DCF with 10.0-year horizon...
   Starting FCF: $94,873,747,456
   Starting growth rate: -2.2%
   Shares outstanding: 14,840,399,872
   Total DCF value: $1,067,339,012,444
   DCF value per share: $71.92

🎯 ENHANCED DCF ANALYSIS RESULT
DCF value per share: $71.92
Current price: $232.78
Valuation gap: +223.7%
Classification: Overvalued (≥ +15% vs DCF)
```

## Why DCF Values Were Too Small

The original system had several issues that caused DCF values to be too small:

1. **Limited Horizon Prediction**: Only predicted {3,5,10} years instead of continuous values
2. **Missing Similar Company Context**: Didn't use patterns from similar companies
3. **Feature Gaps**: Missing margin and leverage data caused poor ML predictions
4. **Classification vs Regression**: Using classification limited horizon choices

## What We Fixed

✅ **Changed to XGBoost Regression** - Can predict any horizon value (3.2, 7.8, 12.1 years)  
✅ **Added Similar Company Analysis** - Finds 20 similar companies for context  
✅ **Dynamic Feature Building** - Uses similar company patterns as fallbacks  
✅ **Enhanced DCF Calculation** - Proper per-share valuation with debugging  
✅ **Robust Error Handling** - Graceful fallbacks when data is missing  

## Technical Details

### Model Architecture
- **Algorithm**: XGBoost Regressor
- **Objective**: reg:squarederror
- **Features**: 9 financial metrics (revenue growth, FCF growth, margins, leverage)
- **Output**: Continuous horizon values (typically 3-15 years)

### Similar Company Selection
- **Primary**: Sector/industry matching
- **Secondary**: Financial similarity scoring
- **Fallback**: All available companies if needed

### DCF Parameters
- **Discount Rate**: 10% (configurable via environment)
- **Terminal Growth**: 2.5% (configurable via environment)
- **Growth Decay**: 85% per year toward terminal rate

## Future Enhancements

- [ ] Add more sectors beyond semiconductors
- [ ] Implement ensemble methods for better horizon prediction
- [ ] Add sensitivity analysis for DCF inputs
- [ ] Include more sophisticated terminal value calculations
- [ ] Add historical backtesting capabilities

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- xgboost
- scikit-learn
- joblib

## Installation

```bash
pip install -r requirements.txt
```

## License

This project is for educational and research purposes.
