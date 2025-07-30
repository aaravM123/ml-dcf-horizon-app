import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Constants
UNDERVALUATION_THRESHOLD = 0.20  # 20% threshold to be considered undervalued

# Load trained model
model = XGBRegressor()
model.load_model("xgb_model.json")  # Update path if needed

# Label encoder for sector
sector_encoder = joblib.load("sector_encoder.pkl")  # Save from training phase

def fetch_financial_data(ticker):
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    info = stock.info

    # Basic fields
    sector = info.get("sector", "Unknown")
    market_cap = info.get("marketCap", None)
    gross_margin = info.get("grossMargins", None)
    roe = info.get("returnOnEquity", None)
    debt_to_equity = info.get("debtToEquity", None)

    # Get FCF directly from cashflow statement
    try:
        cf = stock.cashflow
        
        # Use Free Cash Flow directly if available
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow", cf.columns[0]]  # Use most recent year
        else:
            print(f"   Free Cash Flow not found in data")
            fcf = None
            
    except Exception as e:
        print(f"Error fetching FCF: {e}")
        fcf = None

    features = {
        "sector": sector,
        "market_cap": market_cap,
        "gross_margin": gross_margin,
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "fcf_avg": fcf if fcf is not None else None,
        "fcf_volatility": 0 if fcf is not None else None,  # Simplified for now
        "fcf_growth_3y": 0.05 if fcf is not None else None  # Assume 5% growth for now
    }

    return features, market_cap, fcf

def discounted_cash_flow(fcf, growth, discount, years):
    value = 0
    for t in range(1, years + 1):
        future = fcf * ((1 + growth) ** t)
        discounted = future / ((1 + discount) ** t)
        value += discounted
    return value

def main():
    ticker = input("Enter ticker symbol: ").strip().upper()
    print(f"Fetching data for {ticker}...")
    
    features, market_cap, fcf_series = fetch_financial_data(ticker)
    
    # Check data availability
    if market_cap is None:
        print("❌ Market cap data not available.")
        return
        
    if features['fcf_avg'] is None:
        print("❌ Free cash flow data not available.")
        return

    # Format features for model input
    df = pd.DataFrame([features])
    df["sector"] = sector_encoder.transform(df["sector"])
    df = df.fillna(df.median(numeric_only=True))

    # Predict optimal DCF horizon
    predicted_horizon = model.predict(df)[0]
    rounded_horizon = int(round(predicted_horizon))

    print(f"\nPredicted DCF Horizon for {ticker}: {rounded_horizon} years")

    # Estimate intrinsic value
    avg_fcf = features["fcf_avg"]
    discount_rate = 0.10
    growth_rate = 0.05
    
    # Handle negative FCF
    if avg_fcf < 0:
        print(f"Warning: Negative FCF (${avg_fcf:,.0f}) - using simplified valuation")
        intrinsic_value = market_cap * 0.5  # Conservative estimate for negative FCF
    else:
        intrinsic_value = discounted_cash_flow(avg_fcf, growth_rate, discount_rate, rounded_horizon)

    print(f"Estimated Intrinsic Value: ${intrinsic_value:,.2f}")
    print(f"Market Cap: ${market_cap:,.2f}")

    # Compare to market cap
    diff = intrinsic_value - market_cap
    pct = diff / market_cap

    if pct > 0.10:
        verdict = "Undervalued"
    elif pct < -0.10:
        verdict = "Overvalued"
    else:
        verdict = "Fairly Priced"

    print(f"\nVerdict: {verdict}")
    print(f"Difference: {pct:+.1%} ({diff:+,.0f})")

    # Calculate max price to still be undervalued
    max_undervalued_price = intrinsic_value * (1 - UNDERVALUATION_THRESHOLD)

    # Show recommended buy range based on valuation
    threshold_pct = int(UNDERVALUATION_THRESHOLD * 100)
    print("To qualify as undervalued (by " + str(threshold_pct) + "%), the max price you'd want to pay is: $" + f"{max_undervalued_price:,.2f}")
    print("This gives you a target entry price range for a value-based investment strategy.\n")


if __name__ == "__main__":
    main() 