import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from dcf import DCFInputs, intrinsic_value_from_fcf, TERMINAL_GROWTH
from financials_utils import build_core_metrics_for_app
from features import find_similar_companies, build_dynamic_features_for_ticker, build_training_dataset_from_similar_companies

# -------- Config / constants --------
MODEL_PATH = Path("models/horizon_xgb.pkl")
FEATURES = [
    "rev_cagr_3y","rev_cagr_5y",
    "fcf_cagr_3y","fcf_cagr_5y",
    "gross_margin_last","net_margin_last",
    "fcf_last","revenue_last","net_income_last",
    "leverage_ratio",
]

def _price_now(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).info or {}
        if info.get("currentPrice"):
            return float(info["currentPrice"])
    except Exception:
        pass
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

# -------- Main pipeline --------
def main():
    ticker = input("Enter a company ticker (e.g., NVDA): ").upper().strip()
    print(f"\n🔍 Analyzing {ticker}...")

    # 1) Find similar companies first
    print("🔍 Finding similar companies for context...")
    similar_companies = find_similar_companies(ticker, n_similar=30)
    
    if not similar_companies.empty:
        print(f"✅ Found {len(similar_companies)} similar companies")
        print(f"   Similar companies: {', '.join(similar_companies['ticker'].head(10).tolist())}")
    else:
        print("⚠️ No similar companies found, proceeding with basic analysis")

    # 2) Build features using robust financials utility
    print("📊 Fetching financial statements...")
    core = build_core_metrics_for_app(ticker)
    
    print(f"✅ Company: {core['company_name']}")
    print(f"✅ Sector: {core.get('sector', 'N/A')}")
    print(f"✅ FCF Strategy: {core['fcf_strategy']}")
    
    # 3) Build enhanced feature row using similar companies context
    print("🧠 Building enhanced features using similar companies...")
    
    # Get dynamic features from similar companies
    dynamic_features = build_dynamic_features_for_ticker(ticker, n_similar=30)
    
    # Build feature row for ML model with fallbacks from similar companies
    X = pd.DataFrame([{
        "rev_cagr_3y": core["rev_cagr_3y"] if np.isfinite(core["rev_cagr_3y"]) else dynamic_features.get("rev_cagr_3y_median", 0.0),
        "rev_cagr_5y": core["rev_cagr_5y"] if np.isfinite(core["rev_cagr_5y"]) else dynamic_features.get("rev_cagr_5y_median", 0.0),
        "fcf_cagr_3y": core["fcf_cagr_3y"] if np.isfinite(core["fcf_cagr_3y"]) else dynamic_features.get("fcf_cagr_3y_median", 0.0),
        "fcf_cagr_5y": core["fcf_cagr_5y"] if np.isfinite(core["fcf_cagr_5y"]) else dynamic_features.get("fcf_cagr_5y_median", 0.0),
        "gross_margin_last": dynamic_features.get("gross_margin_last_median", 0.0),
        "net_margin_last": dynamic_features.get("net_margin_last_median", 0.0),
        "fcf_last": core["fcf_last"],
        "revenue_last": core["revenue_last"],
        "net_income_last": core["net_income_last"],
        "leverage_ratio": dynamic_features.get("leverage_median", 0.0),
    }]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4) Build dynamic training dataset and train model for this specific ticker
    print("🧠 Building dynamic training dataset and training model...")
    
    try:
        # Build training dataset from similar companies
        training_df = build_training_dataset_from_similar_companies(ticker, n_similar=30)
        
        if not training_df.empty:
            # Train model on the fly for this ticker
            from train_model import train_model_on_dynamic_data
            
            # Prepare training data
            X_train = training_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = training_df["optimal_forecast_period"].astype(float)
            
            # Train model
            clf = train_model_on_dynamic_data(X_train, y_train)
            
            # Predict optimal forecast period for the target ticker
            pred = clf.predict(X)[0]
            optimal_forecast_period = float(pred)
            print(f"   Model trained and predicted optimal forecast period: {optimal_forecast_period:.1f} years")
            
        else:
            # Fallback to existing model if available
            if MODEL_PATH.exists():
                print("   Using existing pre-trained model as fallback...")
                clf = joblib.load(MODEL_PATH)
                pred = clf.predict(X)[0]
                optimal_forecast_period = float(pred)
                print(f"   Fallback model predicted forecast period: {optimal_forecast_period:.1f} years")
            else:
                # Use default forecast period if no model available
                optimal_forecast_period = 10.0
                print(f"   No model available, using default forecast period: {optimal_forecast_period:.1f} years")
                
    except Exception as e:
        print(f"   Error training dynamic model: {e}")
        # Fallback to existing model
        if MODEL_PATH.exists():
            print("   Using existing pre-trained model as fallback...")
            clf = joblib.load(MODEL_PATH)
            pred = clf.predict(X)[0]
            optimal_forecast_period = float(pred)
            print(f"   Fallback model predicted forecast period: {optimal_forecast_period:.1f} years")
        else:
            optimal_forecast_period = 10.0
            print(f"   Using default forecast period: {optimal_forecast_period:.1f} years")
    
    # Ensure the forecast period is reasonable (between 3 and 15 years)
    optimal_forecast_period = max(3.0, min(15.0, optimal_forecast_period))
    print(f"   Final forecast period: {optimal_forecast_period:.1f} years")

    # 5) Use standard growth rate from company data
    g0 = core.get("fcf_cagr_3y")
    if not np.isfinite(g0) or g0 is None:
        g0 = core.get("rev_cagr_3y")
    
    # Simple fallback if no growth data available
    if g0 is None or not np.isfinite(g0) or g0 < 0.02:
        g0 = 0.12  # 12% standard growth rate for tech companies
        print(f"   Using standard growth rate: {g0:.1%}")
    else:
        print(f"   Using company's actual growth rate: {g0:.1%}")
    
    # Ensure growth rate is reasonable (between 8% and 25%)
    g0 = max(0.08, min(0.25, g0))
    print(f"   Final growth rate: {g0:.1%}")
    
    if not np.isfinite(g0) or g0 is None:
        raise RuntimeError("Insufficient growth data (FCF/Revenue CAGR) to run DCF.")

    fcf0 = core.get("fcf_last")
    if not np.isfinite(fcf0) or (fcf0 is None) or (fcf0 <= 0):
        raise RuntimeError("Missing or non-positive FCF to run DCF.")

    shares_out = core.get("shares_out")
    if not shares_out or shares_out <= 0:
        raise RuntimeError("Missing shares outstanding; cannot compute per-share value.")

    # 6) Run DCF with enhanced inputs
    print(f"💰 Running DCF with {optimal_forecast_period}-year forecast period...")
    print(f"   Starting FCF: ${fcf0:,.0f}")
    print(f"   Starting growth rate: {g0:.1%}")
    print(f"   Shares outstanding: {shares_out:,.0f}")
    
    # Use standard discount rate for all companies
    discount_rate = 0.09  # 9% standard discount rate
    print(f"   Using standard discount rate: {discount_rate:.1%}")
    
    # Set environment variable for DCF calculation
    import os
    os.environ["DISCOUNT_RATE"] = str(discount_rate)
    
    iv_total = intrinsic_value_from_fcf(DCFInputs(
        fcf0=float(fcf0),
        forecast_years=int(optimal_forecast_period),
        growth_initial=float(g0),
    ))

    dcf_per_share = iv_total / float(shares_out)
    
    print(f"   Total DCF value: ${iv_total:,.0f}")
    print(f"   DCF per share: ${dcf_per_share:,.2f}")

    # 7) Compare to current price + 15% undervalued threshold
    price_now = _price_now(ticker)
    threshold_underval = dcf_per_share * 0.85  # 15% margin of safety

    if price_now is None:
        classification = "N/A (no current price)"
    else:
        diff = (price_now - dcf_per_share) / dcf_per_share
        if diff <= -0.15:
            classification = "Undervalued (≤ -15% vs DCF)"
        elif diff >= 0.15:
            classification = "Overvalued (≥ +15% vs DCF)"
        else:
            classification = "Fairly Valued (within ±10–15%)"

    # 8) Print enhanced results
    print("\n" + "="*60)
    print("🎯 ENHANCED DCF ANALYSIS RESULT")
    print("="*60)
    print(f"Ticker: {ticker}")
    print(f"Company: {core['company_name']}")
    print(f"Predicted optimal forecast period: {optimal_forecast_period:.1f} years")
    print(f"FCF Strategy used: {core['fcf_strategy']}")
    print(f"Similar companies analyzed: {len(similar_companies)}")
    if not similar_companies.empty:
        print(f"Similar companies: {', '.join(similar_companies['ticker'].head(8).tolist())}")
    print("-" * 60)
    print(f"Latest FCF: ${fcf0:,.0f}")
    print(f"Growth rate (starting): {g0:.1%}")
    print(f"Shares outstanding: {shares_out:,.0f}")
    print(f"Total DCF value: ${iv_total:,.0f}")
    print("-" * 60)
    print(f"DCF value per share: ${dcf_per_share:,.2f}")
    if price_now is not None:
        print(f"Current price: ${price_now:,.2f}")
        print(f"Valuation gap: {diff:+.1%}")
    print(f"15% undervalued buy-below price: ${threshold_underval:,.2f}")
    print(f"Classification: {classification}")
    print("="*60)
    print("📊 DCF Methodology Used:")
    print(f"   • Forecast period: {optimal_forecast_period:.1f} years")
    print(f"   • Growth rate: {g0:.1%} (decaying to {TERMINAL_GROWTH:.1%})")
    print(f"   • Discount rate: {discount_rate:.1%}")
    print(f"   • Terminal value: Gordon Growth Model")
    print("="*60)

if __name__ == "__main__":
    main()



