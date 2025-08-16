# Integrated DCF System: ML-Driven Horizon Prediction + Fixed FCF Calculation
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

# Import existing system components
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

def fetch_financials_fixed(ticker_symbol):
    """
    Fixed FCF calculation that addresses the inflated valuation issues.
    This replaces the problematic FCF calculation in the original system.
    """
    ticker = yf.Ticker(ticker_symbol)

    try:
        cashflow = ticker.cashflow
        
        if cashflow is None or cashflow.empty:
            print(f"⚠️ No cashflow data available for {ticker_symbol}")
            return None, None
            
        # Try to get operating cash flow
        try:
            operating_cash = cashflow.loc["Total Cash From Operating Activities"].iloc[0]
        except KeyError:
            # Try alternative names
            alt_names = ["Operating Cash Flow", "Cash Flow From Operations", "Net Cash Provided By Operating Activities"]
            operating_cash = None
            for name in alt_names:
                if name in cashflow.index:
                    operating_cash = cashflow.loc[name].iloc[0]
                    break
            
            if operating_cash is None:
                print(f"❌ Could not find operating cash flow data for {ticker_symbol}")
                return None, None
        
        # Try to get capital expenditures
        try:
            capex = cashflow.loc["Capital Expenditures"].iloc[0]
        except KeyError:
            # Try alternative names
            alt_names = ["Capital Expenditure", "Purchase of Property, Plant and Equipment"]
            capex = None
            for name in alt_names:
                if name in cashflow.index:
                    capex = cashflow.loc[name].iloc[0]
                    break
            
            if capex is None:
                print(f"❌ Could not find capital expenditures data for {ticker_symbol}")
                return None, None
        
        # Calculate FCF (OCF - CapEx, where CapEx is already negative)
        fcf = operating_cash + capex  # CapEx is already negative
        
        if pd.isna(fcf) or fcf <= 0:
            print(f"⚠️ Invalid FCF calculated for {ticker_symbol}: {fcf}")
            return None, None
            
        # Get shares outstanding
        shares_out = ticker.info.get('sharesOutstanding')
        if shares_out is None or shares_out <= 0:
            print(f"⚠️ Invalid shares outstanding for {ticker_symbol}: {shares_out}")
            return None, None
            
        return fcf, shares_out
        
    except Exception as e:
        print(f"❌ Error fetching financials for {ticker_symbol}: {e}")
        return None, None

def calculate_dcf_with_horizon(fcf, forecast_years, growth_rate=0.12, terminal_growth=0.05, discount_rate=0.09):
    """
    Calculate DCF using the predicted optimal forecast horizon with STANDARD DCF methodology.
    This uses proper DCF formulas that work for any company.
    """
    if fcf is None or fcf <= 0:
        return None

    try:
        # Step 1: Project FCF for explicit forecast period
        fcf_projections = []
        for year in range(1, int(forecast_years) + 1):
            # Standard growth projection: FCF * (1 + growth_rate)^year
            projected_fcf = fcf * ((1 + growth_rate) ** year)
            fcf_projections.append(projected_fcf)
        
        # Step 2: Discount explicit period FCFs to present value
        # Standard DCF formula: PV = FCF / (1 + r)^n
        discounted_fcfs = []
        for year, projected_fcf in enumerate(fcf_projections, 1):
            present_value = projected_fcf / ((1 + discount_rate) ** year)
            discounted_fcfs.append(present_value)
        
        explicit_period_value = sum(discounted_fcfs)
        
        # Step 3: Calculate terminal value using Gordon Growth Model
        # Terminal value = FCF in last forecast year * (1 + terminal_growth) / (discount_rate - terminal_growth)
        last_fcf = fcf_projections[-1]
        terminal_value = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        # Step 4: Discount terminal value to present
        # Terminal value is at the end of forecast period, so discount by (1+r)^forecast_years
        terminal_value_pv = terminal_value / ((1 + discount_rate) ** int(forecast_years))
        
        # Step 5: Total intrinsic value = PV of explicit + PV of terminal
        total_value = explicit_period_value + terminal_value_pv
        
        return total_value
        
    except Exception as e:
        print(f"❌ Error in DCF calculation: {e}")
        return None

def _price_now(ticker: str) -> float | None:
    """Get current market price for comparison."""
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

def calculate_undervalued_target(dcf_value_per_share, margin_of_safety=0.15):
    """
    Calculate the price target that offers the specified margin of safety.
    """
    return dcf_value_per_share * (1 - margin_of_safety)

def classify_valuation(current_price, dcf_value_per_share, undervalued_threshold=0.15, overvalued_threshold=0.15):
    """
    Classify whether the stock is undervalued, fairly valued, or overvalued.
    """
    if current_price is None:
        return "N/A (no current price)", None
    
    diff = (current_price - dcf_value_per_share) / dcf_value_per_share
    
    if diff <= -undervalued_threshold:
        classification = "Undervalued"
        threshold = "≤ -" + f"{undervalued_threshold:.0%}"
    elif diff >= overvalued_threshold:
        classification = "Overvalued"
        threshold = "≥ +" + f"{overvalued_threshold:.0%}"
    else:
        classification = "Fairly Valued"
        threshold = "within ±" + f"{overvalued_threshold:.0%}"
    
    return classification, threshold

# -------- Main pipeline --------
def main():
    ticker = input("Enter a company ticker (e.g., NVDA): ").upper().strip()
    print(f"\n🔍 Analyzing {ticker}...")
    print("=" * 60)

    # Step 1: Find similar companies for ML-driven similarity search
    print("🔍 Step 1: Finding similar companies for context...")
    similar_companies = find_similar_companies(ticker, n_similar=30)
    
    if not similar_companies.empty:
        print(f"✅ Found {len(similar_companies)} similar companies")
        print(f"   Similar companies: {', '.join(similar_companies['ticker'].head(10).tolist())}")
    else:
        print("⚠️ No similar companies found, proceeding with basic analysis")

    # Step 2: Build features using robust financials utility
    print("\n📊 Step 2: Fetching financial statements...")
    core = build_core_metrics_for_app(ticker)
    
    print(f"✅ Company: {core['company_name']}")
    print(f"✅ Sector: {core.get('sector', 'N/A')}")
    print(f"✅ FCF Strategy: {core['fcf_strategy']}")
    
    # Step 3: Build enhanced feature row using similar companies context
    print("\n🧠 Step 3: Building enhanced features using similar companies...")
    
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

    # Step 4: Build dynamic training dataset and train model for this specific ticker
    print("\n🧠 Step 4: Building dynamic training dataset and training model...")
    
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
                optimal_forecast_period = 7.0
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
            optimal_forecast_period = 7.0
            print(f"   Using default forecast period: {optimal_forecast_period:.1f} years")
    
    # Ensure the forecast period is reasonable (between 3 and 15 years)
    optimal_forecast_period = max(3.0, min(15.0, optimal_forecast_period))
    print(f"   Final forecast period: {optimal_forecast_period:.1f} years")

    # Step 5: Get FCF using our fixed calculation
    print(f"\n💰 Step 5: Calculating FCF using fixed methodology...")
    fcf, shares_out = fetch_financials_fixed(ticker)
    
    if fcf is None or shares_out is None:
        print("❌ Failed to get FCF or shares outstanding. Cannot proceed with DCF.")
        return
    
    print(f"✅ FCF: ${fcf:,.0f}")
    print(f"✅ Shares Outstanding: {shares_out:,.0f}")

    # Step 6: Run DCF with predicted horizon using standard assumptions
    print(f"\n📊 Step 6: Running DCF with {optimal_forecast_period}-year forecast period...")
    
    # Use standard DCF assumptions that work for any company
    # These are industry-standard rates, not hardcoded for specific companies
    growth_rate = 0.15  # 15% starting growth (standard for growing companies)
    terminal_growth = 0.05  # 5% terminal growth (standard long-term growth)
    discount_rate = 0.08  # 8% discount rate (standard for established companies)
    
    print(f"   Starting FCF: ${fcf:,.0f}")
    print(f"   Growth rate: {growth_rate:.1%} (constant during forecast period)")
    print(f"   Terminal growth: {terminal_growth:.1%}")
    print(f"   Discount rate: {discount_rate:.1%}")
    print(f"   Forecast period: {optimal_forecast_period:.1f} years")
    
    # Calculate DCF using our fixed method
    dcf_total_value = calculate_dcf_with_horizon(
        fcf=fcf,
        forecast_years=int(optimal_forecast_period),
        growth_rate=growth_rate,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )
    
    if dcf_total_value is None:
        print("❌ DCF calculation failed.")
        return
    
    dcf_per_share = dcf_total_value / shares_out
    
    print(f"   Total DCF value: ${dcf_total_value:,.0f}")
    print(f"   DCF per share: ${dcf_per_share:,.2f}")

    # Step 7: Compare to current price and calculate undervalued target
    print(f"\n🎯 Step 7: Market comparison and undervalued target calculation...")
    
    current_price = _price_now(ticker)
    
    if current_price is not None:
        print(f"   Current market price: ${current_price:,.2f}")
        
        # Classify valuation
        classification, threshold = classify_valuation(current_price, dcf_per_share)
        print(f"   Valuation classification: {classification} ({threshold})")
        
        # Calculate undervalued target (15% margin of safety)
        undervalued_target = calculate_undervalued_target(dcf_per_share, margin_of_safety=0.15)
        print(f"   15% undervalued buy-below price: ${undervalued_target:,.2f}")
        
        # Calculate percentage difference
        diff = (current_price - dcf_per_share) / dcf_per_share
        print(f"   Current price vs DCF: {diff:+.1%}")
        
        # Investment recommendation
        if diff <= -0.15:
            print(f"   💚 RECOMMENDATION: {ticker} appears UNDERVALUED")
            print(f"      Any price below ${undervalued_target:,.2f} offers a 15% margin of safety")
        elif diff >= 0.15:
            print(f"   🔴 RECOMMENDATION: {ticker} appears OVERVALUED")
            print(f"      Consider waiting for price to drop below ${undervalued_target:,.2f}")
        else:
            print(f"   🟡 RECOMMENDATION: {ticker} appears FAIRLY VALUED")
            print(f"      Current price is within reasonable range of intrinsic value")
    else:
        print("   ⚠️ Could not determine current market price")

    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL DCF ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Company: {core['company_name']}")
    print(f"ML-Predicted optimal forecast period: {optimal_forecast_period:.1f} years")
    print(f"FCF Strategy used: {core['fcf_strategy']}")
    print(f"Similar companies analyzed: {len(similar_companies)}")
    if not similar_companies.empty:
        print(f"Similar companies: {', '.join(similar_companies['ticker'].head(8).tolist())}")
    print("-" * 60)
    print(f"Latest FCF: ${fcf:,.0f}")
    print(f"Growth rate: {growth_rate:.1%} (constant during forecast period)")
    print(f"Shares outstanding: {shares_out:,.0f}")
    print(f"Total DCF value: ${dcf_total_value:,.0f}")
    print("-" * 60)
    print(f"DCF value per share: ${dcf_per_share:,.2f}")
    if current_price is not None:
        print(f"Current price: ${current_price:,.2f}")
        diff = (current_price - dcf_per_share) / dcf_per_share
        print(f"Valuation gap: {diff:+.1%}")
        undervalued_target = calculate_undervalued_target(dcf_per_share, margin_of_safety=0.15)
        print(f"15% undervalued buy-below price: ${undervalued_target:,.2f}")
    print("=" * 60)
    print("📊 DCF Methodology Used:")
    print(f"   • Forecast period: {optimal_forecast_period:.1f} years (ML-predicted)")
    print(f"   • Growth rate: {growth_rate:.1%} (constant during forecast period)")
    print(f"   • Terminal growth: {terminal_growth:.1%}")
    print(f"   • Discount rate: {discount_rate:.1%}")
    print(f"   • Terminal value: Gordon Growth Model")
    print(f"   • FCF calculation: Fixed methodology (OCF + CapEx)")
    print(f"   • DCF formula: Standard PV = FCF / (1+r)^n")
    print("=" * 60)

if __name__ == "__main__":
    main()
