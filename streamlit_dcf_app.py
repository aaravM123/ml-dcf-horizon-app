# Streamlit DCF Analysis App - Simplified Version
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
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



# Top 30 NASDAQ companies by market cap
TOP_NASDAQ_COMPANIES = [
    {"Company": "Apple Inc.", "Ticker": "AAPL", "Sector": "Technology", "Market Cap": "$3.4T"},
    {"Company": "Microsoft Corporation", "Ticker": "MSFT", "Sector": "Technology", "Market Cap": "$3.9T"},
    {"Company": "Alphabet Inc.", "Ticker": "GOOGL", "Sector": "Technology", "Market Cap": "$2.5T"},
    {"Company": "Amazon.com Inc.", "Ticker": "AMZN", "Sector": "Technology", "Market Cap": "$2.5T"},
    {"Company": "NVIDIA Corporation", "Ticker": "NVDA", "Sector": "Technology", "Market Cap": "$4.4T"},
    {"Company": "Meta Platforms Inc.", "Ticker": "META", "Sector": "Technology", "Market Cap": "$1.3T"},
    {"Company": "Tesla Inc.", "Ticker": "TSLA", "Sector": "Technology", "Market Cap": "$0.8T"},
    {"Company": "Broadcom Inc.", "Ticker": "AVGO", "Sector": "Technology", "Market Cap": "$0.8T"},
    {"Company": "Netflix Inc.", "Ticker": "NFLX", "Sector": "Technology", "Market Cap": "$0.3T"},
    {"Company": "Adobe Inc.", "Ticker": "ADBE", "Sector": "Technology", "Market Cap": "$0.3T"},
    {"Company": "Salesforce Inc.", "Ticker": "CRM", "Sector": "Technology", "Market Cap": "$0.3T"},
    {"Company": "Oracle Corporation", "Ticker": "ORCL", "Sector": "Technology", "Market Cap": "$0.3T"},
    {"Company": "Intel Corporation", "Ticker": "INTC", "Sector": "Technology", "Market Cap": "$0.2T"},
    {"Company": "Advanced Micro Devices", "Ticker": "AMD", "Sector": "Technology", "Market Cap": "$0.3T"},
    {"Company": "Cisco Systems Inc.", "Ticker": "CSCO", "Sector": "Technology", "Market Cap": "$0.2T"},
    {"Company": "Qualcomm Inc.", "Ticker": "QCOM", "Sector": "Technology", "Market Cap": "$0.2T"},
    {"Company": "Applied Materials Inc.", "Ticker": "AMAT", "Sector": "Technology", "Market Cap": "$0.2T"},
    {"Company": "Micron Technology Inc.", "Ticker": "MU", "Sector": "Technology", "Market Cap": "$0.2T"},
    {"Company": "Lam Research Corp.", "Ticker": "LRCX", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "KLA Corporation", "Ticker": "KLAC", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Marvell Technology Inc.", "Ticker": "MRVL", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Analog Devices Inc.", "Ticker": "ADI", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "NXP Semiconductors", "Ticker": "NXPI", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "ON Semiconductor", "Ticker": "ON", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Skyworks Solutions", "Ticker": "SWKS", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Monolithic Power Systems", "Ticker": "MPWR", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Microchip Technology", "Ticker": "MCHP", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Synopsys Inc.", "Ticker": "SNPS", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Cadence Design Systems", "Ticker": "CDNS", "Sector": "Technology", "Market Cap": "$0.1T"},
    {"Company": "Autodesk Inc.", "Ticker": "ADSK", "Sector": "Technology", "Market Cap": "$0.1T"},
]

# Page configuration
st.set_page_config(
    page_title="ML-DCF Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .ticker-input-section {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    .company-table {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def fetch_financials_fixed(ticker_symbol):
    """Fixed FCF calculation that addresses the inflated valuation issues."""
    ticker = yf.Ticker(ticker_symbol)

    try:
        cashflow = ticker.cashflow
        
        if cashflow is None or cashflow.empty:
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
                return None, None
        
        # Calculate FCF (OCF - CapEx, where CapEx is already negative)
        fcf = operating_cash + capex  # CapEx is already negative
        
        if pd.isna(fcf) or fcf <= 0:
            return None, None
            
        # Get shares outstanding
        shares_out = ticker.info.get('sharesOutstanding')
        if shares_out is None or shares_out <= 0:
            return None, None
            
        return fcf, shares_out
        
    except Exception:
        return None, None

def calculate_dcf_with_horizon(fcf, forecast_years, growth_rate=0.15, terminal_growth=0.05, discount_rate=0.08):
    """Calculate DCF using the predicted optimal forecast horizon with STANDARD DCF methodology."""
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
        
    except Exception:
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
    """Calculate the price target that offers the specified margin of safety."""
    return dcf_value_per_share * (1 - margin_of_safety)

def classify_valuation(current_price, dcf_value_per_share, undervalued_threshold=0.15, overvalued_threshold=0.15):
    """Classify whether the stock is undervalued, fairly valued, or overvalued."""
    if current_price is None:
        return "N/A", None
    
    diff = (current_price - dcf_value_per_share) / dcf_value_per_share
    
    if diff <= -undervalued_threshold:
        classification = "UNDERVALUED"
        threshold = "≤ -" + f"{undervalued_threshold:.0%}"
    elif diff >= overvalued_threshold:
        classification = "OVERVALUED"
        threshold = "≥ +" + f"{overvalued_threshold:.0%}"
    else:
        classification = "FAIRLY VALUED"
        threshold = "within ±" + f"{overvalued_threshold:.0%}"
    
    return classification, threshold

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚀 ML-DCF Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Intrinsic Value Analysis", unsafe_allow_html=True)
    
    # Ticker input section at the top
    st.markdown('<div class="ticker-input-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Enter Company Ticker to Analyze")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Company Ticker:", value="NVDA", max_chars=10, label_visibility="collapsed").upper().strip()
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button logic
    if analyze_button and ticker:
        # Show loading message
        with st.spinner("Analyzing company... Please wait."):
            run_analysis(ticker)
    
    # Top 30 NASDAQ companies table
    st.markdown("### 📈 Top 30 NASDAQ Companies by Market Cap")
    st.markdown("Click on any company below to copy its ticker, or enter your own above.")
    
    # Convert to DataFrame for better display
    nasdaq_df = pd.DataFrame(TOP_NASDAQ_COMPANIES)
    st.dataframe(
        nasdaq_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Company": st.column_config.TextColumn("Company Name", width="large"),
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Market Cap": st.column_config.TextColumn("Market Cap", width="small")
        }
    )

def run_analysis(ticker):
    """Run the complete DCF analysis."""
    
    try:
        # Step 1: Find similar companies (silent)
        similar_companies = find_similar_companies(ticker, n_similar=30)
        
        # Step 2: Build features (silent)
        core = build_core_metrics_for_app(ticker)
        
        # Step 3: Build enhanced features (silent)
        dynamic_features = build_dynamic_features_for_ticker(ticker, n_similar=30)
        
        # Build feature row for ML model
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
        
        # Step 4: ML model prediction (silent)
        try:
            training_df = build_training_dataset_from_similar_companies(ticker, n_similar=30)
            
            if not training_df.empty:
                from train_model import train_model_on_dynamic_data
                
                X_train = training_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                y_train = training_df["optimal_forecast_period"].astype(float)
                
                clf = train_model_on_dynamic_data(X_train, y_train)
                pred = clf.predict(X)[0]
                optimal_forecast_period = float(pred)
                
            else:
                if MODEL_PATH.exists():
                    clf = joblib.load(MODEL_PATH)
                    pred = clf.predict(X)[0]
                    optimal_forecast_period = float(pred)
                else:
                    optimal_forecast_period = 7.0
                    
        except Exception:
            if MODEL_PATH.exists():
                clf = joblib.load(MODEL_PATH)
                pred = clf.predict(X)[0]
                optimal_forecast_period = float(pred)
            else:
                optimal_forecast_period = 7.0
        
        # Ensure the forecast period is reasonable
        optimal_forecast_period = max(3.0, min(15.0, optimal_forecast_period))
        
        # Step 5: FCF calculation (silent)
        fcf, shares_out = fetch_financials_fixed(ticker)
        
        if fcf is None or shares_out is None:
            st.error("❌ Failed to get financial data. Cannot proceed with analysis.")
            return
        
        # Step 6: DCF calculation (silent)
        # Use standard DCF assumptions
        growth_rate = 0.15
        terminal_growth = 0.05
        discount_rate = 0.08
        
        # Calculate DCF
        dcf_total_value = calculate_dcf_with_horizon(
            fcf=fcf,
            forecast_years=int(optimal_forecast_period),
            growth_rate=growth_rate,
            terminal_growth=terminal_growth,
            discount_rate=discount_rate
        )
        
        if dcf_total_value is None:
            st.error("❌ DCF calculation failed.")
            return
        
        dcf_per_share = dcf_total_value / shares_out
        
        # Step 7: Market comparison
        current_price = _price_now(ticker)
        
        if current_price is not None:
            # Get market cap
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', 0)
            
            # Company-level valuation
            company_diff = (market_cap - dcf_total_value) / dcf_total_value
            
            # Per-share valuation
            classification, threshold = classify_valuation(current_price, dcf_per_share)
            undervalued_target = calculate_undervalued_target(dcf_per_share, margin_of_safety=0.15)
            diff = (current_price - dcf_per_share) / dcf_per_share
            
            # Display results in clean format
            st.header("📊 Analysis Results")
            
            # Company info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Company:** {core.get('company_name', ticker)}")
            with col2:
                st.markdown(f"**Sector:** {core.get('sector', 'N/A')}")
            
            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Value of Business</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${dcf_total_value:,.0f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Current Market Cap</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${market_cap:,.0f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Valuation status
            if diff <= -0.15:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Valuation Status</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">💚 {classification}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif diff >= 0.15:
                st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Valuation Status</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">🔴 {classification}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Valuation Status</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">🟡 {classification}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 15% discount price target
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Buy at 15% Discount</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${undervalued_target:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional details (collapsible)
            with st.expander("📈 Additional Details"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Intrinsic Value", f"${dcf_per_share:.2f}")
                with col3:
                    st.metric("Per-Share Gap", f"{diff:+.1%}")
                with col4:
                    st.metric("Company Gap", f"{company_diff:+.1%}")
                
                st.markdown(f"**ML-Predicted Forecast Period:** {optimal_forecast_period:.1f} years")
                st.markdown(f"**Latest FCF:** ${fcf:,.0f}")
                st.markdown(f"**Shares Outstanding:** {shares_out:,.0f}")
            
        else:
            st.warning("⚠️ Could not determine current market price")
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()
