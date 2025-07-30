import streamlit as st
import yfinance as yf
import numpy as np
import pickle
import xgboost as xgb
import pandas as pd
import joblib

# Load model and encoder
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")
    return model

@st.cache_resource
def load_encoder():
    with open("sector_encoder.pkl", "rb") as f:
        return joblib.load(f)

def process_financials(info, sector_encoder):
    """Process financial data for model prediction"""
    # Extract features
    sector = info.get("sector", "Unknown")
    market_cap = info.get("marketCap", 0)
    gross_margin = info.get("grossMargins", 0)
    roe = info.get("returnOnEquity", 0)
    debt_to_equity = info.get("debtToEquity", 0)
    
    # Get FCF from cashflow statement
    try:
        ticker = yf.Ticker(info.get("symbol", ""))
        cf = ticker.cashflow
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow", cf.columns[0]]
        else:
            fcf = 0
    except:
        fcf = 0
    
    # Create feature array
    features = {
        "sector": sector,
        "market_cap": market_cap,
        "gross_margin": gross_margin,
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "fcf": fcf
    }
    
    # Convert to DataFrame and encode sector
    df = pd.DataFrame([features])
    df["sector"] = sector_encoder.transform(df["sector"])
    
    # Fill missing values with 0
    df = df.fillna(0)
    
    return df.values[0]

def calculate_intrinsic_value(info, dcf_years):
    """Calculate intrinsic value using DCF method"""
    try:
        # Get FCF
        ticker = yf.Ticker(info.get("symbol", ""))
        cf = ticker.cashflow
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow", cf.columns[0]]
        else:
            fcf = info.get("freeCashflow", 0)
        
        # Handle negative FCF
        if fcf < 0:
            return info.get("marketCap", 0) * 0.5  # Conservative estimate
        
        # DCF calculation parameters
        growth_rate = 0.05  # 5% growth rate
        discount_rate = 0.10  # 10% discount rate
        
        # Calculate present value of future cash flows
        intrinsic_value = 0
        for year in range(1, dcf_years + 1):
            future_fcf = fcf * ((1 + growth_rate) ** year)
            discounted_fcf = future_fcf / ((1 + discount_rate) ** year)
            intrinsic_value += discounted_fcf
        
        return intrinsic_value
        
    except Exception as e:
        st.error(f"Error calculating intrinsic value: {e}")
        return 0

# Page configuration
st.set_page_config(
    page_title="ML-Powered DCF Valuation",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 ML-Powered DCF Valuation Tool")
st.markdown("""
This tool uses machine learning to predict optimal DCF horizons and calculate intrinsic values for stocks.
Enter a ticker symbol to get started!
""")

# Load model and encoder
try:
    model = load_model()
    sector_encoder = load_encoder()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "").upper()
with col2:
    analyze_button = st.button("🚀 Run Valuation", type="primary")

if analyze_button and ticker:
    with st.spinner("Fetching data and running prediction..."):
        try:
            # Fetch company data
            company_data = yf.Ticker(ticker)
            info = company_data.info
            
            if not info:
                st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                st.stop()
            
            # Process financial data
            processed_features = process_financials(info, sector_encoder)
            
            # Predict DCF Horizon
            dcf_years = model.predict(np.array(processed_features).reshape(1, -1))[0]
            dcf_years_rounded = round(dcf_years)
            
            # Calculate intrinsic value
            intrinsic_value = calculate_intrinsic_value(info, dcf_years_rounded)
            market_cap = info.get("marketCap", 0)
            
            # Valuation analysis
            if market_cap > 0:
                valuation_ratio = intrinsic_value / market_cap
                
                if valuation_ratio > 1.15:
                    status = "🟢 Undervalued"
                    status_color = "success"
                elif valuation_ratio < 0.85:
                    status = "🔴 Overvalued"
                    status_color = "error"
                else:
                    status = "🟡 Fairly Priced"
                    status_color = "warning"
            else:
                status = "❓ Unable to determine"
                status_color = "info"
            
            # Undervaluation threshold
            undervalue_margin = 0.15
            max_undervalued_price = intrinsic_value * (1 - undervalue_margin)
            
            # Display results
            st.success("✅ Valuation Complete!")
            
            # Main results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted DCF Horizon",
                    value=f"{dcf_years_rounded} years",
                    delta=f"{dcf_years:.1f} → {dcf_years_rounded}"
                )
            
            with col2:
                st.metric(
                    label="Intrinsic Value",
                    value=f"${intrinsic_value:,.0f}",
                    delta=f"{valuation_ratio:.1%}" if market_cap > 0 else None
                )
            
            with col3:
                st.metric(
                    label="Market Cap",
                    value=f"${market_cap:,.0f}"
                )
            
            # Detailed results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Valuation Details")
                st.markdown(f"""
                - **Company**: {info.get('longName', ticker)}
                - **Sector**: {info.get('sector', 'Unknown')}
                - **Industry**: {info.get('industry', 'Unknown')}
                - **Current Price**: ${info.get('currentPrice', 0):.2f}
                - **Valuation Status**: {status}
                """)
            
            with col2:
                st.markdown("### 💡 Investment Insights")
                st.markdown(f"""
                - **Max Undervalued Price**: ${max_undervalued_price:,.0f}
                - **Valuation Ratio**: {valuation_ratio:.2f}x
                - **Margin of Safety**: {undervalue_margin:.0%}
                - **Recommendation**: {'Consider buying' if valuation_ratio > 1.15 else 'Consider selling' if valuation_ratio < 0.85 else 'Hold'}
                """)
            
            # Additional metrics
            if market_cap > 0:
                st.markdown("---")
                st.markdown("### 📈 Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
                
                with col2:
                    st.metric("ROE", f"{info.get('returnOnEquity', 0):.1%}")
                
                with col3:
                    st.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
                
                with col4:
                    st.metric("Gross Margin", f"{info.get('grossMargins', 0):.1%}")
            
        except Exception as e:
            st.error(f"⚠️ Error processing {ticker}: {str(e)}")
            st.info("Please check that the ticker symbol is correct and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by XGBoost • Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True) 