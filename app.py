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
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "", help="This is the short symbol used to represent a company on the stock market. Example: Apple = AAPL, Microsoft = MSFT, Tesla = TSLA.").upper()
with col2:
    analyze_button = st.button("🚀 Run Valuation", type="primary")

# Helper text for users
st.caption("💡 A stock ticker is a company's trading symbol, like AAPL for Apple, TSLA for Tesla, or MSFT for Microsoft.")

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
            
            # Calculate intrinsic value and get FCF data
            intrinsic_value = calculate_intrinsic_value(info, dcf_years_rounded)
            market_cap = info.get("marketCap", 0)
            
            # === FIXED DCF AND MAX UNDERVALUED PRICE CALCULATION ===
            try:
                ticker_obj = yf.Ticker(ticker)
                cf = ticker_obj.cashflow
                fcf = cf.loc["Free Cash Flow", cf.columns[0]] if "Free Cash Flow" in cf.index else info.get("freeCashflow", 0)
                shares_outstanding = info.get("sharesOutstanding", 1)
                
                # ✅ Adjust FCF scale (yfinance often gives in millions or billions)
                if fcf < 1e6:
                    fcf *= 1e9  # assume billions
                elif fcf < 1e9:
                    fcf *= 1e6  # assume millions
                
                # ✅ Calculate total intrinsic value using DCF formula
                growth_rate = 0.05  # 5% growth rate
                discount_rate = 0.10  # 10% discount rate
                intrinsic_value_total = fcf * ((1 + growth_rate)**dcf_years_rounded - 1) / (discount_rate - growth_rate)
                
                # ✅ Compute value per share
                intrinsic_value_per_share = intrinsic_value_total / shares_outstanding
                
                # ✅ Apply margin of safety (15%)
                undervalue_margin = 0.15
                max_undervalued_price_per_share = intrinsic_value_per_share * (1 - undervalue_margin)
                
                # ✅ Display results in Streamlit
                st.metric("📈 Intrinsic Value Per Share", f"${intrinsic_value_per_share:,.2f}")
                st.metric("🟢 Max Undervalued Price", f"${max_undervalued_price_per_share:,.2f}")
                
            except Exception as e:
                st.error(f"❌ Error in DCF calculation: {str(e)}")
                intrinsic_value_per_share = 0
                max_undervalued_price_per_share = 0
            
            # Valuation analysis
            if market_cap > 0:
                valuation_ratio = intrinsic_value_total / market_cap
                
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
                    label="Intrinsic Value Per Share",
                    value=f"${intrinsic_value_per_share:,.2f}",
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
                - **Max Undervalued Price per Share**: ${max_undervalued_price_per_share:,.2f}
                - **Valuation Ratio**: {valuation_ratio:.2f}x
                - **Margin of Safety**: {undervalue_margin:.0%}
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
            
            # PE/G Ratio Section
            st.markdown("---")
            st.subheader("📊 Additional Insights")
            
            try:
                eps = info.get("trailingEps", 0)
                price_per_share = info.get("currentPrice", 0)
                pe_ratio = price_per_share / eps if eps != 0 else None
                fcf_growth = 0.05  # Using the 5% growth rate from DCF calculation
                
                if pe_ratio is not None and fcf_growth > 0:
                    peg_ratio = pe_ratio / (fcf_growth * 100)  # growth as percentage
                    peg_text = f"{peg_ratio:.2f}"
                    peg_description = "The PE/G ratio compares the Price-to-Earnings ratio to the FCF growth rate. Lower values (<1) may indicate undervaluation."
                else:
                    peg_text = "N/A"
                    peg_description = "Insufficient earnings or growth data to calculate PE/G."
                
                st.metric("📉 PE/G Ratio", peg_text, help=peg_description)
                
            except Exception as e:
                st.error(f"❌ Error calculating PE/G: {e}")
            
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