# DCF Patch Script: Fixed Intrinsic Value Calculation
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_financials(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)

    # Enhanced FCF calculation with better debugging
    try:
        cashflow = ticker.cashflow
        print(f"📊 Cashflow data shape: {cashflow.shape if cashflow is not None else 'None'}")
        
        if cashflow is None or cashflow.empty:
            print("⚠️ Cashflow data is empty or None")
            return None, None
            
        # Print available line items for debugging
        print(f"📋 Available cashflow line items: {list(cashflow.index)}")
        
        # Try to get operating cash flow
        try:
            operating_cash = cashflow.loc["Total Cash From Operating Activities"].iloc[0]
            print(f"💰 Operating Cash Flow: ${operating_cash:,.0f}")
        except KeyError:
            # Try alternative names
            alt_names = ["Operating Cash Flow", "Cash Flow From Operations", "Net Cash Provided By Operating Activities"]
            operating_cash = None
            for name in alt_names:
                if name in cashflow.index:
                    operating_cash = cashflow.loc[name].iloc[0]
                    print(f"💰 Operating Cash Flow (alt): ${operating_cash:,.0f}")
                    break
            
            if operating_cash is None:
                print("❌ Could not find operating cash flow data")
                return None, None
        
        # Try to get capital expenditures
        try:
            capex = cashflow.loc["Capital Expenditures"].iloc[0]
            print(f"🏗️ Capital Expenditures: ${capex:,.0f}")
        except KeyError:
            # Try alternative names
            alt_names = ["Capital Expenditure", "Purchase of Property, Plant and Equipment"]
            capex = None
            for name in alt_names:
                if name in cashflow.index:
                    capex = cashflow.loc[name].iloc[0]
                    print(f"🏗️ Capital Expenditures (alt): ${capex:,.0f}")
                    break
            
            if capex is None:
                print("❌ Could not find capital expenditures data")
                return None, None
        
        # Calculate FCF (OCF - CapEx, where CapEx is negative)
        fcf = operating_cash + capex  # CapEx is already negative
        print(f"💵 Free Cash Flow: ${fcf:,.0f}")
        
        if pd.isna(fcf) or fcf <= 0:
            print(f"⚠️ Invalid FCF calculated: {fcf}")
            return None, None
            
    except Exception as e:
        print(f"⚠️ Could not calculate FCF from Yahoo Finance: {e}")
        fcf = None

    # Shares Outstanding with better error handling
    try:
        shares_out = ticker.info.get('sharesOutstanding')
        if shares_out is None or shares_out <= 0:
            print(f"⚠️ Invalid shares outstanding: {shares_out}")
            shares_out = None
        else:
            print(f"📈 Shares Outstanding: {shares_out:,.0f}")
    except Exception as e:
        print(f"⚠️ Could not fetch shares outstanding: {e}")
        shares_out = None

    return fcf, shares_out

def calculate_dcf_improved(fcf, growth_rate=0.12, terminal_growth=0.03, discount_rate=0.09, forecast_years=7):
    """
    Improved DCF calculation with more conservative assumptions and proper growth decay.
    
    Args:
        fcf: Starting free cash flow
        growth_rate: Initial growth rate (more conservative: 12% vs 15%)
        terminal_growth: Terminal growth rate (more conservative: 3% vs 5%)
        discount_rate: Discount rate (higher: 9% vs 7.5%)
        forecast_years: Number of years to forecast (shorter: 7 vs 10)
    """
    if fcf is None:
        return None

    fcf_list = []
    for year in range(1, forecast_years + 1):
        # Linearly decay growth from initial to terminal
        decay = growth_rate - ((growth_rate - terminal_growth) * (year / forecast_years))
        projected_fcf = fcf * ((1 + decay) ** year)
        fcf_list.append(projected_fcf)

    # Discount FCFs to present value
    discounted_fcfs = [cf / ((1 + discount_rate) ** (i+1)) for i, cf in enumerate(fcf_list)]

    # Terminal value using Gordon Growth Model
    final_fcf = fcf_list[-1]
    terminal_value = (final_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / ((1 + discount_rate) ** forecast_years)

    total_value = sum(discounted_fcfs) + terminal_pv
    return total_value

def calculate_dcf_original(fcf, growth_rate=0.15, terminal_growth=0.05, discount_rate=0.075, forecast_years=10):
    """Original DCF calculation for comparison."""
    if fcf is None:
        return None

    fcf_values = []
    for year in range(forecast_years):
        # Linear decay from initial growth to terminal
        decay = growth_rate - ((growth_rate - terminal_growth) / forecast_years) * year
        projected_fcf = fcf * ((1 + decay) ** (year + 1))
        present_value = projected_fcf / ((1 + discount_rate) ** (year + 1))
        fcf_values.append(present_value)

    last_fcf = fcf * ((1 + terminal_growth) ** forecast_years)
    terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_pv = terminal_value / ((1 + discount_rate) ** forecast_years)

    dcf_total = sum(fcf_values) + terminal_value_pv
    return dcf_total

def main(ticker_symbol):
    print(f"\n🔍 Analyzing {ticker_symbol}...")
    print("=" * 60)
    
    fcf, shares_out = fetch_financials(ticker_symbol)
    
    if fcf is None or shares_out is None:
        print("❌ DCF calculation failed due to missing data.")
        return

    # Calculate both original and improved DCF values
    dcf_original = calculate_dcf_original(fcf)
    dcf_improved = calculate_dcf_improved(fcf)
    
    intrinsic_value_original = dcf_original / shares_out
    intrinsic_value_improved = dcf_improved / shares_out

    print("\n" + "=" * 60)
    print("✅ DCF ANALYSIS RESULTS COMPARISON")
    print("=" * 60)
    print(f"Ticker: {ticker_symbol}")
    print(f"FCF: ${fcf:,.0f}")
    print(f"Shares Outstanding: {shares_out:,.0f}")
    print("-" * 60)
    print("📊 ORIGINAL CALCULATION (Inflated):")
    print(f"   Growth Rate: 15% → 5% over 10 years")
    print(f"   Discount Rate: 7.5%")
    print(f"   DCF Total Value: ${dcf_original:,.0f}")
    print(f"   Intrinsic Value Per Share: ${intrinsic_value_original:,.2f}")
    print("-" * 60)
    print("📊 IMPROVED CALCULATION (Conservative):")
    print(f"   Growth Rate: 12% → 3% over 7 years")
    print(f"   Discount Rate: 9%")
    print(f"   DCF Total Value: ${dcf_improved:,.0f}")
    print(f"   Intrinsic Value Per Share: ${intrinsic_value_improved:,.2f}")
    print("-" * 60)
    print(f"💡 Improvement: {((intrinsic_value_original - intrinsic_value_improved) / intrinsic_value_original * 100):.1f}% reduction in inflated value")
    print("=" * 60)

if __name__ == "__main__":
    # Test multiple companies
    test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN"]
    
    print("🚀 TESTING IMPROVED DCF CALCULATION")
    print("=" * 60)
    
    successful_analyses = 0
    
    for ticker in test_tickers:
        try:
            main(ticker)
            successful_analyses += 1
            print(f"\n✅ Successfully analyzed {ticker}")
            print("=" * 60)
            
            # Add a small delay to avoid rate limiting
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to analyze {ticker}: {e}")
            print("=" * 60)
            continue
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print(f"Successfully analyzed: {successful_analyses}/{len(test_tickers)} companies")
    print("=" * 60)
