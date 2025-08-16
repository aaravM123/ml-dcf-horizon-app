# DCF Debug Script - Investigating Low Intrinsic Values
import yfinance as yf
import pandas as pd

def debug_nvda_dcf():
    """Debug why NVDA's intrinsic value is so low."""
    ticker = "NVDA"
    print(f"🔍 Debugging {ticker} DCF calculation...")
    print("=" * 60)
    
    # Get current market data
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice', 0)
    market_cap = stock.info.get('marketCap', 0)
    
    print(f"📊 Current Market Data:")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Market Cap: ${market_cap:,.0f}")
    print()
    
    # Get FCF data
    cashflow = stock.cashflow
    if cashflow is not None and not cashflow.empty:
        print(f"📋 Cashflow Data Available:")
        print(f"   Shape: {cashflow.shape}")
        print(f"   Columns: {list(cashflow.columns)}")
        print(f"   Index: {list(cashflow.index)}")
        print()
        
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
        
        # Calculate FCF
        if operating_cash is not None and capex is not None:
            fcf = operating_cash + capex  # CapEx is already negative
            print(f"💵 Free Cash Flow: ${fcf:,.0f}")
            
            # Get shares outstanding
            shares_out = stock.info.get('sharesOutstanding')
            if shares_out:
                print(f"📈 Shares Outstanding: {shares_out:,.0f}")
                
                # Calculate FCF per share
                fcf_per_share = fcf / shares_out
                print(f"💵 FCF per Share: ${fcf_per_share:.2f}")
                
                # Simple P/FCF ratio
                if current_price > 0:
                    pfcf_ratio = current_price / fcf_per_share
                    print(f"📊 P/FCF Ratio: {pfcf_ratio:.2f}x")
                
                print()
                
                # Now let's test different DCF scenarios
                print("🧮 Testing Different DCF Scenarios:")
                print("-" * 40)
                
                # Scenario 1: Current conservative assumptions
                print("Scenario 1: Current Conservative (12% → 3%, 9% discount, 9.3 years)")
                test_dcf_scenario(fcf, shares_out, 12, 3, 9, 9.3)
                
                # Scenario 2: More realistic for NVDA
                print("\nScenario 2: More Realistic (20% → 5%, 8% discount, 7 years)")
                test_dcf_scenario(fcf, shares_out, 20, 5, 8, 7)
                
                # Scenario 3: Growth company assumptions
                print("\nScenario 3: Growth Company (25% → 6%, 7% discount, 5 years)")
                test_dcf_scenario(fcf, shares_out, 25, 6, 7, 5)
                
                # Scenario 4: Very conservative (current)
                print("\nScenario 4: Very Conservative (8% → 2%, 10% discount, 10 years)")
                test_dcf_scenario(fcf, shares_out, 8, 2, 10, 10)
                
            else:
                print("❌ Could not get shares outstanding")
        else:
            print("❌ Could not calculate FCF")
    else:
        print("❌ No cashflow data available")

def test_dcf_scenario(fcf, shares_out, growth_rate, terminal_growth, discount_rate, forecast_years):
    """Test a specific DCF scenario."""
    if fcf <= 0:
        print("   ❌ Invalid FCF")
        return
    
    try:
        # Calculate DCF
        fcf_list = []
        for year in range(1, int(forecast_years) + 1):
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
        intrinsic_value_per_share = total_value / shares_out
        
        print(f"   📊 DCF Total: ${total_value:,.0f}")
        print(f"   💰 Intrinsic Value/Share: ${intrinsic_value_per_share:.2f}")
        
        # Calculate components
        explicit_value = sum(discounted_fcfs)
        terminal_contribution = terminal_pv
        print(f"   📈 Explicit Period Value: ${explicit_value:,.0f}")
        print(f"   🔮 Terminal Value (PV): ${terminal_contribution:,.0f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    debug_nvda_dcf()
