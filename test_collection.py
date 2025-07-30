import yfinance as yf
import pandas as pd
import time

def test_single_company():
    """Test data collection for a single company"""
    ticker = "AAPL"
    print(f"Testing data collection for {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        print(f"✅ Basic info retrieved for {ticker}")
        print(f"   Sector: {info.get('sector', 'Unknown')}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,.0f}")
        
        # Test FCF extraction
        cashflow = stock.cashflow
        if not cashflow.empty:
            print(f"✅ Cashflow data available")
            print(f"   Available fields: {list(cashflow.index)}")
            
            # Try to get FCF
            if "Free Cash Flow" in cashflow.index:
                fcf = cashflow.loc["Free Cash Flow"].iloc[0]
                print(f"✅ FCF found: ${fcf:,.0f}")
            else:
                print("❌ FCF not found in direct field")
        else:
            print("❌ No cashflow data available")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_single_company() 