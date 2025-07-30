import yfinance as yf
import pandas as pd
import time
import os

def get_major_companies():
    """Get a curated list of major companies"""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JNJ", "V", "UNH",
        "PG", "MA", "XOM", "HD", "PFE", "DIS", "KO", "MRK", "PEP", "AVGO",
        "JPM", "BAC", "WMT", "ABBV", "LLY", "TMO", "VZ", "CMCSA", "ADBE", "CRM",
        "NFLX", "PYPL", "INTC", "AMD", "QCOM", "TXN", "ORCL", "IBM", "CSCO", "INTU",
        "AMGN", "GILD", "BMY", "T", "CME", "SPGI", "BLK", "GS", "MS", "AXP",
        "CAT", "BA", "GE", "UPS", "LIN", "CVX", "COP", "EOG", "SLB", "HAL",
        "COST", "TGT", "LOW", "SBUX", "NKE", "MCD", "YUM", "CMG", "DRI", "DPZ",
        "UNP", "NSC", "CSX", "KSU", "FDX", "LMT", "RTX", "GD", "NOC", "HON",
        "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "XYL", "AME", "CCI",
        "AMT", "PLD", "EQIX", "DLR", "PSA", "SPG", "O", "WELL", "VICI", "RE",
        "EQR", "AVB", "MAA", "UDR", "ESS", "AIV", "CPT", "BXP", "VNO", "FRT",
        "KIM", "REG", "ARE", "BIO", "ILMN", "VRTX", "REGN", "GILD", "BIIB", "ALXN",
        "DXCM", "IDXX", "WST", "COO", "TMO", "DHR", "BDX", "ABT", "ISRG", "SYK",
        "MDT", "BSX", "ZBH", "BAX", "HCA", "DVA", "UHS", "THC", "LVS", "MGM",
        "WYNN", "CZR", "PENN", "BYD", "CHDN", "RCL", "CCL", "NCLH", "ALK", "DAL",
        "UAL", "AAL", "LUV", "JBLU", "SAVE", "HA", "ALGT", "SKYW", "JBLU", "SAVE"
    ]

def get_company_data(ticker):
    """Get comprehensive financial data for a company"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic info
        sector = info.get("sector", "Unknown")
        market_cap = info.get("marketCap")
        gross_margin = info.get("grossMargins")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        
        # Get FCF data
        fcf_avg = None
        fcf_volatility = None
        fcf_growth_3y = None
        
        try:
            cashflow = stock.cashflow
            if not cashflow.empty:
                # Try different methods to get FCF
                fcf_series = None
                if "Free Cash Flow" in cashflow.index:
                    fcf_series = cashflow.loc["Free Cash Flow"]
                elif "Total Cash From Operating Activities" in cashflow.index and "Capital Expenditures" in cashflow.index:
                    fcf_series = cashflow.loc["Total Cash From Operating Activities"] - cashflow.loc["Capital Expenditures"]
                elif "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
                    fcf_series = cashflow.loc["Operating Cash Flow"] - cashflow.loc["Capital Expenditure"]
                
                if fcf_series is not None:
                    fcf_clean = fcf_series.dropna().sort_index(ascending=False)
                    fcf_vals = fcf_clean.values
                    
                    if len(fcf_vals) > 0:
                        fcf_avg = fcf_vals.mean()
                        fcf_volatility = fcf_vals.std() if len(fcf_vals) > 1 else 0
                        if len(fcf_vals) >= 4:
                            fcf_growth_3y = ((fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])) if fcf_vals[-1] != 0 else 0
        except Exception as e:
            print(f"FCF extraction error for {ticker}: {e}")
        
        # Only return data if we have essential fields
        if market_cap and fcf_avg is not None:
            return {
                "ticker": ticker,
                "sector": sector,
                "market_cap": market_cap,
                "gross_margin": gross_margin,
                "roe": roe,
                "debt_to_equity": debt_to_equity,
                "fcf_avg": fcf_avg,
                "fcf_volatility": fcf_volatility,
                "fcf_growth_3y": fcf_growth_3y
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def collect_data():
    """Collect financial data for all companies"""
    tickers = get_major_companies()
    
    print(f"📊 Processing {len(tickers)} major companies...")
    
    records = []
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"🔄 Fetching [{i}/{len(tickers)}]: {ticker}")
        
        data = get_company_data(ticker)
        if data:
            records.append(data)
            successful += 1
            print(f"✅ {ticker}: Success")
        else:
            failed += 1
            print(f"❌ {ticker}: Failed")
        
        # Rate limiting
        time.sleep(1.5)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv("data/company_financials.csv", index=False)
    
    print(f"\n✅ Data collection complete!")
    print(f"📊 Successful: {successful}, Failed: {failed}")
    print(f"💾 Saved to: data/company_financials.csv")
    
    # Show summary statistics
    if not df.empty:
        print(f"\n📈 Summary:")
        print(f"   Companies with FCF data: {df['fcf_avg'].notna().sum()}")
        print(f"   Average market cap: ${df['market_cap'].mean()/1e9:.1f}B")
        print(f"   Sectors represented: {df['sector'].nunique()}")

if __name__ == "__main__":
    collect_data() 