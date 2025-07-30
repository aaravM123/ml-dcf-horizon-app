import yfinance as yf
import pandas as pd
import time

def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table['Symbol'].tolist()

tickers = get_sp500_tickers()
financials = []

for ticker in tickers:
    print(f"Fetching: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cf = stock.cashflow
        fcf = None
        
        # Try to get FCF - extract the most recent value
        if 'Free Cash Flow' in cf.index:
            fcf_series = cf.loc['Free Cash Flow']
            # Get the most recent non-null value
            fcf = fcf_series.dropna().iloc[0] if not fcf_series.dropna().empty else None
        else:
            try:
                # Calculate FCF manually
                op_cf = cf.loc['Total Cash From Operating Activities']
                capex = cf.loc['Capital Expenditures']
                fcf_series = op_cf - capex
                fcf = fcf_series.dropna().iloc[0] if not fcf_series.dropna().empty else None
            except:
                pass
                
        row = {
            'ticker': ticker,
            'sector': info.get('sector'),
            'market_cap': info.get('marketCap'),
            'gross_margin': info.get('grossMargins'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'fcf': fcf
        }
        financials.append(row)
        time.sleep(1)  # avoid hitting rate limit
    except Exception as e:
        print(f"⚠️ Skipping {ticker}: {e}")

df = pd.DataFrame(financials)
df.to_csv("data/company_financials.csv", index=False)
print("✅ Saved expanded dataset.")
