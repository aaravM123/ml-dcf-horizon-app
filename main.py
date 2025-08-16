import os, json, time
from pathlib import Path

import yfinance as yf
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)

DATA_DIR = Path("data/analogs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Step 1: Fetch current company metrics
# -----------------------------
def get_company_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    cashflow = stock.cashflow if hasattr(stock, "cashflow") else pd.DataFrame()

    def safe_fcf(cf: pd.DataFrame):
        try:
            return (
                cf.loc["Total Cash From Operating Activities"].iloc[0]
                - cf.loc["Capital Expenditures"].iloc[0]
            )
        except Exception:
            return None

    metrics = {
        "ticker": ticker.upper(),
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "revenue_ttm": info.get("totalRevenue"),
        "net_income_ttm": info.get("netIncomeToCommon"),
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cash_flow": safe_fcf(cashflow),
        "business_summary": info.get("longBusinessSummary"),
    }
    return metrics

# -----------------------------
# Step 2: LLM similarity (JSON out)
# -----------------------------
def find_similar_companies(metrics: dict) -> list[dict]:
    """
    Returns a list of dicts: [{"company": "...", "ticker": "...", "year": 2011}, ...] length 20
    """
    prompt = f"""
You are an equity analyst. Given the company's current metrics and context,
return EXACTLY 20 historical analogs that were at a similar stage (company + TICKER + YEAR).

Return STRICT JSON ONLY as a list of objects with keys:
company (string), ticker (uppercase string), year (integer).

If unsure of the exact year, choose the best single year (do not return ranges).
If unsure of a ticker, choose the primary listing most commonly used on Yahoo Finance.

COMPANY CONTEXT
---------------
Ticker: {metrics['ticker']}
Company Name: {metrics['company_name']}
Sector: {metrics['sector']}
Industry: {metrics['industry']}
Market Cap: {metrics['market_cap']}
Revenue (TTM): {metrics['revenue_ttm']}
Net Income (TTM): {metrics['net_income_ttm']}
P/E Ratio: {metrics['pe_ratio']}
P/B Ratio: {metrics['pb_ratio']}
Operating Margin: {metrics['operating_margin']}
ROE: {metrics['roe']}
Debt/Equity: {metrics['debt_to_equity']}
Free Cash Flow: {metrics['free_cash_flow']}

Business Summary:
{metrics['business_summary']}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in equity analog selection. Respond with strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content.strip()
    
    print(f"  📝 Raw LLM response ({len(content)} chars):")
    print(f"  {repr(content[:200])}...")
    
    if not content:
        print("  ❌ LLM returned empty response")
        return []

    # Clean up markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    if content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove ```
    
    content = content.strip()
    print(f"  🧹 Cleaned content: {repr(content[:200])}...")

    # Try strict JSON parse; fallback to line-based rescue if needed
    try:
        analogs = json.loads(content)
        # light normalization
        norm = []
        for a in analogs:
            company = (a.get("company") or "").strip()
            ticker = (a.get("ticker") or "").strip().upper()
            year = int(a.get("year"))
            if company and ticker and year:
                norm.append({"company": company, "ticker": ticker, "year": year})
        return norm[:20]
    except Exception as e:
        print(f"  ⚠️ JSON parse failed: {e}")
        print(f"  🔄 Attempting fallback parsing...")
        
        # Fallback: parse various formats the model might return
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        out = []
        
        for ln in lines:
            # Try different separators and formats
            for sep in [" - ", "–", "—", " | ", ":", ","]:
                if sep in ln:
                    parts = [p.strip() for p in ln.split(sep)]
                    if len(parts) >= 2:
                        # Look for year at the end
                        for i, part in enumerate(parts):
                            try:
                                year = int(part)
                                # Assume ticker is before year, company is everything else
                                if i > 0:
                                    ticker = parts[i-1].strip().upper()
                                    company = sep.join(parts[:i-1]).strip()
                                    if company and ticker and len(ticker) <= 5:  # Reasonable ticker length
                                        out.append({"company": company, "ticker": ticker, "year": year})
                                        break
                            except ValueError:
                                continue
        
        print(f"  📊 Fallback parsing found {len(out)} analogs")
        return out[:20]

# -----------------------------
# Step 3: Fetch & store analog financials
# -----------------------------
@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def fetch_statements_for_ticker(ticker: str) -> dict:
    st = yf.Ticker(ticker)
    return {
        "income_stmt": st.income_stmt if hasattr(st, "income_stmt") else pd.DataFrame(),
        "balance_sheet": st.balance_sheet if hasattr(st, "balance_sheet") else pd.DataFrame(),
        "cashflow": st.cashflow if hasattr(st, "cashflow") else pd.DataFrame(),
    }

def save_analog_financials(analog: dict):
    ticker = analog["ticker"]
    year = analog["year"]
    out_path = DATA_DIR / f"{ticker}_{year}.csv"

    try:
        stmts = fetch_statements_for_ticker(ticker)
        # Combine the three statements (long format) with a source column
        frames = []
        for name, df in stmts.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df["__statement__"] = name
                # yfinance statements index are line items; columns are periods (dates)
                # We'll melt to a tidy format: line_item, period, value
                tidy = df.reset_index().melt(id_vars=df.reset_index().columns[0], var_name="period", value_name="value")
                tidy.rename(columns={df.reset_index().columns[0]: "line_item"}, inplace=True)
                tidy["ticker"] = ticker
                frames.append(tidy)

        if frames:
            final = pd.concat(frames, ignore_index=True)
            final["analog_year"] = year
            final.to_csv(out_path, index=False)
            print(f"  📦 Saved {ticker} ({year}) -> {out_path}")
        else:
            print(f"  ⚠️ No statements available for {ticker}; skipping save.")
    except Exception as e:
        print(f"  ❌ Failed for {ticker} ({year}): {e}")

def fetch_and_store_analogs(analogs: list[dict]):
    print("\nFetching financials for analogs (this may take a minute)...")
    for i, a in enumerate(analogs, 1):
        print(f"[{i}/{len(analogs)}] {a['company']} ({a['ticker']}) - {a['year']}")
        save_analog_financials(a)
        time.sleep(0.7)  # light pacing to avoid throttling

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ticker = input("Enter a company ticker (e.g., NVDA): ").upper().strip()
    print("\n🔎 Pulling current company metrics...")
    metrics = get_company_data(ticker)
    print(f"Fetched: {metrics.get('company_name') or ticker} | Sector: {metrics.get('sector')} | Industry: {metrics.get('industry')}")

    print("\n🧠 Asking LLM for 20 historical analogs (company, ticker, year)...")
    analogs = find_similar_companies(metrics)
    if not analogs or len(analogs) < 5:
        print("❌ Could not get enough analogs. Re-run or adjust the prompt.")
        exit(1)

    print("\n✅ Analogs (first 5 shown):")
    for row in analogs[:5]:
        print(f" - {row['company']} ({row['ticker']}) - {row['year']}")
    print(f"...and {max(0, len(analogs)-5)} more.")

    fetch_and_store_analogs(analogs)

    print("\n🎯 Done.")
    print("• Analog CSVs saved in: data/analogs/")
    print("• Your OpenAI API key belongs in .env as:\n  OPENAI_API_KEY=your_api_key_here")
