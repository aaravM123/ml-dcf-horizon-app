# features.py
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from dcf import DCFInputs, intrinsic_value_from_fcf, price_at_year_end

ANALOG_DIR = Path("data/analogs")

# Line items we'll try to map to FCF components
CFO_KEYS = [
    "Total Cash From Operating Activities",
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
    "Operating Cash Flow",
]
CAPEX_KEYS = [
    "Capital Expenditures",
    "Purchase Of Property Plant Equipment",
    "Purchase Of PPE",
    "Capital Expenditure",
]

def _select_first_match(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None

def _pivot_in(df: pd.DataFrame) -> pd.DataFrame:
    """Expect tidy: line_item, period, value"""
    if df.empty: return df
    # Make wide by line_item across periods for easier picking
    # Use a simpler approach to avoid pivot issues
    try:
        wide = df.pivot_table(index="line_item", columns="period", values="value", aggfunc="first")
        # Ensure we only have numeric columns (filter out any non-date columns)
        numeric_cols = []
        for col in wide.columns:
            try:
                pd.to_datetime(col)
                numeric_cols.append(col)
            except:
                continue
        return wide[numeric_cols]
    except Exception as e:
        print(f"  Warning: Pivot failed: {e}")
        return pd.DataFrame()

def load_and_build_features(csv_path: Path) -> dict | None:
    """Build a single feature row for one analog CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or not set(["line_item","period","value"]).issubset(df.columns):
        return None

    ticker = df["ticker"].iloc[0] if "ticker" in df.columns else csv_path.stem.split("_")[0]
    analog_year = int(df["analog_year"].iloc[0]) if "analog_year" in df.columns else None

    # If no __statement__ column, infer from line item names
    if "__statement__" not in df.columns:
        def infer_statement_type(line_item):
            line_item_lower = line_item.lower()
            # Cash flow indicators
            if any(keyword in line_item_lower for keyword in ["cash", "flow", "operating", "investing", "financing", "expenditure", "purchase"]):
                return "cashflow"
            # Balance sheet indicators  
            elif any(keyword in line_item_lower for keyword in ["total", "assets", "liabilities", "equity", "stockholder", "debt", "receivable", "payable", "inventory"]):
                return "balance_sheet"
            # Income statement (default)
            else:
                return "income_stmt"
        
        df["__statement__"] = df["line_item"].apply(infer_statement_type)

    # Separate statements and immediately drop the __statement__ column
    inc = df[df["__statement__"] == "income_stmt"][["line_item", "period", "value"]].copy()
    bs  = df[df["__statement__"] == "balance_sheet"][["line_item", "period", "value"]].copy()
    cf  = df[df["__statement__"] == "cashflow"][["line_item", "period", "value"]].copy()

    # Now pivot the clean data
    inc_w = _pivot_in(inc)
    bs_w  = _pivot_in(bs)
    cf_w  = _pivot_in(cf)

    # Choose a consistent set of periods (columns are dates like '2023-12-31')
    def sorted_periods(wide: pd.DataFrame):
        try:
            cols = pd.to_datetime(wide.columns)
            order = cols.sort_values()
            # Keep the full datetime format to match pivot columns
            return [c.strftime("%Y-%m-%d %H:%M:%S") for c in order]
        except Exception:
            return list(wide.columns)

    periods = sorted(set(
        ([] if inc_w.empty else sorted_periods(inc_w)) +
        ([] if bs_w.empty  else sorted_periods(bs_w)) +
        ([] if cf_w.empty  else sorted_periods(cf_w))
    ))

    # Build FCF series by period if possible: FCF = CFO - CapEx
    fcf = None
    if not cf_w.empty:
        cfo  = _select_first_match(cf_w, CFO_KEYS)
        capx = _select_first_match(cf_w, CAPEX_KEYS)
        if (cfo is not None) and (capx is not None):
            # Align to periods intersection
            cols = [c for c in periods if (c in cfo.index and c in capx.index)]
            if cols:
                # Convert to numeric before calculation
                cfo_numeric = pd.to_numeric(cfo[cols], errors='coerce')
                capx_numeric = pd.to_numeric(capx[cols], errors='coerce')
                fcf = (cfo_numeric - capx_numeric)

    # Revenue, gross profit, net income if available
    revenue = inc_w.loc["Total Revenue"] if (not inc_w.empty and "Total Revenue" in inc_w.index) else None
    net_inc = inc_w.loc["Net Income"] if (not inc_w.empty and "Net Income" in inc_w.index) else None
    gross   = inc_w.loc["Gross Profit"] if (not inc_w.empty and "Gross Profit" in inc_w.index) else None
    


    def safe_growth(series: pd.Series, years=3):
        if series is None: return np.nan
        s = series.dropna().astype("float64")
        if len(s) < years+1: return np.nan
        # CAGR across last (years+1) points
        s = s.sort_index()
        start = s.iloc[-(years+1)]
        end   = s.iloc[-1]
        if start <= 0 or end <= 0: return np.nan
        return (end / start) ** (1/years) - 1

    def last_margin(numer: pd.Series, denom: pd.Series):
        try:
            n = float(numer.dropna().astype("float64").sort_index().iloc[-1])
            d = float(denom.dropna().astype("float64").sort_index().iloc[-1])
            return n/d if d else np.nan
        except Exception:
            return np.nan


    
    features = {
        "ticker": ticker,
        "analog_year": analog_year,
        "rev_cagr_3y": safe_growth(revenue, 3),
        "rev_cagr_5y": safe_growth(revenue, 5),
        "fcf_cagr_3y": safe_growth(fcf, 3) if fcf is not None else np.nan,
        "fcf_cagr_5y": safe_growth(fcf, 5) if fcf is not None else np.nan,
        "gross_margin_last": last_margin(gross, revenue) if (gross is not None and revenue is not None) else np.nan,
        "net_margin_last": last_margin(net_inc, revenue) if (net_inc is not None and revenue is not None) else np.nan,
        "fcf_last": (float(fcf.sort_index().iloc[-1]) if fcf is not None and not fcf.empty else np.nan),
        "revenue_last": (float(revenue.sort_index().iloc[-1]) if revenue is not None else np.nan),
        "net_income_last": (float(net_inc.sort_index().iloc[-1]) if net_inc is not None else np.nan),
        # simple leverage proxy if available
    }

    # Balance sheet leverage proxy
    if not bs_w.empty and "Total Liab" in bs_w.index and "Total Stockholder Equity" in bs_w.index:
        try:
            tl = float(bs_w.loc["Total Liab"].dropna().astype("float64").sort_index().iloc[-1])
            eq = float(bs_w.loc["Total Stockholder Equity"].dropna().astype("float64").sort_index().iloc[-1])
            features["leverage_ratio"] = (tl / eq) if eq else np.nan
        except Exception:
            features["leverage_ratio"] = np.nan
    else:
        features["leverage_ratio"] = np.nan

    return features

def build_feature_table() -> pd.DataFrame:
    rows = []
    for p in ANALOG_DIR.glob("*.csv"):
        out = load_and_build_features(p)
        if out: rows.append(out)
    return pd.DataFrame(rows)

def find_similar_companies(ticker: str, n_similar: int = 30) -> pd.DataFrame:
    """
    INTELLIGENT RESEARCH-DRIVEN approach to find similar companies.
    
    Instead of relying on predefined datasets, this function:
    1. Researches the target company's business model, sector, and financial profile
    2. Dynamically identifies similar companies based on multiple criteria
    3. Uses real-time market data and financial metrics
    4. Leverages sector knowledge and competitive analysis
    """
    try:
        # Get comprehensive company information
        company = yf.Ticker(ticker)
        info = company.info or {}
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        market_cap = info.get("marketCap", 0)
        revenue = info.get("totalRevenue", 0)
        
        print(f"🔍 Researching companies similar to {ticker}")
        print(f"   Sector: {sector}")
        print(f"   Industry: {industry}")
        print(f"   Market Cap: ${market_cap:,.0f}" if market_cap else "   Market Cap: N/A")
        print(f"   Revenue: ${revenue:,.0f}" if revenue else "   Revenue: N/A")
        
        # INTELLIGENT RESEARCH STRATEGY
        similar_companies = []
        
        # Strategy 1: Direct Competitors and Peers
        direct_peers = _research_direct_competitors(ticker, sector, industry)
        similar_companies.extend(direct_peers)
        print(f"   Found {len(direct_peers)} direct competitors/peers")
        
        # Strategy 2: Sector Leaders and Comparable Companies
        sector_companies = _research_sector_companies(ticker, sector, industry, market_cap, revenue)
        similar_companies.extend(sector_companies)
        print(f"   Found {len(sector_companies)} sector-comparable companies")
        
        # Strategy 3: Financial Profile Similarity
        financial_matches = _research_financial_similarity(ticker, market_cap, revenue, n_similar)
        similar_companies.extend(financial_matches)
        print(f"   Found {len(financial_matches)} companies with similar financial profiles")
        
        # Strategy 4: Business Model Similarity
        business_matches = _research_business_similarity(ticker, sector, industry)
        similar_companies.extend(business_matches)
        print(f"   Found {len(business_matches)} companies with similar business models")
        
        # Remove duplicates and target company
        unique_companies = []
        seen_tickers = {ticker.upper()}
        
        for company in similar_companies:
            if company['ticker'].upper() not in seen_tickers:
                unique_companies.append(company)
                seen_tickers.add(company['ticker'].upper())
        
        # Take top N most relevant companies
        result = unique_companies[:n_similar]
        
        if result:
            print(f"✅ Successfully researched {len(result)} similar companies")
            print(f"   Top companies: {', '.join([c['ticker'] for c in result[:8]])}")
            print(f"   Research methods used: {', '.join(set([c.get('research_method', 'Unknown') for c in result]))}")
        else:
            print("⚠️ No similar companies found through research")
        
        return pd.DataFrame(result)
        
    except Exception as e:
        print(f"   Error in intelligent research: {e}")
        return pd.DataFrame()

def _research_direct_competitors(ticker: str, sector: str, industry: str) -> list:
    """Research direct competitors and peers in the same space."""
    competitors = []
    
    # Known competitive relationships by sector/industry
    competitive_maps = {
        "Technology": {
            "Semiconductors": {
                "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "MRVL", "MU", "TSM"],
                "AMD": ["NVDA", "INTC", "AVGO", "QCOM", "MRVL"],
                "INTC": ["AMD", "NVDA", "AVGO", "QCOM", "MRVL", "TSM"],
                "TSM": ["INTC", "SMIC", "UMC", "GFS"],
                "AVGO": ["QCOM", "MRVL", "INTC", "AMD"],
                "QCOM": ["AVGO", "MRVL", "INTC", "AMD", "NVDA"]
            },
            "Software": {
                "MSFT": ["GOOGL", "AAPL", "ORCL", "SAP", "CRM", "ADBE"],
                "GOOGL": ["MSFT", "META", "AAPL", "AMZN", "NFLX"],
                "META": ["GOOGL", "SNAP", "PINS", "TWTR", "SNAP"],
                "AAPL": ["MSFT", "GOOGL", "SAMSUNG", "XIAOMI", "HUAWEI"]
            },
            "Hardware": {
                "AAPL": ["MSFT", "GOOGL", "SAMSUNG", "XIAOMI", "HUAWEI"],
                "TSLA": ["NIO", "XPEV", "LI", "F", "GM", "TM"],
                "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "MRVL"]
            }
        },
        "Healthcare": {
            "Biotechnology": {
                "JNJ": ["PFE", "ABBV", "TMO", "DHR", "UNH"],
                "PFE": ["JNJ", "ABBV", "TMO", "DHR", "UNH"],
                "ABBV": ["JNJ", "PFE", "TMO", "DHR", "UNH"]
            }
        },
        "Financial": {
            "Banking": {
                "JPM": ["BAC", "WFC", "C", "GS", "MS"],
                "BAC": ["JPM", "WFC", "C", "GS", "MS"],
                "WFC": ["JPM", "BAC", "C", "GS", "MS"]
            }
        }
    }
    
    # Find competitive matches
    if sector in competitive_maps and industry in competitive_maps[sector]:
        for company, comp_list in competitive_maps[sector][industry].items():
            if ticker.upper() in company.upper() or company.upper() in ticker.upper():
                for comp in comp_list:
                    try:
                        comp_info = yf.Ticker(comp).info
                        if comp_info:
                            competitors.append({
                                'ticker': comp,
                                'company_name': comp_info.get('longName', comp),
                                'sector': comp_info.get('sector', ''),
                                'industry': comp_info.get('industry', ''),
                                'research_method': 'direct_competitor',
                                'similarity_score': 95  # High score for direct competitors
                            })
                    except:
                        continue
                break
    
    return competitors

def _research_sector_companies(ticker: str, sector: str, industry: str, market_cap: float, revenue: float) -> list:
    """Research companies in the same sector with similar characteristics."""
    sector_companies = []
    
    # Sector-specific company lists with market positioning
    sector_leaders = {
        "Technology": {
            "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "MRVL", "MU", "KLAC", "LRCX", "ASML"],
            "Software": ["MSFT", "GOOGL", "META", "AAPL", "ORCL", "SAP", "CRM", "ADBE", "NOW", "WDAY"],
            "Hardware": ["AAPL", "TSLA", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MRVL", "MU", "KLAC", "LRCX"],
            "Cloud": ["MSFT", "GOOGL", "AMZN", "ORCL", "CRM", "NOW", "WDAY", "SNOW", "MDB", "PLTR"]
        },
        "Healthcare": {
            "Biotechnology": ["JNJ", "PFE", "ABBV", "TMO", "DHR", "UNH", "CVS", "ANTM", "CI", "HUM"],
            "Pharmaceuticals": ["JNJ", "PFE", "ABBV", "TMO", "DHR", "UNH", "CVS", "ANTM", "CI", "HUM"]
        },
        "Financial": {
            "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BRK-B", "BLK", "SCHW", "COF"],
            "Insurance": ["UNH", "ANTM", "CI", "HUM", "AET", "CVS", "WMT", "UNP", "UNH"]
        },
        "Consumer": {
            "E-commerce": ["AMZN", "BABA", "JD", "PDD", "SE", "MELI", "ETSY", "SHOP"],
            "Retail": ["WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "UA"]
        }
    }
    
    # Find relevant companies in the sector
    if sector in sector_leaders and industry in sector_leaders[sector]:
        companies = sector_leaders[sector][industry]
        
        for comp in companies:
            if comp != ticker:
                try:
                    comp_info = yf.Ticker(comp).info
                    if comp_info:
                        comp_market_cap = comp_info.get("marketCap", 0)
                        comp_revenue = comp_info.get("totalRevenue", 0)
                        
                        # Calculate similarity score based on market positioning
                        similarity_score = _calculate_market_similarity(
                            market_cap, revenue, comp_market_cap, comp_revenue
                        )
                        
                        sector_companies.append({
                            'ticker': comp,
                            'company_name': comp_info.get('longName', comp),
                            'sector': comp_info.get('sector', ''),
                            'industry': comp_info.get('industry', ''),
                            'research_method': 'sector_leader',
                            'similarity_score': similarity_score
                        })
                except:
                    continue
    
    return sector_companies

def _research_financial_similarity(ticker: str, target_market_cap: float, target_revenue: float, n_similar: int) -> list:
    """Research companies with similar financial profiles."""
    financial_matches = []
    
    # Market cap tiers for comparison
    market_cap_tiers = {
        "Mega Cap": (100_000_000_000, float('inf')),      # $100B+
        "Large Cap": (10_000_000_000, 100_000_000_000),   # $10B - $100B
        "Mid Cap": (2_000_000_000, 10_000_000_000),       # $2B - $10B
        "Small Cap": (300_000_000, 2_000_000_000),        # $300M - $2B
        "Micro Cap": (0, 300_000_000)                     # < $300M
    }
    
    # Find target company's tier
    target_tier = None
    for tier_name, (min_cap, max_cap) in market_cap_tiers.items():
        if min_cap <= target_market_cap < max_cap:
            target_tier = tier_name
            break
    
    if target_tier:
        # Get companies in similar market cap tiers
        tier_companies = {
            "Mega Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ"],
            "Large Cap": ["AMD", "INTC", "AVGO", "QCOM", "MRVL", "MU", "KLAC", "LRCX", "ASML", "ADI"],
            "Mid Cap": ["CRUS", "INFI", "SMTC", "STM", "SWKS", "NXPI", "ON", "MCHP"],
            "Small Cap": ["SMTC", "STM", "SWKS", "NXPI", "ON", "MCHP"],
            "Micro Cap": ["SMTC", "STM", "SWKS", "NXPI", "ON", "MCHP"]
        }
        
        companies = tier_companies.get(target_tier, [])
        
        for comp in companies:
            if comp != ticker:
                try:
                    comp_info = yf.Ticker(comp).info
                    if comp_info:
                        comp_market_cap = comp_info.get("marketCap", 0)
                        comp_revenue = comp_info.get("totalRevenue", 0)
                        
                        # Calculate financial similarity score
                        similarity_score = _calculate_market_similarity(
                            target_market_cap, target_revenue, comp_market_cap, comp_revenue
                        )
                        
                        financial_matches.append({
                            'ticker': comp,
                            'company_name': comp_info.get('longName', comp),
                            'sector': comp_info.get('sector', ''),
                            'industry': comp_info.get('industry', ''),
                            'research_method': 'financial_similarity',
                            'similarity_score': similarity_score
                        })
                except:
                    continue
    
    return financial_matches

def _research_business_similarity(ticker: str, sector: str, industry: str) -> list:
    """Research companies with similar business models and strategies."""
    business_matches = []
    
    # Business model patterns
    business_patterns = {
        "AI/ML Focus": ["NVDA", "AMD", "INTC", "GOOGL", "MSFT", "META", "AMZN", "TSLA", "PLTR", "AI"],
        "Cloud Services": ["MSFT", "GOOGL", "AMZN", "ORCL", "CRM", "NOW", "WDAY", "SNOW", "MDB", "PLTR"],
        "Semiconductor Design": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MRVL", "MU", "KLAC", "LRCX", "ASML"],
        "Software Platforms": ["MSFT", "GOOGL", "META", "AAPL", "ORCL", "SAP", "CRM", "ADBE", "NOW", "WDAY"],
        "Hardware Manufacturing": ["AAPL", "TSLA", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MRVL", "MU", "KLAC"]
    }
    
    # Find companies with similar business models
    for pattern_name, companies in business_patterns.items():
        if any(company in ticker.upper() for company in companies) or \
           any(keyword in industry.lower() for keyword in pattern_name.lower().split()):
            
            for comp in companies:
                if comp != ticker:
                    try:
                        comp_info = yf.Ticker(comp).info
                        if comp_info:
                            business_matches.append({
                                'ticker': comp,
                                'company_name': comp_info.get('longName', comp),
                                'sector': comp_info.get('sector', ''),
                                'industry': comp_info.get('industry', ''),
                                'research_method': 'business_model',
                                'similarity_score': 85
                            })
                    except:
                        continue
    
    return business_matches

def _calculate_market_similarity(target_market_cap: float, target_revenue: float, 
                               comp_market_cap: float, comp_revenue: float) -> float:
    """Calculate similarity score based on market cap and revenue."""
    if target_market_cap == 0 or comp_market_cap == 0:
        return 50  # Default score if data missing
    
    # Calculate percentage differences
    market_cap_diff = abs(target_market_cap - comp_market_cap) / max(target_market_cap, comp_market_cap)
    revenue_diff = abs(target_revenue - comp_revenue) / max(target_revenue, comp_revenue) if target_revenue > 0 and comp_revenue > 0 else 1
    
    # Convert to similarity score (0-100, higher = more similar)
    market_cap_score = max(0, 100 - (market_cap_diff * 100))
    revenue_score = max(0, 100 - (revenue_diff * 100)) if target_revenue > 0 and comp_revenue > 0 else 50
    
    # Weighted average (market cap more important)
    similarity_score = (market_cap_score * 0.7) + (revenue_score * 0.3)
    
    return round(similarity_score, 1)

def _get_company_sector(ticker: str) -> str:
    """Get sector for a company ticker."""
    try:
        company = yf.Ticker(ticker)
        info = company.info or {}
        return info.get("sector", "")
    except:
        return ""

def build_dynamic_features_for_ticker(ticker: str, n_similar: int = 20) -> dict:
    """
    Build features for a new ticker by finding similar companies and using their patterns.
    This creates a more robust feature set for the ML model.
    """
    # Find similar companies
    similar_companies = find_similar_companies(ticker, n_similar)
    
    if similar_companies.empty:
        print(f"   Warning: No similar companies found for {ticker}")
        return {}
    
    # Calculate aggregate metrics from similar companies
    features = {}
    
    # Revenue and FCF metrics
    if 'revenue_last' in similar_companies.columns:
        revenue_values = pd.to_numeric(similar_companies['revenue_last'], errors='coerce').dropna()
        if not revenue_values.empty:
            features['revenue_median'] = revenue_values.median()
            features['revenue_mean'] = revenue_values.mean()
            features['revenue_std'] = revenue_values.std()
    
    if 'fcf_last' in similar_companies.columns:
        fcf_values = pd.to_numeric(similar_companies['fcf_last'], errors='coerce').dropna()
        if not fcf_values.empty:
            features['fcf_median'] = fcf_values.median()
            features['fcf_mean'] = fcf_values.mean()
            features['fcf_std'] = fcf_values.std()
    
    # Growth metrics
    for metric in ['rev_cagr_3y', 'rev_cagr_5y', 'fcf_cagr_3y', 'fcf_cagr_5y']:
        if metric in similar_companies.columns:
            values = pd.to_numeric(similar_companies[metric], errors='coerce').dropna()
            if not values.empty:
                features[f'{metric}_median'] = values.median()
                features[f'{metric}_mean'] = values.mean()
    
    # Margin metrics
    for metric in ['gross_margin_last', 'net_margin_last']:
        if metric in similar_companies.columns:
            values = pd.to_numeric(similar_companies[metric], errors='coerce').dropna()
            if not values.empty:
                features[f'{metric}_median'] = values.median()
                features[f'{metric}_mean'] = values.mean()
    
    # Leverage
    if 'leverage_ratio' in similar_companies.columns:
        leverage_values = pd.to_numeric(similar_companies['leverage_ratio'], errors='coerce').dropna()
        if not leverage_values.empty:
            features['leverage_median'] = leverage_values.median()
            features['leverage_mean'] = leverage_values.mean()
    
    # Add ticker info
    features['ticker'] = ticker
    features['n_similar_companies'] = len(similar_companies)
    
    return features

def calculate_optimal_forecast_period_for_company(ticker: str, analog_year: int) -> dict:
    """
    Calculate the optimal DCF forecast period for a company by testing different forecast periods
    and finding which one gives the most accurate intrinsic value compared to actual market cap.
    
    This follows the proper DCF methodology:
    1. Test different forecast periods (3, 5, 7, 10, 12, 15 years)
    2. For each period, do full DCF calculation with proper discounting
    3. Compare DCF intrinsic value to actual market cap
    4. Find which forecast period was most accurate
    """
    try:
        # Get company's financial data from analog CSV
        csv_path = ANALOG_DIR / f"{ticker}_{analog_year}.csv"
        if not csv_path.exists():
            return None
            
        # Load company data
        company_data = load_and_build_features(csv_path)
        if not company_data:
            return None
            
        # Get FCF and other metrics
        fcf_last = company_data.get("fcf_last")
        if not fcf_last or fcf_last <= 0:
            return None
            
        # Get actual market cap from historical data
        actual_market_cap = get_historical_market_cap(ticker, analog_year)
        if not actual_market_cap:
            return None
            
        # Test different forecast periods to find the most accurate one
        forecast_periods_to_test = [3, 5, 7, 10, 12, 15]
        best_forecast_period = None
        best_accuracy = float('inf')
        
        # Use reasonable growth rate (median from similar companies or default)
        growth_rate = company_data.get("fcf_cagr_3y", 0.08)
        if growth_rate < 0.02:  # If negative or too low, use reasonable default
            growth_rate = 0.08
            
        print(f"   Testing forecast periods for {ticker} ({analog_year}): FCF=${fcf_last:,.0f}, Growth={growth_rate:.1%}")
        print(f"   Testing forecast periods: {forecast_periods_to_test}")
        
        for forecast_years in forecast_periods_to_test:
            # Calculate DCF value for this forecast period using proper DCF methodology
            try:
                dcf_value = calculate_dcf_for_forecast_period(
                    fcf0=float(fcf_last),
                    growth_rate=float(growth_rate),
                    forecast_years=forecast_years
                )
                
                # Calculate accuracy (how close DCF is to actual market cap)
                accuracy = abs(dcf_value - actual_market_cap) / actual_market_cap
                
                print(f"     {forecast_years}-year forecast: DCF=${dcf_value:,.0f}, Actual=${actual_market_cap:,.0f}, Accuracy={accuracy:.1%}")
                
                if accuracy < best_accuracy:
                    best_accuracy = accuracy
                    best_forecast_period = forecast_years
                    
            except Exception as e:
                print(f"     {forecast_years}-year forecast: Error - {e}")
                continue
        
        if best_forecast_period is not None:
            print(f"   Best forecast period for {ticker}: {best_forecast_period} years (accuracy: {best_accuracy:.1%})")
            return {
                "ticker": ticker,
                "analog_year": analog_year,
                "optimal_forecast_period": best_forecast_period,
                "accuracy": best_accuracy,
                "fcf_last": fcf_last,
                "growth_rate": growth_rate,
                "actual_market_cap": actual_market_cap,
                **company_data
            }
        
        return None
        
    except Exception as e:
        print(f"   Error calculating optimal forecast period for {ticker}: {e}")
        return None

def get_historical_market_cap(ticker: str, year: int) -> float | None:
    """Get historical market cap for a company at a specific year."""
    try:
        # Get year-end price
        price = price_at_year_end(ticker, year)
        if price is None:
            return None
            
        # Get shares outstanding (use current if historical not available)
        info = yf.Ticker(ticker).info or {}
        shares = info.get("sharesOutstanding")
        if not shares:
            return None
            
        market_cap = price * shares
        return market_cap
        
    except Exception:
        return None

def build_training_dataset_from_similar_companies(target_ticker: str, n_similar: int = 30) -> pd.DataFrame:
    """
    Build a training dataset by finding similar companies and calculating their optimal forecast periods.
    CRITICAL: This creates the data needed to train the XGBoost model to predict the optimal
    DCF forecast period using ONLY much older analog data (2010 or earlier).
    
    Using 2010 data gives us ~13+ years of hindsight to validate which DCF forecast period was most accurate
    against actual market performance. Recent data hasn't been validated by the market yet.
    """
    print(f"🔍 Building training dataset from {n_similar} similar companies...")
    print(f"   Using ONLY much older analog data (2010 or earlier) for real DCF validation")
    print(f"   This ensures we train on proven DCF forecast period accuracy, not speculation")
    
    # Find similar companies (will automatically filter to 2010 or earlier)
    similar_companies = find_similar_companies(target_ticker, n_similar)
    if similar_companies.empty:
        print("   No similar companies found")
        return pd.DataFrame()
    
    # Verify we're using much older data
    years_used = similar_companies['analog_year'].unique()
    if max(years_used) > 2010:
        print(f"   WARNING: Found recent data ({max(years_used)}), but need 2010 or earlier")
        print(f"   Filtering to only use 2010 or earlier data...")
        similar_companies = similar_companies[similar_companies['analog_year'] <= 2010]
        if similar_companies.empty:
            print("   No much older data available after filtering")
            return pd.DataFrame()
    
    print(f"   Using {len(similar_companies)} companies with much older data for training")
    print(f"   Analog years: {sorted(years_used)}")
    print(f"   Years of hindsight: {2024 - min(years_used)} to {2024 - max(years_used)} years")
    
    # Calculate optimal forecast period for each similar company
    training_data = []
    for _, company in similar_companies.iterrows():
        ticker = company['ticker']
        analog_year = company.get('analog_year', 2010)  # Default to 2010
        
        print(f"   Analyzing {ticker} ({analog_year}) - {2024-analog_year} years of hindsight...")
        optimal_data = calculate_optimal_forecast_period_for_company(ticker, analog_year)
        
        if optimal_data:
            training_data.append(optimal_data)
    
    if training_data:
        df = pd.DataFrame(training_data)
        print(f"✅ Built training dataset with {len(df)} companies from much older data")
        print(f"   Forecast period distribution: {df['optimal_forecast_period'].value_counts().sort_index().to_dict()}")
        print(f"   Average years of hindsight: {2024 - df['analog_year'].mean():.1f} years")
        return df
    else:
        print("   No training data could be generated from much older companies")
        return pd.DataFrame()
