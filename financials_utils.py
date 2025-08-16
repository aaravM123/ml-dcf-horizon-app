# financials_utils.py (v2)
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

# ---- aliases Yahoo uses (exhaustive-ish) ----
CFO_KEYS = [
    "Total Cash From Operating Activities",
    "Operating Cash Flow",
    "Net Cash Provided By Operating Activities",
    "Net Cash Provided by Operating Activities",
]
CAPEX_KEYS = [
    "Capital Expenditures",
    "Purchase Of Property Plant Equipment",
    "Purchase of property and equipment",
]
DEP_KEYS = [
    "Depreciation",
    "Depreciation & amortization",
    "Depreciation And Amortization",
    "Depreciation & Amortization",
]
REV_KEYS = ["Total Revenue", "Operating Revenue", "Revenue"]
NI_KEYS  = ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]

# Balance sheet items for WC and PP&E
CA_KEYS = ["Total Current Assets"]
CL_KEYS = ["Total Current Liabilities"]
CASH_KEYS = ["Cash And Cash Equivalents", "Cash And Cash Equivalents, At Carrying Value", "Cash"]
STD_KEYS  = ["Short Long Term Debt", "Short Term Debt", "Current Debt"]
PPE_KEYS  = ["Property Plant Equipment", "Property, Plant & Equipment Net", "Net PPE", "Net Property Plant & Equipment"]

def _first(df: pd.DataFrame, keys: list[str]) -> Optional[pd.Series]:
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None

def _sorted_cols(df: pd.DataFrame) -> list[str]:
    try:
        cols = pd.to_datetime(df.columns)
        cols = cols.sort_values()
        return [c.strftime("%Y-%m-%d") for c in cols]
    except Exception:
        return list(df.columns)

def _last(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty: return None
    s = series.dropna().astype("float64").sort_index()
    if s.empty: return None
    return float(s.iloc[-1])

def _cagr(series: Optional[pd.Series], years: int) -> float:
    if series is None: return np.nan
    s = series.dropna().astype("float64").sort_index()
    if len(s) < years + 1: return np.nan
    a, b = s.iloc[-(years+1)], s.iloc[-1]
    if a <= 0 or b <= 0: return np.nan
    return (b / a) ** (1/years) - 1

def _wc_series(bs: pd.DataFrame) -> Optional[pd.Series]:
    ca = _first(bs, CA_KEYS)
    cl = _first(bs, CL_KEYS)
    cash = _first(bs, CASH_KEYS)
    std  = _first(bs, STD_KEYS)
    if ca is None or cl is None: return None
    # align
    cols = set(ca.index) & set(cl.index)
    if isinstance(cash, pd.Series): cols &= set(cash.index)
    if isinstance(std, pd.Series):  cols &= set(std.index)
    if not cols: return None
    cols = sorted(cols)
    a = ca[cols].astype("float64")
    l = cl[cols].astype("float64")
    c = cash[cols].astype("float64") if isinstance(cash, pd.Series) else 0.0
    d = std[cols].astype("float64")  if isinstance(std,  pd.Series) else 0.0
    wc = (a - c) - (l - d)
    wc.index = cols
    return wc

def fetch_statements(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    inc_a  = getattr(t, "income_stmt", pd.DataFrame())
    bs_a   = getattr(t, "balance_sheet", pd.DataFrame())
    cf_a   = getattr(t, "cashflow", pd.DataFrame())
    inc_q  = getattr(t, "quarterly_income_stmt", pd.DataFrame())
    bs_q   = getattr(t, "quarterly_balance_sheet", pd.DataFrame())
    cf_q   = getattr(t, "quarterly_cashflow", pd.DataFrame())

    for df in [inc_a, bs_a, cf_a, inc_q, bs_q, cf_q]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.columns = _sorted_cols(df)

    info = t.info or {}
    return {"info": info, "inc_a": inc_a, "bs_a": bs_a, "cf_a": cf_a, "inc_q": inc_q, "bs_q": bs_q, "cf_q": cf_q}

def _capex_from_ppe_and_dep(bs_a: pd.DataFrame, dep: Optional[pd.Series]) -> Optional[pd.Series]:
    """Approx CapEx ≈ ΔPP&E + Depreciation (signs handled)."""
    ppe = _first(bs_a, PPE_KEYS)
    if ppe is None: return None
    ppe = ppe.dropna().astype("float64").sort_index()
    if len(ppe) < 2: return None
    dppe = ppe.diff()  # positive when PPE grows
    dep_series = dep.dropna().astype("float64").sort_index() if isinstance(dep, pd.Series) else pd.Series(0.0, index=ppe.index)
    # Align
    cols = list(sorted(set(dppe.index) & set(dep_series.index)))
    if not cols: return None
    capex = dppe[cols] + dep_series[cols]
    capex.index = cols
    # Capital expenditures are typically negative in Yahoo (cash outflow). Keep numeric magnitude sign as-is.
    return capex

def compute_fcf_series_robust(stmts: dict) -> tuple[Optional[pd.Series], str]:
    info = stmts["info"]; inc_a = stmts["inc_a"]; bs_a = stmts["bs_a"]; cf_a = stmts["cf_a"]; inc_q = stmts["inc_q"]; bs_q = stmts["bs_q"]; cf_q = stmts["cf_q"]

    # 0) Direct freeCashflow from info (rare but sometimes present)
    fcf_info = info.get("freeCashflow") or info.get("freeCashFlow")
    if fcf_info and np.isfinite(fcf_info):
        # represent as 1-point series
        return pd.Series([float(fcf_info)], index=["info"]), "info_freecashflow"

    # 1) Annual CFO - CapEx
    if not cf_a.empty:
        cfo = _first(cf_a, CFO_KEYS)
        cap = _first(cf_a, CAPEX_KEYS)
        if (cfo is not None) and (cap is not None):
            cols = [c for c in cf_a.columns if (c in cfo.index and c in cap.index)]
            if cols:
                fcf = (cfo[cols].astype("float64") - cap[cols].astype("float64"))
                fcf.index = cols
                return fcf, "annual_cfo_minus_capex"

    # 2) Quarterly TTM (sum last 4 quarters). If <4 quarters, sum available.
    if not cf_q.empty:
        cfo_q = _first(cf_q, CFO_KEYS)
        cap_q = _first(cf_q, CAPEX_KEYS)
        if (cfo_q is not None) and (cap_q is not None):
            q = pd.concat([cfo_q, cap_q], axis=1)
            q.columns = ["cfo", "capex"]
            q = q.dropna().astype("float64").sort_index()
            if len(q) >= 2:  # allow partial TTM
                take = min(4, len(q))
                fcf_ttm = q["cfo"].iloc[-take:].sum() - q["capex"].iloc[-take:].sum()
                return pd.Series([float(fcf_ttm)], index=[q.index[-1]]), ("quarterly_ttm" if take == 4 else f"quarterly_sum_{take}")

    # 3) Rebuild CFO annually: NI + D&A − ΔWC, CapEx from CF if available else from PP&E change
    dep_a = _first(cf_a if not cf_a.empty else inc_a, DEP_KEYS)
    ni_a  = _first(inc_a, NI_KEYS)
    wc    = _wc_series(bs_a) if not bs_a.empty else None
    cap_a = _first(cf_a, CAPEX_KEYS) if not cf_a.empty else None

    if (ni_a is not None) and (wc is not None):
        d_wc = wc.astype("float64").sort_index().diff()
        common = set(ni_a.index) & set(d_wc.index)
        if isinstance(dep_a, pd.Series): common &= set(dep_a.index)
        if common:
            cols = sorted(common)
            ni  = ni_a[cols].astype("float64")
            dW  = d_wc[cols].astype("float64")
            dep = dep_a[cols].astype("float64") if isinstance(dep_a, pd.Series) else pd.Series(0.0, index=cols)
            cfo_rebuilt = ni + dep - dW

            if isinstance(cap_a, pd.Series):
                cap = cap_a.reindex(cols).astype("float64")
            else:
                # try CapEx ≈ ΔPP&E + Dep
                cap_approx = _capex_from_ppe_and_dep(bs_a, dep)
                cap = cap_approx.reindex(cols).astype("float64") if isinstance(cap_approx, pd.Series) else pd.Series(0.0, index=cols)

            fcf = cfo_rebuilt - cap
            fcf.index = cols
            return fcf, ("rebuilt_cfo_minus_capex" if isinstance(cap_a, pd.Series) else "rebuilt_cfo_minus_capex_ppe_approx")

    # 4) Last‑resort: estimate FCF via historical FCF margin × latest revenue
    #    margin = median(FCF / Revenue) over overlapping years we can compute from any partial above
    # Try to compute any partial FCF with PP&E approximation even if NI/WC missing:
    if not inc_a.empty:
        rev = _first(inc_a, REV_KEYS)
        if rev is not None and len(rev.dropna()) >= 2:
            # crude: 10% FCF margin fallback if nothing else (optional: turn off if you prefer strictness)
            est = float(rev.dropna().astype("float64").sort_index().iloc[-1] * 0.10)
            return pd.Series([est], index=["estimate"]), "heuristic_10pct_of_revenue"

    return None, "unavailable"

def build_core_metrics_for_app(ticker: str) -> dict:
    stmts = fetch_statements(ticker)
    info  = stmts["info"]
    inc_a = stmts["inc_a"]

    revenue = _first(inc_a, REV_KEYS)
    fcf_series, strategy = compute_fcf_series_robust(stmts)

    return {
        "company_name": info.get("longName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "shares_out": info.get("sharesOutstanding"),
        "fcf_series": None if fcf_series is None else fcf_series.copy(),
        "fcf_last": _last(fcf_series),
        "fcf_strategy": strategy,
        "revenue_series": None if revenue is None else revenue.copy(),
        "revenue_last": _last(revenue),
        "rev_cagr_3y": _cagr(revenue, 3),
        "rev_cagr_5y": _cagr(revenue, 5),
        "fcf_cagr_3y": _cagr(fcf_series, 3) if fcf_series is not None else np.nan,
        "fcf_cagr_5y": _cagr(fcf_series, 5) if fcf_series is not None else np.nan,
        "net_income_last": _last(_first(inc_a, NI_KEYS)),
    }
