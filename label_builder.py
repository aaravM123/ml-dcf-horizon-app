# label_builder.py
from pathlib import Path
import pandas as pd
import numpy as np

from features import build_feature_table
from dcf import DCFInputs, intrinsic_value_from_fcf, market_cap_approx

DATA_DIR = Path("data")
OUT_DIR  = DATA_DIR / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3,5,7,10]

def choose_best_horizon_row(row: pd.Series) -> tuple[int, dict]:
    """Return (label_horizon, errors_by_horizon)."""
    fcf0 = row.get("fcf_last", np.nan)
    print(f"    FCF0: {fcf0}")
    if not np.isfinite(fcf0) or fcf0 <= 0:
        print(f"    ❌ FCF0 invalid: {fcf0}")
        return None, {}
    
    # crude starting growth: prefer fcf CAGR, fall back to revenue CAGR
    g0 = row.get("fcf_cagr_3y")
    if not np.isfinite(g0):
        g0 = row.get("rev_cagr_3y")
    print(f"    Growth rate: {g0}")
    if not np.isfinite(g0):
        print(f"    ❌ Growth rate invalid: {g0}")
        return None, {}

    ticker = row["ticker"]
    year   = int(row["analog_year"])
    print(f"    Getting market cap for {ticker} in {year}...")
    mcap = market_cap_approx(ticker, year)
    print(f"    Market cap: {mcap}")
    if mcap is None or mcap <= 0:
        print(f"    ❌ Market cap invalid: {mcap}")
        return None, {}

    errs = {}
    for H in HORIZONS:
        try:
            iv = intrinsic_value_from_fcf(DCFInputs(
                fcf0=fcf0,
                horizon_years=H,
                growth_initial=float(g0),
            ))
            # value is enterprise-like; we compare rough level to market cap proxy
            # Use absolute percentage error
            err = abs(iv - mcap) / mcap
            errs[H] = float(err)
        except Exception:
            continue

    if not errs:
        return None, {}

    best = min(errs, key=errs.get)
    return best, errs

def build_labeled_dataset() -> pd.DataFrame:
    feats = build_feature_table()
    if feats.empty:
        raise RuntimeError("No features available. Did you run main.py to fetch analog CSVs?")
    print(f"📊 Built feature table with {len(feats)} rows")
    
    labels = []
    for i, (_, row) in enumerate(feats.iterrows()):
        print(f"[{i+1}/{len(feats)}] Processing {row['ticker']} ({row['analog_year']})...")
        label, errs = choose_best_horizon_row(row)
        if label is None:
            print(f"  ❌ Failed to get label for {row['ticker']}")
            continue
        print(f"  ✅ Best horizon: {label} years")
        rec = row.to_dict()
        rec["label_horizon"] = int(label)
        for H in HORIZONS:
            rec[f"ape_{H}y"] = errs.get(H, np.nan)
        labels.append(rec)
    
    df = pd.DataFrame(labels)
    df.to_csv(OUT_DIR / "train_dataset.csv", index=False)
    print(f"✅ Saved labeled dataset -> {OUT_DIR/'train_dataset.csv'}  (rows={len(df)})")
    return df

if __name__ == "__main__":
    build_labeled_dataset()
