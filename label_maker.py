# ============================================
# ML-Powered DCF Horizon Labeler
# ============================================

import pandas as pd
import os

# ----- CONFIG -----
INPUT_PATH = "data/company_financials.csv"
OUTPUT_PATH = "data/labeled_company_financials.csv"
DISCOUNT_RATE = 0.10
GROWTH_RATE = 0.05
HORIZON_RANGE = range(1, 51)  # From 1 to 50 years

# ----- DCF CALC -----
def discounted_cash_flow(fcf, growth_rate, discount_rate, years):
    total_value = 0
    for t in range(1, years + 1):
        future_fcf = fcf * ((1 + growth_rate) ** t)
        discounted = future_fcf / ((1 + discount_rate) ** t)
        total_value += discounted
    return total_value

def choose_best_dcf_window(fcf, market_cap):
    errors = {}
    for window in HORIZON_RANGE:
        estimated_value = discounted_cash_flow(fcf, GROWTH_RATE, DISCOUNT_RATE, window)
        error = abs(estimated_value - market_cap)
        errors[window] = error
    return min(errors, key=errors.get)

# ----- MAIN SCRIPT -----
def label_companies():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: File not found — {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    labels = []
    skipped = 0

    for i, row in df.iterrows():
        fcf = row.get("fcf")
        market_cap = row.get("market_cap")

        if pd.notnull(fcf) and pd.notnull(market_cap):
            label = choose_best_dcf_window(fcf, market_cap)
            print(f"✅ {row['ticker']}: Best match is {label}-year DCF horizon")
        else:
            label = None
            skipped += 1
            print(f"⚠️  {row['ticker']}: Skipped (missing FCF or Market Cap)")

        labels.append(label)

    df["optimal_dcf_years"] = labels
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved labeled file to: {OUTPUT_PATH}")
    print(f"🧠 Companies labeled: {len(df) - skipped}, Skipped: {skipped}")

if __name__ == "__main__":
    label_companies()
