# dcf.py
import os
from dataclasses import dataclass
from typing import List
import numpy as np
import yfinance as yf
import pandas as pd

# Standard DCF parameters - same for all companies
DISCOUNT_RATE = float(os.getenv("DISCOUNT_RATE", "0.09"))  # 9% standard discount rate
TERMINAL_GROWTH = float(os.getenv("TERMINAL_GROWTH", "0.05"))  # 5% terminal growth

@dataclass
class DCFInputs:
    fcf0: float                 # last actual FCF
    forecast_years: int         # number of years to forecast FCF (from ML algorithm)
    growth_initial: float       # starting growth rate (from company data)

def _project_fcfs(inp: DCFInputs) -> List[float]:
    """
    Project FCF for the forecast period using simple linear decay to terminal growth.
    Standard approach: gradual decline from initial growth to terminal growth.
    """
    g = inp.growth_initial
    f = inp.fcf0
    projected_fcfs = []
    
    # Calculate decay rate to reach terminal growth by end of forecast period
    if inp.forecast_years > 1:
        decay_factor = (g - TERMINAL_GROWTH) / (inp.forecast_years - 1)
    else:
        decay_factor = 0
    
    for year in range(1, inp.forecast_years + 1):
        # Project FCF for this year
        f = f * (1 + g)
        projected_fcfs.append(f)
        
        # Linear decay toward terminal growth rate
        g = max(TERMINAL_GROWTH, g - decay_factor)
    
    return projected_fcfs

def intrinsic_value_from_fcf(inp: DCFInputs) -> float:
    """
    Calculate intrinsic value using standard DCF methodology:
    1. Forecast FCF for explicit period
    2. Discount each FCF using: PV = FCF / (1+r)^n
    3. Calculate terminal value for years beyond forecast period
    4. Discount terminal value to present
    5. Sum all present values
    """
    # Get projected FCFs for forecast period
    projected_fcfs = _project_fcfs(inp)
    discount_rate = DISCOUNT_RATE
    
    # Step 1: Calculate PV of explicit forecast period
    # Use standard DCF formula: PV = FCF / (1+r)^n
    pv_explicit = 0
    for year, fcf in enumerate(projected_fcfs, 1):
        pv = fcf / ((1 + discount_rate) ** year)
        pv_explicit += pv
    
    # Step 2: Calculate terminal value using Gordon Growth Model
    # Terminal value = FCF in last forecast year * (1 + terminal growth) / (discount rate - terminal growth)
    last_fcf = projected_fcfs[-1]
    terminal_value = (last_fcf * (1 + TERMINAL_GROWTH)) / (discount_rate - TERMINAL_GROWTH)
    
    # Step 3: Discount terminal value to present
    # Terminal value is at the end of forecast period, so discount by (1+r)^forecast_years
    pv_terminal = terminal_value / ((1 + discount_rate) ** inp.forecast_years)
    
    # Step 4: Total intrinsic value = PV of explicit + PV of terminal
    total_intrinsic_value = pv_explicit + pv_terminal
    
    return total_intrinsic_value

def calculate_dcf_for_forecast_period(fcf0: float, growth_rate: float, forecast_years: int, discount_rate: float = None) -> float:
    """
    Calculate DCF value for a specific forecast period.
    This is the main function used to test different forecast periods.
    """
    if discount_rate is None:
        discount_rate = DISCOUNT_RATE
    
    # Set environment variable for this calculation
    import os
    os.environ["DISCOUNT_RATE"] = str(discount_rate)
    
    dcf_inputs = DCFInputs(
        fcf0=fcf0,
        forecast_years=forecast_years,
        growth_initial=growth_rate
    )
    
    return intrinsic_value_from_fcf(dcf_inputs)

def price_at_year_end(ticker: str, year: int) -> float | None:
    """Close price nearest to Dec 31 of `year`."""
    start = f"{year}-12-01"
    end   = f"{year+1}-01-31"
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty: return None
        # nearest to 12/31
        hist = hist.sort_index()
        # pick last trading day of that window
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

def market_cap_approx(ticker: str, year: int) -> float | None:
    """Approximate market cap = year-end price * shares outstanding (current if we can't get historical)."""
    price = price_at_year_end(ticker, year)
    if price is None: return None
    info = yf.Ticker(ticker).info or {}
    shares = info.get("sharesOutstanding")
    if not shares: return None
    return float(price) * float(shares)
