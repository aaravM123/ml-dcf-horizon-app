# Enhanced DCF Patch Script: Advanced Intrinsic Value Calculation
import yfinance as yf
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class EnhancedDCFCalculator:
    """
    Enhanced DCF calculator with multiple growth models and better error handling.
    This can be used as a standalone tool or integrated with the existing DCF system.
    """
    
    def __init__(self, 
                 default_growth_rate: float = 0.15,
                 default_terminal_growth: float = 0.05,
                 default_discount_rate: float = 0.075,
                 default_forecast_years: int = 10):
        self.default_growth_rate = default_growth_rate
        self.default_terminal_growth = default_terminal_growth
        self.default_discount_rate = default_discount_rate
        self.default_forecast_years = default_forecast_years
    
    def fetch_financials(self, ticker_symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch financial data from Yahoo Finance with robust error handling.
        
        Returns:
            Tuple of (FCF, shares_outstanding) or (None, None) if failed
        """
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get cashflow statement
            cashflow = ticker.cashflow
            if cashflow is None or cashflow.empty:
                print(f"⚠️ No cashflow data available for {ticker_symbol}")
                return None, None
            
            # Calculate FCF
            try:
                operating_cash = cashflow.loc["Total Cash From Operating Activities"].iloc[0]
                capex = cashflow.loc["Capital Expenditures"].iloc[0]
                fcf = operating_cash - capex
                
                if pd.isna(fcf) or fcf <= 0:
                    print(f"⚠️ Invalid FCF calculated for {ticker_symbol}: {fcf}")
                    return None, None
                    
            except KeyError as e:
                print(f"⚠️ Missing required cashflow line items for {ticker_symbol}: {e}")
                return None, None
            
            # Get shares outstanding
            try:
                shares_out = ticker.info.get('sharesOutstanding')
                if shares_out is None or shares_out <= 0:
                    print(f"⚠️ Invalid shares outstanding for {ticker_symbol}: {shares_out}")
                    return None, None
            except Exception:
                print(f"⚠️ Could not fetch shares outstanding for {ticker_symbol}")
                return None, None
            
            return fcf, shares_out
            
        except Exception as e:
            print(f"❌ Error fetching financials for {ticker_symbol}: {e}")
            return None, None
    
    def calculate_dcf_linear_decay(self, 
                                  fcf: float, 
                                  growth_rate: Optional[float] = None,
                                  terminal_growth: Optional[float] = None,
                                  discount_rate: Optional[float] = None,
                                  forecast_years: Optional[int] = None) -> Optional[float]:
        """
        Calculate DCF using linear decay growth model.
        
        Args:
            fcf: Starting free cash flow
            growth_rate: Initial growth rate (defaults to instance default)
            terminal_growth: Terminal growth rate (defaults to instance default)
            discount_rate: Discount rate (defaults to instance default)
            forecast_years: Number of years to forecast (defaults to instance default)
        
        Returns:
            Total DCF value or None if calculation fails
        """
        if fcf is None or fcf <= 0:
            return None
            
        # Use defaults if not provided
        growth_rate = growth_rate or self.default_growth_rate
        terminal_growth = terminal_growth or self.default_terminal_growth
        discount_rate = discount_rate or self.default_discount_rate
        forecast_years = forecast_years or self.default_forecast_years
        
        try:
            fcf_values = []
            for year in range(forecast_years):
                # Linear decay from initial growth to terminal
                decay = growth_rate - ((growth_rate - terminal_growth) / forecast_years) * year
                projected_fcf = fcf * ((1 + decay) ** (year + 1))
                present_value = projected_fcf / ((1 + discount_rate) ** (year + 1))
                fcf_values.append(present_value)
            
            # Calculate terminal value
            last_fcf = fcf * ((1 + terminal_growth) ** forecast_years)
            terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** forecast_years)
            
            dcf_total = sum(fcf_values) + terminal_value_pv
            return dcf_total
            
        except Exception as e:
            print(f"❌ Error in DCF calculation: {e}")
            return None
    
    def calculate_dcf_constant_growth(self, 
                                     fcf: float, 
                                     growth_rate: Optional[float] = None,
                                     terminal_growth: Optional[float] = None,
                                     discount_rate: Optional[float] = None,
                                     forecast_years: Optional[int] = None) -> Optional[float]:
        """
        Calculate DCF using constant growth model (simpler alternative).
        """
        if fcf is None or fcf <= 0:
            return None
            
        growth_rate = growth_rate or self.default_growth_rate
        terminal_growth = terminal_growth or self.default_terminal_growth
        discount_rate = discount_rate or self.default_discount_rate
        forecast_years = forecast_years or self.default_forecast_years
        
        try:
            # Calculate explicit period
            explicit_value = 0
            for year in range(1, forecast_years + 1):
                projected_fcf = fcf * ((1 + growth_rate) ** year)
                present_value = projected_fcf / ((1 + discount_rate) ** year)
                explicit_value += present_value
            
            # Calculate terminal value
            last_fcf = fcf * ((1 + growth_rate) ** forecast_years)
            terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** forecast_years)
            
            return explicit_value + terminal_value_pv
            
        except Exception as e:
            print(f"❌ Error in constant growth DCF calculation: {e}")
            return None
    
    def analyze_company(self, 
                        ticker_symbol: str, 
                        growth_model: str = "linear_decay",
                        custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete company analysis with DCF calculation.
        
        Args:
            ticker_symbol: Stock ticker symbol
            growth_model: "linear_decay" or "constant_growth"
            custom_params: Dictionary of custom parameters to override defaults
        
        Returns:
            Dictionary with analysis results
        """
        print(f"\n🔍 Analyzing {ticker_symbol}...")
        
        # Fetch financial data
        fcf, shares_out = self.fetch_financials(ticker_symbol)
        
        if fcf is None or shares_out is None:
            return {"error": "Failed to fetch financial data"}
        
        # Prepare parameters
        params = {
            "growth_rate": self.default_growth_rate,
            "terminal_growth": self.default_terminal_growth,
            "discount_rate": self.default_discount_rate,
            "forecast_years": self.default_forecast_years
        }
        
        if custom_params:
            params.update(custom_params)
        
        # Calculate DCF based on selected model
        if growth_model == "linear_decay":
            dcf_value = self.calculate_dcf_linear_decay(fcf, **params)
        elif growth_model == "constant_growth":
            dcf_value = self.calculate_dcf_constant_growth(fcf, **params)
        else:
            return {"error": f"Unknown growth model: {growth_model}"}
        
        if dcf_value is None:
            return {"error": "DCF calculation failed"}
        
        # Calculate per-share value
        intrinsic_value_per_share = dcf_value / shares_out
        
        # Prepare results
        results = {
            "ticker": ticker_symbol,
            "fcf": fcf,
            "shares_outstanding": shares_out,
            "dcf_total_value": dcf_value,
            "intrinsic_value_per_share": intrinsic_value_per_share,
            "growth_model": growth_model,
            "parameters": params
        }
        
        # Print results
        self._print_analysis_results(results)
        
        return results
    
    def _print_analysis_results(self, results: Dict[str, Any]):
        """Print formatted analysis results."""
        print("---\n✅ ANALYSIS RESULTS")
        print(f"Ticker: {results['ticker']}")
        print(f"FCF: ${results['fcf']:,.0f}")
        print(f"Shares Outstanding: {results['shares_outstanding']:,.0f}")
        print(f"DCF Total Value: ${results['dcf_total_value']:,.0f}")
        print(f"Intrinsic Value Per Share: ${results['intrinsic_value_per_share']:,.2f}")
        print(f"Growth Model: {results['growth_model']}")
        print(f"Parameters: {results['parameters']}")
        print("---")

def main():
    """Main function for standalone usage."""
    # Create calculator instance
    calculator = EnhancedDCFCalculator(
        default_growth_rate=0.15,
        default_terminal_growth=0.05,
        default_discount_rate=0.075,
        default_forecast_years=10
    )
    
    # Get ticker from user
    ticker = input("Enter a company ticker (e.g., AAPL): ").upper().strip()
    
    # Run analysis
    results = calculator.analyze_company(ticker)
    
    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
    else:
        print("✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
