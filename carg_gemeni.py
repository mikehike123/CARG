import yfinance as yf
import pandas as pd
import numpy as np

def calculate_cagr_stddev(symbol, years=10):
    """
    Calculates CAGR and standard deviation of annual returns for a given stock symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL").
        years (int): The number of years to analyze.

    Returns:
        tuple: A tuple containing two DataFrames: one for annual returns and one for the 10-year summary.
    """
    try:
        # Get historical data with auto_adjust set to False
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

        if data.empty:
            print(f"No data found for symbol: {symbol}")
            return None, None

        # Calculate annual returns
        annual_data = data['Close'].resample('YE').last()  # Use 'YE' for year-end
        annual_returns = annual_data.pct_change().dropna()

        # Ensure annual_returns is a 1D Series
        if isinstance(annual_returns, pd.DataFrame):
            annual_returns = annual_returns.squeeze()  # Convert to Series if it's a DataFrame

        # Calculate CAGR
        first_year_value = annual_data.iloc[0]
        last_year_value = annual_data.iloc[-1]
        cagr = (last_year_value / first_year_value) ** (1 / years) - 1
        cagr_percentage = cagr * 100

        # Ensure cagr_percentage is a scalar value
        if isinstance(cagr_percentage, pd.Series):
            cagr_percentage = cagr_percentage.iloc[0]  # Extract the scalar value

        # Calculate standard deviation
        std_dev = annual_returns.std() * 100

        # Create DataFrames
        annual_returns_df = pd.DataFrame({
            'Year': annual_returns.index.year,
            'Annual Return (%)': annual_returns.values * 100
        })

        # Format CAGR and standard deviation as strings with one decimal place and a percent sign
        cagr_str = f"{float(cagr_percentage):.1f}%"  # Ensure cagr_percentage is a float
        std_dev_str = f"{float(std_dev):.1f}%"  # Ensure std_dev is a float

        summary_df = pd.DataFrame({
            'Ticker': [symbol],
            f'{years} Year CAGR (%)': [cagr_str],
            'Standard Deviation (%)': [std_dev_str]
        })

        return annual_returns_df, summary_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Calculate CAGR and standard deviation for SPY over the last 15 years
duration = 15 
annual_returns_df, summary_df = calculate_cagr_stddev("SPY", 15)

if annual_returns_df is not None and summary_df is not None:
    print("Annual Returns:")
    print(annual_returns_df.to_string(index=False))
    print(f"\n{duration}-Year Summary:")
    print(summary_df.to_string(index=False))