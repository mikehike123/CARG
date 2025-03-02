import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_mixed_portfolio(initial_capital, treasury_pct, treasury_yield, 
                            tqqq_ticker, start_date, end_date, rebalance_mode):
    """
    Simulate a portfolio with fixed treasury yield and TQQQ ETF with various rebalancing strategies
    
    Parameters:
    initial_capital (float): Initial investment amount
    treasury_pct (float): Percentage to allocate to treasury (0-1)
    treasury_yield (float): Annual yield of treasury (0-1)
    tqqq_ticker (str): Ticker symbol for the ETF component
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    rebalance_mode (str): Rebalancing strategy ('tqqq_only', 'treasury_only', 'none', 'both')
    
    Returns:
    tuple: (yearly_results, cagr, std_dev, portfolio_history)
    """
    # Download historical data for TQQQ
    tqqq_data = yf.download(tqqq_ticker, start=start_date, end=end_date)['Close']
    tqqq_data.columns = ['Close']
    
    # Convert dates to pandas datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create a date range for each year-end
    year_ends = pd.date_range(start=start_dt, end=end_dt, freq='Y')
    if year_ends[-1] < end_dt:
        year_ends = year_ends.append(pd.DatetimeIndex([end_dt]))
    
    # Initialize portfolio
    tqqq_allocation = initial_capital * (1 - treasury_pct)
    treasury_allocation = initial_capital * treasury_pct
    
    # Track portfolio value over time
    portfolio_dates = tqqq_data.index
    portfolio_values = pd.Series(index=portfolio_dates)
    
    # Track yearly results
    yearly_results = []
    
    # For each year period
    for i in range(len(year_ends) - 1):
        year_start = year_ends[i] + pd.Timedelta(days=1)  # Adjust to start from January 1st
        year_end = year_ends[i+1]
        
        # Get relevant TQQQ data for this year
        year_mask = (tqqq_data.index >= year_start) & (tqqq_data.index <= year_end)
        year_tqqq = tqqq_data[year_mask].reset_index()
        
        if len(year_tqqq) == 0:
            continue
        
        # Calculate initial TQQQ shares
        tqqq_start_price = year_tqqq.iloc[0]['Close']
        tqqq_shares = tqqq_allocation / tqqq_start_price
        
        # Set the index back to datetime
        year_tqqq.set_index('Date', inplace=True)
        
        # Calculate daily portfolio value for the year
        for date in year_tqqq.index:
            tqqq_value = year_tqqq.loc[date, 'Close'] * tqqq_shares
            
            # Calculate treasury value with daily compounding
            days_passed = (date - year_start).days
            daily_rate = (1 + treasury_yield) ** (1/365) - 1
            treasury_value = treasury_allocation * (1 + daily_rate) ** days_passed
            
            portfolio_values[date] = tqqq_value + treasury_value
        
        # Calculate year-end values
        year_end_value = portfolio_values[year_tqqq.index[-1]]
        year_start_value = initial_capital if i == 0 else portfolio_values[year_tqqq.index[0]]
        yearly_return = (year_end_value / year_start_value) - 1
        
        # Store yearly result
        yearly_results.append({
            'Year': year_start.year,
            'Start Value': year_start_value,
            'End Value': year_end_value,
            'Return (%)': yearly_return * 100,
            'TQQQ Allocation (%)': (tqqq_value / year_end_value) * 100,
            'Treasury Allocation (%)': (treasury_value / year_end_value) * 100
        })
        
        # Rebalancing logic based on mode
        tqqq_pct = tqqq_value / year_end_value
        treasury_pct_actual = treasury_value / year_end_value
        
        if rebalance_mode == 'tqqq_only' and tqqq_pct < (1 - treasury_pct):
            # Scenario 1: Rebalance TQQQ only
            desired_tqqq = year_end_value * (1 - treasury_pct)
            transfer_amount = desired_tqqq - tqqq_value
            tqqq_allocation = tqqq_value + transfer_amount
            treasury_allocation = treasury_value - transfer_amount
        elif rebalance_mode == 'treasury_only' and treasury_pct_actual < treasury_pct:
            # Scenario 2: Rebalance Treasury only
            desired_treasury = year_end_value * treasury_pct
            transfer_amount = desired_treasury - treasury_value
            treasury_allocation = treasury_value + transfer_amount
            tqqq_allocation = tqqq_value - transfer_amount
        elif rebalance_mode == 'both':
            # Scenario 4: Rebalance both
            tqqq_allocation = year_end_value * (1 - treasury_pct)
            treasury_allocation = year_end_value * treasury_pct
        else:
            # Scenario 3: No rebalancing
            tqqq_allocation = tqqq_value
            treasury_allocation = treasury_value
    
    # Calculate overall portfolio metrics
    total_years = (end_dt - start_dt).days / 365.25
    final_value = portfolio_values.iloc[-1]
    cagr = (final_value / initial_capital) ** (1 / total_years) - 1
    
    # Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    annual_std_dev = daily_returns.std() * np.sqrt(252)  # Annualized standard deviation

    # Calculate maximum drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    
    return pd.DataFrame(yearly_results), cagr * 100, annual_std_dev * 100, portfolio_values, max_drawdown

def plot_portfolio_performance(portfolio_values, tqqq_ticker, treasury_yield, start_date, end_date):
    """Plot the performance of the portfolio versus TQQQ and Treasury"""
    # Download TQQQ data for comparison
    tqqq_data = yf.download(tqqq_ticker, start=start_date, end=end_date)['Close']
    
    # Normalize to starting values
    normalized_portfolio = portfolio_values / portfolio_values.iloc[0] * 100
    normalized_tqqq = tqqq_data / tqqq_data.iloc[0] * 100
    
    # Calculate theoretical treasury growth
    start_dt = pd.to_datetime(start_date)
    treasury_values = pd.Series(index=portfolio_values.index)
    for i, date in enumerate(treasury_values.index):
        days = (date - start_dt).days
        daily_rate = (1 + treasury_yield) ** (1/365) - 1
        treasury_values.iloc[i] = (1 + daily_rate) ** days * 100
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(normalized_portfolio.index, normalized_portfolio, 'b-', linewidth=2, label='Mixed Portfolio')
    plt.plot(treasury_values.index, treasury_values, 'g-', linewidth=1, label=f'Treasury ({treasury_yield*100:.1f}%)')
    
    plt.title('Comparative Performance (Starting Value = 100)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Set pandas display options to prevent clipping
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Portfolio parameters
    initial_capital = 33000  # $33K
    treasury_pct = 0.70  # 90% in treasury
    treasury_yield = 0.045  # 4% annual yield
    tqqq_ticker = 'spy'
    rebalance_flag = 'both'
    
    # Define time period (10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
    
    # Run simulation
    yearly_results, cagr, std_dev, portfolio_values, max_drawdown = simulate_mixed_portfolio(
        initial_capital, treasury_pct, treasury_yield, tqqq_ticker, start_date, end_date, rebalance_flag
    )

    # Print results
    print(f"\nPORTFOLIO SIMULATION: {treasury_pct*100:.0f}% Treasury ({treasury_yield*100:.1f}%) + {(1-treasury_pct)*100:.0f}% {tqqq_ticker}")
    print("=" * 80)
    
    print("\nYearly Performance:")
    print(yearly_results.round(2))
    
    print("\nOverall Portfolio Metrics:")
    print(f"Starting Value: ${initial_capital:,.2f}")
    print(f"Ending Value: ${portfolio_values.iloc[-1]:,.2f}")
    print(f"Total Growth: {(portfolio_values.iloc[-1]/initial_capital - 1)*100:.2f}%")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Standard Deviation: {std_dev:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Plot performance
    portfolio_values.dropna(inplace=True)
    
    plot_portfolio_performance(portfolio_values, tqqq_ticker, treasury_yield, start_date, end_date)

if __name__ == "__main__":
    main()