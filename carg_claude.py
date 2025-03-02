import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_portfolio_metrics(tickers, weights, start_date, end_date):
    """
    Calculate CAGR and standard deviation for each ticker and the weighted portfolio
    
    Parameters:
    tickers (list): List of ticker symbols
    weights (list): List of portfolio weights corresponding to tickers
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pd.DataFrame: DataFrame with CAGR and standard deviation for each ticker and portfolio
    """
    # Download historical data
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate annual returns (252 trading days in a year)
    annual_returns = returns.mean() * 252
    annual_std_dev = returns.std() * np.sqrt(252)
    
    # Calculate CAGR
    total_return = (data.iloc[-1] / data.iloc[0]) - 1
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Create results DataFrame
    results = pd.DataFrame({
        'CAGR (%)': cagr * 100,
        'Standard Deviation (%)': annual_std_dev * 100,
        'Weight (%)': pd.Series(weights, index=tickers) * 100
    })
    
    # Calculate portfolio metrics
    weighted_cagr = np.sum(cagr * weights)
    
    # For portfolio std dev, we need the covariance matrix
    cov_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    
    # Add portfolio row
    portfolio_metrics = pd.Series({
        'CAGR (%)': weighted_cagr * 100,
        'Standard Deviation (%)': portfolio_std_dev * 100,
        'Weight (%)': 100.0
    }, name='PORTFOLIO')
    
    results = pd.concat([results, portfolio_metrics.to_frame().T])
    
    # Add some additional metrics
    max_drawdown = calculate_max_drawdown(data, weights)
    sharpe_ratio = calculate_sharpe_ratio(weighted_cagr, portfolio_std_dev)
    
    return results, max_drawdown, sharpe_ratio

def calculate_max_drawdown(data, weights):
    """Calculate the maximum drawdown for the portfolio"""
    # Calculate portfolio value over time
    portfolio_value = (data * weights).sum(axis=1)
    portfolio_value = portfolio_value / portfolio_value.iloc[0]  # Normalize to 1
    
    # Calculate drawdown
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    return max_drawdown * 100  # Convert to percentage

def calculate_sharpe_ratio(cagr, std_dev, risk_free_rate=0.01):
    """Calculate the Sharpe ratio"""
    return (cagr - risk_free_rate) / std_dev

def plot_performance(tickers, weights, start_date, end_date):
    """Plot the cumulative performance of individual components and the portfolio"""
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Normalize to starting value
    normalized_data = data.div(data.iloc[0]) * 100
    
    # Calculate portfolio value
    portfolio_value = (normalized_data * weights).sum(axis=1)
    
    # Plot
    plt.figure(figsize=(12, 8))
    for ticker in tickers:
        plt.plot(normalized_data.index, normalized_data[ticker], label=ticker)
    
    plt.plot(portfolio_value.index, portfolio_value, 'k--', linewidth=3, label='Portfolio')
    plt.title('Cumulative Performance (Starting Value = 100)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_inflation_data(start_date, end_date):
    """Get inflation data from FRED (would need to use a different API)"""
    # This is a placeholder - in a real implementation you'd use FRED API or similar
    # For now, return an estimate based on historical averages
    return 2.8  # Estimated average inflation 2015-2024

def main():
    # Define portfolio components
    tickers = ['TIP', 'VTIP', 'SCHP', 'VNQ', 'FREL', 'INDS', 'PDBC', 'COMT', 'NOBL', 'QQQ', 'MSFT', 'SHY']
    # tickers = ['TQQQ' ]
    # SHY serves as a proxy for fixed income and money market components
    
    # Define weights (corresponding to recommended allocation)
    weights = [0.0, 0.047, 0.038, 0.023, 0.015, 0.009, 0.019, 0.013, 0.034, 0.015, 0.016, 0.771]
    #weights = ["1.0"]
    # Note: The final weight (0.771) combines fixed income (0.60) and money market (0.171)
    
    # Define time period (10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
    # Run the analysis
    results, max_drawdown, sharpe_ratio = calculate_portfolio_metrics(tickers, weights, start_date, end_date)
    
    # Print results
    print("\nPORTFOLIO PERFORMANCE ANALYSIS (10-Year Period)")
    print("=" * 80)
    print(results.round(2).sort_values('Weight (%)', ascending=False))
    print("\nAdditional Portfolio Metrics:")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Get inflation data
    avg_inflation = get_inflation_data(start_date, end_date)
    portfolio_cagr = results.loc['PORTFOLIO', 'CAGR (%)']
    print(f"\nInflation Comparison:")
    print(f"Average Annual Inflation: {avg_inflation:.2f}%")
    print(f"Portfolio Real Return: {portfolio_cagr - avg_inflation:.2f}%")
    
    # Plot the performance
    plot_performance(tickers, weights, start_date, end_date)

if __name__ == "__main__":
    main()