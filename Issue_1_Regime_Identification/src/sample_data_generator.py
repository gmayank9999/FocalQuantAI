"""
Sample Data Generator for Testing Regime Classification
Generates realistic OHLCV data when API is not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_stock_data(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-15",
    initial_price: float = 2500.0,
    stock_name: str = "SAMPLE"
) -> pd.DataFrame:
    """
    Generate realistic sample stock OHLCV data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    initial_price : float
        Starting price for the stock
    stock_name : str
        Name of the stock (for reference)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with realistic OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate date range (exclude weekends)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start=start, end=end, freq='B')  # Business days only
    
    n_days = len(dates)
    
    # Generate price movements with realistic patterns
    # Use geometric brownian motion with regime changes
    returns = []
    volatility = 0.02  # Base volatility
    
    for i in range(n_days):
        # Create regime changes (volatility clusters)
        if i % 100 == 0:
            volatility = np.random.choice([0.01, 0.02, 0.03, 0.04])
        
        # Add trend changes
        if i % 150 == 0:
            trend = np.random.choice([-0.001, 0, 0.002])
        elif i < 50:
            trend = 0.001  # Initial uptrend
        else:
            trend = 0
        
        # Generate daily return
        daily_return = np.random.normal(trend, volatility)
        returns.append(daily_return)
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC
        daily_volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + daily_volatility)
        low = close * (1 - daily_volatility)
        
        # Open is influenced by previous close
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        # Ensure OHLC logic: high >= max(open, close), low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume (higher volume during volatile periods)
        base_volume = 10000000
        volume = int(base_volume * (1 + abs(np.random.normal(0, 0.5))))
        
        data.append({
            'datetime': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df


def generate_multiple_scenarios(stock_name: str = "SAMPLE") -> dict:
    """
    Generate multiple market scenarios for testing
    
    Returns:
    --------
    dict : Dictionary with different scenario DataFrames
    """
    scenarios = {}
    
    # Scenario 1: Trending market with increasing volatility
    dates = pd.bdate_range(start='2023-01-01', end='2024-12-15', freq='B')
    n = len(dates)
    prices = 2500 + np.cumsum(np.random.normal(2, 20, n))
    
    scenarios['trending'] = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.normal(0, 10, n),
        'high': prices + abs(np.random.normal(20, 10, n)),
        'low': prices - abs(np.random.normal(20, 10, n)),
        'close': prices,
        'volume': np.random.randint(5000000, 15000000, n)
    })
    
    # Scenario 2: Sideways market with low volatility
    prices = 2500 + np.cumsum(np.random.normal(0, 5, n))
    
    scenarios['sideways'] = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.normal(0, 3, n),
        'high': prices + abs(np.random.normal(5, 2, n)),
        'low': prices - abs(np.random.normal(5, 2, n)),
        'close': prices,
        'volume': np.random.randint(3000000, 8000000, n)
    })
    
    # Scenario 3: High volatility bear market
    prices = 3000 - np.cumsum(abs(np.random.normal(1, 25, n)))
    
    scenarios['volatile_falling'] = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.normal(0, 30, n),
        'high': prices + abs(np.random.normal(40, 20, n)),
        'low': prices - abs(np.random.normal(40, 20, n)),
        'close': prices,
        'volume': np.random.randint(10000000, 25000000, n)
    })
    
    return scenarios


if __name__ == "__main__":
    print("Sample Data Generator module loaded successfully!")
