"""
Validation utilities for stock data simulation
"""

import pandas as pd
from typing import Tuple, Union
from datetime import datetime


def validate_ohlcv(open_price: float, high: float, low: float, 
                   close: float, volume: float) -> Tuple[bool, str]:
    """
    Validate OHLCV data integrity.
    
    Parameters
    ----------
    open_price : float
        Opening price
    high : float
        High price
    low : float
        Low price
    close : float
        Closing price
    volume : float
        Volume
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check for non-negative values
    if any(x < 0 for x in [open_price, high, low, close, volume]):
        return False, "All OHLCV values must be non-negative"
    
    # Check for zero prices (volume can be zero)
    if any(x <= 0 for x in [open_price, high, low, close]):
        return False, "All price values must be positive"
    
    # High should be the highest
    if high < max(open_price, close, low):
        return False, f"High ({high}) must be >= Open ({open_price}), Close ({close}), Low ({low})"
    
    # Low should be the lowest
    if low > min(open_price, close, high):
        return False, f"Low ({low}) must be <= Open ({open_price}), Close ({close}), High ({high})"
    
    # Open and Close should be between High and Low
    if not (low <= open_price <= high):
        return False, f"Open ({open_price}) must be between Low ({low}) and High ({high})"
    
    if not (low <= close <= high):
        return False, f"Close ({close}) must be between Low ({low}) and High ({high})"
    
    return True, "Valid OHLCV"


def validate_simulation_params(start_time: Union[str, datetime],
                               end_time: Union[str, datetime],
                               granularity_minutes: int,
                               volatility_index: float) -> Tuple[bool, str]:
    """
    Validate simulation parameters.
    
    Parameters
    ----------
    start_time : str or datetime
        Start timestamp
    end_time : str or datetime
        End timestamp
    granularity_minutes : int
        Granularity in minutes
    volatility_index : float
        Volatility index (1-25)
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Parse timestamps if strings
    if isinstance(start_time, str):
        try:
            start_time = pd.to_datetime(start_time)
        except:
            return False, f"Invalid start_time format: {start_time}"
    
    if isinstance(end_time, str):
        try:
            end_time = pd.to_datetime(end_time)
        except:
            return False, f"Invalid end_time format: {end_time}"
    
    # Check time order
    if start_time >= end_time:
        return False, "start_time must be before end_time"
    
    # Check granularity
    if granularity_minutes <= 0:
        return False, "granularity_minutes must be positive"
    
    # Check if time window is sufficient for at least 2 data points
    time_diff_minutes = (end_time - start_time).total_seconds() / 60
    if time_diff_minutes < granularity_minutes:
        return False, f"Time window ({time_diff_minutes:.1f} min) is smaller than granularity ({granularity_minutes} min)"
    
    # Check volatility index range
    if not (1 <= volatility_index <= 25):
        return False, f"volatility_index must be between 1 and 25, got {volatility_index}"
    
    return True, "Valid parameters"
