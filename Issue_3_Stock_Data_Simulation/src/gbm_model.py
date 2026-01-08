"""
GBM (Geometric Brownian Motion) Volatility Model
=================================================

Implements the standard GBM formula: dS = μ*S*dt + σ*S*dW
"""

import numpy as np
from typing import Optional


def scale_volatility(volatility_index: float) -> float:
    """
    Convert volatility index (1-25) to annualized volatility.
    
    Parameters
    ----------
    volatility_index : float
        Volatility index on scale 1-25
        
    Returns
    -------
    float
        Annual volatility as decimal (e.g., 0.15 for 15%)
        
    Scale Mapping
    -------------
    Index 1  -> 5% annual volatility (very low)
    Index 5  -> 15% annual volatility (low)
    Index 10 -> 25% annual volatility (moderate)
    Index 15 -> 40% annual volatility (high)
    Index 25 -> 80% annual volatility (extreme)
    """
    return 0.05 + (volatility_index - 1) * 0.03125


def generate_gbm_returns(n_steps: int, 
                        volatility_index: float,
                        drift: float = 0.0,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Generate price returns using Geometric Brownian Motion.
    
    Formula: dS = μ*S*dt + σ*S*dW
    
    Parameters
    ----------
    n_steps : int
        Number of time steps to simulate
    volatility_index : float
        Volatility level (1-25 scale)
    drift : float, optional
        Annual drift rate (default: 0.0 for no trend)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of returns for each time step
        
    Implementation Details
    ----------------------
    - Assumes 252 trading days per year
    - Assumes 375 minutes per trading day (6.25 hours)
    - Scales annual volatility to per-minute using sqrt(252 * 375)
    - Uses standard normal distribution N(0,1) for Brownian motion
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert annual parameters to per-step (per-minute)
    annual_vol = scale_volatility(volatility_index)
    step_vol = annual_vol / np.sqrt(252 * 375)
    step_drift = drift / (252 * 375)
    
    # Generate random shocks from standard normal distribution
    random_shocks = np.random.normal(0, 1, n_steps)
    
    # Calculate returns: drift component + volatility component
    returns = step_drift + step_vol * random_shocks
    
    return returns


def generate_gbm_path(n_steps: int,
                      start_value: float,
                      end_value: float,
                      volatility_index: float,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Generate price path using GBM with endpoint constraint.
    
    Creates a random walk path from start_value to end_value using
    linear interpolation plus GBM-style random deviations.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps (will generate n_steps + 1 points)
    start_value : float
        Starting price
    end_value : float
        Ending price
    volatility_index : float
        Volatility level (1-25 scale)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of n_steps + 1 price points
        
    Implementation Details
    ----------------------
    Uses linear interpolation between start and end with added
    GBM-style random deviations scaled by volatility and sqrt(time).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert volatility index to per-step volatility
    annual_vol = scale_volatility(volatility_index)
    step_vol = annual_vol / np.sqrt(252 * 375)
    
    # Initialize path with n_steps + 1 points
    path = np.zeros(n_steps + 1)
    path[0] = start_value
    path[n_steps] = end_value
    
    # Generate path using linear interpolation + random noise
    for i in range(1, n_steps):
        # Linear interpolation component
        t = i / n_steps
        drift_price = start_value + (end_value - start_value) * t
        
        # Add GBM-style random noise scaled by sqrt(time)
        noise = np.random.normal(0, step_vol * start_value * np.sqrt(i))
        path[i] = drift_price + noise
    
    return path
