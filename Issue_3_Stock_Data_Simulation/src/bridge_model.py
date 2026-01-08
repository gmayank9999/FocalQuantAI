"""
Brownian Bridge Model
=====================

Implements Brownian Bridge for constrained price path generation.
The bridge ensures the price path starts at a specific value and ends 
at another specific value, with random variation in between.

Formula: B(t) = a + (b-a)*(t/T) + sqrt(t*(T-t)/T) * W(t)
where W(t) ~ N(0,1) is standard Brownian motion
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
        Annual volatility as decimal
        
    Scale Mapping
    -------------
    Index 1  -> 5% annual volatility (very low)
    Index 5  -> 15% annual volatility (low)
    Index 10 -> 25% annual volatility (moderate)
    Index 15 -> 40% annual volatility (high)
    Index 25 -> 80% annual volatility (extreme)
    """
    return 0.05 + (volatility_index - 1) * 0.03125


def generate_bridge_path(n_steps: int,
                         start_value: float,
                         end_value: float,
                         volatility_index: float,
                         seed: Optional[int] = None) -> np.ndarray:
    """
    Generate price path using Brownian Bridge.
    
    The Brownian Bridge ensures the path starts at start_value and 
    ends at end_value, with volatility-controlled variation in between.
    
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
        Array of n_steps + 1 price points forming the bridge
        
    Implementation Details
    ----------------------
    Uses standard Brownian Bridge construction:
    - Linear interpolation between start and end
    - Add volatility-scaled random deviations
    - Conditional on endpoints i.e must reach exact start and end values
    
    Formula at time step i:
    B(t_i) = start + (end - start) * (i/n) + sigma * sqrt(t_i * (T - t_i) / T) * Z_i
    
    where Z_i ~ N(0,1) and t_i = i/n, T = 1
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert volatility index to scaling factor
    annual_vol = scale_volatility(volatility_index)
    
    # Scale for per-step volatility
    # Assuming 252 trading days, 375 minutes per day
    vol_scaling = annual_vol / np.sqrt(252 * 375)
    
    # Initialize path with n_steps + 1 points
    path = np.zeros(n_steps + 1)
    path[0] = start_value
    path[n_steps] = end_value
    
    # Generate bridge path for intermediate points
    for i in range(1, n_steps):
        # Time fraction
        t = i / n_steps
        T_minus_t = (n_steps - i) / n_steps
        
        # Linear interpolation component
        linear_part = start_value + (end_value - start_value) * t
        
        # Bridge variance: t * (T - t) / T
        bridge_variance = t * T_minus_t
        bridge_std = np.sqrt(bridge_variance)
        
        # Random component scaled by volatility
        random_shock = np.random.normal(0, 1)
        random_component = vol_scaling * bridge_std * random_shock * start_value * np.sqrt(n_steps)
        
        # Combine components
        path[i] = linear_part + random_component
    
    return path


def generate_bridge_returns(n_steps: int,
                            start_value: float,
                            end_value: float,
                            volatility_index: float,
                            seed: Optional[int] = None) -> np.ndarray:
    """
    Generate returns using Brownian Bridge path.
    
    This is a convenience function that generates a bridge path
    and returns the step-by-step returns.
    
    Parameters
    ----------
    n_steps : int
        Number of return steps
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
        Array of n_steps returns
    """
    # Generate bridge path (n_steps + 1 points)
    path = generate_bridge_path(n_steps, start_value, end_value, 
                                volatility_index, seed)
    
    # Calculate returns from path
    returns = np.zeros(n_steps)
    for i in range(n_steps):
        if path[i] > 0:
            returns[i] = (path[i + 1] - path[i]) / path[i]
        else:
            returns[i] = 0.0
    
    return returns
