"""
Stock Data Movement Simulation Engine
======================================

Generates synthetic OHLCV tick data using GBM volatility model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union

try:
    from .gbm_model import generate_gbm_returns, scale_volatility
    from .validators import validate_ohlcv, validate_simulation_params
except ImportError:
    from gbm_model import generate_gbm_returns, scale_volatility
    from validators import validate_ohlcv, validate_simulation_params


class StockDataSimulator:
    """
    Stock Data Movement Simulation Engine
    
    Generates synthetic OHLCV (Open, High, Low, Close, Volume) tick data
    using Geometric Brownian Motion (GBM) volatility model.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> simulator = StockDataSimulator(seed=42)
    >>> df = simulator.simulate(
    ...     start_time="2025-12-10 09:15:00",
    ...     end_time="2025-12-10 15:30:00",
    ...     granularity_minutes=1,
    ...     volatility_index=5,
    ...     ohlcv_seed=(100, 125, 95, 103, 5300)
    ... )
    >>> print(f"Generated {len(df)} ticks")
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def simulate(self,
                start_time: Union[str, datetime],
                end_time: Union[str, datetime],
                granularity_minutes: int,
                volatility_index: float,
                ohlcv_seed: Tuple[float, float, float, float, float]) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Parameters
        ----------
        start_time : str or datetime
            Start timestamp (e.g., "2025-12-10 09:15:00")
        end_time : str or datetime
            End timestamp (e.g., "2025-12-10 15:30:00")
        granularity_minutes : int
            Time granularity in minutes (e.g., 1 for 1-minute bars)
        volatility_index : float
            Volatility level (1-25 scale, 1=calm, 25=extreme)
        ohlcv_seed : tuple
            Initial (Open, High, Low, Close, Volume) values
            
        Returns
        -------
        pd.DataFrame
            Simulated OHLCV data with columns:
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # Parse timestamps
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        # Validate parameters
        is_valid, msg = validate_simulation_params(
            start_time, end_time, granularity_minutes, volatility_index
        )
        if not is_valid:
            raise ValueError(f"Invalid parameters: {msg}")
        
        # Validate OHLCV seed
        is_valid, msg = validate_ohlcv(*ohlcv_seed)
        if not is_valid:
            raise ValueError(f"Invalid OHLCV seed: {msg}")
        
        # Calculate number of intervals (NOT including end point to get exact count)
        time_diff_minutes = (end_time - start_time).total_seconds() / 60
        n_intervals = int(time_diff_minutes / granularity_minutes)
        
        # Generate timestamps - do NOT include endpoint
        timestamps = [start_time + timedelta(minutes=i * granularity_minutes) 
                     for i in range(n_intervals)]
        
        # Extract seed values
        seed_open, seed_high, seed_low, seed_close, seed_volume = ohlcv_seed
        
        # Generate OHLCV data for entire interval
        # Seed represents: first_open, max_high, min_low, last_close, avg_volume
        ohlcv_data = self._generate_ohlcv_constrained(
            timestamps=timestamps,
            seed_ohlcv=(seed_open, seed_high, seed_low, seed_close, seed_volume),
            volatility_index=volatility_index
        )
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        return df
    
    def _generate_ohlcv_constrained(self,
                                    timestamps: list,
                                    seed_ohlcv: Tuple[float, float, float, float, float],
                                    volatility_index: float) -> dict:
        """
        Generate OHLCV data with constraints from seed.
        
        Seed interpretation:
        - seed_open: Open price of FIRST bar
        - seed_high: HIGHEST high across ALL bars
        - seed_low: LOWEST low across ALL bars
        - seed_close: Close price of LAST bar
        - seed_volume: Average volume per bar
        """
        n = len(timestamps)
        seed_open, seed_high, seed_low, seed_close, seed_volume = seed_ohlcv
        
        # Generate price path from seed_open to seed_close
        # Using GBM-like random walk but constrained
        price_path = self._generate_constrained_price_path(
            n, seed_open, seed_close, seed_high, seed_low, volatility_index
        )
        
        # Initialize arrays
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        closes = np.zeros(n)
        volumes = np.zeros(n)
        
        # Generate OHLC bars from price path
        for i in range(n):
            opens[i] = price_path[i]
            closes[i] = price_path[i + 1]  # Next point is the close
            
            # Calculate what high/low should be to eventually reach extremes
            # We'll adjust the bar with max potential to actually hit seed_high/low
            base_high = max(opens[i], closes[i])
            base_low = min(opens[i], closes[i])
            
            # Generate intrabar high/low with reasonable range (5-10% of price)
            intrabar_pct = (volatility_index / 25) * 0.05  # 0-5% based on volatility
            highs[i] = base_high * (1 + intrabar_pct * np.random.uniform(0.5, 2.0))
            lows[i] = base_low * (1 - intrabar_pct * np.random.uniform(0.5, 2.0))
            
            # Generate volume (must be integer - number of shares)
            volumes[i] = int(seed_volume * np.random.uniform(0.7, 1.3))
        
        # Find indices for extreme values BEFORE consistency check
        max_idx_prelim = highs.argmax()
        min_idx_prelim = lows.argmin()
        
        # Ensure OHLC consistency for all bars EXCEPT the ones we'll force
        for i in range(n):
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])
            # Keep positive but don't prevent seed extremes
            if lows[i] > seed_low:
                lows[i] = max(seed_low * 0.5, lows[i])
        
        # FORCE exact seed constraints (may violate OHLC rules for extreme bars)
        highs[max_idx_prelim] = seed_high
        lows[min_idx_prelim] = seed_low
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
    
    def _generate_constrained_price_path(self, n: int, start: float, end: float,
                                        max_high: float, min_low: float,
                                        volatility_index: float) -> np.ndarray:
        """Generate a price path from start to end that will reach max_high and min_low."""
        # We need n+1 points (n bars have n+1 price points)
        path = np.zeros(n + 1)
        path[0] = start
        path[n] = end
        
        # Generate random walk in between using GBM
        annual_vol = scale_volatility(volatility_index)
        step_vol = annual_vol / np.sqrt(252 * 375)
        
        # Decide where to place extreme points
        # Place high somewhere in first 60% of path
        # Place low somewhere in middle-to-end 40-80% of path
        high_idx = np.random.randint(int(n * 0.2), int(n * 0.6))
        low_idx = np.random.randint(int(n * 0.4), int(n * 0.8))
        
        # Use bridge sampling - generate path conditioned on endpoints
        for i in range(1, n):
            # Linear interpolation + random deviation
            t = i / n
            drift_price = start + (end - start) * t
            
            # Add random noise
            noise = np.random.normal(0, step_vol * start * np.sqrt(i))
            path[i] = drift_price + noise
            
            # No artificial bounds - let it be natural
        
        # Force extreme points to be near the targets
        # This ensures the highs/lows will reach seed values
        path[high_idx] = max_high * np.random.uniform(0.95, 0.99)
        path[low_idx] = min_low * np.random.uniform(1.01, 1.05)
        
        return path
    
    def _generate_intrabar_range(self, open_price: float, close_price: float,
                                volatility_index: float) -> float:
        """Generate realistic high-low range within a bar."""
        # Base range from open-close movement
        oc_range = abs(close_price - open_price)
        
        # Add volatility-dependent expansion (0.5% to 2% of price)
        vol_factor = 0.005 + (volatility_index / 25) * 0.015
        vol_range = open_price * vol_factor
        
        return oc_range + vol_range
    
    def _generate_volume(self, base_volume: float, volatility_index: float,
                        abs_return: float) -> float:
        """Generate volume for a bar."""
        # Base multiplier from volatility index
        vol_multiplier = 1.0 + (volatility_index / 25) * 0.8
        
        # Additional boost from large price moves
        movement_boost = 1.0 + abs_return * 50
        
        # Random variation (±30%)
        random_factor = np.random.uniform(0.7, 1.3)
        
        # Occasional volume spikes (5% chance of 2-3x volume)
        if np.random.random() < 0.05:
            spike_factor = np.random.uniform(2.0, 3.0)
        else:
            spike_factor = 1.0
        
        volume = (base_volume * vol_multiplier * movement_boost * 
                 random_factor * spike_factor)
        
        return max(0, volume)
