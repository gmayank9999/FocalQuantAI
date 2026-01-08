"""
Stock Data Movement Simulation Engine
======================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union

try:
    from .gbm_model import generate_gbm_path
    from .gbm_model import scale_volatility as gbm_scale_volatility
    from .bridge_model import generate_bridge_path
    from .bridge_model import scale_volatility as bridge_scale_volatility
    from .validators import validate_ohlcv, validate_simulation_params
except ImportError:
    from gbm_model import generate_gbm_path
    from gbm_model import scale_volatility as gbm_scale_volatility
    from bridge_model import generate_bridge_path
    from bridge_model import scale_volatility as bridge_scale_volatility
    from validators import validate_ohlcv, validate_simulation_params


class StockDataSimulator:
    """
    Stock Data Movement Simulation Engine
    
    Generates synthetic OHLCV (Open, High, Low, Close, Volume) tick data
    using either GBM (Geometric Brownian Motion) or Brownian Bridge models.
    
    Parameters
    ----------
    model : str, optional
        Volatility model to use: 'gbm' or 'bridge' (default: 'gbm')
    seed : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> # Using GBM model
    >>> simulator = StockDataSimulator(model='gbm', seed=42)
    >>> df = simulator.simulate(
    ...     start_time="2025-12-10 09:15:00",
    ...     end_time="2025-12-10 15:30:00",
    ...     granularity_minutes=1,
    ...     volatility_index=5,
    ...     ohlcv_seed=(100, 125, 95, 103, 5300)
    ... )
    >>> print(f"Generated {len(df)} ticks")
    
    >>> # Using Brownian Bridge model
    >>> simulator = StockDataSimulator(model='bridge', seed=42)
    >>> df = simulator.simulate(...)
    """
    
    def __init__(self, model: str = 'gbm', seed: Optional[int] = None):
        if model not in ['gbm', 'bridge']:
            raise ValueError(f"Invalid model '{model}'. Choose 'gbm' or 'bridge'")
        
        self.model = model
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
        
        # Generate timestamps - do NOT include endpoint [2025-12-10 09:15:00 ,2025-12-10 09:16:00, ..., 2025-12-10 15:29:00]
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
        
        # Generate OHLC bars from price path with natural high/low progression
        for i in range(n):
            # Add small noise to open price (gap from previous close)
            if i == 0:
                opens[i] = price_path[i]
            else:
                gap_pct = np.random.uniform(-0.001, 0.001)
                gap_noise = price_path[i] * gap_pct
                opens[i] = price_path[i] + gap_noise
            
            closes[i] = price_path[i + 1]  # Next point is the close
            
            # Base high/low from open/close
            base_high = max(opens[i], closes[i])
            base_low = min(opens[i], closes[i])
            
            # Generate intrabar high/low with volatility-based range
            intrabar_pct = (volatility_index / 25) * 0.05  # 0-5% based on volatility
            highs[i] = base_high * (1 + intrabar_pct * np.random.uniform(0.5, 2.0))
            lows[i] = base_low * (1 - intrabar_pct * np.random.uniform(0.5, 2.0))
            
            # Generate volume - random variation around seed (avg will be ~seed_volume)
            volumes[i] = int(seed_volume * np.random.uniform(0.8, 1.2))
        
        # Now find which bars should reflect the extremes
        # If price_path[j] is maximum, then bars j-1 and j will touch that price
        max_path_idx = np.argmax(price_path)
        min_path_idx = np.argmin(price_path)
        
        # Ensure OHLC consistency for all bars first
        for i in range(n):
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])
            # Ensure positive prices
            lows[i] = max(0.01, lows[i])
        
        # Now set seed extremes for bars that touch those prices in price_path
        # Bar i uses price_path[i] as open and price_path[i+1] as close
        for i in range(n):
            # Check if this bar touches the maximum price
            if i == max_path_idx or (i > 0 and i == max_path_idx - 1):
                highs[i] = seed_high
            
            # Check if this bar touches the minimum price  
            if i == min_path_idx or (i > 0 and i == min_path_idx - 1):
                lows[i] = seed_low
        
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
        """Generate a price path from start to end that naturally reaches max_high and min_low."""
        # We need n+1 points (n bars have n+1 price points)
        path = np.zeros(n + 1)
        path[0] = start
        path[n] = end
        
        # Decide where to place extreme points (natural positions in timeline)
        # Place high somewhere in first 60% of path
        # Place low somewhere in middle-to-end 40-80% of path
        high_idx = np.random.randint(int(n * 0.2), int(n * 0.6))
        low_idx = np.random.randint(int(n * 0.4), int(n * 0.8))
        
        if self.model == 'bridge':
            # Use Brownian Bridge - directly generates constrained path
            path = generate_bridge_path(n, start, end, volatility_index, self.seed)
            
            # Naturally inject extremes at chosen points
            path[high_idx] = max_high
            path[low_idx] = min_low
            
            # Smooth the path around extremes to make transitions natural
            # Before high peak
            if high_idx > 2:
                for j in range(max(1, high_idx - 3), high_idx):
                    path[j] = path[j] * 0.7 + max_high * 0.3 * ((high_idx - j) / 3)
            # After high peak
            if high_idx < n - 2:
                for j in range(high_idx + 1, min(n, high_idx + 4)):
                    path[j] = path[j] * 0.7 + max_high * 0.3 * ((4 - (j - high_idx)) / 3)
            
            # Before low dip
            if low_idx > 2:
                for j in range(max(1, low_idx - 3), low_idx):
                    path[j] = path[j] * 0.7 + min_low * 0.3 * ((low_idx - j) / 3)
            # After low dip
            if low_idx < n - 2:
                for j in range(low_idx + 1, min(n, low_idx + 4)):
                    path[j] = path[j] * 0.7 + min_low * 0.3 * ((4 - (j - low_idx)) / 3)
            
            return path
        
        # GBM model
        path = generate_gbm_path(n, start, end, volatility_index, self.seed)
        
        # Inject exact extreme values at chosen points
        path[high_idx] = max_high
        path[low_idx] = min_low
        
        # Smooth transitions around extremes for natural movement
        # Gradual build-up to high
        if high_idx > 2:
            for j in range(max(1, high_idx - 3), high_idx):
                blend_factor = (high_idx - j) / 3
                path[j] = path[j] * 0.6 + max_high * 0.4 * (1 - blend_factor)
        # Gradual descent from high
        if high_idx < n - 2:
            for j in range(high_idx + 1, min(n, high_idx + 4)):
                blend_factor = (j - high_idx) / 3
                path[j] = path[j] * 0.6 + max_high * 0.4 * (1 - blend_factor)
        
        # Gradual descent to low
        if low_idx > 2:
            for j in range(max(1, low_idx - 3), low_idx):
                blend_factor = (low_idx - j) / 3
                path[j] = path[j] * 0.6 + min_low * 0.4 * (1 - blend_factor)
        # Gradual recovery from low
        if low_idx < n - 2:
            for j in range(low_idx + 1, min(n, low_idx + 4)):
                blend_factor = (j - low_idx) / 3
                path[j] = path[j] * 0.6 + min_low * 0.4 * (1 - blend_factor)
        
        return path
    

