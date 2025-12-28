"""
Unit Tests for Stock Data Simulation Engine
============================================

Test suite covering:
1. Granularity calculation
2. OHLCV validation
3. Volatility effects
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

from stock_data_simulator import StockDataSimulator
from validators import validate_ohlcv, validate_simulation_params


class TestGranularityCalculation(unittest.TestCase):
    """Test correct granularity and tick count calculation."""
    
    def test_exact_example_case(self):
        """Test the exact example: 375 ticks from 9:15 AM to 3:30 PM."""
        simulator = StockDataSimulator(seed=42)
        
        df = simulator.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=1,
            volatility_index=5,
            ohlcv_seed=(100, 125, 95, 103, 5300)
        )
        
        # 9:15 AM to 3:30 PM = 375 minutes = 375 intervals
        self.assertEqual(len(df), 375)
    
    def test_one_hour_one_minute_granularity(self):
        """Test 1-hour period with 1-minute bars."""
        simulator = StockDataSimulator(seed=42)
        
        df = simulator.simulate(
            start_time="2025-12-10 10:00:00",
            end_time="2025-12-10 11:00:00",
            granularity_minutes=1,
            volatility_index=5,
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        self.assertEqual(len(df), 60)  # 60 minutes = 60 intervals
    
    def test_five_minute_granularity(self):
        """Test 1-hour period with 5-minute bars."""
        simulator = StockDataSimulator(seed=42)
        
        df = simulator.simulate(
            start_time="2025-12-10 10:00:00",
            end_time="2025-12-10 11:00:00",
            granularity_minutes=5,
            volatility_index=5,
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        self.assertEqual(len(df), 12)  # 60 minutes / 5 = 12 intervals


class TestOHLCVValidation(unittest.TestCase):
    """Test OHLCV data validation."""
    
    def test_valid_ohlcv(self):
        """Test valid OHLCV data."""
        is_valid, msg = validate_ohlcv(100, 105, 95, 102, 1000)
        self.assertTrue(is_valid)
    
    def test_high_less_than_open(self):
        """Test invalid case: high < open."""
        is_valid, msg = validate_ohlcv(100, 95, 90, 102, 1000)
        self.assertFalse(is_valid)
    
    def test_low_greater_than_close(self):
        """Test invalid case: low > close."""
        is_valid, msg = validate_ohlcv(100, 105, 103, 102, 1000)
        self.assertFalse(is_valid)
    
    def test_negative_values(self):
        """Test invalid case: negative values."""
        is_valid, msg = validate_ohlcv(-100, 105, 95, 102, 1000)
        self.assertFalse(is_valid)
    
    def test_simulator_produces_valid_ohlcv(self):
        """Test that simulator always produces valid OHLCV data."""
        simulator = StockDataSimulator(seed=42)
        
        df = simulator.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 10:00:00",
            granularity_minutes=1,
            volatility_index=10,
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        # Check every row
        for _, row in df.iterrows():
            is_valid, _ = validate_ohlcv(
                row['open'], row['high'], row['low'], 
                row['close'], row['volume']
            )
            self.assertTrue(is_valid)


class TestVolatilityEffects(unittest.TestCase):
    """Test that volatility index affects price movements."""
    
    def test_low_vs_high_volatility_price_movement(self):
        """Test that higher volatility produces larger price movements."""
        # Low volatility
        sim_low = StockDataSimulator(seed=42)
        df_low = sim_low.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=1,
            volatility_index=1,  # Low volatility
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        # High volatility
        sim_high = StockDataSimulator(seed=42)
        df_high = sim_high.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=1,
            volatility_index=25,  # High volatility
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        # Calculate price ranges
        price_range_low = df_low['high'].max() - df_low['low'].min()
        price_range_high = df_high['high'].max() - df_high['low'].min()
        
        # High volatility should have larger price range
        self.assertGreater(price_range_high, price_range_low)
    
    def test_volatility_affects_intrabar_spread(self):
        """Test that volatility affects high-low spread within bars."""
        # Low volatility
        sim_low = StockDataSimulator(seed=42)
        df_low = sim_low.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 10:00:00",
            granularity_minutes=1,
            volatility_index=1,
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        # High volatility
        sim_high = StockDataSimulator(seed=42)
        df_high = sim_high.simulate(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 10:00:00",
            granularity_minutes=1,
            volatility_index=25,
            ohlcv_seed=(100, 125, 95, 103, 1000)
        )
        
        # Calculate average intrabar spread
        spread_low = (df_low['high'] - df_low['low']).mean()
        spread_high = (df_high['high'] - df_high['low']).mean()
        
        # High volatility should have larger spreads
        self.assertGreater(spread_high, spread_low)


class TestParameterValidation(unittest.TestCase):
    """Test input parameter validation."""
    
    def test_valid_parameters(self):
        """Test valid simulation parameters."""
        is_valid, msg = validate_simulation_params(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=1,
            volatility_index=5
        )
        self.assertTrue(is_valid)
    
    def test_invalid_time_order(self):
        """Test invalid case: start_time >= end_time."""
        is_valid, msg = validate_simulation_params(
            start_time="2025-12-10 15:30:00",
            end_time="2025-12-10 09:15:00",
            granularity_minutes=1,
            volatility_index=5
        )
        self.assertFalse(is_valid)
    
    def test_invalid_volatility_range(self):
        """Test invalid case: volatility_index out of range."""
        is_valid, msg = validate_simulation_params(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=1,
            volatility_index=30  # > 25
        )
        self.assertFalse(is_valid)
    
    def test_invalid_granularity(self):
        """Test invalid case: negative granularity."""
        is_valid, msg = validate_simulation_params(
            start_time="2025-12-10 09:15:00",
            end_time="2025-12-10 15:30:00",
            granularity_minutes=-1,
            volatility_index=5
        )
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()
