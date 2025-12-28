"""
Stock Data Movement Simulation Engine
======================================

A simulation engine for generating synthetic OHLCV tick data
using GBM (Geometric Brownian Motion) volatility model.
"""

from .stock_data_simulator import StockDataSimulator
from .gbm_model import generate_gbm_returns, scale_volatility
from .validators import validate_ohlcv, validate_simulation_params

__version__ = "1.0.0"
__all__ = [
    "StockDataSimulator",
    "generate_gbm_returns",
    "scale_volatility",
    "validate_ohlcv",
    "validate_simulation_params"
]
