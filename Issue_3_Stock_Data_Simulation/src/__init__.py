"""
Stock Data Movement Simulation Engine
======================================

A simulation engine for generating synthetic OHLCV tick data
using GBM (Geometric Brownian Motion) or Brownian Bridge models.
"""

from .stock_data_simulator import StockDataSimulator
from .gbm_model import generate_gbm_returns, generate_gbm_path, scale_volatility
from .bridge_model import generate_bridge_path, generate_bridge_returns
from .validators import validate_ohlcv, validate_simulation_params

__version__ = "1.0.0"
__all__ = [
    "StockDataSimulator",
    "generate_gbm_returns",
    "generate_gbm_path",
    "generate_bridge_path",
    "generate_bridge_returns",
    "scale_volatility",
    "validate_ohlcv",
    "validate_simulation_params"
]
