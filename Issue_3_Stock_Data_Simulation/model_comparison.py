"""
Model Comparison Example
========================

Demonstrates switching between GBM and Brownian Bridge models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stock_data_simulator import StockDataSimulator
import pandas as pd

def main():
    print("="*70)
    print("Stock Data Simulation - Model Comparison")
    print("="*70)
    
    # Common parameters
    params = {
        'start_time': "2025-12-10 09:15:00",
        'end_time': "2025-12-10 09:30:00",
        'granularity_minutes': 1,
        'volatility_index': 10,
        'ohlcv_seed': (100, 125, 95, 103, 5300)
    }
    
    print("\nSimulation Parameters:")
    print(f"  Time Range: {params['start_time']} to {params['end_time']}")
    print(f"  Granularity: {params['granularity_minutes']} minute")
    print(f"  Volatility Index: {params['volatility_index']}")
    print(f"  OHLCV Seed: {params['ohlcv_seed']}")
    
    # Generate using GBM model
    print("\n" + "-"*70)
    print("1. GBM (Geometric Brownian Motion) Model")
    print("-"*70)
    sim_gbm = StockDataSimulator(model='gbm', seed=42)
    df_gbm = sim_gbm.simulate(**params)
    
    print(f"Generated: {len(df_gbm)} ticks")
    print(f"Start Price: ${df_gbm['open'].iloc[0]:.2f}")
    print(f"End Price: ${df_gbm['close'].iloc[-1]:.2f}")
    print(f"Price Range: ${df_gbm['low'].min():.2f} - ${df_gbm['high'].max():.2f}")
    print(f"Average Volume: {df_gbm['volume'].mean():.0f}")
    
    print("\nFirst 3 ticks (GBM):")
    print(df_gbm[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(3).to_string(index=False))
    
    # Generate using Brownian Bridge model
    print("\n" + "-"*70)
    print("2. Brownian Bridge Model")
    print("-"*70)
    sim_bridge = StockDataSimulator(model='bridge', seed=42)
    df_bridge = sim_bridge.simulate(**params)
    
    print(f"Generated: {len(df_bridge)} ticks")
    print(f"Start Price: ${df_bridge['open'].iloc[0]:.2f}")
    print(f"End Price: ${df_bridge['close'].iloc[-1]:.2f}")
    print(f"Price Range: ${df_bridge['low'].min():.2f} - ${df_bridge['high'].max():.2f}")
    print(f"Average Volume: {df_bridge['volume'].mean():.0f}")
    
    print("\nFirst 3 ticks (Bridge):")
    print(df_bridge[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(3).to_string(index=False))
    
    # Comparison
    print("\n" + "="*70)
    print("Model Comparison Summary")
    print("="*70)
    
    print("\nKey Differences:")
    print(f"  GBM Model:")
    print(f"    - Uses pure random walk with drift")
    print(f"    - Price moves freely (constrained after generation)")
    print(f"    - Good for unconstrained simulations")
    
    print(f"\n  Brownian Bridge Model:")
    print(f"    - Constrained to start and end values")
    print(f"    - Price path 'knows' where it needs to end")
    print(f"    - Better for scenarios with fixed endpoints")
    
    print(f"\nBoth models:")
    print(f"  - Generate exactly {len(df_gbm)} ticks")
    print(f"  - Start at ${params['ohlcv_seed'][0]:.2f}")
    print(f"  - End at ${params['ohlcv_seed'][3]:.2f}")
    print(f"  - Respect OHLCV constraints")
    print(f"  - Include realistic volume patterns")
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
