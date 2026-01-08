"""
Simple Usage Example
====================

Demonstrates the usage of the Stock Data Simulation Engine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stock_data_simulator import StockDataSimulator
import pandas as pd

def main():
    print("="*70)
    print("Stock Data Movement Simulation Engine - Example")
    print("="*70)
    
    # Initialize simulator with seed for reproducibility
    model_type = 'gbm'  # Use model_type='gbm' for GBM or model_type='bridge' for Brownian Bridge
    simulator = StockDataSimulator(model=model_type, seed=42)
    
    # Define parameters
    start_time = "2025-12-10 09:15:00"
    end_time = "2025-12-10 15:30:00"
    granularity_minutes = 1
    volatility_index = 5
    ohlcv_seed = (100, 125, 95, 103, 5300)
    
    # Example parameters from requirements
    print("\nInput Parameters:")
    print("-" * 70)
    print(f"Model: {model_type.upper()}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Granularity: {granularity_minutes} minute")
    print(f"Volatility Index: {volatility_index} (scale 1-25)")
    print(f"OHLCV Seed: {ohlcv_seed}")
    
    # Generate synthetic data
    print("\nGenerating synthetic tick data...")
    df = simulator.simulate(
        start_time=start_time,
        end_time=end_time,
        granularity_minutes=granularity_minutes,
        volatility_index=volatility_index,
        ohlcv_seed=ohlcv_seed
    )
    
    print(f"[OK] Generated {len(df)} data points")
    
    # Display sample data
    print("\n" + "="*70)
    print("Sample Output (First 5 rows):")
    print("="*70)
    print(df.head(5).to_string(index=False))
    
    print("\n" + "="*70)
    print("Sample Output (Last 5 rows):")
    print("="*70)
    print(df.tail(5).to_string(index=False))
    
    # Basic statistics
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"Total Ticks: {len(df)}")
    print(f"Initial Price: ${df['open'].iloc[0]:.2f}")
    print(f"Final Price: ${df['close'].iloc[-1]:.2f}")
    print(f"Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Total Volume: {df['volume'].sum():,.0f}")
    
    # Export to CSV with model name
    output_path = f'data/example_output_{model_type}.csv'
    try:
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Data exported to {output_path}")
    except PermissionError:
        # File is locked - try alternative filename with timestamp
        import time
        alt_path = f'data/example_output_{model_type}_{int(time.time())}.csv'
        try:
            df.to_csv(alt_path, index=False)
            print(f"\n[OK] Data exported to {alt_path}")
            print(f"  (Original file was locked, used alternative name)")
        except Exception as e:
            print(f"\n[WARNING] Could not export CSV: {e}")
            print(f"  (File may be open in Excel/Editor - close it and try again)")
    except Exception as e:
        print(f"\n[ERROR] Error exporting CSV: {e}")
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
