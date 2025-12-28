"""
Simple Usage Example
====================

Demonstrates basic usage of the Stock Data Simulation Engine.
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
    simulator = StockDataSimulator(seed=42)
    
    # Example parameters from requirements
    print("\nInput Parameters:")
    print("-" * 70)
    print("Start Time: 2025-12-10 09:15:00")
    print("End Time: 2025-12-10 15:30:00")
    print("Granularity: 1 minute")
    print("Volatility Index: 5 (scale 1-25)")
    print("OHLCV Seed: (100, 125, 95, 103, 5300)")
    
    # Generate synthetic data
    print("\nGenerating synthetic tick data...")
    df = simulator.simulate(
        start_time="2025-12-10 09:15:00",
        end_time="2025-12-10 15:30:00",
        granularity_minutes=1,
        volatility_index=20,
        ohlcv_seed=(100, 1510, 80, 90, 5300)
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
    
    # Export to CSV
    output_path = 'data/example_output.csv'
    try:
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Data exported to {output_path}")
    except PermissionError:
        # File is locked - try alternative filename
        import time
        alt_path = f'data/example_output_{int(time.time())}.csv'
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
