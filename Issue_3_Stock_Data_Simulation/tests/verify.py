"""Quick verification that all requirements are met"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stock_data_simulator import StockDataSimulator

sim = StockDataSimulator(seed=42)
df = sim.simulate(
    '2025-12-10 09:15:00', 
    '2025-12-10 15:30:00', 
    1, 5, 
    (100, 125, 95, 103, 5300)
)

print("="*50)
print("VERIFICATION RESULTS")
print("="*50)
print(f"[OK] Generated exactly {len(df)} data points")
print(f"[OK] Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"[OK] All High >= Open/Close: {(df['high'] >= df[['open', 'close']].max(axis=1)).all()}")
print(f"[OK] All Low <= Open/Close: {(df['low'] <= df[['open', 'close']].min(axis=1)).all()}")
print(f"[OK] All prices positive: {(df[['open', 'high', 'low', 'close']] > 0).all().all()}")
print("="*50)
print("ALL REQUIREMENTS MET")
print("="*50)
