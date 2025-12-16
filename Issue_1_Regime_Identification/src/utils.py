"""
Utility functions for regime identification
"""

import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to weekly OHLCV
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: datetime, open, high, low, close, volume
    
    Returns:
    --------
    pd.DataFrame : Weekly OHLCV data
    """
    df = df.copy()
    df.set_index('datetime', inplace=True)
    
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    weekly.reset_index(inplace=True)
    return weekly


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to monthly OHLCV
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: datetime, open, high, low, close, volume
    
    Returns:
    --------
    pd.DataFrame : Monthly OHLCV data
    """
    df = df.copy()
    df.set_index('datetime', inplace=True)
    
    monthly = df.resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    monthly.reset_index(inplace=True)
    return monthly


def plot_regime_matrix(matrix: pd.DataFrame, title: str = "Regime Distribution"):
    """
    Plot regime classification matrix as heatmap
    
    Parameters:
    -----------
    matrix : pd.DataFrame
        Regime classification matrix
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Remove 'All' row and column for cleaner visualization
    plot_matrix = matrix.iloc[:-1, :-1]
    
    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Count'},
        linewidths=0.5
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Volatility Regime', fontsize=12)
    plt.ylabel('Direction Regime', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_price_with_regimes(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    stock_name: str = "Stock"
):
    """
    Plot price chart with regime classifications overlaid
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original OHLCV data
    regimes_df : pd.DataFrame
        Regime classifications
    stock_name : str
        Stock name for title
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Price with regimes
    ax1.plot(df['datetime'], df['close'], label='Close Price', color='black', linewidth=1)
    
    # Color background by regime
    colors_vol = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    
    for _, regime in regimes_df.iterrows():
        ax1.axvspan(
            regime['start_date'],
            regime['end_date'],
            alpha=0.2,
            color=colors_vol.get(regime['volatility_regime'], 'gray'),
            label=f"{regime['volatility_regime']} Vol" if _ == 0 else ""
        )
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{stock_name} - Price with Volatility Regimes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Direction Regimes
    direction_map = {'Falling': -1, 'Sideways': 0, 'Rising': 1}
    regimes_df['direction_numeric'] = regimes_df['direction_regime'].map(direction_map)
    
    ax2.fill_between(
        regimes_df['start_date'],
        0,
        regimes_df['direction_numeric'],
        step='post',
        alpha=0.7,
        color=['red' if x < 0 else 'green' if x > 0 else 'gray' 
               for x in regimes_df['direction_numeric']]
    )
    ax2.set_ylabel('Direction', fontsize=12)
    ax2.set_title('Direction Regimes', fontsize=14, fontweight='bold')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Falling', 'Sideways', 'Rising'])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility Z-Scores
    ax3.plot(
        regimes_df['start_date'],
        regimes_df['volatility_z_score'],
        marker='o',
        label='Volatility Z-Score',
        color='purple'
    )
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=-0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_ylabel('Z-Score', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Volatility Z-Score', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def get_regime_statistics(regimes_df: pd.DataFrame) -> Dict:
    """
    Calculate statistics for regime classifications
    
    Parameters:
    -----------
    regimes_df : pd.DataFrame
        Regime classifications
    
    Returns:
    --------
    Dict : Statistics summary
    """
    stats = {
        'total_periods': len(regimes_df),
        'volatility_distribution': regimes_df['volatility_regime'].value_counts().to_dict(),
        'direction_distribution': regimes_df['direction_regime'].value_counts().to_dict(),
        'avg_volatility_z_score': regimes_df['volatility_z_score'].mean(),
        'avg_direction_z_score': regimes_df['direction_z_score'].mean(),
        'most_common_regime': f"{regimes_df['direction_regime'].mode()[0]} + {regimes_df['volatility_regime'].mode()[0]}"
    }
    
    return stats


def export_regime_report(
    regimes_df: pd.DataFrame,
    matrix: pd.DataFrame,
    output_path: str,
    stock_name: str = "Stock"
):
    """
    Export regime analysis to CSV
    
    Parameters:
    -----------
    regimes_df : pd.DataFrame
        Regime classifications
    matrix : pd.DataFrame
        Regime distribution matrix
    output_path : str
        Output file path
    stock_name : str
        Stock name
    """
    with open(output_path, 'w') as f:
        f.write(f"Regime Analysis Report - {stock_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Statistics
        stats = get_regime_statistics(regimes_df)
        f.write("STATISTICS\n")
        f.write("-" * 60 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Regime Matrix
        f.write("REGIME DISTRIBUTION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(matrix.to_string())
        f.write("\n\n")
        
        # Detailed Classifications
        f.write("DETAILED REGIME CLASSIFICATIONS\n")
        f.write("-" * 60 + "\n")
    
    # Append regime details
    regimes_df.to_csv(output_path, mode='a', index=False)
    print(f"Report exported to {output_path}")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
