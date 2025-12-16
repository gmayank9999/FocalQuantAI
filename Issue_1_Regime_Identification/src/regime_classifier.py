"""
Regime Classification Module
Classifies market periods into volatility and directional regimes
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class RegimeClassifier:
    """
    Classify market periods into regimes based on:
    1. Volatility: Low | Medium | High
    2. Direction: Falling | Sideways | Rising
    
    Uses Relative Z-Scoring methodology:
    - Reference period: Last 30 days for mean and std calculation
    - Focus period: Current + 90 days data to be classified
    """
    
    def __init__(
        self,
        reference_window: int = 30,
        focus_window: int = 90,
        volatility_thresholds: Tuple[float, float] = (-0.5, 0.5),
        direction_thresholds: Tuple[float, float] = (-0.5, 0.5)
    ):
        """
        Initialize the regime classifier
        
        Parameters:
        -----------
        reference_window : int
            Number of days for reference period (default 30)
        focus_window : int
            Number of days for focus period (default 90)
        volatility_thresholds : Tuple[float, float]
            Z-score thresholds for volatility (low_threshold, high_threshold)
        direction_thresholds : Tuple[float, float]
            Z-score thresholds for direction (falling_threshold, rising_threshold)
        """
        self.reference_window = reference_window
        self.focus_window = focus_window
        self.vol_low_thresh, self.vol_high_thresh = volatility_thresholds
        self.dir_fall_thresh, self.dir_rise_thresh = direction_thresholds
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from close prices"""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        return df
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate volatility (standard deviation of returns)"""
        return returns.std()
    
    def calculate_direction(self, prices: pd.Series) -> float:
        """
        Calculate direction (trend) using linear regression slope
        Positive slope = Rising, Negative = Falling
        """
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        y = prices.values
        
        # Linear regression: y = mx + b
        m = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
            (len(x) * np.sum(x**2) - np.sum(x)**2)
        
        return m
    
    def relative_z_score(
        self,
        value: float,
        reference_mean: float,
        reference_std: float
    ) -> float:
        """
        Calculate relative Z-score
        Z = (value - reference_mean) / reference_std
        """
        if reference_std == 0:
            return 0.0
        return (value - reference_mean) / reference_std
    
    def classify_volatility(self, z_score: float) -> str:
        """
        Classify volatility based on Z-score
        
        Returns:
        --------
        str : 'Low', 'Medium', or 'High'
        """
        if z_score < self.vol_low_thresh:
            return 'Low'
        elif z_score > self.vol_high_thresh:
            return 'High'
        else:
            return 'Medium'
    
    def classify_direction(self, z_score: float) -> str:
        """
        Classify direction based on Z-score
        
        Returns:
        --------
        str : 'Falling', 'Sideways', or 'Rising'
        """
        if z_score < self.dir_fall_thresh:
            return 'Falling'
        elif z_score > self.dir_rise_thresh:
            return 'Rising'
        else:
            return 'Sideways'
    
    def identify_regime(
        self,
        df: pd.DataFrame,
        start_idx: int
    ) -> Dict:
        """
        Identify regime for a specific period
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        start_idx : int
            Index where focus period starts
        
        Returns:
        --------
        Dict : Contains regime classification and metrics
        """
        # Calculate reference period boundaries
        ref_start = max(0, start_idx - self.reference_window)
        ref_end = start_idx
        
        # Calculate focus period boundaries
        focus_start = start_idx
        focus_end = min(len(df), start_idx + self.focus_window)
        
        # Check if we have enough data
        # For small windows (monthly), allow smaller minimums
        # Need at least the window size itself
        min_ref_points = max(1, self.reference_window)
        min_focus_points = max(2, self.focus_window)
        
        if ref_end - ref_start < min_ref_points or focus_end - focus_start < min_focus_points:
            return None
        
        # Extract data
        ref_data = df.iloc[ref_start:ref_end].copy()
        focus_data = df.iloc[focus_start:focus_end].copy()
        
        # Calculate returns
        ref_data = self.calculate_returns(ref_data)
        focus_data = self.calculate_returns(focus_data)
        
        # Calculate volatility for both periods
        ref_volatility = self.calculate_volatility(ref_data['returns'].dropna())
        focus_volatility = self.calculate_volatility(focus_data['returns'].dropna())
        
        # Calculate direction for both periods
        ref_direction = self.calculate_direction(ref_data['close'])
        focus_direction = self.calculate_direction(focus_data['close'])
        
        # Calculate reference statistics
        ref_vol_mean = ref_volatility
        ref_vol_std = ref_data['returns'].rolling(5).std().std()
        
        ref_dir_mean = ref_direction
        ref_dir_std = abs(ref_direction) * 0.5  # Simple std estimate
        
        # Calculate Z-scores
        vol_z_score = self.relative_z_score(focus_volatility, ref_vol_mean, ref_vol_std)
        dir_z_score = self.relative_z_score(focus_direction, ref_dir_mean, ref_dir_std)
        
        # Classify regimes
        volatility_regime = self.classify_volatility(vol_z_score)
        direction_regime = self.classify_direction(dir_z_score)
        
        return {
            'start_date': df.iloc[focus_start]['datetime'],
            'end_date': df.iloc[focus_end - 1]['datetime'],
            'volatility_regime': volatility_regime,
            'direction_regime': direction_regime,
            'volatility_z_score': vol_z_score,
            'direction_z_score': dir_z_score,
            'focus_volatility': focus_volatility,
            'focus_direction': focus_direction,
            'ref_volatility': ref_volatility,
            'ref_direction': ref_direction
        }
    
    def classify_full_timeseries(
        self,
        df: pd.DataFrame,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Classify regimes for entire time series using rolling windows
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        step_size : int
            Number of days to step forward for each classification
        
        Returns:
        --------
        pd.DataFrame : DataFrame with regime classifications
        """
        results = []
        
        # Start from reference_window to have enough historical data
        for idx in range(self.reference_window, len(df), step_size):
            regime = self.identify_regime(df, idx)
            if regime:
                results.append(regime)
        
        return pd.DataFrame(results)
    
    def create_regime_matrix(self, regimes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime classification matrix showing distribution
        
        Returns:
        --------
        pd.DataFrame : Matrix showing count of each regime combination
        """
        matrix = pd.crosstab(
            regimes_df['direction_regime'],
            regimes_df['volatility_regime'],
            margins=True
        )
        
        # Reorder columns and rows
        col_order = ['Low', 'Medium', 'High', 'All']
        row_order = ['Falling', 'Sideways', 'Rising', 'All']
        
        # Ensure all columns and rows exist
        for col in col_order:
            if col not in matrix.columns:
                matrix[col] = 0
        for row in row_order:
            if row not in matrix.index:
                matrix.loc[row] = 0
        
        matrix = matrix[col_order].loc[row_order]
        
        return matrix


if __name__ == "__main__":
    print("Regime Classifier module loaded successfully!")
