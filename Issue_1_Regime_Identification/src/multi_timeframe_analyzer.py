"""
Multi-Timeframe Regime Identification
Supports Daily, Weekly, and Monthly regime analysis
"""

import pandas as pd
from typing import Dict, Tuple
from regime_classifier import RegimeClassifier
from utils import resample_to_weekly, resample_to_monthly


class MultiTimeframeRegimeAnalyzer:
    """
    Perform regime identification across multiple timeframes
    """
    
    def __init__(
        self,
        reference_window: int = 30,
        focus_window: int = 90,
        volatility_thresholds: Tuple[float, float] = (-0.5, 0.5),
        direction_thresholds: Tuple[float, float] = (-0.5, 0.5)
    ):
        """
        Initialize multi-timeframe analyzer
        
        Parameters:
        -----------
        reference_window : int
            Reference period window
        focus_window : int
            Focus period window
        volatility_thresholds : Tuple[float, float]
            Volatility classification thresholds
        direction_thresholds : Tuple[float, float]
            Direction classification thresholds
        """
        self.classifier = RegimeClassifier(
            reference_window=reference_window,
            focus_window=focus_window,
            volatility_thresholds=volatility_thresholds,
            direction_thresholds=direction_thresholds
        )
    
    def analyze_daily(
        self,
        df: pd.DataFrame,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Perform daily regime analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Daily OHLCV data
        step_size : int
            Step size for rolling analysis
        
        Returns:
        --------
        pd.DataFrame : Daily regime classifications
        """
        print("Analyzing Daily regimes...")
        regimes = self.classifier.classify_full_timeseries(df, step_size=step_size)
        regimes['timeframe'] = 'Daily'
        return regimes
    
    def analyze_weekly(
        self,
        df: pd.DataFrame,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Perform weekly regime analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Daily OHLCV data (will be resampled to weekly)
        step_size : int
            Step size for rolling analysis
        
        Returns:
        --------
        pd.DataFrame : Weekly regime classifications
        """
        print("Analyzing Weekly regimes...")
        weekly_df = resample_to_weekly(df)
        
        print(f"Weekly data points: {len(weekly_df)}")
        
        # Check if we have enough weekly data
        if len(weekly_df) < 26:
            print(f"⚠️ Warning: Not enough weekly data for analysis (have {len(weekly_df)}, need at least 26 weeks)")
            return pd.DataFrame()
        
        # Adjust windows for weekly data (approximately)
        weekly_classifier = RegimeClassifier(
            reference_window=6,  # ~6 weeks = ~1.5 months
            focus_window=13,     # ~13 weeks = ~3 months (reduced from 18)
            volatility_thresholds=(self.classifier.vol_low_thresh, self.classifier.vol_high_thresh),
            direction_thresholds=(self.classifier.dir_fall_thresh, self.classifier.dir_rise_thresh)
        )
        
        # Use step size of 4 for weekly (analyze every ~month)
        regimes = weekly_classifier.classify_full_timeseries(weekly_df, step_size=4)
        
        if len(regimes) > 0:
            regimes['timeframe'] = 'Weekly'
            print(f"Weekly regimes identified: {len(regimes)}")
        else:
            print(f"⚠️ Warning: No weekly regimes could be identified")
        
        return regimes
    
    def analyze_monthly(
        self,
        df: pd.DataFrame,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Perform monthly regime analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Daily OHLCV data (will be resampled to monthly)
        step_size : int
            Step size for rolling analysis
        
        Returns:
        --------
        pd.DataFrame : Monthly regime classifications
        """
        print("Analyzing Monthly regimes...")
        monthly_df = resample_to_monthly(df)
        
        print(f"Monthly data points: {len(monthly_df)}")
        
        # Check if we have enough monthly data
        if len(monthly_df) < 5:
            print(f"⚠️ Warning: Not enough monthly data for analysis (have {len(monthly_df)}, need at least 5 months)")
            return pd.DataFrame()
        
        # Adjust windows for monthly data
        # 30 days reference ≈ 1 month, 90 days focus ≈ 3 months
        monthly_classifier = RegimeClassifier(
            reference_window=1,  # 1 month reference (30 days)
            focus_window=3,      # 3 months focus (90 days)
            volatility_thresholds=(self.classifier.vol_low_thresh, self.classifier.vol_high_thresh),
            direction_thresholds=(self.classifier.dir_fall_thresh, self.classifier.dir_rise_thresh)
        )
        
        # Analyze every month (step_size=1)
        regimes = monthly_classifier.classify_full_timeseries(monthly_df, step_size=1)
        
        if len(regimes) > 0:
            regimes['timeframe'] = 'Monthly'
            print(f"Monthly regimes identified: {len(regimes)}")
        else:
            print(f"⚠️ Warning: No monthly regimes could be identified (need ref+focus = 4 months minimum)")
        
        return regimes
    
    def analyze_all_timeframes(
        self,
        df: pd.DataFrame,
        step_size: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform regime analysis on all timeframes
        
        Parameters:
        -----------
        df : pd.DataFrame
            Daily OHLCV data
        step_size : int
            Step size for rolling analysis
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : Regime classifications for each timeframe
        """
        results = {}
        
        # Daily analysis
        daily_results = self.analyze_daily(df, step_size=step_size)
        if len(daily_results) > 0:
            results['daily'] = daily_results
        
        # Weekly analysis
        weekly_results = self.analyze_weekly(df, step_size=step_size)
        if len(weekly_results) > 0:
            results['weekly'] = weekly_results
        
        # Monthly analysis
        monthly_results = self.analyze_monthly(df, step_size=step_size)
        if len(monthly_results) > 0:
            results['monthly'] = monthly_results
        
        return results
    
    def get_current_regime(
        self,
        df: pd.DataFrame,
        timeframe: str = 'daily'
    ) -> Dict:
        """
        Get current regime for specified timeframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
        timeframe : str
            'daily', 'weekly', or 'monthly'
        
        Returns:
        --------
        Dict : Current regime classification
        """
        if timeframe.lower() == 'daily':
            regimes = self.analyze_daily(df, step_size=30)
        elif timeframe.lower() == 'weekly':
            regimes = self.analyze_weekly(df, step_size=5)
        elif timeframe.lower() == 'monthly':
            regimes = self.analyze_monthly(df, step_size=1)
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        if len(regimes) == 0:
            return None
        
        # Return the most recent regime
        return regimes.iloc[-1].to_dict()
    
    def compare_timeframes(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare current regimes across all timeframes
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
        
        Returns:
        --------
        pd.DataFrame : Comparison of current regimes
        """
        comparison = []
        
        for timeframe in ['daily', 'weekly', 'monthly']:
            current = self.get_current_regime(df, timeframe)
            if current:
                comparison.append({
                    'timeframe': timeframe.capitalize(),
                    'volatility_regime': current['volatility_regime'],
                    'direction_regime': current['direction_regime'],
                    'start_date': current['start_date'],
                    'end_date': current['end_date']
                })
        
        return pd.DataFrame(comparison)


if __name__ == "__main__":
    print("Multi-timeframe Regime Analyzer module loaded successfully!")
