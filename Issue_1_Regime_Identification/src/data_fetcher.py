"""
Data Fetcher Module for ICICI Breeze API
Fetches OHLCV data for Indian market stocks
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
from dotenv import load_dotenv
from breeze_connect import BreezeConnect

# Load environment variables
load_dotenv()


class BreezeDataFetcher:
    """Fetch historical market data using ICICI Breeze API"""
    
    def __init__(self):
        """Initialize Breeze API connection"""
        self.api_key = os.getenv('BREEZE_API_KEY')
        self.api_secret = os.getenv('BREEZE_SECRET')
        self.api_session = os.getenv('MY_API_SESSION')
        
        if not all([self.api_key, self.api_secret, self.api_session]):
            raise ValueError("Missing API credentials. Check your .env file.")
        
        # Initialize Breeze connection
        self.breeze = BreezeConnect(api_key=self.api_key)
        self.breeze.generate_session(
            api_secret=self.api_secret,
            session_token=self.api_session
        )
    
    def fetch_historical_data(
        self,
        stock_code: str,
        exchange: str = "NSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a stock
        
        Parameters:
        -----------
        stock_code : str
            Stock symbol (e.g., 'RELIANCE', 'TCS')
        exchange : str
            Exchange name ('NSE', 'BSE')
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        interval : str
            Data interval ('1minute', '5minute', '1day', '1week', '1month')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Try the standard method first (as per ICICI Breeze documentation)
            response = self.breeze.get_historical_data(
                interval=interval,
                from_date=f"{start_date}T07:00:00.000Z",
                to_date=f"{end_date}T07:00:00.000Z",
                stock_code=stock_code,
                exchange_code=exchange,
                product_type="cash"
            )
            
            # If that fails, try v2
            if not response.get('Success') or len(response.get('Success', [])) == 0:
                response = self.breeze.get_historical_data_v2(
                    interval=interval,
                    from_date=f"{start_date}T07:00:00.000Z",
                    to_date=f"{end_date}T07:00:00.000Z",
                    stock_code=stock_code,
                    exchange_code=exchange,
                    product_type="cash"
                )
            
            # Check response
            if response['Status'] != 200:
                error_msg = response.get('Error', 'API returned non-200 status')
                raise Exception(f"API Error: {error_msg}")
            
            # Check if data is empty
            if not response.get('Success') or len(response['Success']) == 0:
                raise Exception(
                    f"No data returned for {stock_code}. "
                    f"Possible issues:\n"
                    f"1. Check if stock code is correct (try without .NS suffix)\n"
                    f"2. Verify date range is valid\n"
                    f"3. Ensure API session is active\n"
                    f"4. Check if market was open during this period"
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(response['Success'])
            
            # Rename columns to standard format
            df = df.rename(columns={
                'datetime': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert datetime to pandas datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert price and volume columns to numeric (API returns strings!)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values after conversion
            df = df.dropna(subset=['close'])
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            print(f"Error fetching data for {stock_code}: {str(e)}")
            raise
    
    def fetch_multiple_stocks(
        self,
        stock_list: List[str],
        exchange: str = "NSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1day"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks
        
        Parameters:
        -----------
        stock_list : List[str]
            List of stock symbols
        exchange : str
            Exchange name
        start_date : str, optional
            Start date
        end_date : str, optional
            End date
        interval : str
            Data interval
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with stock symbols as keys and DataFrames as values
        """
        data_dict = {}
        
        for stock in stock_list:
            print(f"Fetching data for {stock}...")
            try:
                data_dict[stock] = self.fetch_historical_data(
                    stock_code=stock,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
            except Exception as e:
                print(f"Failed to fetch {stock}: {str(e)}")
                continue
        
        return data_dict
    
    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        data_dir: str = "data/raw"
    ):
        """Save DataFrame to CSV"""
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


if __name__ == "__main__":
    print("Breeze Data Fetcher module loaded successfully!")
