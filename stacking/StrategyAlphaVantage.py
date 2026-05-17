from StrategyData import StrategyData

import time
import requests
import pandas as pd
import datetime
import yaml
import os

class StrategyAlphaVantage(StrategyData):

    def download_data(self, ticker, days, interval):
        with open("api_key.yml", "r") as file:
            api_key = yaml.safe_load(file)
            url = f'https://www.alphavantage.co/query'

            if interval == '1d':
                querystring = {
                            "function": f"TIME_SERIES_DAILY",
                            "symbol": ticker,
                            "apikey": api_key['alpha_vantage'],
                            "outputsize": "full"
                            }
            else:
                querystring = {
                            "function": f"TIME_SERIES_INTRADAY",
                            "symbol": ticker,
                            "interval": self.translate_interval(interval),
                            "apikey": api_key['alpha_vantage'],
                            "outputsize": "full"
                            }

            response = requests.get(url, params=querystring)

            if response.status_code == 200:
                data = response.json()

                # Extract the time series data
                time_series_key = next(key for key in data.keys() if f'Time Series ({self.translate_interval(interval)})' in key)
                time_series_data = data[time_series_key]

                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series_data, orient='index')

                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
                df = df.apply(pd.to_numeric)

                return self.filter_data_by_date(df, days=days)
            else:
                raise ValueError("Data hasn't been downloaded")
            
    def filter_data_by_date(self, df, days):
        # Convert start and end date to datetime objects
        start = pd.to_datetime(datetime.datetime.now() - datetime.timedelta(days=days))
        end = pd.to_datetime(datetime.datetime.now())

        # Filter the DataFrame by the date range
        filtered_df = df.loc[start:end]

        return filtered_df
    
    def translate_interval(self, interval):
        """Converts the past in `interval` string to an alpha vantage interval

        Raises:
            ValueError: If the interval is not found

        Returns:
            pd.Timedelta: The converted timedelta
        """
        interval_mapping = {
            '1d': 'Daily',
            '1h': '60min',
            '30m': '30min',
            '15m': '15min',
            '5m': '5min',
            '1m': '1min'
        }

        if interval in interval_mapping:
            return interval_mapping[interval]
        else:
            raise ValueError(f"Interval mapping not found for the specified interval: {interval}")
