from StrategyData import StrategyData

import time
import requests
import pandas as pd
import datetime
import yaml
import os

class StrategyFinazon(StrategyData):

    def download_data(self, ticker, days, interval):
        url = "https://api.finazon.io/latest/finazon/us_stocks_essential/time_series"
        ls = []
        i = 0
        with open("api_key.yml", "r") as file:
            api_key = yaml.safe_load(file)
            while True:
                querystring = {"ticker":f"{ticker}",
                            "interval": interval,
                            "page":f"{i}",
                            "page_size":"1000",
                            "adjust":"all",
                            "start_at":int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp()),
                            "end_at":int(datetime.datetime.now().timestamp())
                            }
                
                headers = {"Authorization": api_key["api_key"]}
                
                while True:
                    response = requests.get(url, headers=headers, params=querystring)
                    if response.status_code == 429:
                        print(f"waiting 1 second. {i} calls completed")
                        time.sleep(1)
                    else:
                        break
                data = response.json()
                
                if len(data["data"]) == 0:
                    break

                pandas_data = [{"Datetime": datetime.datetime.fromtimestamp(x["t"]),
                                "Open": x["o"],
                                "High": x["h"],
                                "Low": x["l"],
                                "Close": x["c"],
                                "Volume": x["v"]
                                } for x in data["data"]]
                for item in pandas_data:
                    ls.append(item)

                i += 1

        df = pd.DataFrame(ls)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

        return df