from StackingModel import StackingModel
from ContextData import ContextData
from StrategyYfinance import StrategyYfinance
from StrategyFinazon import StrategyFinazon
from StrategyAlphaVantage import StrategyAlphaVantage
from StrategyAlpaca import StrategyAlpaca
import seaborn as sns
import matplotlib.pyplot as plt

import time

def plot_price_time(context):
    plt.figure(figsize=(12, 6))
    plt.plot(context.data['Close'], label=f'{context.ticker} Closing Price', linewidth=1)
    plt.title(f'{context.ticker} Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.show()

def plot_price_change_time(context):
    plt.figure(figsize=(12, 6))
    plt.plot(context.data['price_change'], label=f'{context.ticker} Price Change in %', linewidth=1)
    plt.title(f'{context.ticker} % Price Change Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price Change (%)')
    plt.legend()
    plt.show()

def plot_price_change_distrubution(context):
    sns.displot(data=context.data, x="price_change", kde=True)
    plt.show()

def main():
    print('>Fetching data')
    context = ContextData(ticker="SPY", strategy=StrategyAlpaca(), days=20, interval="1m", market_hours_only=False)
    context.write_csv()
    start_time = time.time()

    plot_price_time(context)
    plot_price_change_time(context)
    plot_price_change_distrubution(context)


    model = StackingModel(context.data, context.interval)
    cv_scores, predicted_price, last_price, prediction_time = model.run()

    print('---------------------------')
    # print("RMSE: ", rmse)
    # print(f"RRMSE: {rrmse:.8f}%")
    # print(f"R^2: {r2:.8f}%")
    print(cv_scores)
    print(f"Price in next interval ({prediction_time}): {predicted_price}")
    print(f"Last price: {last_price}")

    end_time = time.time()
    print('---------------------------')
    print("Time taken: ", end_time-start_time)

if __name__ == "__main__":
    # Enable interactive mode for non-blocking plotting
    plt.ion()

    # Display the plot window in non-blocking mode
    plt.show(block=False)

    main()

    # while True:
    #     input("> Press key to exit")
    #     break