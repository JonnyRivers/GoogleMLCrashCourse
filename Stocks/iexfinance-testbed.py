# https://pypi.org/project/iexfinance/

# Stock
from iexfinance import Stock

tsla = Stock('TSLA')
print(tsla.get_open())
print(tsla.get_price())

# Historical Data
from iexfinance import get_historical_data
from datetime import datetime

start = datetime(2017, 2, 9)
end = datetime(2017, 5, 24)

df = get_historical_data("AAPL", start=start, end=end, output_format='pandas')
print(df.head())

# Chart it
import matplotlib.pyplot as plt

df.plot()
plt.show()