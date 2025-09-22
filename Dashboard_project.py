import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download("SPY")['Close']
data.to_csv("spy_data.csv")
print(data.index)
print('n\n\n\n')

#df.ffillna(method="ffill")

def data_download (ticker, filename):
    data = yf.download(ticker)['Close']
    data.to_csv(filename)
    return data

ticker_filename = {
    "SPY":"spy.csv",
    "DX-Y.NYB":"usd.csv",
    "GC=F":"gold.csv",
    "WTI":"wti.csv",
    "ZW=F":"wheat.csv",
    "^TNX":"bond.csv",
}

for ticker, filename in ticker_filename.items():
    data_download(ticker, filename)


spy = (pd.read_csv("spy_data.csv", index_col = 0, parse_dates = True)['SPY'].pct_change() + 1).cumprod()

## check missing values
def check_data(data):
    null_sum = data.isna().sum()
    null_percentage  = null_sum / len(data)
    print(f"Ratio of missing values: {null_percentage}n\ Number of missing values: {null_sum}") 

missing_values = data.isnull().sum() # number is zero, no missing values

# Plot the data
spy.plot(label='SPY')
plt.ylabel("SPY Closing Price")
plt.title("SPY Closing Price over time")

# Forex using USD index 
usd = (yf.download('DX-Y.NYB', start = spy.index.min())['Close'].pct_change() + 1).cumprod()
usd['DX-Y.NYB'].plot(label='USD Index')
usd.to_csv("usd_data.csv")


plt.legend
plt.show()

# for next time: we need to deal with the missing values / scale the data properly (log) / write one mother function that will call all the others / 
# function that plots in a systematic way 