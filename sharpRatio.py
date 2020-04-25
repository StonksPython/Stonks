import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
def get_dataframe(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
    df = df.rename(columns={"timestamp":"Date"})
    df = df.set_index(df['Date'])
    df = df.sort_index()
    df = df.drop(columns=['open', 'low', 'high', 'volume'])
    return df
def run_prophet(df):
    df = df.rename(columns={"timestamp": "Date"})
    df = df.reset_index(0)
    df = df.drop(columns=['open', 'low', 'close', 'volume'])
    df = df.rename(columns={"Date": "ds", "high": "y"})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=5)
    forecast = m.predict(future)
    i = forecast[['yhat']].iloc[-1]
    return i['yhat']
def get_series(names):
    series = []
    for name in names:
        df = get_dataframe(name)
        series.append(df)
    return series
#main is here
names = ['AAPL', 'GOOGL', 'FB', 'IBM', 'AMZN']
series = get_series(names)
stocks = pd.concat(series, axis = 1)
stocks = stocks.drop(columns={'Date'})
stocks.columns = ['aapl','googl','fb','ibm', 'amzn']

weights = np.array(np.random.random(5))
print('Random Weights:')
print(weights)

print('Rebalance')
weights = weights/np.sum(weights)
print(weights)

# expected return
print('Expected Portfolio Return')
exp_ret = np.sum((log_return.mean()*weights)*252)
print(exp_ret)

# expected volatility
print('Expected Volatility')
exp_vol = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))
print(exp_vol)

# Sharpe Ratio
print('Sharpe Ratio')
SR = exp_ret/exp_vol
print(SR)

import multiprocessing

log_return = np.log(stocks/stocks.shift(1))

ports = 5000
all_weights = np.zeros((ports, len(stocks.columns)))
ret_arr = np.zeros(ports)
vol_arr = np.zeros(ports)
sharpe_arr = np.zeros(ports)

for i in range(ports): 
    # weights 
    weights = np.array(np.random.random(5)) 
    weights = weights/np.sum(weights)  
	
    # save the weights
    all_weights[i,:] = weights
	
    # expected return 
    ret_arr[i] = np.sum((log_return.mean()*weights)*252)

    # expected volatility 
    vol_arr[i] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))

    # Sharpe Ratio 
    sharpe_arr[i] = ret_arr[i]/vol_arr[i]
