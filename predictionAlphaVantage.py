import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
from tqdm import tqdm
import time
import requests
#Predicting Stock Price with Prophet
predictedPrices = {}
def get_dataframe(name):
    
    df = pd.read_csv(('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=JA1VCTFBG7378ZB7&datatype=csv'))
    return df
def run_prophet(df):
    df = df.rename(columns={"timestamp": "Date"})
    #fig3 = df.plot(y='high')
    #fig3.figure.savefig('/home/homuser/Stonks/preIndexReset.png')
    df = df.reset_index(0)
    df = df.drop(columns=['open', 'low', 'close', 'volume'])
    df = df.rename(columns={"Date": "ds", "high": "y"})
    #fig = df.plot(x='ds', y='y')
    #fig.figure.savefig('/home/homeuser/Stonks/preProphet.png')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=5)
    forecast = m.predict(future)
    #fig1 = m.plot(forecast)
    #fig1.savefig('/home/homeuser/Stonks/postProphet.png')
    i = forecast[['yhat']].iloc[-1]
    predictedPrices['name'].append(i['yhat'])
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

#multiprocessing implemented - https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245
from multiprocessing import Pool, cpu_count
start_time = time.time()
p = Pool(cpu_count())
predictions = list(tqdm(p.imap(run_prophet, series), total=len(series)))
p.close()
p.join()
print("--- %s seconds ---" % (time.time() - start_time))