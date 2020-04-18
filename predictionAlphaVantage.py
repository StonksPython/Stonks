import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
#Predicting Stock Price with Prophet
def prophetPredict(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
    df = df.rename(columns={"timestamp": "Date"})
    fig3 = df.plot(y='high')
    fig3.figure.savefig('/home/dev/Stonks/preIndexReset.png')
    df = df.reset_index(0)
    df = df.drop(columns=['open', 'low', 'close', 'volume'])
    df = df.rename(columns={"Date": "ds", "high": "y"})
    fig = df.plot(x='ds', y='y')
    fig.figure.savefig('/home/dev/Stonks/preProphet.png')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=5)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    fig1.savefig('/home/dev/Stonks/postProphet.png')
    i = forecast[['yhat']].iloc[-1]
    return i['yhat']
y = prophetPredict('AAPL')
print(y)