import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
import quandl
quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
#Predicting Stock Price with Prophet
df = quandl.get('WIKI/GOOGL')
fig3 = df.plot(y='High')
fig3.figure.savefig('/home/dev/Stonks/preIndexReset.png')
df = df.reset_index(0)
df = df.drop(columns=['Open', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])
df = df.rename(columns={"Date": "ds", "High": "y"})
fig = df.plot(x='ds', y='y')
fig.figure.savefig('/home/dev/Stonks/preProphet.png')
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
print(forecast[['yhat']].iloc[-1])
fig1.savefig('/home/dev/Stonks/postProphet.png')