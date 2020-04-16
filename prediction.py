import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
from numpy import sqrt,mean,log,diff
import quandl
quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
#Predicting Stock Price with Prophet
df = quandl.get('WIKI/GOOGL')
df = df.reset_index()
df = df.drop(columns=['Open', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])
df = df.rename(columns={"Date": "ds", "High": "y"})
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
print(forecast[['yhat']].iloc[-1])
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
fig = px.scatter(df, x="ds", y="y")
fig.show()