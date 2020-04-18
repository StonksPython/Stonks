import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
fb = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'FB' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
ibm = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'IBM' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
amzn = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AMZN' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
google = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'GOOGL' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
apple = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AAPL' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
for df in (fb, ibm, amzn, google, apple):
    df['Normalized Return'] = df['close']/df.iloc[-1]['close']
    