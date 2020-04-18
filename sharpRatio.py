import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
fb = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'FB' +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
ibm = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'IBM' +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
amzn = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AMZN' +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
google = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'GOOGL' +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
apple = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AAPL' +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
for df in (fb, ibm, amzn, google, apple):
    df = df.rename(columns={"timestamp":"Date"})
    df = df.setIndex(df['Date'])
    df = df.sort_index()
    df['Normalized Return'] = df['close']/df.iloc[0]['close']
