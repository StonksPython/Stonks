import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
from tqdm import tqdm
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pyESN import ESN 

#Implementation based off: https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4

df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AMZN' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
df = df.rename(columns={"timestamp":"Date"})
df = df.set_index(df['Date'])
df = df.sort_index()
df = df.drop(columns=['open', 'low', 'high', 'volume'])
df = df.drop(columns=['Date'])
df = df.reset_index()
df = df.drop(columns=['Date'])
df = df.reset_index()
df = df.rename(columns={"index":"x", "close":"y"})
print(df.head())
print(df.tail())

n_resevoir = 500
sparsity = 0.2
rand_seed = 23
spectral_radius = 1.2
noise = 0.0005

esn = ESN(n_inputs=1,n_outputs=1,n_reservoir=n_resevoir,sparsity=sparsity,random_state=rand_seed,spectral_radius=spectral_radius,noise=noise)
#First training will be with 2,000 datapoints to test accuracy
trainlen = 2000
#We want to predict the next day
future = 1
#we want to keep predicting this way for the next 3032 points
futureTotal = 3032
predictedTotal = np.zeroes(futureTotal)

#travers futureTotal by future days at a time
for i in range(0, futureTotal, future):
    predictedTraining = esn.fit(np.ones(trainlen), df['y'][i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    predictedTotal[i:i+future] = prediction[:,0]