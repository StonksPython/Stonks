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
from matplotlib import rc


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
predictedTotal = np.zeros(futureTotal)
y = df['y']
y = y.to_numpy()
print('y data:')
print(type(y))
print(y)
#travers futureTotal by future days at a time

def predict(i, future, trainlen):
    predictedTraining = esn.fit(np.ones(trainlen), y[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    predictedTotal[i:i+future] = prediction[:,0]
start_time = time.time()

import multiprocessing
for i in range(trainlen):
    p = multiprocessing.Process(target=predict, args=(i,future,trainlen))
    p.start()
print("--- %s seconds ---" % (time.time() - start_time))
rc('text', usetex=False)
fig  = plt.figure(figsize=(16,8))
fig.plot(range(1000,trainlen+futureTotal),y[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
fig.plot(range(trainlen,trainlen+futureTotal),predictedTotal,'k',  alpha=0.8, label='Free Running ESN')

lo,hi = fig.ylim()
fig.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

fig.title(r'Ground Truth and Echo State Network Output', fontsize=25)
fig.xlabel(r'Time (Days)', fontsize=20,labelpad=10)
fig.ylabel(r'Price ($)', fontsize=20,labelpad=10)
fig.legend(fontsize='xx-large', loc='best')
fig.figure.savefig('/home/homeuser/Stonks/ESN.png')

sns.despine()
