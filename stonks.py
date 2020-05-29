#!/usr/bin/env python3
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
import time
#BackupKey = JA1VCTFBG7378ZB7
def get_dataframe(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
    df = df.rename(columns={"timestamp":"Date"})
    df = df.set_index(df['Date'])
    df = df.sort_index()
    df = df.drop(columns=['open', 'low', 'high', 'volume'])
    return df

def get_series(names):
    series = []
    for name in names:
        df = get_dataframe(name)
        series.append(df)
    return series

def getStocks(names):
    series = get_series(names)
    stocks = pd.concat(series, axis = 1)
    stocks = stocks.drop(columns={'Date'})
    stocks.columns = ['aapl','googl','fb','ibm', 'amzn']
    return stocks

def logReturn(stocks):
    log_return = np.log(stocks/stocks.shift(1))
    return log_return

def getRandomWeights():
    weights = np.array(np.random.random(5))
    print('Random Weights:')
    print(weights)
    return weights

def rebalanceWeights(weights):
    print('Rebalance')
    weights1 = weights/np.sum(weights)
    print(weights1)
    return weights1

def expectedReturn(log_return, weights):
    # expected return
    print('Expected Portfolio Return')
    exp_ret = np.sum((log_return.mean()*weights)*252)
    print(exp_ret)
    return exp_ret


def expectedVolatility(log_return, weights):
    # expected volatility
    print('Expected Volatility')
    exp_vol = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))
    print(exp_vol)
    return exp_vol

# Sharpe Ratio
def sharpeRatio(exp_ret, exp_vol):
    print('Sharpe Ratio')
    SR = exp_ret/exp_vol
    print(SR)
    return SR

def getArrayStats(stocks, ports):
    import multiprocessing
    all_weights = np.zeros((ports, len(stocks.columns)))
    ret_arr = np.zeros(ports)
    vol_arr = np.zeros(ports)
    sharpe_arr = np.zeros(ports)
    start_time = time.time()
    def getStats(i, log_return):
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
    for i in range(ports):
        p = multiprocessing.Process(target=getStats, args=(i,))
        p.start()
    print("--- %s seconds ---" % (time.time() - start_time))   
    arrayStats = {}
    arrayStats['weights'].append(all_weights)
    arrayStats['return'].append(ret_arr)
    arrayStats['volatility'].append(vol_arr)
    arrayStats['sharpe'].append(sharpe_arr)
    return arrayStats

from pandas_datareader import data
#Return Market Capitalization
def marketCap(name):
    marketCap = data.get_quote_yahoo(name)['marketCap']
    temp =  marketCap[name]
    return temp

def historicalVolatility(name):
    df = get_dataframe(name)
    close = df['close']
    r = np.diff(np.log(close))
    r_mean = np.mean(r)
    diff_square = [(r[i]-r_mean)**2 for i in range(0,len(r))]
    std = np.sqrt(sum(diff_square)*(1.0/(len(r)-1)))
    vol = std*np.sqrt(252)
    return vol

def allTimePointChange(name):
    df = get_dataframe(name)
    last = df['close'][-1]
    first = df['close'][0]
    return first/last * 100

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
    return i['yhat']

def get_series(names):
    series = []
    for name in names:
        df = get_dataframe(name)
        series.append(df)
    return series

from fbprophet import Prophet
import numpy as np
from tqdm import tqdm
import time
import requests
from multiprocessing import Pool, cpu_count

#multiprocessing implemented - https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245
def predictedPrices(names):
    series = get_series(names)
    start_time = time.time()
    p = Pool(cpu_count())
    predictions = list(tqdm(p.imap(run_prophet, series), total=len(series)))
    predictedPrices = {}
    count = 0
    for name in names:
        predictedPrices[name].append(predictions[count])
        count+=1
    p.join()
    print(predictions)
    print("--- %s seconds ---" % (time.time() - start_time))
    return predictedPrices

def calculate_ESN(name, rand_seed, nReservoir, spectralRadius, future, futureTotal):
    data = open(name+".txt").read().split()
    data = np.array(data).astype('float64')
    sparsity=0.2
    noise = .0005
    nReservoir = nReservoir *1
    spectralRadius = spectralRadius * 1
    future = future * 1
    futureTotal = futureTotal * 1



    esn = ESN(n_inputs = 1,
        n_outputs = 1, 
        n_reservoir = nReservoir,
        sparsity=sparsity,
        random_state=rand_seed,
        spectral_radius = spectralRadius,
        noise=noise)

    trainlen = data.__len__()-futureTotal
    pred_tot=np.zeros(futureTotal)

    for i in range(0,futureTotal,future):
        pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
        prediction = esn.predict(np.ones(future))
        pred_tot[i:i+future] = prediction[:,0]
    return pred_tot

def predictedPricesESN(names,rand_seed, nReservoir, spectralRadius, future, futureTotal):
    predictedPrices = {}
    for name in names:
        predictedPrices[name].append(calculate_ESN(name,rand_seed, nReservoir, spectralRadius, future, futureTotal))
    return predictedPrices


