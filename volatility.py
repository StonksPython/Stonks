#!/usr/bin/env python3
#Code is Based of this site: https://www.quantconnect.com/tutorials/introduction-to-options/historical-volatility-and-implied-volatility
import pandas as pd
import numpy as np
#Historical Volitiliy is based on past performance - how much stock varies from Market
def get_dataframe(name):
    df = pd.read_csv(('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=JA1VCTFBG7378ZB7&datatype=csv'))
    return df
def historicalVolatility(name):
    df = get_dataframe(name)
    close = df['close']
    r = np.diff(np.log(close))
    r_mean = np.mean(r)
    diff_square = [(r[i]-r_mean)**2 for i in range(0,len(r))]
    std = np.sqrt(sum(diff_square)*(1.0/(len(r)-1)))
    vol = std*np.sqrt(252)
    return vol