#!/usr/bin/env python3
import pandas as pd

def get_dataframe(name):
    
    df = pd.read_csv(('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=JA1VCTFBG7378ZB7&datatype=csv'))
    return df

def allTimePointChange(name):
    df = get_dataframe(name)
    last = df['close'][-1]
    first = df['close'][0]
    return first/last * 100