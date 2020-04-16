import pandas as pd
from numpy import sqrt,mean,log,diff
import quandl
from datetime import datetime
from dateutil.relativedelta import relativedelta

one_yr_ago = (datetime.today()- relativedelta(years=1)).strftime('%Y-%m')
six_months_ago = (datetime.today()- relativedelta(months=6)).strftime('%Y-%m')

print(one_yr_ago)
print(six_months_ago)
quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'

def allTimePointChange(name):
    stock_table = quandl.get(name)
    close = stock_table[['Adj. Close']]
    first = close['Adj. Close'].iloc[0]
    last = close['Adj. Close'].iloc[-1]
    percentageChange = last/first * 100
    print(percentageChange)
    return percentageChange
oneYearPointTrend('WIKI/GOOGL')