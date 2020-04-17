import pandas as pd
from numpy import sqrt,mean,log,diff
from pandas_datareader import data
import quandl
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
#Filter out API-Warnings
if sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#Return Stock Volatility
def marketCap(name):
    name = name[5:]
    marketCap = data.get_quote_yahoo(name)['marketCap']
    temp =  marketCap[name]
    return temp
i = marketCap('WIKI/GOOGL')
print(i)