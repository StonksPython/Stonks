import pandas as pd
from pandas_datareader import data
#Return Market Capitalization
def marketCap(name):
    marketCap = data.get_quote_yahoo(name)['marketCap']
    temp =  marketCap[name]
    return temp
