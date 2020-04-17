import pandas as pd
from pandas_datareader import data
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
#Filter out API-Warnings
if sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#Return Market Capitalization
def marketCap(name):
    marketCap = data.get_quote_yahoo(name)['marketCap']
    temp =  marketCap[name]
    return temp
i = marketCap('GOOGL')
print(i)