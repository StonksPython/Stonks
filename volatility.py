#Code is Based of this site: https://www.quantconnect.com/tutorials/introduction-to-options/historical-volatility-and-implied-volatility
import pandas as pd
from numpy import sqrt,mean,log,diff
import quandl
quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
#Historical Volitiliy is based on past performance - how much stock varies from Market
def historicalVolatility(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
    # use the daily data of Google(NASDAQ: GOOG) from 01/2016 to 08/2016
    close = df['close']
    r = diff(log(close))
    r_mean = mean(r)
    diff_square = [(r[i]-r_mean)**2 for i in range(0,len(r))]
    std = sqrt(sum(diff_square)*(1.0/(len(r)-1)))
    vol = std*sqrt(252)
    return vol

i = historicalVolatility('IBM')
print(i)