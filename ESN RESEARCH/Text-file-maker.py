import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN 
#DATE OF COLLECTION IS JULY 20TH
df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'CNBS' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
df.astype({'close':'float64'}).dtypes
columns = df['close']
columns = columns[::-1]
np.savetxt('CNBS.txt', columns.values, fmt = '%f')