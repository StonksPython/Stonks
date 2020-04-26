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

n_resevoir = 500
sparsity = 0.2
rand_seed = 23
spectral_radius = 1.2
noise = 0.0005

