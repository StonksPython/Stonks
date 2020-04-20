import datetime as dt
import os
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from flask_caching import Cache

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60

@cache.memoize(timeout=TIMEOUT)
def get_dataframe(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
    df = df.rename(columns={"timestamp":"Date"})
    df = df.set_index(df['Date'])
    df = df.sort_index()
    df = df.drop(columns=['open', 'low', 'high', 'volume', 'Date'])
    return df
def get_series():
    names = ['AAPL', 'GOOGL', 'FB', 'IBM', 'AMZN']
    series = []
    for name in names:
        df = get_dataframe(name)
        series.append(df)
    for df in series:
        df['Normalized Return'] = df['close']/df.iloc[0]['close']
    stocks = pd.concat(series, axis = 1)
    stocks.columns = ['AAPL','GOOGL','FB','IBM', 'AMZN']
    stocks['Date'] = stocks.index
    return stocks


app.layout = html.Div([
    html.Div('Data was updated within the last {} seconds'.format(TIMEOUT)),
    dcc.Dropdown(
        id='live-dropdown',
        value='AAPL',
        options=[{'label': i, 'value': i} for i in get_series().columns]
    ),
    dcc.Graph(id='live-graph')
])

@app.callback(Output('live-graph', 'figure'),
              [Input('live-dropdown', 'value')])
def update_live_graph(value):
    df = get_series()
    return {
        'data': [{
            'x': df['Date'],
            'y': df[value],
            'line': {
                'width': 1,
                'color': '#0074D9',
                'shape': 'spline'
            }
        }],
    }




if __name__ == '__main__':
    app.run_server(debug=True)
