import pandas as pd
import plotly.express as px

df = pd.read_csv("esn.csv")
fig = px.box(df, y="Avg. 5 Day Percent Return (ESN)", points=False)
fig.show()
