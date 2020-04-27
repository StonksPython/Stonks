# Stonks
Trading Algorithm I am Developing. Made Entirely from Scratch, does not use existing competition framework like Quantopian or Quantisi Blueshift. The Algorithm takes advantage of the Alpaca API to conduct 0% Commision Trades.

### Disclaimer:
I know next to nothing about trading. This algorithm would probably be lousy in a corporate setting, and this project serves mostly for me to learn about different techniques for Data Analysis and Data Visualization in Python. If it sounds like I don't know what I'm writing about, I probably didn't at the time. This file is just a rough space for me to plan out my next steps. Please try not to roast me too hard, but leave constructive criticism!

## General Proccess behind the Algorithm is as follows:

 - Identify Buying Power
 - Adjust max/min share price accordingly
 - Predict next day share price
 - Calculate return
 - Market Cap
 - Volatility
 - Liquidity
 - Identify Sentiment
 
 Classify as Hold, Buy, or Sell

## Possible Bottlenecks

The main bottleneck is API Call Frequency. AlphaVantage only allows 5 API Calls a minute, and 500 a day. This bottleneck is reached very easily when dealing with multiple stocks.

### Possible Solution to Bottleneck

Possible Solution would include collected Data for 5 Stocks at a time, every minute, until all Data is collected, and Stored Locally on CSV Files. This would make it very easy to read data, but would require an organized file system, with a standard naming convention.

Another possible solution would be to read the data into a SQL Database. This would keep all the data stored in one, nice centralized place, and you wouldn't really have to worry about CSV file corruption, or naming issues, or whatnot. Since the algorithm is initially small scale, I don't see a real benefit to implementing this over the CSV File Storage, especially since the CSV Files are so much easier to work with.
