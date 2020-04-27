# Stonks
Trading Algorithm I am Developing. Made Entirely from Scratch, does not use existing competition framework like Quantopian or Quantisi Blueshift. The Algorithm takes advantage of the Alpaca API to conduct 0% Commision Trades.

### Disclaimer:
I know next to nothing about trading. This algorithm would probably be lousy in a corporate setting, and this project serves mostly for me to learn about different techniques for Data Analysis and Data Visualization in Python. If it sounds like I don't know what I'm writing about, I probably didn't at the time. This file is just a rough space for me to plan out my next steps. Please try not to roast me too hard, but leave constructive criticism!

## General Proccess behind the Algorithm is as follows:

0. Run the Algorithm after the market closes each Day
1. Identify Buying Power
2. Based of buying power, calculate maximum and minimum share price.
3. Collect Data for a cluster of stocks with that share price (For example, all Tech stocks with a price between 100 and 1,000 dollars
4. Run Prophet Prediction on each Stock
5. Run ARIMA Predictions on each Stock
6. Apply an ESN Model to predict stock prices
7. Store most accurate share price prediction for the next day
8. Get Twitter Sentiment Score for each Stock
9. Calculate Volatility of each Stock
10. Calculate Market Cap of each Stock
11. Compare each of the collected values across Stocks, and rank each stock out of 10 in each category (Market Cap, Volatility, Sentiment Score, Predicted Share Price)
12. Average Rankings to determine final ranking score
13. Chose top 10 stocks to be in the portfolio.
14. Calculate best weightage using a Sharpe Ratio and Monte Carlo (Test random weightages to determine best returns)
15. After Identifying best weightage, place orders that will be ran 15 minutes after Market Opens next day


On second glance, the steps above seem cumbersome, redundant, and slow. I will probably adjust to placing heavy weightage on the ESN prediction, put less weightage on the twitter sentiment score, and keep Market Cap and Volatility Ratings. 

Then, a decision tree would be implemented to classify a stock as Good, Neutral, and Bad. Then I would test different portfolio weightage returns on the predicted values, and see what portfolio weightage would work out the best. 

After that, adjust the portfolio to match by buying or selling shares.

## Possible Bottlenecks

The main bottleneck is API Call Frequency. AlphaVantage only allows 5 API Calls a minute, and 500 a day. This bottleneck is reached very easily when dealing with multiple stocks.

### Possible Solution to Bottleneck

Possible Solution would include collected Data for 5 Stocks at a time, every minute, until all Data is collected, and Stored Locally on CSV Files. This would make it very easy to read data, but would require an organized file system, with a standard naming convention.

Another possible solution would be to read the data into a SQL Database. This would keep all the data stored in one, nice centralized place, and you wouldn't really have to worry about CSV file corruption, or naming issues, or whatnot. Since the algorithm is initially small scale, I don't see a real benefit to implementing this over the CSV File Storage, especially since the CSV Files are so much easier to work with.
