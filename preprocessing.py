from __future__ import print_function
import pandas as pd
import pandas_datareader.data as web
import io
import numpy as np
import requests
import time
import datetime
import math

#function to get stock data
def yahoo_stocks(symbol, start, end):
    return web.DataReader(symbol, 'yahoo', start, end)

#function to add rows for missing dates
def add_missing_dates(dataframe, start, end):
    idx = pd.date_range(start, end)
    dataframe.index = pd.DatetimeIndex(dataframe.index)
    dataframe = dataframe.reindex(idx, fill_value='np.nan')
    return dataframe

#function to convert the columns to numeric
def convert_to_numeric(dataframe):
    for col in dataframe:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe

#function to perform interpolation
def interpolate(dataframe):
    features = list(dataframe)
    for feature in features:
        dataframe[feature] = dataframe[feature].interpolate()
    return dataframe

#function to add difference between the previous day and todays closing value as a feature
def prev_diff(dataframe):
    close = dataframe['Close']
    prev_diff = [0]
    for i in range(1, len(dataframe)):
        prev_diff.append(round((close[i]-close[i-1]),6))
    return prev_diff

def moving_avg(dataframe):
    dataframe["50d"] = np.round(dataframe["Close"].rolling(window = 50, center = False).mean(), 2)
    z = dataframe['Close'][0:49]
    dataframe['50d'][0:49] = z
    return dataframe

def ten_day_volatility(dataframe):
    volatility = dataframe['Close'].rolling(window=10,center=False).std(ddof=0)
    # daily_pct_change = stockstats_df['close'] / stockstats_df['close'].shift(1) - 1
    # volatility2 = daily_pct_change.rolling(window=10,center=False).std(ddof=0) * math.sqrt(10)
    top = dataframe[0:9]
    top_vol = top['Close'].rolling(window=2,center=False).std(ddof=0)
    top_vol[0] = 0
    volatility[0:9] = top_vol
    return volatility

def combine_datasets(data1, data2):
    return pd.concat([data1, data2], axis=1)

def preprocessing(data, startDate, endDate):
    data = add_missing_dates(data, startDate, endDate)
    dataNumeric = convert_to_numeric(data)
    dataInterpolated = interpolate(dataNumeric)
    dataInterpolated['prev_diff'] = prev_diff(dataInterpolated)
    return dataInterpolated

def getting_final_data(symbol):
    #get 7 year stock data for Apple
    startDate = datetime.datetime(2010, 1, 4)
    endDate = datetime.date.today()
    stockData = yahoo_stocks(symbol, startDate, endDate)

    #getting stock market data for 7 years
    stockMarketData = yahoo_stocks('^GSPC', startDate, endDate)

    #preprocess the data
    stockDataFinal = preprocessing(stockData, startDate, endDate)
    stockDataFinal = moving_avg(stockDataFinal)
    stockDataFinal['10d_vol'] = ten_day_volatility(stockDataFinal)
    stockMarketDataFinal = preprocessing(stockData, startDate, endDate)
    stockMarketDataFinal.columns = ['sm_open', 'sm_high', 'sm_low', 'sm_close', 'sm_adj_close', 'sm_volume', 'sm_prev_diff']
    finalData = combine_datasets(stockDataFinal, stockMarketDataFinal)
    return finalData

#main function to call to get data
def main():
    symbol = raw_input("Enter the Company Symbol: ")
    finalData = getting_final_data(symbol)
    return finalData
