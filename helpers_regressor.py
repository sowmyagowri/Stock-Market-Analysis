import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def reg_data_preprocessing(data):
    labels = data['Close'][1:-1]
    today = data.iloc[-2]
    data = data.iloc[:-2]
    return data, labels, today

def split_dataset(data, label, ratio):
    index = int(np.floor(data.shape[0]*ratio))
    X_train, X_test = data[:index], data[index:]
    y_train, y_test = label[:index], label[index:]
    return X_train, y_train, X_test, y_test

def nn_data_preprocessing(data):
    stockClose = pd.DataFrame()
    stockClose['Close'] = data['Close']
    today = data.iloc[-3].tolist()
    stockData_actual = today[-4]
    stockLabel = pd.DataFrame()
    stockLabel = stockClose.copy()
    stockLabel.drop([0])
    stockLabel.drop([1])
    # stockLabel.drop([2])
    stockData = data[0:len(data) - 3]
    data = stockData
    return data, stockLabel, today, stockData_actual

def get_mse(actual, prediction):
    return mean_squared_error(actual, prediction)
