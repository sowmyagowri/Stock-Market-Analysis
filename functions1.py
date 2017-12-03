
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import io
import requests


# In[9]:


# Import data
#function to get stock data
def yahoo_stocks(symbol, start, end):
    return web.DataReader(symbol, 'yahoo', start, end)

def get_historical_stock_price(stock):
    print ("Getting historical stock prices for stock ", stock)
    
    #get 7 year stock data for Apple
    startDate = datetime.datetime(2005, 1, 4)
    endDate = datetime.datetime(2017, 11, 29)
    #endDate = pd.to_datetime(date)
    stockData = yahoo_stocks(stock, startDate, endDate)
    stockMarketData = yahoo_stocks('^GSPC', startDate, endDate)
    return stockData, stockMarketData

def predict_prices(company):
    #stock = input("Enter stock name(ex:GOOGL, AAPL): ")
    stockData, stockMarketData = get_historical_stock_price(company)
    
    print (stockData.shape)
    print (stockMarketData.shape)
    
    stockData = stockData.reset_index()
    del stockData['Date']
    
    print (stockData.head(5))
    print (stockData.shape)
    # 
    stockMarketData = stockMarketData.reset_index()
    del stockMarketData['Date']
    print (stockMarketData.head(5))
    print (stockMarketData.shape)
    
    # 
    stockData['S&P'] =  stockMarketData['Close'] 
    print (stockData.head(5))
    print (stockData.shape)
    
    stockClose = pd.DataFrame()
    stockClose['Close'] = stockData['Close']
    
    #stockData_predict = stockData.iloc[-1].tolist()
    stockData_predict = stockData.iloc[-2].tolist()
    print ("Last row")
    print (type(stockData_predict))
    #stockData_predict = stockData_predict.reshape(1,-1)[0].reshape(1,-1)[0]
    stockData_predict = np.asarray(stockData_predict).reshape(1,-1)
    #print (len(stockData_predict))
    #print(np.asarray(stockData_predict).reshape(1,-1).shape)
    # stockData_predict_array = stockData_predict.values
    # print(stockData_predict_array.shape)
    # print (type(stockData_predict_array))
    stockLabel = pd.DataFrame()
    #stockLabel.index = np.arange(1, len(stockClose))
    #stockLabel['Close'] = stockClose.shift()[1:]
    stockLabel = stockClose.copy()
    stockLabel.drop([0])
    print (stockLabel.shape)
    print (stockLabel)
    
    #stockData = stockData[0:len(stockData) - 1]
    stockData = stockData[0:len(stockData) - 2]
    #print (stockData.shape)
    #print (stockData.head(5))
    # 
    # print(stockData.shape)
    # 
    
    #print (stockClose_new.shape)
    # 
    # stockClose_array = stockClose_new.values
    # l = []
    # 
    # for item in stockClose_array:
    #     l.append(item[0])
    # 
    # print (type(l))
    # stockLabel = pd.DataFrame()
    # #dtype = [('Col1','float32')]
    # #values = np.zeros(20, dtype=dtype)
    # #index = range(0, stockClose.shape[0]-1)
    # stockLabel = pd.DataFrame({'y':l})
    # #stockLabel = pd.DataFrame(stockClose_array, index=index, columns=['0'])
    # print (stockLabel)
    # 
    # #stockData.dropna(axis=0)
    # #stockData_new = stockData[0:len(stockData) - 1]
    # 
    # #data['S&P'] =  stockMarketData['Close'][0:len(stockMarketData) - 2]
    # #plt.plot(stockData['Close'])
    # #plt.show()
    # 
    data = stockData
    
    print (data.head(5))
    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # 
    print (n)
    print (p)
    # 
    #Make data a numpy array
    data = data.values
    print (type(data))
    # 
    # 
    # 
    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end 
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    # 
    print (type(data_train))
    print (data_train.shape)
    print (type(data_train))
    print (data_test.shape)
    
    y_train = stockLabel['Close'][np.arange(train_start, train_end)].as_matrix()
    y_test = stockLabel['Close'][np.arange(test_start, test_end)].as_matrix()
    
    print (y_train)
    
    print (type(y_train))
    print (y_train.shape)
    print (type(y_test))
    print (y_test.shape) 
    # 
    # #print (data_train.shape)
    # #print (y_train)
    # # Scale data
    from sklearn.preprocessing import MinMaxScaler
    # 
    # #y_train_min = min(data_train[:,7])
    # #y_train_max = max(data_train[:,7])
    # 
    # #print (y_train_min)
    # #print (y_train_max)
    # 
    
    print ("After scaling.... ")
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    
    print (type(data_train))
    print (data_train.shape)
    print (type(data_train))
    print (data_test.shape)
    
    #
    #print ("y train")
    #print (y_train[1200])
    #print ("y_tset")
    #print (y_test)
    #scaler_y = MinMaxScaler()
    #scaler_y.fit(y_train.reshape(-1,1))
    #y_train = scaler_y.transform(y_train.reshape(-1,1))
    #y_test = scaler_y.transform(y_test.reshape(-1,1))
    
    #print ("y train")
    #print (y_train)
    #print ("y_tset")
    #print (y_test)
    
    print (type(y_train))
    print (y_train.shape)
    print (type(y_test))
    print (y_test.shape) 
    # 
    # # Build X and y
    # #X_train = data_train[:, 0:7]
    # X_train = data_train
    # #y_train = data_train[:,7]
    # #X_test = data_test[:, 0:7]
    # X_test = data_test
    # print (X_test[0].shape)
    # print (type(X_test[0]))
    # #y_test = data_test[:,7]
    # 
    # #x_scaler = MinMaxScaler()
    # #y_scaler = MinMaxScaler()
    # 
    # #X_train = x_scaler.fit_transform(X_train)
    # #X_test = x_scaler.fit_transform(X_test)
    # 
    # #y_train = y_scaler.fit_transform(y_train)
    # #y_test = y_scaler.fit_transform(y_test)
    # 
    # 
    # #y_test1 = ********give data here
    # # Import TensorFlow
    
    X_train = data_train
    X_test = data_test
    print (type(X_test[0]))
    print (X_test[0])
    print (X_test[0].shape)
    import tensorflow as tf
    # 
    # 
    # # Model architecture parameters
    n_stocks = 7
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_neurons_5 = 64
    n_target = 1
    # 
    # # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    # # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()
    # # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    # # Layer 5: Variables for hidden weights and biases
    W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
    bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
    # 
    # # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_5, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    # 
    # # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
    # 
    # # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out))
    # 
    # # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    # 
    # # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    # 
    # # Make Session
    net = tf.Session()
    # # Run initializer
    net.run(tf.global_variables_initializer())
    # 
    # # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test*0.5)
    plt.show()
    # 
    # # Number of epochs and batch size
    epochs = 20
    batch_size = 64
    # 
    for e in range(epochs):
    # 
    #     # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
    # 
    #     # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
    #         # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    # 
    #         # Show progress
            if np.mod(i, 5) == 0:
    #             # Predictio
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
                plt.savefig(file_name)
                plt.pause(0.01)
    #             
    
    stockData_predict_transformed = scaler.transform(stockData_predict)
    
    print ("stockData_predict_transformed")
    print (stockData_predict_transformed)
    print (type(stockData_predict_transformed))
    print (stockData_predict_transformed.shape)
    print (type(stockData_predict_transformed[0]))
    print (stockData_predict_transformed[0].shape)
    stockData_predict_transformed_0 = stockData_predict_transformed[0].reshape(1,-1)
    #stockData_predict_transformed_reshape = stockData_predict_transformed[0].reshape(1, -1)
    #print (stockData_predict_transformed_reshape)
    #print (stockData_predict_transformed_reshape.shape)
    #y_train_min = min(data_train[:,7])
    #y_train_max = max(data_train[:,7])
    # 
    # print (y_train_min)
    # print (y_train_max)
    # 
    # #stockData_predict_array_reshape = stockData_predict_array.reshape(1,7)
    # #stockData_predict_transform = scaler.transform(stockData_predict_array)
    # #print ("Prediction for tomorrow")
    # 
    # #stockData_predict_array_fit
    # #stockData_predict_array_fit_reshape = stockData_predict_array_fit.reshape(1,7)
    # #print (stockData_predict_array_fit_reshape.shape)
    print ("1-day prediction")
    pred1 = net.run(out, feed_dict={X: stockData_predict_transformed_0})
    # 
    print (pred1)
    # 
    # #pred_rescaled = (pred1* (y_train_max - y_train_min)) + (y_train_min)
    # 
    # #print (pred_rescaled)
    # 
    # #real_prediction = y_scaler.inverse_transform(pred1)
    # #print (real_prediction)
    # #pred1_reshape = pred1.reshape(1,-1)
    # #pred1_reshape = pred1_reshape[0].reshape(1,8)
    # #print (pred1)
    # #print (pred1_reshape)
    # #print (scaler.inverse_transform(pred1_reshape))
    #     
    # #mse_final1 = net.run(mse, feed_dict={X: stockData_predict, Y: y_test})
    # # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)
    result = pred1.tolist()
    
    return round(result[0][0],2)