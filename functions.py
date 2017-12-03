import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import io
import requests
from preprocessing import *
from helpers_regressor import *
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def predict_prices(company):
    finalData = getting_final_data(company)

    data = finalData
    data = data.reset_index()

    del data['index']
    # del data['prev_diff']
    # del data['50d']
    # del data['10d_vol']
    del data['sm_open']
    del data['sm_high']
    del data['sm_low']
    del data['sm_adj_close']
    del data['sm_volume']
    # del data['sm_prev_diff']

    data, stockLabel, stockData_predict, stockData_actual = nn_data_preprocessing(data)
    stockData_predict = np.asarray(stockData_predict).reshape(1,-1)

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]

    print (n)
    print (p)

    #Make data a numpy array
    data = data.values

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    y_train = stockLabel['Close'][np.arange(train_start, train_end)].as_matrix()
    y_test = stockLabel['Close'][np.arange(test_start, test_end)].as_matrix()

    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    X_train = data_train
    X_test = data_test

    # Model architecture parameters
    n_stocks = X_train.shape[1]
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_neurons_5 = 64
    # n_neurons_6 = 32
    n_target = 1

    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    # Layer 5: Variables for hidden weights and biases
    W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
    bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
    # # Layer 6: Variables for hidden weights and biases
    # W_hidden_6 = tf.Variable(weight_initializer([n_neurons_5, n_neurons_6]))
    # bias_hidden_6 = tf.Variable(bias_initializer([n_neurons_6]))

    # Output layer: Variables for output weights and biases
    # W_out = tf.Variable(weight_initializer([n_neurons_6, n_target]))
    W_out = tf.Variable(weight_initializer([n_neurons_5, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
    # hidden_6 = tf.nn.relu(tf.add(tf.matmul(hidden_5, W_hidden_6), bias_hidden_6))

    # Output layer (must be transposed)
    # out = tf.transpose(tf.add(tf.matmul(hidden_6, W_out), bias_out))
    out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test*0.5)
    plt.show()

    # Number of epochs and batch size
    epochs = 20
    batch_size = 64

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
                plt.savefig(file_name)
                plt.pause(0.01)

    stockData_predict_transformed = scaler.transform(stockData_predict)
    stockData_predict_transformed_0 = stockData_predict_transformed[0].reshape(1,-1)

    print ("1-day prediction")
    pred1 = net.run(out, feed_dict={X: stockData_predict_transformed_0})

    print (pred1)

    # # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)
    result = pred1.tolist()

    return round(result[0][0],2)

predict_prices("aapl")
