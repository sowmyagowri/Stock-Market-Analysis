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

    #get the dataset
    finalData = getting_final_data(company)

    data = finalData
    #remove the date from index
    data = data.reset_index()

    #remove the correlated columns
    del data['index']
    del data['Open']
    del data['High']
    del data['Low']
    del data['sm_open']
    del data['sm_high']
    del data['sm_low']
    del data['sm_adj_close']
    del data['sm_prev_diff']

    #get the train features and labels
    data, stockLabel, stockData_predict, stockData_prev, stockData_actual = nn_data_preprocessing(data)
    stockData_predict = np.asarray(stockData_predict).reshape(1,-1)

    # Dataset dimensions
    numberOfRows = data.shape[0]
    numberOfColumns = data.shape[1]

    #converting dataframe to numpy array
    data = data.values

    # Splitting the dataset into train and test
    start_index_train = 0
    end_index_train = int(np.floor(0.9*numberOfRows))
    start_index_test = end_index_train
    end_index_test = numberOfRows
    train_data = data[np.arange(start_index_train, end_index_train), :]
    test_data = data[np.arange(start_index_test, end_index_test), :]

    y_train = stockLabel['Close'][np.arange(start_index_train, end_index_train)].as_matrix()
    y_test = stockLabel['Close'][np.arange(start_index_test, end_index_test)].as_matrix()

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    X_train = train_data
    X_test = test_data

    # Model architecture parameters
    numberOfFeatures = X_train.shape[1]
    layer_nodes_1 = 1024
    layer_nodes_2 = 512
    layer_nodes_3 = 256
    layer_nodes_4 = 128
    layer_nodes_5 = 64

    #number of nodes in the output
    output_node = 1

    # Declare Placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, numberOfFeatures])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Initializers
    sigma = 1
    weight_init = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_init = tf.zeros_initializer()

    #Declare the weights and biases for each hidden layer
    # Layer 1
    weight_layer_1 = tf.Variable(weight_init([numberOfFeatures, layer_nodes_1]))
    bias_layer_1 = tf.Variable(bias_init([layer_nodes_1]))

    # Layer 2
    weight_layer_2 = tf.Variable(weight_init([layer_nodes_1, layer_nodes_2]))
    bias_layer_2 = tf.Variable(bias_init([layer_nodes_2]))

    # Layer 3
    weight_layer_3 = tf.Variable(weight_init([layer_nodes_2, layer_nodes_3]))
    bias_layer_3 = tf.Variable(bias_init([layer_nodes_3]))

    # Layer 4
    weight_layer_4 = tf.Variable(weight_init([layer_nodes_3, layer_nodes_4]))
    bias_layer_4 = tf.Variable(bias_init([layer_nodes_4]))

    # Layer 5
    weight_layer_5 = tf.Variable(weight_init([layer_nodes_4, layer_nodes_5]))
    bias_layer_5 = tf.Variable(bias_init([layer_nodes_5]))

    #Declare the weight and bias for output layer
    weight_output = tf.Variable(weight_init([layer_nodes_5, output_node]))
    bias_output = tf.Variable(bias_init([output_node]))

    # Define all the hidden layers
    layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weight_layer_1), bias_layer_1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weight_layer_2), bias_layer_2))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weight_layer_3), bias_layer_3))
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weight_layer_4), bias_layer_4))
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weight_layer_5), bias_layer_5))

    # Output layer
    output_layer = tf.transpose(tf.add(tf.matmul(layer_5, weight_output), bias_output))

    # Cost function
    mean_squared_error = tf.reduce_mean(tf.squared_difference(output_layer, Y))

    # Define Optimizer (Adam Optimizer)
    optimizer = tf.train.AdamOptimizer().minimize(mean_squared_error)

    # Make Session
    session = tf.Session()

    # Run initializer
    session.run(tf.global_variables_initializer())

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
            session.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                prediction = session.run(output_layer, feed_dict={X: X_test})
                line2.set_ydata(prediction)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
                plt.savefig(file_name)
                plt.pause(0.01)
    plt.close()

    stockData_predict_transformed = scaler.transform(stockData_predict)
    stockData_predict_transformed_0 = stockData_predict_transformed[0].reshape(1,-1)

    prediction1 = session.run(output_layer, feed_dict={X: stockData_predict_transformed_0})

    mean_squared_error_final = session.run(mean_squared_error, feed_dict={X: X_test, Y: y_test})
    print("The Mean Squared Error is " + str(mean_squared_error_final))
    result = list()
    result.append(round(prediction1.tolist()[0][0],2))
    result.append(round(stockData_prev,2))
    result.append(round(stockData_actual,2))

    return result
