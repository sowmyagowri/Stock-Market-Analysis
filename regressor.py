from preprocessing import *
from sklearn.preprocessing import StandardScaler

finalData = main()

print (finalData.head(5))

def reg_data_preprocessing(data):
    labels = data['Close'][1:]
    today = data.iloc[-1]
    data = data.iloc[:-1]
    return data, labels

def split_dataset(data, label, ratio):
    index = int(np.floor(data.shape[0]*ratio))
    X_train, X_test = data[:index], data[index:]
    y_train, y_test = label[:index], label[index:]
    return X_train, y_train, X_test, y_test

#shifting the dataset
X, y = reg_data_preprocessing(finalData)

# print ("Type and shape of X")
# print (type(X))
# print (X.shape)
# print ("Type and shape of y")
# print (type(y))
# print (y.shape)

#splitting the splitting the dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.95)

# print('Size of train set: ', X_train.shape)
# print('Size of test set: ', X_test.shape)
# print('Size of train set: ', y_train.shape)
# print('Size of test set: ', y_test.shape)
#
# print('Type of train set: ', type(X_train))
# print('type of test set: ', type(X_test))
# print('Type of train set: ', type(y_train))
# print('Type of test set: ', type(y_test))

scaler = StandardScaler()
print(scaler.fit(X_train))
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)

scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1,1))
y_train_transform = scaler_y.transform(y_train.reshape(-1,1))
y_test_transform = scaler_y.transform(y_test.reshape(-1,1))
