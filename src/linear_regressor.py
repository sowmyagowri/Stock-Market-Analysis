from preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from helpers_regressor import *

finalData = main()

#shifting the dataset,
X, y, today = reg_data_preprocessing(finalData)

#splitting the splitting the dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.80)

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)
today_transform = scaler.transform(today.reshape(1,-1))

scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1,1))
y_train_transform = scaler_y.transform(y_train.reshape(-1,1))
y_test_transform = scaler_y.transform(y_test.reshape(-1,1))

linearReg = LinearRegression(normalize=False)
linearReg.fit(X_train_transform, y_train_transform)
predictions = linearReg.predict(X_test_transform)

prediction_today = linearReg.predict(today_transform)
actual_today = scaler_y.inverse_transform(prediction_today)

actual = scaler_y.inverse_transform(predictions)
error = get_mse(y_test, actual)
print ("MSE:")
print (error)
