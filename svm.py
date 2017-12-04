from preprocessing import *
from helpers_regressor import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

finalData = main()

#shifting the dataset,
X, y, today = reg_data_preprocessing(finalData)

#splitting the splitting the dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.90)

scaler = StandardScaler()
print(scaler.fit(X_train))
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)
today_transform = scaler.transform(today.reshape(1,-1))

scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1,1))
y_train_transform = scaler_y.transform(y_train.reshape(-1,1))
y_test_transform = scaler_y.transform(y_test.reshape(-1,1))

svm_linear = SVR(kernel='linear', C=2500)

svm_linear.fit(X_train_transform, y_train_transform)
predictions = svm_linear.predict(X_test_transform)

prediction_today = svm_linear.predict(today_transform)
actual_today = scaler_y.inverse_transform(prediction_today)
print ("Tomorrow: ")
print (actual_today)

scaled_predictions = scaler_y.inverse_transform(predictions)
error = get_mse(y_test, scaled_predictions)
print ("MSE:")
print (error)

# plot
plt.plot(y_test.as_matrix(), color='blue')
plt.plot(scaled_predictions, color='red')
plt.savefig("images/svm.png")
plt.show()
