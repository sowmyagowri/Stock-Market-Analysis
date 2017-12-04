# from fbprophet import Prophet
from fbprophet import Prophet
from preprocessing import *
from helpers_regressor import *

# dataProphet = getting_final_data('aapl')
dataProphet = main()
dataProphet = dataProphet['Close']
dataProphet = dataProphet.reset_index()
# print (dataProphet)

dataProphetRed = dataProphet.rename(columns={"index": "ds", "Close": "y"})
# print (dataProphetRed.head(5))
dataProphetRed['y_orig'] = dataProphetRed['y']

#log transform y
dataProphetRed['y'] = np.log(dataProphetRed['y'])

splitIndex = int(np.floor(dataProphetRed.shape[0]*0.95))
X_train_prophet, X_test_prophet = dataProphetRed[:splitIndex], dataProphetRed[splitIndex:]
print ("No. of samples in the training set: ", len(X_train_prophet))
print ("No. of samples in the test set", len(X_test_prophet))

model=Prophet(daily_seasonality=True)
# model.fit(dataProphetRed)
model.fit(X_train_prophet)

future_data = model.make_future_dataframe(periods=30)
forecast_data = model.predict(future_data)

test = X_test_prophet

del test['y_orig']
test.set_index('ds')
test1 = model.predict(test)

MSE = get_mse(np.exp(test['y']), np.exp(test1['yhat']))
print ("Mean Squared Error: ", MSE)

# model.plot(forecast_data)
