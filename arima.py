from statsmodels.tsa.arima_model import ARIMA
from preprocessing import *
import matplotlib.pyplot as plt

finalData = main()

df_arima = finalData['Close']

X = df_arima.values
size = int(len(X) * 0.85)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test, color='blue')
plt.plot(predictions, color='red')
plt.savefig('images/arima.png')
