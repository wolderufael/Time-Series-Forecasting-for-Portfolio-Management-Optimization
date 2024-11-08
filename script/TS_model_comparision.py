import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Sample time series data
date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='M')
data = pd.DataFrame(date_rng, columns=['date'])
data['value'] = np.sin(np.linspace(0, 10, len(date_rng))) + np.random.normal(scale=0.5, size=len(date_rng))

# Split data into train and test sets
train = data.iloc[:int(0.8*len(data))]
test = data.iloc[int(0.8*len(data)):]

# ARIMA Model
arima_model = ARIMA(train['value'], order=(1, 1, 1)).fit()
arima_forecast = arima_model.forecast(steps=len(test))
arima_rmse = sqrt(mean_squared_error(test['value'], arima_forecast))
arima_mape = mean_absolute_percentage_error(test['value'], arima_forecast)

# Exponential Smoothing Model
es_model = ExponentialSmoothing(train['value'], trend='add', seasonal='add', seasonal_periods=12).fit()
es_forecast = es_model.forecast(steps=len(test))
es_rmse = sqrt(mean_squared_error(test['value'], es_forecast))
es_mape = mean_absolute_percentage_error(test['value'], es_forecast)

# Prophet Model
prophet_df = train.rename(columns={'date': 'ds', 'value': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
prophet_future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
prophet_forecast = prophet_model.predict(prophet_future)
prophet_rmse = sqrt(mean_squared_error(test['value'], prophet_forecast['yhat'][-len(test):]))
prophet_mape = mean_absolute_percentage_error(test['value'], prophet_forecast['yhat'][-len(test):])

# Display results
print("ARIMA RMSE:", arima_rmse, "| MAPE:", arima_mape)
print("Exponential Smoothing RMSE:", es_rmse, "| MAPE:", es_mape)
print("Prophet RMSE:", prophet_rmse, "| MAPE:", prophet_mape)

# Plotting results for visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['value'], label='Train')
plt.plot(test['date'], test['value'], label='Test')
plt.plot(test['date'], arima_forecast, label='ARIMA Forecast')
plt.plot(test['date'], es_forecast, label='Exponential Smoothing Forecast')
plt.plot(test['date'], prophet_forecast['yhat'][-len(test):], label='Prophet Forecast')
plt.legend(loc='upper left')
plt.title('Time Series Model Comparison')
plt.show()
