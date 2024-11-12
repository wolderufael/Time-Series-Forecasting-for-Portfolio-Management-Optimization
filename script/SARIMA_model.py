import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
import itertools
import gc
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s') 
class Modelling:
    def load_data(self,file_path):
        df =pd.read_csv(file_path)
        df['Date']=pd.to_datetime(df['Date'])
        df.set_index('Date',inplace=True)
        
        return df
    
    def train_test_split(self,stoke_data,ticker):
        training_data=stoke_data[[ticker]].resample('W').mean()
        train_size = int(len(training_data) * 0.8)
        train, test = training_data[ticker][:train_size], training_data[ticker][train_size:]
        logging.info("Train-Test split is done with a ratio of 0.8.")
        
        return train, test
    
    def grid_search(self,train,test,best_params):
        # Seasonal parameter ranges for yearly seasonality
        P = range(0, 2)
        D = range(0, 2)
        Q = range(0, 2)
        s =52   # Yearly seasonality

        # Define the seasonal parameter grid
        seasonal_grid = list(ParameterGrid({'P': P, 'D': D, 'Q': Q}))

        # Initialize variables to track the best seasonal model
        best_seasonal_mae = np.inf
        best_seasonal_params = None
        best_model = None

        # Step 2: Use the best (p, d, q) and find the best (P, D, Q)
        for seasonal_params in seasonal_grid:
            try:
                model = SARIMAX(train,
                                order=(best_params['p'], best_params['d'], best_params['q']),
                                seasonal_order=(seasonal_params['P'], seasonal_params['D'], seasonal_params['Q'], s),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                
                sarima_model = model.fit(disp=False)
                predictions = sarima_model.forecast(steps=len(test))
                mae = mean_absolute_error(test, predictions)
                
                if mae < best_seasonal_mae:
                    best_seasonal_mae = mae
                    best_seasonal_params = seasonal_params
                    best_model = sarima_model
                    
            except Exception as e:
                print(f"Error with seasonal params {seasonal_params}: {e}")

        # Clear memory
        gc.collect()

        # Print the best seasonal parameters
        print(f"Best seasonal parameters: {best_seasonal_params} with MAE: {best_seasonal_mae}")

        # Display the final model and predictions
        print(f"Best overall SARIMA model parameters: (p, d, q): {best_params}, (P, D, Q, s): {best_seasonal_params}")
        logging.info("Grid search is use to find the best seasonal orders of the SARIMA model.")
        return best_seasonal_params
    
    def sarima_train(self,train,test,best_params,best_seasonal_params):
        s =52 # the weekly samoled data have yearly seasonal cycle 
        sarima_model = SARIMAX(train,
                order=(best_params['p'], best_params['d'], best_params['q']),
                seasonal_order=(best_seasonal_params['P'], best_seasonal_params['D'], best_seasonal_params['Q'], s),
                enforce_stationarity=False,
                enforce_invertibility=False)
        model_fit=sarima_model.fit()
        forecast = model_fit.forecast(steps=len(test))
        logging.info("The best orders are used to fit the SARIMA model.")
        
        return forecast
    
    def evaluate_sarima_model(self,y_true,y_pred):
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2_Score=r2_score(y_true,y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # MAPE as a percentage

        # Print metrics
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R Square Score (r2_score):", r2_Score)
        print("Mean Absolute Percentage Error (MAPE):", mape, "%")
        
        logging.info("Evaluatio Metrics to assses the performance of the model.")

    def plot_result(self,ticker,train,test,forecast):
        # Plot the final forecast
        plt.figure(figsize=(14,7))
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(test.index, forecast, label='SARIMA Forecast')
        plt.title(f"{ticker} Test Vs Prediction")
        plt.legend()
        plt.show()
        
        logging.info("Plot of the resulting prediction.")