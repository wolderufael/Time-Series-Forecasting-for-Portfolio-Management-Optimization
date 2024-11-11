import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
import itertools
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
    
    def check_stationarity(self,df,col):
        result=adfuller(df[col],autolag="AIC")
        
        print(f"Test Statistics: {result[0]}")
        print(f"p-Value: {result[1]}")
        print(f"Lag used: {result[2]}")
        print(f"Number of observation: {result[2]}")
        print(f"Critical Values': {result[4]}")
        print(f"Conclusion: {'Stationary' if result[1] < 0.05 else 'Non-Stationary'}")
        
        logging.info("Stationarity of the time series data is checked.")
        
    def train_test_split(self,stoke_data,ticker):
        training_data=stoke_data[[ticker]]
        train_size = int(len(training_data) * 0.8)
        train, test = training_data[ticker][:train_size], training_data[ticker][train_size:]
        logging.info("Train-Test split is done with a ratio of 0.8.")
        
        return train, test
    

    def grid_search(self,train):
        # Grid search over p, d, q parameters
        p_values = range(0, 5)
        d_values = range(0, 3)
        q_values = range(0, 5)

        best_aic = float('inf')
        best_order = None
        best_model = None

        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

        print(f"Best ARIMA parameters: {best_order}")
        logging.info("Grid search is use to find the best orders of the ARIMA model.")
        
        return best_order
        
        
    def arima_train(self,train,test,best_order):
        arima_model=ARIMA(train,order=best_order)
        model_fit=arima_model.fit()
        
        # forecast = model_fit.predict(n_periods=len(test))
        forecast = model_fit.forecast(steps=len(test))
        logging.info("The best orders are used to fit the ARIMA model.")
        
        return forecast

    def evaluate_arima_model(self,y_true,y_pred):
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
        plt.plot(test.index, forecast, label='ARIMA Forecast')
        plt.title(f"{ticker} Test Vs Prediction")
        plt.legend()
        plt.show()
        
        logging.info("Plot of the resulting prediction.")