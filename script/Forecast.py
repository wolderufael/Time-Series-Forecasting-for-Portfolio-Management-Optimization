import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

class Forecast:
    def load_data(self,file_path):
        df =pd.read_csv(file_path)
        df['Date']=pd.to_datetime(df['Date'])
        df.set_index('Date',inplace=True)
        
        return df

    def lstm_predict_future(self,data, ticker, model_path, scaler_path, predict_days):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        scaler = joblib.load(scaler_path)
        time_step = 252  # Last 252 trading days
        last_data = data[[ticker]].values[-time_step:]  # Get the last 252 days of data
        last_data_scaled = scaler.transform(last_data.reshape(-1, 1))  # Scale the data
        input_seq = last_data_scaled.reshape(1, time_step, 1)  # Reshape for LSTM input
        predictions = []
        current_date = pd.to_datetime(data.index[-1]) + timedelta(days=1)  # Start date for prediction

        for _ in range(predict_days):
            predicted_price_scaled = model.predict(input_seq)  # Predict the next day's price
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]  # Inverse scaling to get the actual price
            
            # Append the predicted value and date to the predictions list
            predictions.append((current_date, predicted_price))
            
            # Update the input sequence by removing the oldest value and adding the new prediction
            input_seq = np.append(input_seq[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)
            
            # Increment the current date by 1 day
            current_date += timedelta(days=1)

            # Update the last 365 days by appending the predicted value to the data
            # This ensures the model uses the last 365 days (including predictions) for the next forecast
            last_data = np.append(last_data, predicted_price)  # Add predicted value
            last_data = last_data[-time_step:]  # Keep only the last 365 days
            last_data_scaled = scaler.transform(last_data.reshape(-1, 1))  # Rescale new data
            input_seq = last_data_scaled.reshape(1, time_step, 1)  # Reshape for LSTM input

        prediction_df = pd.DataFrame(predictions, columns=['Date', f'{ticker}'])
        prediction_df.set_index('Date',inplace=True)
        
        return prediction_df

    def plot_forecast(self,ticker,historical_data, forecasted_data, confidence_interval): 
        conf_int_pct= 100-confidence_interval* 100 
        # Confidence Intervals
        lower_bound = forecasted_data[ticker].values * (1 - confidence_interval)
        upper_bound = forecasted_data[ticker].values * (1 + confidence_interval)

        # Plotting
        plt.figure(figsize=(14, 8))
        plt.plot(historical_data.index, historical_data[ticker], color='blue', label=f'{ticker} Historical Data')
        plt.plot(forecasted_data.index,forecasted_data[ticker], color='orange', label=f'{ticker} Forecasted Price')
        plt.fill_between(forecasted_data.index, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')
        
        plt.title(f"{ticker} Stock Price Forecast with {conf_int_pct} % Confidence Interval ")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        
        
    def analyze_forecast(self,ticker,forecasted_data, confidence_interval=0.05):
        # Trend Analysis
        trend = "upward" if forecasted_data[ticker][-1] > forecasted_data[ticker][0] else "downward"
        print(f"Trend Analysis: The forecast shows a {trend} trend.")
        variance=forecasted_data[ticker].var()
        volatility = np.sqrt(variance * 252)
        # Volatility and Risk Analysis
        # volatility = (forecasted_data.max() - forecasted_data.min()) / forecasted_data.mean()
        print(f"Volatility Analysis: The forecasted data shows a volatility level of {volatility:.2f}.")
        
        # Market Opportunities and Risks
        if trend == "upward":
            print("Market Opportunity: There is potential for growth, suggesting possible price increases.")
        else:
            print("Market Risk: The downward trend suggests potential price declines, indicating high risk.")
            