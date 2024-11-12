import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

#Load model and scalar
model_path='public/lstm-06-11-2024-04-52-41-00.pkl'
scaler_path='public/scaler.joblib'
with open(model_path, 'rb') as file:
        model = pickle.load(file)
scaler = joblib.load(scaler_path)

class Forecast:
    def lstm_predict_future(data,model, scaler,start_date, predict_days=30, time_step=60):
        last_data = data[['Price']].values[-time_step:]
        last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
        input_seq = last_data_scaled.reshape(1, time_step, 1)
        predictions = []
        current_date = pd.to_datetime(data['Date'].iloc[-1]) + timedelta(days=1)

        for _ in range(predict_days):
            predicted_price_scaled = model.predict(input_seq)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
            predictions.append((current_date, predicted_price))
            input_seq = np.append(input_seq[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)
            current_date += timedelta(days=1)
        prediction_df = pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])
        
        return prediction_df
    def forecast_prices(model, data, steps=180, sequence_length=60):
        predictions = []
        last_sequence = data[-sequence_length:]
        
        for _ in range(steps):
            pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
            predictions.append(pred[0, 0])
            
            # Update the last sequence with the predicted value
            last_sequence = np.append(last_sequence[1:], pred)
        
        return np.array(predictions)

    def plot_forecast(historical_data, forecasted_data, scaler, confidence_interval=0.05):
        # Inverse scaling for original price scale
        historical_data = scaler.inverse_transform(historical_data)
        forecasted_data = scaler.inverse_transform(forecasted_data.reshape(-1, 1))

        # Create a date index for the forecast
        dates = pd.date_range(start=historical_data.index[-1], periods=len(forecasted_data)+1, freq='B')[1:]

        # Confidence Intervals
        lower_bound = forecasted_data * (1 - confidence_interval)
        upper_bound = forecasted_data * (1 + confidence_interval)

        # Plotting
        plt.figure(figsize=(14, 8))
        plt.plot(historical_data.index, historical_data, color='blue', label='Historical Data')
        plt.plot(dates, forecasted_data, color='orange', label='Forecasted Price')
        plt.fill_between(dates, lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.3, label='Confidence Interval')
        
        plt.title("Tesla Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        
        
    def analyze_forecast(forecasted_data, confidence_interval=0.05):
        # Trend Analysis
        trend = "upward" if forecasted_data[-1] > forecasted_data[0] else "downward"
        print(f"Trend Analysis: The forecast shows a {trend} trend.")

        # Volatility and Risk Analysis
        volatility = (forecasted_data.max() - forecasted_data.min()) / forecasted_data.mean()
        print(f"Volatility Analysis: The forecasted data shows a volatility level of {volatility:.2f}.")

        # Confidence Interval Insights
        risk_periods = np.where((forecasted_data * (1 + confidence_interval) - forecasted_data * (1 - confidence_interval)) > forecasted_data.mean() * 0.05)[0]
        print("Market Risk Periods:", risk_periods)

        # Market Opportunities and Risks
        if trend == "upward":
            print("Market Opportunity: There is potential for growth, suggesting possible price increases.")
        else:
            print("Market Risk: The downward trend suggests potential price declines, indicating high risk.")