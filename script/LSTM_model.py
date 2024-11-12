import os
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s') 

class LSTM_Modelling:
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
    
    def train_lstm(self,train_data,test_data):
        # Scale the 'Price' column
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

        # Function to create dataset with time steps
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        # Set time step (e.g., 63 days for three month stock price )
        time_step = 63
        X_train, y_train = create_dataset(train_scaled, time_step)
        X_test, y_test = create_dataset(test_scaled, time_step)

        # Reshape data to be compatible with LSTM input (samples, time steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Output layer for regression

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        # history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions on the test set
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform predictions to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        logging.info("Model trainning with LSTM model")

        return y_test,test_predict
    
    def evaluate_lstm_model(self,y_true,y_pred):        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2_Score=r2_score(y_true,y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
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
        plt.plot(test.index, forecast, label='LSTM Forecast')
        plt.title(f"{ticker} Test Vs Prediction")
        plt.legend()
        plt.show()
        
        logging.info("Plot of the resulting prediction.")
