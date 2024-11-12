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
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
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
        training_data=stoke_data[[ticker]]
        train_size = int(len(training_data) * 0.8)
        train, test = training_data[ticker][:train_size], training_data[ticker][train_size:]
        logging.info("Train-Test split is done with a ratio of 0.8.")
        
        return train, test
    
    def train_lstm(self, train_data, test_data, ticker):
        # Scale the 'Price' column
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

        # Save the fitted scaler for future use
        joblib.dump(scaler, f'models/{ticker}-scaler.joblib')

        # Function to create dataset with time steps
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        # Set time step (e.g., 63 days for three-month stock price)
        time_step = 63
        X_train, y_train = create_dataset(train_scaled, time_step)
        X_test, y_test = create_dataset(test_scaled, time_step)

        # Reshape data to be compatible with LSTM input (samples, time steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define model-building function for Keras Tuner
        def build_model(hp):
            model = Sequential()
            # First LSTM layer
            model.add(LSTM(
                units=hp.Int('units_1', min_value=32, max_value=128, step=32),
                return_sequences=True,
                input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(rate=hp.Float('dropout_1', 0.1, 0.5, step=0.1)))

            # Second LSTM layer
            model.add(LSTM(
                units=hp.Int('units_2', min_value=32, max_value=128, step=32),
                return_sequences=False))
            model.add(Dropout(rate=hp.Float('dropout_2', 0.1, 0.5, step=0.1)))

            # Output layer
            model.add(Dense(units=1))

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                loss='mean_squared_error')
            
            return model

        # Initialize Keras Tuner
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=5,  # You can increase this for more extensive tuning
            directory='keras_tuner_dir',
            project_name=f'{ticker}_lstm_tuning'
        )

        # Search for the best hyperparameters
        tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        # Retrieve the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        history = best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions on the test set
        train_predict = best_model.predict(X_train)
        test_predict = best_model.predict(X_test)

        # Inverse transform predictions to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Convert test_predict to a DataFrame with appropriate index
        forecast_index = test_data.index[time_step + 1:]
        forecast = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': test_predict.flatten()}, index=forecast_index)

        # Save the model
        folder_path = 'models/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
        filename = f'{folder_path}{ticker}-{timestamp}.pkl'
        
        with open(filename, 'wb') as file:
            pickle.dump(best_model, file)

        print(f"Model saved as {filename}")
        logging.info("Model training with LSTM model")

        return forecast
    
    
    
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
    
    def plot_result(self,ticker,train,forecast):
        # Plot the final forecast
        plt.figure(figsize=(14,7))
        plt.plot(train, label='Train')
        plt.plot(forecast['Actual'], label='Test')
        plt.plot(forecast['Predicted'], label='LSTM Forecast')
        plt.title(f"{ticker} Test Vs Prediction")
        plt.legend()
        plt.show()
        
        logging.info("Plot of the resulting prediction.")
