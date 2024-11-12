import numpy as np
import tensorflow as tf


class Forecast:
    def forecast_prices(model, data, steps=180, sequence_length=60):
        predictions = []
        last_sequence = data[-sequence_length:]
        
        for _ in range(steps):
            pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
            predictions.append(pred[0, 0])
            
            # Update the last sequence with the predicted value
            last_sequence = np.append(last_sequence[1:], pred)
        
        return np.array(predictions)
