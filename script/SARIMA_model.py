import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
import itertools
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s') 

def train_test_split(self,stoke_data,ticker):
    stoke_data['Date']=pd.to_datetime(stoke_data['Date'])
    stoke_data.set_index('Date',inplace=True)
    training_data=stoke_data[[ticker]]
    train_size = int(len(training_data) * 0.8)
    train, test = training_data[ticker][:train_size], training_data[ticker][train_size:]
    logging.info("Train-Test split is done with a ratio of 0.8.")
    
    return train, test