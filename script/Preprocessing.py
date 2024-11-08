import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf 
from scipy.stats import zscore
from IPython.display import display, HTML
from statsmodels.tsa.seasonal import seasonal_decompose

class Preprocessor:
    def dload_save_yfinance_data(self,tickers):
        stoke_arr=[]
        stoke_data= pd.DataFrame(stoke_arr)
        for ticker in tickers:
            closing_price=yf.download(ticker,start="2015-01-01",end="2024-10-31")['Close']
            stoke_data[f'{ticker}']=closing_price

        stoke_data.index = stoke_data.index.date
        stoke_data = stoke_data.rename_axis("Date").reset_index()
        stoke_data.to_csv("data/stoke_data.csv",index=False)
        
        return stoke_data
    
    def check_missing(self,df):
        null_value=df.isnull().sum().sum()
        
        display(HTML(f"<h2>There is no misssing value </h2>"))if null_value==0 else display(HTML(f"<h2>The number of missing values: {null_value}</h2>"))
        
    def timeseries_plot(self,stoke_data):
        plt.figure(figsize=(14,7))
        for col in stoke_data.columns:
            stoke_data[col].plot(label=col)
        plt.ylabel('Stoke Price($)')
        plt.title("Daily Closing Stoke Price($) Over Time")
        plt.legend()
        plt.show()
        
    def daily_pct_return(self,stoke_data):
        plt.figure(figsize=(14,7))
        columns=['TSLA','BND','SPY']
        for col in columns:
            returns=stoke_data[col].pct_change()*100
            returns_summary=returns.describe()
            print(f"{col} summary :")
            print(returns_summary)
            
            plt.plot(stoke_data.index,returns,label=col)
        plt.ylabel('Daily % Return')
        plt.title("Daily % Return")
        plt.legend()
        plt.show()
        
    def return_distribution(self,stoke_data):
        columns=['TSLA','BND','SPY']
        plt.figure(figsize=(15,5))
        for i, col in enumerate(columns):
            returns=stoke_data[col].pct_change()

            plt.subplot(1,3,i + 1)
            plt.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title(f'{col} Returns')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')


        plt.tight_layout()  
        plt.show()
        
    def rolling_mean_std(self,stoke_data):
        columns=['TSLA','BND','SPY']
        window={'Weekly':5,'Monthly':21,'Yearly':252}

        plt.figure(figsize=(14,7))
        for col in columns:
            plt.figure(figsize=(20, 30))
            plt.suptitle(f'{col} - Rolling Mean and Standard Deviation', fontsize=30)
            for i, (label, win_size) in enumerate(window.items(), 1):
                # Calculate rolling mean and standard deviation
                rolling_mean = stoke_data[col].rolling(window=win_size).mean()
                rolling_std = stoke_data[col].rolling(window=win_size).std()

                # Create a subplot for each window
                plt.subplot(len(window), 1, i)
                plt.plot(stoke_data.index, stoke_data[col], label=f'{col} Price', color='blue')
                plt.plot(stoke_data.index, rolling_mean, label=f'{label} Rolling Mean', color='orange')
                plt.plot(stoke_data.index, rolling_std, label=f'{label} Rolling Std Dev', color='green')
                
                # Titles and labels
                plt.title(f'{label} Window Size ({win_size} Days)',fontsize=20)
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
        plt.subplots_adjust(hspace=2)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()
        
    def outliers_plot(self,stoke_data):
        columns=['TSLA','BND','SPY']

        for col in columns:
            stoke_data[f'{col}_zscore']= zscore(stoke_data[col])
            outliers = stoke_data[(stoke_data[f'{col}_zscore'] > 2) | (stoke_data[f'{col}_zscore'] < -2)]
            print(f'{col} Outliers Date :', outliers.index)
            plt.figure(figsize=(12, 6))
            plt.scatter(stoke_data.index, stoke_data[col], color='blue', label='Normal', alpha=0.6)
            plt.scatter(outliers.index, outliers[col], color='red', label='Outliers', marker='o', s=50)
            plt.xlabel("Date")
            plt.ylabel("Stoke Price")
            plt.title(f"{col} Stock Prices with Outliers Highlighted")
            plt.legend()

            plt.show()
            
    def seasonal_decompose(self,stoke_data):
        columns=['TSLA','BND','SPY']
        for col in columns:
            decomposition = seasonal_decompose(stoke_data[col], model='additive', period=252 )
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            plt.figure(figsize=(12, 10))
            plt.suptitle(f'{col} - Seasonal Decompose', fontsize=30)
            # Plot original data
            plt.subplot(4, 1, 1)
            plt.plot(stoke_data[col], label='Original')
            plt.legend(loc='upper left')
            plt.title('Original Close Price')

            # Plot trend
            plt.subplot(4, 1, 2)
            plt.plot(trend, label='Trend', color='orange')
            plt.legend(loc='upper left')
            plt.title('Trend Component')

            # Plot seasonal
            plt.subplot(4, 1, 3)
            plt.plot(seasonal, label='Seasonality', color='green')
            plt.legend(loc='upper left')
            plt.title('Seasonal Component')

            # Plot residuals
            plt.subplot(4, 1, 4)
            plt.plot(residual, label='Residuals', color='red')
            plt.legend(loc='upper left')
            plt.title('Residual Component')

            plt.tight_layout()
            plt.show()
    def var_ratio(self,stoke_data):
        columns=['TSLA','BND','SPY']
        for col in columns:
            returns = stoke_data[col].pct_change()*100
            
            # Calculate VaR at 95% confidence level
            VaR_95 = np.percentile(returns.dropna(), 5)  
            print(f"{col} Value at Risk (VaR) at 95% confidence: {VaR_95}")
            
    def sharpe_ratio(self,stoke_data):
        columns=['TSLA','BND','SPY']
        risk_free_rate = 0.02  # Assume a 2% risk-free rate
        for col in columns:
            returns = stoke_data[col].pct_change()*100
            returns.dropna()
            # Calculate the Sharpe Ratio
            excess_return = np.mean(returns) - risk_free_rate  # Excess return
            volatility = np.std(returns)  # Standard deviation (volatility)
            sharpe_ratio = excess_return / volatility
            print(f"{col} Sharpe Ratio: {sharpe_ratio}")