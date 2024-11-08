import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf 

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
        null_value=df.isnull().sum()
        
        "There is no misssing valuee" if null_value==0 else f"The number of missing values: {null_value}"
        
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
        for col in stoke_data.columns:
            stoke_data[f'{col}_daily_%_return']=stoke_data[col].pct_change()*100
            plt.plot(stoke_data[f'{col}_daily_%_return'],label=col)
        plt.ylabel('Daily % Return')
        plt.title("Daily % Return")
        plt.legend()
        plt.show()
    