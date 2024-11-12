import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Optimization:
    def load_data(self,file_path):
        df =pd.read_csv(file_path)
        df['Date']=pd.to_datetime(df['Date'])
        df.set_index('Date',inplace=True)
        
        return df
    
    def log_volitility(self,prediction):
        prediction.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252)).plot(kind='bar')
        
    def log_covaviance(self,prediction):
        log_return = prediction.pct_change().apply(lambda x: np.log(1+x))
        cov_matrix=log_return.cov_()
        cov_matrix
        return cov_matrix
        
    def annual_return(self,prediction):
        # Yearly returns for individual companies
        ind_er = prediction.resample('Y').last().pct_change().mean()
        ann_sd = prediction.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
        
        return ind_er,ann_sd
    
    def efficient_frontier_plot(self,df,cov_matrix,ind_er,ann_sd):
        assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
        assets.columns = ['Returns', 'Volatility']
        
        p_ret = [] # Define an empty array for portfolio returns
        p_vol = [] # Define an empty array for portfolio volatility
        p_weights = [] # Define an empty array for asset weights

        num_assets = len(df.columns)
        num_portfolios = 10000

        for portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            p_weights.append(weights)
            returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                            # weights 
            p_ret.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
            sd = np.sqrt(var) # Daily standard deviation
            ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
            p_vol.append(ann_sd)
            
        data = {'Returns':p_ret, 'Volatility':p_vol}

        for counter, symbol in enumerate(df.columns.tolist()):
            #print(counter, symbol)
            data[symbol+' weight'] = [w[counter] for w in p_weights]
            
            
        portfolios  = pd.DataFrame(data)
        portfolios.head() # Dataframe of the 10000 portfolios created
        
        #  Plot efficient frontier
        ax = portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10],label='Portfolios')
        plt.title("Portfolio Risk vs. Returns")  
        plt.xlabel("Volatility (Risk)")          
        plt.ylabel("Expected Returns")  
        plt.legend()         
        plt.show()
        
        return portfolios
        
    def min_vol_port(self,portfolios):
        min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
        # idxmin() gives us the minimum value in the column specified.                               
        min_vol_port
        # plotting the minimum volatility portfolio
        plt.figure(figsize=[10,10])
        plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
        plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500,label='Minimum Volatility Portfolio')
        plt.title("Minimum Volatility Portfolio")
        plt.legend()
        return min_vol_port
        
    def optimal_port(self,portfolios,min_vol_port):
        rf = 0.01 # risk factor
        optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
        optimal_risky_port
        
        # Plotting optimal portfolio
        plt.subplots(figsize=(10, 10))
        plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
        plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500,label='Minimum Volatility Portfolio')
        plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500,label='Optimal Portfolio')
        plt.title("Optimal Portfolio")
        plt.legend()
        
        