# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:33:10 2022

@author: JuliJaramillo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px

class PerformanceMetrics:
    
    def __init__(self, returns, risk_free, market_returns):
        '''
        Returns is a pd.DataFrame with rows being the dates and columns being the different securites
        Parameters:
            returns: pd.DataFrame with returns. NO excess returns
            risk_free: pd.Series with cash
            market_returns: EXCESS market returns
        '''
        # if (returns.shape[0] != risk_free.shape[0]):
        #     raise Exception('Number of rows of returns and risk free rate should match')
            
        self.returns = returns
        self.T = returns.shape[0]
        # self.risk_free = risk_free
        # self.market_returns = market_returns
        self.risk_free = risk_free[risk_free.index.isin(returns.index)]
        self.market_returns = market_returns[market_returns.index.isin(returns.index)]
        
    def get_averageReturn(self, arithmetic = True, annualize = 1, excessReturn = False):
        '''
        Computes avg return
        Parameters:
            arithmetic: if true, is arithmetic, if false, geometric
            annualize: if 1, not annualizing, if it is different, then the annualize should be the constant to perform
            the annualization
            excessReturn: average excess return or no excess return
        returns:
            avg return per column
        '''
        avg_return = ''
        excessReturns = self.returns.copy()
        
        if excessReturn:
            excessReturns = excessReturns.subtract(self.risk_free, axis = 0)
        
        if arithmetic:
            avg_return = np.mean( excessReturns, axis = 0) * annualize
        else:
            avg_return = np.prod( (1 + excessReturns), axis = 0) **(1/self.T) - 1
            avg_return = (1 + avg_return)**annualize - 1
            
        
        return avg_return    

    def get_volatility(self, annualize = 1, excessReturn = False):
        
        if excessReturn:
            data = self.returns.subtract(self.risk_free, axis = 0)
        else:
            data = self.returns
        
        vol = np.std(data, ddof = 1, axis = 0) * np.sqrt(annualize)
        
        return vol

    def get_rolling_volatility(self, window = 12, annualize = 1, excessReturn = False):
        
        if excessReturn:
            data = self.returns.subtract(self.risk_free, axis = 0)
        else:
            data = self.returns
        
        roll_vol = data.rolling(window).std(ddof=1) * np.sqrt(annualize)
        
        return roll_vol

    def get_sharpeRatio(self, annualize = 1, excessReturn = False):
        
        '''
        Aritmethic returns / volatility, not geometric
        '''
        
        num = self.get_averageReturn(arithmetic = True, annualize = 1, excessReturn = excessReturn)
        den = self.get_volatility(annualize = 1, excessReturn = excessReturn)
        
        sharpeRatio = (num/den) * np.sqrt(annualize) 
        
        return sharpeRatio
        
    def get_Alpha_Beta_IR(self,  annualize = 1, excessReturn = False):
        
        '''Market Returns are assumed to be already excess return'''
        theMarketReturns = self.market_returns
        
        if excessReturn:
            data = self.returns.subtract(self.risk_free, axis = 0)
            
        else:
            data = self.returns
        
        columns = list(data.columns)
        sizeCol = len(columns)
        
        results = np.zeros(shape = (sizeCol, 7))
        
        for idx, col in enumerate(columns): #Iterate through the columns
            y = data.iloc[:, idx].copy()
            X = theMarketReturns.copy()
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            results[idx, 0] = model.params.iloc[0] * annualize #alpha
            results[idx, 1] = model.params.iloc[1] #beta
            
            den = np.std(model.resid,ddof = 1)
            results[idx, 2] = (model.params.iloc[0] / den) * np.sqrt(annualize)
            
            results[idx, 3] = model.tvalues.iloc[0] #t value of alpha
            results[idx, 4] = model.tvalues.iloc[1] #t value of beta
            results[idx, 5] = model.pvalues.iloc[0] #p value of alpha
            results[idx, 6] = model.pvalues.iloc[1] #p value of beta
            
        return pd.DataFrame(results, columns = ['Annualized Alpha', 'Beta', 'IR', 'Alpha T value', 'Beta T Value', 'Alpha P value', 'Beta P Value'], index = columns)
    
    def get_cumulativeReturn(self):
        '''These are non excess returns'''
        return np.cumprod( 1 + self.returns, axis = 0)
    
    def get_HWM(self):
        cumProd = self.get_cumulativeReturn()
        return np.maximum.accumulate(cumProd, axis = 0)
        
        
    def get_drawdown(self):
        
        HWM = self.get_HWM()
        cumProd = self.get_cumulativeReturn()
        
        return (HWM - cumProd)/HWM
    
    def get_maxDrawdown(self):
        drawdown = self.get_drawdown()
        
        return np.max(drawdown, axis = 0)
    
    def get_skew(self, excessReturn = False):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)
            
        skew_ = skew(data, axis = 0, bias = True)
        
        return pd.Series(skew_, index = self.returns.columns)
    
    def get_kurtosis(self, excessReturn = False):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)
            
        kurt = kurtosis(data, axis = 0, bias = True)
        
        return pd.Series(kurt, index = self.returns.columns)

    def port_returns(self, weights, excessReturn = False):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)
        return data @ weights#pd.Series(data @ weights)
           
    def get_basic_metrics(self, annualize, excessReturn = False):
        #Averages
        avg_return_arithmetic = self.get_averageReturn(arithmetic = True, annualize = annualize, 
                                                       excessReturn = False) 
        avg_excess_return_arithmetic = self.get_averageReturn(arithmetic = True, annualize = annualize, 
                                                       excessReturn = True) 
        
        avg_return_geometric = self.get_averageReturn(arithmetic = False, annualize = annualize, 
                                                      excessReturn = False) 
        avg_excess_return_geometric = self.get_averageReturn(arithmetic = False, annualize = annualize, 
                                                      excessReturn = True) 
        
        #vol
        vol = self.get_volatility(annualize = annualize, excessReturn = excessReturn)
        
        #SR
        sr = self.get_sharpeRatio(annualize = annualize, excessReturn = excessReturn)

        #max DD
        max_dd = self.get_maxDrawdown()
        
        #Skew and kurt
        skew_ = self.get_skew(excessReturn = False)
        ExSkew_ = self.get_skew(excessReturn = True)
        
        kurt = self.get_kurtosis(excessReturn = False)
        Exkurt = self.get_kurtosis(excessReturn = True)
        #Create a DF
        toConcat = [avg_return_arithmetic, avg_excess_return_arithmetic, avg_return_geometric, avg_excess_return_geometric,
                    vol, sr, max_dd, skew_, ExSkew_, kurt, Exkurt]
        toReturn = pd.concat(toConcat, axis = 1)
        
        part1 = ['RetArith', 'ExcRetArith', 'AvgRetGeo', 'ExcRetGet', 'Vol', 'SR']
        part2 = ['MaxDD', 'Skew','ExcRetSkew', 'Kurt', 'ExcRetKurt']
        toReturn.columns =  part1 + part2
        
        return toReturn

    def get_metrics(self, annualize, excessReturn = False):
        basic = self.get_basic_metrics(annualize = annualize, excessReturn = excessReturn)
        #Alpha beta
        alpha_beta = self.get_Alpha_Beta_IR(annualize = annualize, excessReturn = excessReturn)

        toReturn = pd.concat([basic, alpha_beta], axis = 1)

        return toReturn

    
    #def get_metrics(self, annualize, excessReturn = False):
        
    #     #Averages
    #     avg_return_arithmetic = self.get_averageReturn(arithmetic = True, annualize = annualize, 
    #                                                    excessReturn = False) 
    #     avg_excess_return_arithmetic = self.get_averageReturn(arithmetic = True, annualize = annualize, 
    #                                                    excessReturn = True) 
        
    #     avg_return_geometric = self.get_averageReturn(arithmetic = False, annualize = annualize, 
    #                                                   excessReturn = False) 
    #     avg_excess_return_geometric = self.get_averageReturn(arithmetic = False, annualize = annualize, 
    #                                                   excessReturn = True) 
        
    #     #vol
    #     vol = self.get_volatility(annualize = annualize, excessReturn = excessReturn)
        
    #     #SR
    #     sr = self.get_sharpeRatio(annualize = annualize, excessReturn = excessReturn)
        
    #     #Alpha beta
    #     alpha_beta = self.get_Alpha_Beta_IR(annualize = annualize, excessReturn = excessReturn)
        
    #     #max DD
    #     max_dd = self.get_maxDrawdown()
        
    #     #Skew and kurt
    #     skew_ = self.get_skew(excessReturn = False)
    #     ExSkew_ = self.get_skew(excessReturn = True)
        
    #     kurt = self.get_kurtosis(excessReturn = False)
    #     Exkurt = self.get_kurtosis(excessReturn = True)
        
    #     #Create a DF
    #     toConcat = [avg_return_arithmetic, avg_excess_return_arithmetic, avg_return_geometric, avg_excess_return_geometric,
    #                 vol, sr, alpha_beta, max_dd, skew_, ExSkew_, kurt, Exkurt]
    #     toReturn = pd.concat(toConcat, axis = 1)
        
    #     part1 = ['RetArith', 'ExcRetArith', 'AvgRetGeo', 'ExcRetGet', 'Vol', 'SP'] + list(alpha_beta.columns) 
    #     part2 = ['MaxDD', 'Skew','ExcRetSkew', 'Kurt', 'ExcRetKurt']
    #     toReturn.columns =  part1 + part2
        
    #     return toReturn
    
    def get_regression(self, X, annualize, excessReturn):
        '''
        Do a linear regression with X as dependent variable and returns as the y. Excess return applies for 
        the returns, not for X. X HAS TO BE ALREADY EXCESS RETURNS.
        Parameters:
            X : EXCESS RETURNS, rows should match rows of returns. X Does not have intercept
            excessReturn: calculate excess returns of returns or not
        Returns:
            array with alphas and betas.
        '''
        
        data = self.returns
        if excessReturn:
            data = self.returns.subtract(self.risk_free, axis = 0)
            
        columns = list(data.columns)
        sizeCol = len(columns)
        
        #each row has the regression of each of the columns in the returns dataframe
        #each column has alpha, and all the betas.
        coefficients = np.zeros(shape = (sizeCol, X.shape[1] + 1)) 
        t_values = np.zeros(shape = (sizeCol, X.shape[1] + 1))
        
        X = sm.add_constant(X)

        for idx, col in enumerate(columns):
            y = data.iloc[:, idx].copy()

            model = sm.OLS(y, X).fit()
            coefficients[idx, :] = model.params.values
            t_values[idx, :] = model.tvalues.values
            
        coefficients[:, 0] = coefficients[:, 0] * annualize
        coefficients = pd.DataFrame(coefficients, index = columns, columns = X.columns)
        t_values = pd.DataFrame(t_values, index = columns, columns = X.columns)
        
        return coefficients, t_values

    def dd_control_fun(self, madd, dd, vol):
        wgt = (madd - dd)/(3*vol)

        if (madd - dd) >= (3*vol) :
            return 1

        else:
            return wgt

    def dd_control_weights(self, madd, window = 12, annualize=1, excessReturn = False):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)

        rolling_vol = self.get_rolling_volatility(window=window, annualize=annualize)
        dd = self.get_drawdown()

        dd_model = np.vectorize(self.dd_control_fun)
        wgt = dd_model(madd, dd, rolling_vol)
        wgt = pd.DataFrame(data = wgt, columns=data.columns, index=data.index)
        # fill the nan in the beginning with 1
        return wgt.fillna(1)

    def stretgy_analytics(self, annualize=1, excessReturn = False):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)
        
        sr = self.get_sharpeRatio(annualize = annualize, excessReturn = True)
        avg_ret = self.get_averageReturn(arithmetic = True, annualize = annualize, excessReturn = excessReturn)
        max_dd = self.get_maxDrawdown()

        df = pd.concat([sr, avg_ret, max_dd], axis=1)
        df.columns = ["Sharpe Ratio", "Avg Return", "Max Drawdown"]
        return df

    def ind_mom_analytics(self, annualize = 1, excessReturn = True):
        data = self.returns.copy()
        if excessReturn:
            data = data.subtract(self.risk_free, axis = 0)

        avg_mth_ret = self.get_averageReturn(annualize=annualize/12, excessReturn=excessReturn)
        std_mth = self.get_volatility(annualize=annualize/12, excessReturn=excessReturn)
        mth_sr = self.get_sharpeRatio(annualize=annualize/12, excessReturn=excessReturn)
        ann_sr = self.get_sharpeRatio(annualize=annualize, excessReturn=excessReturn)
        
        df = pd.concat([avg_mth_ret, std_mth, mth_sr, ann_sr], axis=1)
        df.columns = ["Avg Monthly Returns", "Monthly Std Dev", "Monthly SR", "Annualized SR"]
        return df

