import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

def get_fama_french_monthly():
    ff_factors = pd.read_csv('data/F-F_Research_Data_Factors.csv',index_col=0,header=3).iloc[:-100].astype(float)/100
    ff_factors.index = pd.to_datetime(ff_factors.index,format='%Y%m') + pd.tseries.offsets.MonthEnd()
    ff_factors.to_csv('data/F-F_Research_Data_Factors_Monthly.csv')

def get_portfolio_excess_returns():
    crsp_m = pd.read_csv('data/crspmsf.csv', parse_dates=[0])
    crsp_m['permno'] = crsp_m['permno'].astype(int)
    crsp_m['mktcap'] = abs(crsp_m['shrout']*crsp_m['altprc'])
    returndata = crsp_m[['date','permno','ret','mktcap']]

    # Remove the stocks where we dont have returns data
    returndata = returndata[~returndata.ret.isna()]

    # Sort the data date wise
    returndata = returndata.sort_values(['date', 'permno'])

    # Update the date to retain ony the year and month string
    returndata['date'] = returndata['date'].apply(lambda x: x.replace('-', '')[:6])

    # ff_factors = pd.read_csv('data/F-F_Research_Data_Factors_Monthly.csv', names='date,mkt_rf,smb,hml,rf'.split(','),skiprows=1)
    ff_factors = pd.read_csv('data/F-F_Research_Data_Factors_Monthly.csv', names='date,mkt_rf,smb,hml,rf'.split(','),skiprows=1,parse_dates=[0])
    ff_factors['date'] = ff_factors['date'].dt.strftime('%Y%m')
    # Join the ff factors and the CRSP returns
    returndata = pd.merge(returndata, ff_factors[['date', 'rf']], how='left', on=['date'])
    # Get the excess returns
    returndata['excess_ret'] = returndata['ret'] - returndata['rf'].astype(float)

    # Create a portfolio and get the portfolio excess return
    def get_portfolio_excess_ret(data_dt, num_stocks):
        data_dt_temp = data_dt.copy()
        data_dt_temp = data_dt_temp.sort_values('mktcap', ascending=False)[:num_stocks]
        total_mkt_cap = np.sum(data_dt_temp.mktcap)
        return np.sum(data_dt_temp.excess_ret*data_dt_temp.mktcap)/total_mkt_cap

    # Number of stocks for the Benchmark
    N = 500

    portfolio_excess_ret = returndata.groupby('date').apply(lambda x: get_portfolio_excess_ret(x, N))
    returndata = pd.merge(returndata, portfolio_excess_ret.reset_index().rename(columns={0: 'benchmark_excess_ret'}), on='date', how='left')
    portfolio_excess_ret.index = pd.to_datetime(portfolio_excess_ret.index,format='%Y%m') + pd.tseries.offsets.MonthEnd()
    portfolio_excess_ret = portfolio_excess_ret.reset_index().rename(columns={0: 'benchmark_excess_ret'})

    portfolio_excess_ret.to_csv('data/portfolio_excess_ret.csv')

def get_universe(N=500):
    crsp_m = pd.read_csv('data/crspmsf.csv',index_col=0)
    crsp_m['permno'] = crsp_m['permno'].astype(int)
    crsp_m['mktcap'] = abs(crsp_m['shrout']*crsp_m['altprc'])
    returndata = crsp_m[['date','permno','ret','mktcap']]
    data = returndata.pivot('date','permno')
    data.index = pd.to_datetime(data.index)
    universe = data['mktcap'].apply(lambda ser: ser.sort_values(ascending=False)[:N].index.to_numpy(),axis=1)
    universe = pd.DataFrame(list(universe.values),index=universe.index)
    universe.to_csv(f'data/TOP{N}_universe.csv')

def ff_loadings(exp_weight_months=36,rolling_window_months=60,universe_N=2000):
    ff_factors = pd.read_csv('data/F-F_Research_Data_Factors_Monthly.csv')
    crsp_m = pd.read_csv('data/crspmsf.csv',index_col=0)
    crsp_m['permno'] = crsp_m['permno'].astype(int)
    crsp_m['mktcap'] = abs(crsp_m['shrout']*crsp_m['altprc'])
    returndata = crsp_m[['date','permno','ret','mktcap']]
    data = returndata.pivot('date','permno')
    data.index = pd.to_datetime(data.index)
    
    universe = pd.read_csv(f'data/TOP{universe_N}_universe.csv',parse_dates=[0],index_col=0)

    def get_exp_weights(exp_halflife,N=2000,weights_dict = {}):
        if (exp_halflife,N) in weights_dict:
            return weights_dict[exp_halflife,N]
        else:
            x = N - np.ones(N).cumsum()
            alpha = 0.5**(1/exp_halflife)
            weights_dict[exp_halflife,N] = alpha**x
            return weights_dict[exp_halflife,N]

    def compute_beta(stock,factors,exp_halflife = None):
        if exp_halflife is not None:
            weights = get_exp_weights(exp_halflife,len(stock))
            weights = pd.Series(weights,index=stock.index)
            # display(weights)
        stock = stock.dropna()
        if len(stock) > 0:
            stock, factors = stock.align(factors,join='inner')
            X,y = factors.values,stock.values
            X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
            if exp_halflife is not None:
                w = weights.reindex_like(stock).values
                X,y = X*np.sqrt(w[:,None]), y*np.sqrt(w)
            betas = np.linalg.lstsq(X,y)[0]
        else:
            betas = [np.nan]*(len(factors.columns)+1)
        return pd.Series(betas,index=['alpha'] + list(factors.columns))
    
    def get_beta(sub,exp_halflife=None):
        return sub.apply(lambda ser: compute_beta(ser-ff_factors['RF'],ff_factors[['Mkt-RF','SMB','HML']],exp_halflife))

    roll_data = data['ret'].loc['1986-02-01':].iloc[:]
    concat_lst = []
    for i,sub in tqdm(enumerate(roll_data.rolling(rolling_window_months,axis=0)),total=roll_data.shape[0]):
        cur_date = roll_data.index[i]
        concat_lst.append(get_beta(sub[universe.loc[cur_date]],exp_weight_months).unstack(0))
    beta_df = pd.DataFrame(concat_lst,index=roll_data.index)
    beta_df.to_csv(f'data/ffloadings_halflife{exp_weight_months}_TOP{universe_N}.csv')

    systematic_returns = None
    for col in ['Mkt-RF','SMB','HML']:
        tmp = beta_df.reorder_levels([1,0],axis=1)[col].multiply(ff_factors[col],axis=0).tail(20)
        if systematic_returns is None:
            systematic_returns = tmp
        else:
            systematic_returns += tmp
    systematic_returns = systematic_returns.add(ff_factors['RF'],axis=0)
    specific_returns = data['ret'] - systematic_returns

    systematic_returns.to_csv(f'data/systematic_returns_halflife{exp_weight_months}_TOP{universe_N}.csv')
    specific_returns.to_csv(f'data/specific_returns_halflife{exp_weight_months}_TOP{universe_N}.csv')
    


if __name__ == "__main__":
    print("Fama-French Monthly...")
    get_fama_french_monthly()
    # print("Benchmark Portfolio Excess Returns...")
    # get_portfolio_excess_returns()
    print("TOP500 Universe...")
    get_universe(500)
    print("TOP2000 Universe...")
    get_universe(2000)
    print("Fama French Loadings, Systematic and Specific Returns...")
    ff_loadings()