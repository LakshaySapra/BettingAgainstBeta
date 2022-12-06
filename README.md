# BettingAgainstBeta

Files Order:

1. BetaEstimates.ipynb: get the returns data and estimate Betas
2. compute_datasets.py: Computes the universe of stocks, fama-french loadings, systematic and specific returns. Just run `python compute_datasets.py`
3. BAB_ExpectedReturns.ipynb: 

                              (i) Load beta, computed with benchmark. The benchmark was computed with top 500 by market cap

                              (ii) Load specific returns (e_t + alpha = r_t - B.T*F_t). There is an e_t over time per stock (historical)
                                Take into account that the 3 factors are the Fama-French, so the market here is different to (i)

                              (iii) Lag the betas computed with benchmark. 

                              (iv) Training period: 1970 - 1999 / Test: 1999 - until the end. 

                                    E[e_t] = alpha 

                                    E[e_t|beta_{t-1}] = lambda * f(beta_{t-1}) + error term

                                    Different functions of beta:

                                        - Z score cross sectional. 

                                    *Note: betas here are the ones computed with the benchmark. 

                                    At time t, estimate lambda

                                    E[alpha_{i,t+1}|beta_{t}] = lambda_{i,t}* f(beta_{i,t}) + error term

                                    Shirnk the coeff toward 1. 

                                    Robust Linear regr with Huber loss. 

                                    After getting the time series of lambdas, get the EWM of them with 36 months. 

                              (v) The last section outputs the expected returns for all stocks for the whole time period.
4. CovarianceMatrix.ipynb: Computes covariance between fama-french factors, and the specific variance estimates. These are used as the covariance matrix in the portfolio optimization.
5. FactorMimickingOpt.ipynb: Uses the expected returns and covariance matrix to compute and backtest two Factor-mimicking portfolios with ex-ante target volatility=0.05: 

   a. Equal-Weighted: Long smallest 10% beta stocks, short largest 10% beta stocks. Balance the long and short portfolios to be beta-neutral, and volatility equal to target volatility.
   
   b. Max-Sharpe: Use Max-Sharpe optimization to calculate the max sharpe portfolio, with an additional constraint on individual alpha contribution. Then scale the portfolio to have volatility equal to target volatility.

6. InvestorPortfolioOpt: Computes investor portfolios to beat the benchmark using the BAB factor. Backtests both portfolios and compute metrics

   a. Benchmark portfolio: Value-weighted portfolio of top500 largest stocks

   b. Mean Variance Optimization using Active Weights: Use mean variance optimization and several other constraints to calculate a portfolio that aims to beat the benchmark. Takes as input the expected returns and covariance matrix.


                                    


