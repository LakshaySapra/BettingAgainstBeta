# BettingAgainstBeta

Files Order:

1. BetaEstimates.ipynb: get the returns data and estimate Betas
2. BAB_ExpectedReturns.ipynb: (i) Load beta, computed with benchmark. The benchmark was computed with top 500 by market cap
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
                                    E[alpha_{i,t+1}|beta_{t}] = lambda_{i,t}* f(beta_{i,t}) + erro term
                                    Shirnk the coeff toward 1. 
                                    Robust Linear regr with Huber loss. 
                                    After getting the time series of lambdas, get the EWM of them with 36 months. 


                                    


