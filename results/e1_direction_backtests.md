Out-of-sample backtests. DSR: deflated Sharpe ratio over the 18-configuration grid (probability that true Sharpe > 0 after the multiple-testing correction).

|                                              |   Sharpe |   ann. return |   max DD |   trades |     DSR |
|:---------------------------------------------|---------:|--------------:|---------:|---------:|--------:|
| ('BTC', 'Buy & hold')                        |    0.837 |         0.373 |   -0.772 |  nan     | nan     |
| ('BTC', 'v1 (Pine), 1 bp')                   |   -0.102 |        -0.002 |   -0.069 |  542.000 | nan     |
| ('BTC', 'v1 (Pine), 5 bp')                   |   -0.439 |        -0.009 |   -0.079 |  542.000 | nan     |
| ('BTC', 'v2 defaults, 5 bp')                 |   -0.330 |        -0.151 |   -0.688 | 2633.000 | nan     |
| ('BTC', 'Momentum, 5 bp')                    |   -0.941 |        -0.224 |   -0.835 | 3821.000 | nan     |
| ('BTC', 'v2 walk-forward grid, 5 bp')        |   -0.120 |        -0.075 |   -0.657 |  nan     |   0.014 |
| ('ETH', 'Buy & hold')                        |    0.778 |         0.352 |   -0.813 |  nan     | nan     |
| ('ETH', 'v1 (Pine), 1 bp')                   |   -0.783 |        -0.021 |   -0.151 |  529.000 | nan     |
| ('ETH', 'v1 (Pine), 5 bp')                   |   -1.037 |        -0.028 |   -0.172 |  529.000 | nan     |
| ('ETH', 'v2 defaults, 5 bp')                 |   -0.855 |        -0.325 |   -0.927 | 2772.000 | nan     |
| ('ETH', 'Momentum, 5 bp')                    |   -0.066 |        -0.044 |   -0.590 | 3782.000 | nan     |
| ('ETH', 'v2 walk-forward grid, 5 bp')        |   -0.761 |        -0.287 |   -0.923 |  nan     |   0.000 |
| ('PAXG', 'Buy & hold')                       |    0.948 |         0.214 |   -0.292 |  nan     | nan     |
| ('PAXG', 'v1 (Pine), 1 bp')                  |    0.000 |        -0.000 |   -0.024 |  389.000 | nan     |
| ('PAXG', 'v1 (Pine), 5 bp')                  |   -0.848 |        -0.007 |   -0.046 |  389.000 | nan     |
| ('PAXG', 'v2 defaults, 5 bp')                |   -1.839 |        -0.192 |   -0.609 |  916.000 | nan     |
| ('PAXG', 'Momentum, 5 bp')                   |   -2.420 |        -0.345 |   -0.839 | 2548.000 | nan     |
| ('PAXG', 'v2 walk-forward grid, 5 bp')       |   -1.145 |        -0.083 |   -0.318 |  nan     |   0.000 |
| ('S&P 500', 'Buy & hold')                    |    0.546 |         0.078 |   -0.568 |  nan     | nan     |
| ('S&P 500', 'v1 (Pine), 1 bp')               |    0.141 |         0.001 |   -0.038 |  229.000 | nan     |
| ('S&P 500', 'v1 (Pine), 5 bp')               |    0.099 |         0.001 |   -0.040 |  229.000 | nan     |
| ('S&P 500', 'v2 defaults, 5 bp')             |    0.077 |         0.003 |   -0.582 |  596.000 | nan     |
| ('S&P 500', 'Momentum, 5 bp')                |    0.188 |         0.007 |   -0.258 | 1227.000 | nan     |
| ('S&P 500', 'v2 walk-forward grid, 5 bp')    |    0.079 |         0.003 |   -0.572 |  nan     |   0.317 |
| ('Nasdaq-100', 'Buy & hold')                 |    0.958 |         0.192 |   -0.356 |  nan     | nan     |
| ('Nasdaq-100', 'v1 (Pine), 1 bp')            |    0.275 |         0.002 |   -0.025 |   50.000 | nan     |
| ('Nasdaq-100', 'v1 (Pine), 5 bp')            |    0.246 |         0.002 |   -0.025 |   50.000 | nan     |
| ('Nasdaq-100', 'v2 defaults, 5 bp')          |    0.288 |         0.018 |   -0.228 |  196.000 | nan     |
| ('Nasdaq-100', 'Momentum, 5 bp')             |    0.199 |         0.008 |   -0.143 |  309.000 | nan     |
| ('Nasdaq-100', 'v2 walk-forward grid, 5 bp') |    0.445 |         0.030 |   -0.209 |  nan     |   0.583 |
| ('Gold', 'Buy & hold')                       |    0.715 |         0.108 |   -0.264 |  nan     | nan     |
| ('Gold', 'v1 (Pine), 1 bp')                  |    0.370 |         0.003 |   -0.022 |   32.000 | nan     |
| ('Gold', 'v1 (Pine), 5 bp')                  |    0.339 |         0.002 |   -0.024 |   32.000 | nan     |
| ('Gold', 'v2 defaults, 5 bp')                |    0.174 |         0.008 |   -0.138 |   72.000 | nan     |
| ('Gold', 'Momentum, 5 bp')                   |    0.301 |         0.014 |   -0.121 |  224.000 | nan     |
| ('Gold', 'v2 walk-forward grid, 5 bp')       |    0.399 |         0.021 |   -0.103 |  nan     |   0.424 |
| ('Treasuries', 'Buy & hold')                 |   -0.164 |        -0.033 |   -0.542 |  nan     | nan     |
| ('Treasuries', 'v1 (Pine), 1 bp')            |    0.069 |         0.001 |   -0.024 |   49.000 | nan     |
| ('Treasuries', 'v1 (Pine), 5 bp')            |    0.033 |         0.000 |   -0.024 |   49.000 | nan     |
| ('Treasuries', 'v2 defaults, 5 bp')          |    0.047 |         0.001 |   -0.220 |  115.000 | nan     |
| ('Treasuries', 'Momentum, 5 bp')             |   -0.011 |        -0.002 |   -0.132 |  258.000 | nan     |
| ('Treasuries', 'v2 walk-forward grid, 5 bp') |   -0.293 |        -0.019 |   -0.292 |  nan     |   0.003 |
| ('EUR/USD', 'Buy & hold')                    |   -0.154 |        -0.015 |   -0.311 |  nan     | nan     |
| ('EUR/USD', 'v1 (Pine), 1 bp')               |   -0.296 |        -0.001 |   -0.016 |   37.000 | nan     |
| ('EUR/USD', 'v1 (Pine), 5 bp')               |   -0.354 |        -0.001 |   -0.018 |   37.000 | nan     |
| ('EUR/USD', 'v2 defaults, 5 bp')             |   -0.372 |        -0.015 |   -0.195 |   60.000 | nan     |
| ('EUR/USD', 'Momentum, 5 bp')                |   -0.341 |        -0.014 |   -0.218 |  204.000 | nan     |
| ('EUR/USD', 'v2 walk-forward grid, 5 bp')    |   -0.433 |        -0.019 |   -0.231 |  nan     |   0.002 |
| ('NIFTY 50', 'Buy & hold')                   |    0.621 |         0.094 |   -0.384 |  nan     | nan     |
| ('NIFTY 50', 'v1 (Pine), 1 bp')              |    0.032 |         0.000 |   -0.016 |   25.000 | nan     |
| ('NIFTY 50', 'v1 (Pine), 5 bp')              |   -0.003 |        -0.000 |   -0.017 |   25.000 | nan     |
| ('NIFTY 50', 'v2 defaults, 5 bp')            |   -0.145 |        -0.012 |   -0.208 |   95.000 | nan     |
| ('NIFTY 50', 'Momentum, 5 bp')               |    0.019 |        -0.000 |   -0.196 |  147.000 | nan     |
| ('NIFTY 50', 'v2 walk-forward grid, 5 bp')   |   -0.636 |        -0.045 |   -0.390 |  nan     |   0.002 |
