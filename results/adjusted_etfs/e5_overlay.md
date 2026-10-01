Out-of-sample volatility-managed long exposure (no leverage, 5 bp per unit turnover, 0.1 rebalance band). p: paired stationary-bootstrap test of the Sharpe difference.

|                                                             |   Sharpe |   ΔSharpe vs B&H |       p |   ann. vol |   max DD |   CVaR 5% |   avg exposure |   turnover / yr |
|:------------------------------------------------------------|---------:|-----------------:|--------:|-----------:|---------:|----------:|---------------:|----------------:|
| ('Nasdaq-100', 'Buy & hold')                                |    0.785 |          nan     | nan     |      0.223 |   -0.534 |    -0.033 |          1.000 |           0.000 |
| ('Nasdaq-100', 'Vol target: EWMA')                          |    0.904 |            0.119 |   0.110 |      0.150 |   -0.327 |    -0.022 |          0.802 |           1.014 |
| ('Nasdaq-100', 'Vol target: GARCH')                         |    0.901 |            0.116 |   0.092 |      0.153 |   -0.348 |    -0.023 |          0.818 |           1.605 |
| ('Nasdaq-100', 'Vol target: HAR')                           |    0.873 |            0.088 |   0.178 |      0.159 |   -0.412 |    -0.024 |          0.835 |           2.092 |
| ('Nasdaq-100', 'Vol target: Analogue')                      |    0.879 |            0.094 |   0.128 |      0.161 |   -0.403 |    -0.024 |          0.840 |           4.486 |
| ('Nasdaq-100', 'Vol target: Analogue+HAR')                  |    0.882 |            0.097 |   0.110 |      0.161 |   -0.409 |    -0.024 |          0.844 |           2.226 |
| ('Nasdaq-100', 'Kelly (1/var): Analogue+HAR')               |    0.909 |            0.124 |   0.256 |      0.134 |   -0.327 |    -0.020 |          0.754 |           5.209 |
| ('Nasdaq-100', 'Vol target: Analogue+HAR + Hawkes gate')    |    0.882 |            0.097 |   0.110 |      0.161 |   -0.409 |    -0.024 |          0.844 |           2.226 |
| ('Treasuries', 'Buy & hold')                                |    0.198 |          nan     | nan     |      0.149 |   -0.484 |    -0.020 |          1.000 |           0.000 |
| ('Treasuries', 'Vol target: EWMA')                          |    0.192 |           -0.006 |   0.808 |      0.129 |   -0.445 |    -0.018 |          0.924 |           0.380 |
| ('Treasuries', 'Vol target: GARCH')                         |    0.190 |           -0.008 |   0.788 |      0.129 |   -0.441 |    -0.018 |          0.930 |           0.595 |
| ('Treasuries', 'Vol target: HAR')                           |    0.182 |           -0.016 |   0.608 |      0.130 |   -0.443 |    -0.018 |          0.925 |           0.627 |
| ('Treasuries', 'Vol target: Analogue')                      |    0.145 |           -0.053 |   0.106 |      0.127 |   -0.471 |    -0.017 |          0.897 |           3.098 |
| ('Treasuries', 'Vol target: Analogue+HAR')                  |    0.179 |           -0.019 |   0.532 |      0.131 |   -0.446 |    -0.018 |          0.933 |           0.918 |
| ('Treasuries', 'Kelly (1/var): Analogue+HAR')               |    0.143 |           -0.055 |   0.304 |      0.121 |   -0.436 |    -0.017 |          0.882 |           2.665 |
| ('Treasuries', 'Vol target: Analogue+HAR + Hawkes gate')    |    0.138 |           -0.060 |   0.308 |      0.127 |   -0.447 |    -0.017 |          0.919 |           1.277 |
| ('Russell 2000', 'Buy & hold')                              |    0.481 |          nan     | nan     |      0.247 |   -0.543 |    -0.037 |          1.000 |           0.000 |
| ('Russell 2000', 'Vol target: EWMA')                        |    0.494 |            0.013 |   0.868 |      0.172 |   -0.298 |    -0.025 |          0.845 |           1.014 |
| ('Russell 2000', 'Vol target: GARCH')                       |    0.525 |            0.043 |   0.570 |      0.173 |   -0.298 |    -0.025 |          0.842 |           1.549 |
| ('Russell 2000', 'Vol target: HAR')                         |    0.513 |            0.031 |   0.684 |      0.178 |   -0.336 |    -0.026 |          0.858 |           1.820 |
| ('Russell 2000', 'Vol target: Analogue')                    |    0.448 |           -0.033 |   0.574 |      0.177 |   -0.353 |    -0.026 |          0.843 |           3.343 |
| ('Russell 2000', 'Vol target: Analogue+HAR')                |    0.480 |           -0.002 |   0.936 |      0.178 |   -0.352 |    -0.026 |          0.851 |           1.826 |
| ('Russell 2000', 'Kelly (1/var): Analogue+HAR')             |    0.444 |           -0.037 |   0.714 |      0.151 |   -0.289 |    -0.022 |          0.776 |           4.592 |
| ('Russell 2000', 'Vol target: Analogue+HAR + Hawkes gate')  |    0.480 |           -0.002 |   0.936 |      0.178 |   -0.352 |    -0.026 |          0.851 |           1.826 |
| ('Emerging mkts', 'Buy & hold')                             |    0.332 |          nan     | nan     |      0.216 |   -0.398 |    -0.031 |          1.000 |           0.000 |
| ('Emerging mkts', 'Vol target: EWMA')                       |    0.319 |           -0.014 |   0.230 |      0.208 |   -0.383 |    -0.030 |          0.976 |           0.078 |
| ('Emerging mkts', 'Vol target: GARCH')                      |    0.323 |           -0.010 |   0.374 |      0.207 |   -0.383 |    -0.030 |          0.976 |           0.087 |
| ('Emerging mkts', 'Vol target: HAR')                        |    0.325 |           -0.007 |   0.416 |      0.212 |   -0.398 |    -0.031 |          0.998 |           0.062 |
| ('Emerging mkts', 'Vol target: Analogue')                   |    0.329 |           -0.004 |   0.156 |      0.215 |   -0.394 |    -0.031 |          0.995 |           0.014 |
| ('Emerging mkts', 'Vol target: Analogue+HAR')               |    0.325 |           -0.007 |   0.110 |      0.213 |   -0.393 |    -0.031 |          0.992 |           0.021 |
| ('Emerging mkts', 'Kelly (1/var): Analogue+HAR')            |    0.320 |           -0.013 |   0.194 |      0.207 |   -0.387 |    -0.030 |          0.967 |           0.071 |
| ('Emerging mkts', 'Vol target: Analogue+HAR + Hawkes gate') |    0.219 |           -0.114 |   0.096 |      0.201 |   -0.411 |    -0.029 |          0.975 |           2.922 |
