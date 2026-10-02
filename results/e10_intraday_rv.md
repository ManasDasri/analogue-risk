Hourly crypto with realised variance from 5-minute returns as target and HAR/analogue input.

|                                                |   QLIKE |   QLIKE / HAR |   MCS p (QLIKE) |
|:-----------------------------------------------|--------:|--------------:|----------------:|
| ('BTC (h=24)', 'EWMA')                         |   0.365 |         1.300 |           0.046 |
| ('BTC (h=24)', 'GARCH')                        |   0.341 |         1.213 |           0.046 |
| ('BTC (h=24)', 'HAR')                          |   0.281 |         1.000 |           1.000 |
| ('BTC (h=24)', 'GBM')                          |   0.335 |         1.193 |           0.046 |
| ('BTC (h=24)', 'Analogue')                     |   0.310 |         1.103 |           0.046 |
| ('BTC (h=24)', 'Analogue+HAR')                 |   0.285 |         1.013 |           0.727 |
| ('BTC (h=24)', 'HAR (rolling 2190)')           |   0.303 |         1.077 |           0.112 |
| ('BTC (h=24)', 'Analogue+HAR (rolling 2190)')  |   0.296 |         1.052 |           0.046 |
| ('BTC (h=24)', 'HAR (rolling 8760)')           |   0.285 |         1.013 |           0.727 |
| ('BTC (h=24)', 'Analogue+HAR (rolling 8760)')  |   0.287 |         1.020 |           0.120 |
| ('ETH (h=24)', 'EWMA')                         |   0.339 |         1.336 |           0.071 |
| ('ETH (h=24)', 'GARCH')                        |   0.297 |         1.171 |           0.074 |
| ('ETH (h=24)', 'HAR')                          |   0.254 |         1.000 |           1.000 |
| ('ETH (h=24)', 'GBM')                          |   0.317 |         1.250 |           0.025 |
| ('ETH (h=24)', 'Analogue')                     |   0.282 |         1.112 |           0.071 |
| ('ETH (h=24)', 'Analogue+HAR')                 |   0.259 |         1.022 |           0.247 |
| ('ETH (h=24)', 'HAR (rolling 2190)')           |   0.273 |         1.075 |           0.239 |
| ('ETH (h=24)', 'Analogue+HAR (rolling 2190)')  |   0.270 |         1.062 |           0.079 |
| ('ETH (h=24)', 'HAR (rolling 8760)')           |   0.258 |         1.016 |           0.616 |
| ('ETH (h=24)', 'Analogue+HAR (rolling 8760)')  |   0.262 |         1.032 |           0.247 |
| ('PAXG (h=24)', 'EWMA')                        |   0.624 |         2.164 |           0.000 |
| ('PAXG (h=24)', 'GARCH')                       |   0.536 |         1.859 |           0.000 |
| ('PAXG (h=24)', 'HAR')                         |   0.289 |         1.000 |           0.526 |
| ('PAXG (h=24)', 'GBM')                         |   0.320 |         1.108 |           0.021 |
| ('PAXG (h=24)', 'Analogue')                    |   0.318 |         1.104 |           0.021 |
| ('PAXG (h=24)', 'Analogue+HAR')                |   0.288 |         0.998 |           0.526 |
| ('PAXG (h=24)', 'HAR (rolling 2190)')          |   0.279 |         0.967 |           1.000 |
| ('PAXG (h=24)', 'Analogue+HAR (rolling 2190)') |   0.279 |         0.967 |           0.996 |
| ('PAXG (h=24)', 'HAR (rolling 8760)')          |   0.280 |         0.969 |           0.969 |
| ('PAXG (h=24)', 'Analogue+HAR (rolling 8760)') |   0.282 |         0.976 |           0.905 |
