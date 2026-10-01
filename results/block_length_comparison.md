Robustness: fixed bootstrap block lengths (2h for volatility and overlay, 10 days for VaR) vs Politis-White (2004) automatic block lengths.

|                                              |   cases |   in 90% MCS: fixed blocks |   in 90% MCS: Politis-White |   verdicts changed |   significant at 5%: fixed blocks |   significant at 5%: Politis-White |
|:---------------------------------------------|--------:|---------------------------:|----------------------------:|-------------------:|----------------------------------:|-----------------------------------:|
| ('Development', 'Volatility MCS (e2)')       |  90.000 |                     61.000 |                      61.000 |              0.000 |                           nan     |                            nan     |
| ('Development', 'VaR/ES MCS (e8)')           | 162.000 |                    135.000 |                     135.000 |              2.000 |                           nan     |                            nan     |
| ('Development', 'Overlay Sharpe test (e5)')  |  63.000 |                    nan     |                     nan     |              0.000 |                             4.000 |                              4.000 |
| ('Confirmatory', 'Volatility MCS (e2)')      | 150.000 |                     99.000 |                     105.000 |              6.000 |                           nan     |                            nan     |
| ('Confirmatory', 'VaR/ES MCS (e8)')          | 270.000 |                    211.000 |                     220.000 |             11.000 |                           nan     |                            nan     |
| ('Confirmatory', 'Overlay Sharpe test (e5)') | 105.000 |                    nan     |                     nan     |              3.000 |                            14.000 |                             13.000 |
