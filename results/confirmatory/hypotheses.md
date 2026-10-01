Pre-registered confirmatory tests (Holm-adjusted over H1-H8).

| id   | hypothesis                                                                                   | statistic                     |         p |   series | note                                            |    Holm p |
|:-----|:---------------------------------------------------------------------------------------------|:------------------------------|----------:|---------:|:------------------------------------------------|----------:|
| H1   | Analogue direction forecasts have zero mean IC across assets (null expected)                 | mean IC +0.0031; max DSR 0.46 | 0.6026    |       15 | confirmed if p >= 0.05 and no DSR >= 0.95       | 1         |
| H2   | Analogue+HAR QLIKE differs from HAR                                                          | median ratio 1.001            | 0.6721    |       25 | in 90% MCS: Analogue+HAR 23, HAR 24             | 1         |
| H3   | Analogue alone QLIKE differs from HAR                                                        | median ratio 1.087            | 8.804e-05 |       25 |                                                 | 0.0003521 |
| H4   | Gaussian-kernel analogue beats kNN analogue (one-sided)                                      | median ratio 0.953            | 5.96e-08  |       25 |                                                 | 4.768e-07 |
| H5   | Gradient boosting QLIKE differs from HAR                                                     | median ratio 1.423            | 5.96e-08  |       25 |                                                 | 4.768e-07 |
| H6   | Hawkes tail forecasts beat static Poisson (one-sided log-loss)                               | median ratio 0.847            | 2.086e-07 |       25 | Logistic-EWMA vs Hawkes p = 0.0136              | 1.252e-06 |
| H7   | Analogue-FHS blend FZ0 differs from FHS at alpha = 2.5%                                      | median diff -0.0005           | 0.9341    |       15 | Analogue vs FHS p = 0.0125; vs GARCH-t p = 0.89 | 1         |
| H8   | Volatility targeting (Analogue+HAR) reduces max drawdown vs buy & hold (one-sided sign test) | 15/15 improve                 | 3.052e-05 |       15 | Sharpe difference p = 0.0256                    | 0.0001526 |
