Pre-registered confirmatory tests (Holm-adjusted over H1-H8).

| id   | hypothesis                                                                                   | statistic                     |         p |   series | note                                              |    Holm p |
|:-----|:---------------------------------------------------------------------------------------------|:------------------------------|----------:|---------:|:--------------------------------------------------|----------:|
| H1   | Analogue direction forecasts have zero mean IC across assets (null expected)                 | mean IC -0.0043; max DSR 0.58 | 0.5622    |        9 | confirmed if p >= 0.05 and no DSR >= 0.95         | 1         |
| H2   | Analogue+HAR QLIKE differs from HAR                                                          | median ratio 1.012            | 0.3028    |       15 | in 90% MCS: Analogue+HAR 14, HAR 14               | 0.9084    |
| H3   | Analogue alone QLIKE differs from HAR                                                        | median ratio 1.103            | 0.00116   |       15 |                                                   | 0.005798  |
| H4   | Gaussian-kernel analogue beats kNN analogue (one-sided)                                      | median ratio 0.967            | 3.052e-05 |       15 |                                                   | 0.0002441 |
| H5   | Gradient boosting QLIKE differs from HAR                                                     | median ratio 1.522            | 6.104e-05 |       15 |                                                   | 0.0003662 |
| H6   | Hawkes tail forecasts beat static Poisson (one-sided log-loss)                               | median ratio 0.930            | 3.052e-05 |       15 | Logistic-EWMA vs Hawkes p = 6.1e-05               | 0.0002441 |
| H7   | Analogue-FHS blend FZ0 differs from FHS at alpha = 2.5%                                      | median diff -0.0003           | 0.9102    |        9 | Analogue vs FHS p = 0.00391; vs GARCH-t p = 0.301 | 1         |
| H8   | Volatility targeting (Analogue+HAR) reduces max drawdown vs buy & hold (one-sided sign test) | 9/9 improve                   | 0.001953  |        9 | Sharpe difference p = 0.652                       | 0.007812  |
