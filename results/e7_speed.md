Analogue forecast wall time on 10 cores (includes all O(n^2) distance work).

|                                      |   seconds |   bars / s |
|:-------------------------------------|----------:|-----------:|
| ('kNN, history 5000', 1, 25000)      |      2.85 |    8760.34 |
| ('kNN, history 5000', 1, 50000)      |      7.20 |    6944.99 |
| ('kNN, history 5000', 1, 100000)     |     16.19 |    6175.96 |
| ('kNN, history 5000', 1, 200000)     |     33.86 |    5905.95 |
| ('kNN, history 5000', 10, 25000)     |      0.52 |   47980.37 |
| ('kNN, history 5000', 10, 50000)     |      1.45 |   34513.89 |
| ('kNN, history 5000', 10, 100000)    |      3.22 |   31094.19 |
| ('kNN, history 5000', 10, 200000)    |      6.77 |   29558.28 |
| ('kNN, full history', 1, 25000)      |      6.35 |    3934.94 |
| ('kNN, full history', 1, 50000)      |     22.09 |    2263.17 |
| ('kNN, full history', 1, 100000)     |     75.60 |    1322.78 |
| ('kNN, full history', 1, 200000)     |    297.05 |     673.28 |
| ('kNN, full history', 10, 25000)     |      1.28 |   19468.69 |
| ('kNN, full history', 10, 50000)     |      5.17 |    9671.11 |
| ('kNN, full history', 10, 100000)    |     20.77 |    4814.71 |
| ('kNN, full history', 10, 200000)    |     92.54 |    2161.18 |
| ('kernel, full history', 1, 25000)   |      3.80 |    6572.60 |
| ('kernel, full history', 1, 50000)   |     16.40 |    3049.63 |
| ('kernel, full history', 1, 100000)  |     66.71 |    1499.02 |
| ('kernel, full history', 1, 200000)  |    276.86 |     722.39 |
| ('kernel, full history', 10, 25000)  |      0.85 |   29431.18 |
| ('kernel, full history', 10, 50000)  |      4.15 |   12053.63 |
| ('kernel, full history', 10, 100000) |     19.66 |    5086.21 |
| ('kernel, full history', 10, 200000) |     88.70 |    2254.71 |
