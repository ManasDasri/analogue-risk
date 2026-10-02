Diebold-Mariano tests on QLIKE (lag 24). Positive stat: the second model has lower loss.

|                                                                      |   DM stat |     p |
|:---------------------------------------------------------------------|----------:|------:|
| ('BTC (h=24)', 'Analogue+HAR', 'HAR')                                |    -1.748 | 0.081 |
| ('BTC (h=24)', 'Analogue', 'HAR')                                    |    -6.655 | 0.000 |
| ('BTC (h=24)', 'Analogue+HAR (rolling 8760)', 'HAR (rolling 8760)')  |    -0.704 | 0.482 |
| ('ETH (h=24)', 'Analogue+HAR', 'HAR')                                |    -1.776 | 0.076 |
| ('ETH (h=24)', 'Analogue', 'HAR')                                    |    -4.207 | 0.000 |
| ('ETH (h=24)', 'Analogue+HAR (rolling 8760)', 'HAR (rolling 8760)')  |    -2.205 | 0.027 |
| ('PAXG (h=24)', 'Analogue+HAR', 'HAR')                               |     0.086 | 0.931 |
| ('PAXG (h=24)', 'Analogue', 'HAR')                                   |    -2.152 | 0.031 |
| ('PAXG (h=24)', 'Analogue+HAR (rolling 8760)', 'HAR (rolling 8760)') |    -0.319 | 0.750 |
