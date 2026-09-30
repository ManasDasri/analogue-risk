# Pre-registration: confirmatory evaluation of analogue risk forecasting

**Status:** frozen. This file, the analysis code (`python/sq/hypotheses.py`) and all model code are
committed and pushed to GitHub *before* any confirmatory data is downloaded. The Git commit
timestamp is the registration date. No data for the confirmatory assets had been downloaded by
this project at the time of registration (`data/` held only the development series listed below).

## 1. Background

All model design (architecture, hyperparameters, bias corrections, the Analogue+HAR and
Analogue/FHS combinations, the bad-tick filter) was developed on the **development set**:
BTC, ETH, PAXG (Binance, hourly); S&P 500, Nasdaq-100 (QQQ), Gold (GLD), Treasuries (TLT),
EUR/USD, NIFTY 50 (Yahoo, daily). Some development-set out-of-sample numbers were inspected during
development, so development results are **exploratory**. This registration defines a
**confirmatory** test on series never used in development.

## 2. Confirmatory data

| Group | Series (source ticker) |
|---|---|
| Hourly crypto (Binance spot, from 2018-01-01 or listing) | SOL (SOLUSDT), BNB (BNBUSDT), XRP (XRPUSDT), ADA (ADAUSDT), LTC (LTCUSDT) |
| Daily (Yahoo, from 1950-01-01 or first available) | DAX (^GDAXI), Nikkei 225 (^N225), FTSE 100 (^FTSE), Hang Seng (^HSI), Russell 2000 (IWM), Emerging markets (EEM), Silver (SLV), Oil (USO), GBP/USD (GBPUSD=X), USD/JPY (JPY=X) |

Data end: the latest complete bar at download time. Cleaning is exactly `sq/data.py` as
committed (the incomplete final bar is dropped; Yahoo series pass through `drop_bad_ticks`).
**Exclusion rule:** a series with fewer than 20,000 hourly or 3,000 daily bars after cleaning is
excluded and reported. No other series may be added or removed.

## 3. Models and protocol (frozen)

Exactly the code at the registration commit, run with
`python -m sq.experiments --universe confirmatory` (experiments e1, e2, e3, e4, e5, e8) and
analysed with `python -m sq.hypotheses results/confirmatory`.

- Split: calibration = first 30% of bars (at least warm-up + 500); out-of-sample = remainder in
  5 walk-forward folds; every fitted quantity uses only data before its fold.
- Direction (e1): `model.Config` defaults (hourly) / `model.DAILY` (daily); 18-config grid.
- Volatility (e2, e3): `vol.HOURLY` (h = 24) for hourly series; `vol.DAILY_W` (h = 5) and
  `vol.DAILY_M` (h = 22) for daily series. Models: EWMA, GARCH(1,1), HAR, GBM, Analogue (kNN,
  k = 20 per regime, Markov regime mixture, exclusion zone h), Analogue+HAR (equal log weights).
- Tail events (e4): raw jumps |r| > 3 long-run sigma; Constant, Poisson (v1), Hawkes,
  Logistic-EWMA, Analogue.
- Overlay (e5): long-only volatility targeting, no leverage, 5 bp per unit turnover.
- VaR/ES (e8): daily (hourly series resampled to UTC days); HS, GARCH-N, GARCH-t, FHS, Analogue,
  Blend = mean of Analogue and FHS quantiles/ES; alpha in {1%, 2.5%, 5%}; `k = 250, min_hist =
  2000` for series with >= 6,000 bars, else `k = 125, min_hist = 1000`.
- Bootstrap: 1,000 stationary-bootstrap draws; seeds as in the code.

## 4. Hypotheses and primary endpoints

Unit of analysis: one series (asset x horizon for e2/e3/e4; asset x alpha for e8, primary alpha =
2.5%; asset for e1/e5). Tests are across series. Holm correction over H1-H8 at a family-wise 5%.

| ID | Hypothesis | Test | Development result (exploratory; `results/hypotheses.md`) |
|---|---|---|---|
| H1 | Analogue direction forecasts have zero mean IC (m = 4) | two-sided one-sample t-test on per-asset IC; plus no grid DSR >= 0.95 | null holds: mean IC -0.004, p = 0.56; max DSR 0.58 |
| H2 | Analogue+HAR QLIKE differs from HAR | two-sided Wilcoxon on log QLIKE ratios | tie: median ratio 1.012, p = 0.30 |
| H3 | Analogue alone QLIKE differs from HAR | two-sided Wilcoxon | worse: median ratio 1.103, p = 0.001 |
| H4 | Gaussian-kernel analogue (bw 0.6) has lower QLIKE than kNN analogue | one-sided Wilcoxon | better: median ratio 0.967, p < 0.001 (post-hoc finding) |
| H5 | GBM QLIKE differs from HAR | two-sided Wilcoxon | worse: median ratio 1.52, p < 0.001 |
| H6 | Hawkes log-loss lower than static Poisson (v1) | one-sided Wilcoxon on log-loss ratios | better: median ratio 0.930, p < 0.001 |
| H7 | Blend FZ0 loss differs from FHS (alpha = 2.5%) | two-sided Wilcoxon on FZ0 differences | no difference: p = 0.91 |
| H8 | Volatility targeting reduces max drawdown vs buy & hold | one-sided sign test | reduces: 9/9, p = 0.002 |

Secondary (reported, not part of the family): per-series Diebold-Mariano tests and MCS
membership; Logistic-EWMA vs Hawkes; Analogue and Blend vs GARCH-t; Kupiec and Christoffersen
coverage rates; Sharpe differences in the overlay.

## 5. Interpretation rules

A hypothesis "differs"/"beats" claim is supported only if its Holm-adjusted p < 0.05. H1 is
confirmed if its p >= 0.05 and no DSR >= 0.95. Results are reported in full whichever way they
fall; development and confirmatory results are reported separately and never pooled for the
confirmatory claims.

## 6. Deviations

Any deviation from this plan (for example a download failure, a code fix needed to run on the
new series) will be listed in `results/confirmatory/DEVIATIONS.md` with its reason, and results
with and without the deviation will be reported where possible.
