# analogue-risk

[![CI](https://github.com/ManasDasri/analogue-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/ManasDasri/analogue-risk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg?logo=rust)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/bindings-PyO3%20%2B%20maturin-6b4fbb.svg)](https://pyo3.rs/)

**Regime-conditioned analogue forecasting of market risk, with self-exciting tail-risk gating.**
(StochQuant v2.)

A Rust + Python rebuild of the *Stock_pine* TradingView strategy ("Dynamic Stochastic Quantitative
Trading Model Using 1D K-Nearest Neighbors and Poisson Risk Gating"), re-evaluated under a strict
out-of-sample protocol, and redirected from **direction** forecasting to **risk** forecasting,
which is where the data shows predictability.

The architecture of the original is kept: analogue (k-nearest-neighbour) matching, a Markov
volatility regime, a Poisson-family tail-risk gate, and Kelly-style sizing. Each part was made
statistically sound, then tested against standard benchmarks.

## Quick start

```python
from analogue_risk import Analogue, volatility_forecast
from analogue_risk.data import binance

prices = binance("BTCUSDT", "1h")                    # open/high/low/close DataFrame
vf = volatility_forecast(prices, horizon=24)         # next-day variance from volatility analogues
print(vf.tail())

# the general engine: any causal features, any targets
model = Analogue(horizon=5, k=20)
F = model.forecast(embedding, targets, state=regime)  # mean/se per target, analogues used
idx, w = model.neighbours(embedding, targets, state=regime)  # the analogues behind each forecast
```

## Results summary

The study was **pre-registered**: after a development study on 9 assets, eight hypotheses, the
models, a list of 15 new assets and the analysis code were committed and tagged
[`prereg-v1`](PREREGISTRATION.md) before any of those assets' data was downloaded.
**All eight replicated** ([`results/confirmatory/hypotheses.md`](results/confirmatory/hypotheses.md);
no deviations: [`DEVIATIONS.md`](results/confirmatory/DEVIATIONS.md)).

| Question | Development (9 assets) | Confirmatory (15 new assets) |
|---|---|---|
| Does analogue matching predict **direction**? (H1) | No: mean IC −0.004, best deflated Sharpe 0.58 | **No**: mean IC +0.003 (p = 0.60), best deflated Sharpe 0.46 |
| Analogue+HAR vs HAR for **volatility** (H2) | tie (median QLIKE ratio 1.012); in 90% MCS 14/15 | **tie** (1.001); in MCS 23/25 (HAR 24, GARCH 20) |
| Analogue alone vs HAR (H3) | worse (1.103) | **worse** (1.087, Holm p < 0.001) |
| Gaussian kernel vs kNN analogue (H4) | better (0.967), found post hoc | **better** (0.953, Holm p < 10⁻⁶) |
| Gradient-boosted trees vs HAR (H5) | worse (1.52), in MCS 0/15 | **worse** (1.42), in MCS 2/25 |
| Hawkes vs the v1 static-Poisson crash gate (H6) | better (log-loss ratio 0.93) | **better** (0.85); v1 Poisson is worse than a constant in 20/25; a logistic on EWMA volatility beats both |
| Analogue/FHS blend vs FHS for **VaR/ES** (H7) | no difference | **no difference**; analogue VaR is as well calibrated as FHS, GARCH-t/FHS most accurate |
| Volatility targeting vs buy & hold (H8) | drawdown lower on 9/9 | drawdown lower on **15/15** (median −5.6 pp); Sharpe slightly lower (median −0.015) |

Also: the original "low-pass filter" (timeframe) claim gets only weak, non-significant support
(`e6_timeframe`); the analogue engine runs at ~30k bars/s with capped history and 200k bars in
~90 s with the full O(n²) search on 10 cores (`e7_speed`).

> **Package name.** The Python package is `analogue_risk` (it was `sq` up to the pre-registration;
> `PREREGISTRATION.md` and `results/confirmatory/DEVIATIONS.md` keep the original commands, e.g.
> `python -m sq.experiments`, which today is `python -m analogue_risk.experiments`).

## Method

**Analogue engine** (`src/analog.rs`). For every bar, a causal embedding is compared with all past
anchors whose targets are fully observed. The *k* nearest per volatility regime are chosen greedily
under an exclusion zone (no two analogues within *h* bars, so their outcome windows do not overlap).
Regime groups are mixed by the *h*-step Markov forecast of the regime. A config guard rejects
settings where the exclusion zone would force non-neighbours into the set.

- *Direction model*: embedding = volatility-scaled log returns in *m* segments (m = 1 is the v1
  feature); target = the R-multiple of the exact trade the strategy would take (same simulator as
  the backtester, so labels and execution cannot disagree).
- *Volatility model*: embedding = log variance of *m* past segments relative to the long-run
  level; target = log mean squared return over the next *h* bars (h = 24 hourly, 5 and 22 daily).

**Benchmarks**: EWMA (half-life selected per fold), GARCH(1,1) by MLE with variance targeting,
HAR (Corsi 2009, log form), gradient-boosted trees (fixed hyperparameters), static Poisson (v1),
Hawkes (exponential kernel, MLE), logistic on EWMA variance. For one-day VaR/ES: historical
simulation, GARCH-N, GARCH-t, filtered historical simulation and analogue-conditioned FHS.

**Protocol**: first 30% of each series = calibration, the rest = 5 walk-forward folds. Every fitted
quantity uses only data observed before the fold. Model design was chosen on a validation slice
*inside* the calibration period (`results/dev/design_study.*`). Note for the paper: earlier pilot
runs looked at OOS numbers for BTC and the S&P 500 before the protocol was frozen; the frozen design
(kNN, k = 20, regime mixture, Analogue+HAR combination) was selected on calibration data.

**Statistics**: QLIKE and log-MSE; Diebold–Mariano with Newey–West (lag h); Hansen–Lunde–Nason
Model Confidence Set (T_max, stationary bootstrap, 1000 draws); log-loss / Brier / AUC for tail
probabilities; deflated Sharpe ratio for the direction grid; paired stationary-bootstrap tests of
Sharpe differences.

**Execution** (direction backtests): signal at close, fill at next open, stops/targets intrabar with
stop-first on ambiguous bars, gaps fill at the open, 5 bp per side (1 bp shown for v1 as in the
original). **Overlay**: long-only, no leverage, 5 bp per unit turnover, 0.1 rebalance band.

## Robustness: dividend-adjusted prices

The main study uses Yahoo's unadjusted prices. For the four distribution-paying funds (QQQ, TLT,
IWM, EEM; Yahoo's price indices have no adjusted series)
`python -m analogue_risk.experiments --universe adjusted_etfs --only e2 e5` reruns the volatility and overlay
experiments on dividend-adjusted prices ([`results/adjusted_etfs/comparison.md`](results/adjusted_etfs/comparison.md)).
Volatility results barely move (QLIKE ratios change by at most 0.026). Sharpe ratio levels rise
for both buy-and-hold and volatility targeting, most for Treasuries (+0.20, coupons), but every
comparison keeps its sign: volatility targeting still cuts drawdowns, lowers Sharpe slightly for
TLT, IWM and EEM, and raises it for QQQ.

## Robustness: bootstrap block lengths

The model confidence sets and Sharpe tests use fixed bootstrap block lengths (2h; 10 days for
VaR). `--auto-block` replaces them with Politis-White (2004) estimates
([`results/block_length_comparison.md`](results/block_length_comparison.md)). No headline claim
moves: Analogue+HAR stays in the 90% MCS for 14/15 development and 23/25 confirmatory series, HAR
for 14/15 and 24/25. The automatic lengths are slightly more conservative, admitting a few more
models to some confidence sets (confirmatory: analogue alone 12 -> 14, GARCH 20 -> 22); 13 of 432
VaR/ES MCS verdicts and 3 of 168 overlay Sharpe-test verdicts change.

## Recommended settings for new work

The paper's results use the pre-registered kNN analogue (`vol.HOURLY`, `vol.DAILY_W`,
`vol.DAILY_M`), which stay unchanged so every number reproduces. The Gaussian kernel won the
pre-registered comparison with kNN (H4), so new work should start from `vol.RECOMMENDED`
(`bandwidth=0.6`, `kernel_block=True`). `kernel_block` treats each block of `h` consecutive anchors
as one observation when computing the effective sample size, because their outcome windows overlap;
it changes standard errors and the minimum-sample gate, not the forecast mean.

## Data

**Binance access.** Binance's REST API refuses some regions (e.g. HTTP 451 in the US). The loader
then falls back to the public bulk archive at `data.binance.vision` (or set
`SQ_BINANCE_SOURCE=archive`). The two official sources agree on 99.94% of bars but not all (the
archive contains a misaligned window during a February 2018 outage and a few bars differing around
exchange maintenance), so an archive-built series fails the snapshot check below and results can
differ marginally from the paper; the loader warns when this happens.

**Pinned snapshot.** `python/analogue_risk/data_manifest.json` records, for every series in the paper, the
last bar used, the row count and a SHA-256 of the cleaned data. Loaders truncate to that bar and
verify the hash, so re-running the experiments later reproduces the published numbers; if a
source has revised its history, a warning says so.


Binance spot klines (BTC, ETH, PAXG as a gold proxy; hourly, plus BTC 15-minute) and Yahoo daily
bars (S&P 500 from 1927, Nasdaq-100 ETF, GLD, TLT, EUR/USD, NIFTY 50), cached in `data/`. Yahoo
bars are passed through a documented bad-tick filter (|r| > 8 × 20-day σ **and** reversed > 70% on
the next bar); it removes three EUR/USD prints from 2008 and one S&P 500 print from 1935, and keeps
genuine events such as 2020-03-12.

## Reproduce

```bash
git clone https://github.com/ManasDasri/analogue-risk && cd analogue-risk
curl https://sh.rustup.rs -sSf | sh             # Rust toolchain
uv venv --python 3.12 && uv pip install maturin numpy pandas scipy matplotlib tabulate pytest
.venv/bin/maturin develop --release              # builds the Rust core into analogue_risk._core
.venv/bin/python -m pytest tests                 # 15 correctness tests
.venv/bin/python -m analogue_risk.experiments               # development tables -> results/  (~35 min on an M4)
.venv/bin/python -m analogue_risk.experiments --universe confirmatory --only e1 e2 e3 e4 e5 e8
.venv/bin/python -m analogue_risk.hypotheses results/confirmatory   # pre-registered tests
.venv/bin/python -m analogue_risk.figures                   # figures  -> results/figures/
.venv/bin/python paper/make_tables.py            # LaTeX tables for the manuscripts
cd paper/research && tectonic main.tex           # or any LaTeX engine; also paper/softwarex
```

Tests cover: no look-ahead (forecasts, labels and the v1 port are identical on truncated data),
Hawkes and GARCH likelihoods against brute force, Hawkes parameter recovery, label = executed
trade, planted-signal recovery vs. random walk, MCS behaviour, and the bootstrap.

## Checking the Pine port against TradingView

The Rust port of the original strategy (`src/legacy.rs`) can be checked against TradingView's own
backtest. In TradingView, add `codes/script.pine` (default inputs) to `BINANCE:BTCUSDT`, 1h, set
the chart timezone to UTC, open *Strategy Tester -> List of Trades*, export the CSV, save it as
`tests/data/tradingview_btcusdt_1h.csv`, and run

```bash
.venv/bin/python -m analogue_risk.pine_parity tests/data/tradingview_btcusdt_1h.csv   # report
.venv/bin/python -m pytest tests -k tradingview                            # >= 95% of trades must match
```

## Layout

```
src/            Rust core: analog.rs (engine), backtest.rs (simulator), hawkes.rs, garch.rs,
                features.rs, legacy.rs (bar-for-bar port of the Pine v1 strategy), lib.rs (PyO3)
python/analogue_risk/      data.py, model.py (direction + v1 + gates), vol.py (risk models, overlay),
                var.py (VaR/ES), stats.py, experiments.py, hypotheses.py, figures.py
paper/          research/ (journal manuscript), softwarex/ (software paper), refs.bib,
                make_tables.py (every table generated from results/*.csv)
PREREGISTRATION.md   frozen confirmatory plan (tag prereg-v1)
tests/          test_core.py
results/        development e1..e8 tables (.md + .csv), confirmatory/, figures/, config.json,
                dev/ (calibration-period design study)
```

## Limitations

- Direction results are for price-only analogue features at 20-bar horizons; they do not rule
  out predictability from other information.
- Daily S&P 500 data before 1962 has close prices only (ATR regime uses close-to-close ranges).
- The overlay ignores financing and cash yield (no leverage, cash earns zero).
- The Analogue+HAR combination uses fixed equal weights; learned weights were not tested.
- PAXG is a gold-backed token, not spot XAU/USD; early PAXG history is thinly traded.

## Citation

If you use this code, please cite the accompanying papers (in preparation) by M. G. Dasari,
V. KVS and K. N. Meera, Amrita Vishwa Vidyapeetham, Bengaluru.

## License

[MIT](LICENSE) © 2026 Manas Ganesh Dasari. The MIT license applies to the source code in this
repository; the underlying method is also the subject of a patent by the authors.
