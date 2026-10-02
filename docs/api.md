# API reference

```python
import analogue_risk as ar
```

## High level

### `ar.volatility_forecast(prices, horizon=24, kernel=True, **overrides)`
Analogue forecast of the mean squared log return over the next `horizon` bars.

| Argument | Meaning |
|---|---|
| `prices` | DataFrame with `open`, `high`, `low`, `close` columns, any bar size |
| `horizon` | forecast horizon in bars (24 hourly = one day; 5 or 22 daily) |
| `kernel` | `True`: recommended Gaussian-kernel analogue (bandwidth 0.6, overlap-aware effective sample size). `False`: the paper's pre-registered kNN analogue |
| `**overrides` | any field of `vol.VolConfig`, e.g. `m`, `k`, `min_hist`, `long_halflife` |

Returns a DataFrame indexed like `prices` with `variance`, `log_variance`, `realised` (where already
known) and `n_analogues` (number, or effective number, of analogues used). Rows during warm-up are NaN.

### `ar.Analogue(horizon, k=20, exclusion=None, regime=True, bandwidth=0.0, kernel_block=False, lookback=None, min_history=1000, k_min=5)`
The general engine: forecasts any targets from any causal embedding.

| Field | Meaning |
|---|---|
| `horizon` | horizon of the targets; also the default `delay` and Markov-regime horizon |
| `k` | analogues per regime (kNN mode) |
| `exclusion` | minimum spacing between analogues (default `horizon`: non-overlapping outcomes) |
| `regime` | condition on the 0/1 `state` and mix regime groups by the Markov forecast; ignored without `state` |
| `bandwidth` | `> 0` switches to Gaussian-kernel weights over all anchors |
| `kernel_block` | kernel mode: count each block of `horizon` anchors as one observation in the effective sample size |
| `lookback` | candidate history in rows (`None` = all past rows) |
| `min_history` | rows before the first forecast |
| `k_min` | minimum analogues (or effective sample) for a regime group to be used |

**`forecast(embedding, targets, state=None, delay=None, index=None)`** → DataFrame with
`mean_<i>` and `se_<i>` for each target column, `n_analogues` and `p_high_regime`.
`embedding` is `(n, m)`; row *t* may only use information up to *t*. `targets` is `(n,)` or
`(n, p)`; the target of anchor *j* must be observed by *j + delay* (default `horizon`), and only
such anchors are used at each *t*, so forecasts never see the future.

**`neighbours(embedding, targets, state=None, delay=None)`** (kNN mode) → `(indices, weights)`,
each `(n, 2k)`, padded with `-1` / `0`: the analogues behind each forecast and their weights
(summing to 1). Use it to explain a forecast or to compute any functional of the analogue
distribution (quantiles, expected shortfall).

## Data — `ar.data`

| Function | Description |
|---|---|
| `binance(symbol, interval="1h", start="2018-01-01")` | Binance spot klines; falls back to `data.binance.vision` when the API is geo-blocked (`SQ_BINANCE_SOURCE=archive` forces it) |
| `yahoo(symbol, start="1990-01-01", adjusted=False)` | Yahoo daily bars; `adjusted=True` includes distributions (funds only) |
| `resample(df, rule)` | aggregate bars, e.g. `"4h"`, `"1D"` |
| `drop_bad_ticks(df, k=8, reversal=0.7, window=20)` | remove isolated bad prints (applied to Yahoo data) |

Series used in the paper are pinned in `data_manifest.json`; loaders truncate to the pinned bar and
warn if the source has revised its history. Data are cached in `~/.cache/analogue-risk` (installed
package; override with `ANALOGUE_RISK_DATA`) or `data/` (source checkout).

## Risk models — `ar.vol`, `ar.var`, `ar.model`

| Object | Description |
|---|---|
| `vol.VolConfig`, `vol.HOURLY`, `vol.DAILY_W`, `vol.DAILY_M` | the paper's volatility configurations (h = 24, 5, 22) |
| `vol.RECOMMENDED` | the same with the Gaussian kernel and overlap-aware sample size |
| `vol.VolData(model.Market(prices), cfg)` | causal embedding, targets and realised variance for a series |
| `vol.analogue(d, cfg)` | analogue log-variance forecast, jump probability and raw engine output |
| `vol.walk_forward(d, folds)` | walk-forward EWMA, GARCH, HAR, GBM, Analogue, Analogue+HAR forecasts |
| `vol.tail_forecasts(d, folds, ewma_log)` | jump probabilities: constant, static Poisson, Hawkes, logistic-EWMA, analogue |
| `vol.overlay(d, log_forecast, start, kind="vol" or "kelly")` | volatility-managed long exposure |
| `var.forecasts(d, var.VarConfig(), folds)` | one-day VaR/ES: HS, GARCH-N, GARCH-t, FHS, Analogue |
| `var.kupiec`, `var.christoffersen`, `var.tick_loss`, `var.fz0_loss` | VaR/ES backtests and scoring rules |
| `model.run_v1(Market(prices), cost_bps)` | line-by-line port of the original TradingView strategy (check against TradingView with `pine_parity`) |

## Statistics — `ar.stats`

`diebold_mariano`, `model_confidence_set`, `stationary_indices`, `politis_white_block`,
`newey_west_var`, `sharpe`, `max_drawdown`, `cvar`, `summary`, `psr`, `dsr`, `sharpe_diff_test`,
`log_loss`, `brier`, `auc`.

## Reproducing the paper — command line

| Command | Output |
|---|---|
| `python -m analogue_risk.experiments [--universe development\|confirmatory\|adjusted_etfs] [--only e1 … e8] [--auto-block]` | result tables in `results/` |
| `python -m analogue_risk.hypotheses results/confirmatory` | the pre-registered tests |
| `python -m analogue_risk.figures` | paper figures |
| `python -m analogue_risk.pine_parity trades.csv` | port vs TradingView trade-list comparison |

## Low level — `ar._core`

The Rust functions (`forecast`, `neighbours`, `labels`, `backtest`, `hawkes_loglik`, `hawkes_prob`,
`garch_filter`, `expanding_tail`, `rolling_tail`, `legacy_v1`, `features`, `embed`) take NumPy
arrays and plain arguments; prefer the classes above.
