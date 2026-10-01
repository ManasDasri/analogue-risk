"""Public API.

    from analogue_risk import Analogue, volatility_forecast

`Analogue` is the general engine: any causal embedding, any targets. `volatility_forecast` is the
one-call risk forecaster used in the paper (volatility-shape analogues, optionally combined with
HAR in the walk-forward study).
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import _core as core


@dataclass(frozen=True)
class Analogue:
    """Regime-conditioned analogue (nearest-neighbour or Gaussian-kernel) forecaster.

    At each row t, anchors j <= t - delay (whose targets are fully observed by t) are compared with
    the embedding at t. kNN mode picks the `k` nearest anchors per volatility regime, at least
    `exclusion` rows apart; kernel mode (bandwidth > 0) weights all anchors. Regime groups are mixed
    by the `horizon`-step Markov forecast of the regime.

    Parameters
    ----------
    k : analogues per regime (kNN mode).
    exclusion : minimum spacing between analogues; defaults to `horizon` (non-overlapping outcomes).
    regime : condition on the regime state and mix groups; False pools all analogues.
    bandwidth : > 0 switches to Gaussian-kernel weights with this bandwidth (embedding units).
    kernel_block : kernel mode; count each block of `horizon` anchors as one observation in the
        effective sample size (recommended, since their outcomes overlap).
    lookback : candidate history in rows (None = all past rows).
    min_history : rows before the first forecast.
    k_min : minimum analogues (or effective sample) for a regime group to be used.
    """
    horizon: int
    k: int = 20
    exclusion: int | None = None
    regime: bool = True
    bandwidth: float = 0.0
    kernel_block: bool = False
    lookback: int | None = None
    min_history: int = 1000
    k_min: int = 5

    def _args(self, embedding, targets, state, delay):
        emb = np.ascontiguousarray(np.asarray(embedding, float).reshape(len(embedding), -1))
        y = np.ascontiguousarray(np.asarray(targets, float).reshape(len(targets), -1))
        if len(emb) != len(y):
            raise ValueError(f"embedding has {len(emb)} rows but targets has {len(y)}")
        st = np.zeros(len(emb), np.uint8) if state is None else np.ascontiguousarray(state, np.uint8)
        if len(st) != len(emb):
            raise ValueError("state must have one entry per row")
        delay = self.horizon if delay is None else delay
        excl = self.horizon if self.exclusion is None else self.exclusion
        regime = self.regime and state is not None
        lookback = 10 ** 9 if self.lookback is None else self.lookback
        return (emb, st, y, 0, delay, self.horizon, lookback, self.k, self.k_min, excl, regime,
                self.min_history), y.shape[1]

    def forecast(self, embedding, targets, state=None, delay=None, index=None):
        """Forecast every target column at every row.

        embedding : (n, m) causal features; row t may only use information up to t.
        targets   : (n,) or (n, p) outcome attached to each anchor row, observed `delay` rows later
                    (default `horizon`); NaN where unknown.
        state     : optional (n,) 0/1 volatility regime.
        Returns a DataFrame with columns mean_<i>, se_<i> per target, n_analogues and p_high_regime
        (NaN rows where no forecast exists, e.g. during warm-up).
        """
        args, p = self._args(embedding, targets, state, delay)
        out = core.forecast(*args, self.bandwidth, args[4] if (self.kernel_block and self.bandwidth > 0) else 0)
        cols = [f"mean_{i}" for i in range(p)] + [f"se_{i}" for i in range(p)] + ["n_analogues", "p_high_regime"]
        return pd.DataFrame(out, index=index, columns=cols)

    def neighbours(self, embedding, targets, state=None, delay=None):
        """kNN mode: (indices, weights), each (n, 2k), padded with -1 / 0. Row t lists the anchors
        behind the forecast at t and their weights (summing to 1)."""
        if self.bandwidth > 0:
            raise ValueError("neighbours() is defined for kNN mode (bandwidth = 0)")
        args, _ = self._args(embedding, targets, state, delay)
        return core.neighbours(*args)


def volatility_forecast(prices, horizon=24, kernel=True, **overrides):
    """Analogue forecast of mean squared log return over the next `horizon` bars.

    prices : DataFrame with open/high/low/close columns (any bar size).
    kernel : use the recommended Gaussian-kernel settings (bandwidth 0.6, overlap-aware sample size);
             False gives the paper's pre-registered kNN analogue.
    overrides : any field of vol.VolConfig (e.g. m, k, min_hist, long_halflife).
    Returns a DataFrame indexed like `prices` with the variance forecast, its log, the realised
    value where already known, and the number (or effective number) of analogues used.
    """
    from . import model as M, vol as V

    base = V.HOURLY if horizon == 24 else replace(V.DAILY_W, h=horizon)
    cfg = replace(base, bandwidth=0.6 if kernel else 0.0, kernel_block=kernel, **overrides)
    d = V.VolData(M.Market(prices), cfg)
    log_f, _, F = V.analogue(d, cfg)
    return pd.DataFrame({"variance": np.exp(log_f), "log_variance": log_f, "realised": d.realised,
                         "n_analogues": F[:, -2]}, index=prices.index)
