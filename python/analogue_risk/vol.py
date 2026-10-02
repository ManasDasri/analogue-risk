"""Volatility and tail-risk forecasting: regime-conditioned analogues and standard benchmarks.

Target at bar t: log of the mean squared log return over bars t+1..t+h (realised variance per bar).
Every forecaster is walk-forward: parameters for fold [a, b) are fitted only on rows whose
targets are fully observed before a (t + h < a).
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import _core as core
from .model import check_exclusion, fit_hawkes, poisson_prob

FLOOR = 1e-3  # variance floor, as a fraction of the long-run level (stale bars have zero returns)


@dataclass(frozen=True)
class VolConfig:
    h: int = 24  # horizon (bars)
    m: int = 4  # segments in the volatility-shape embedding
    long_halflife: float = 500
    lookback: int = 10**9  # candidate history for analogues (default: all past bars)
    bandwidth: float = 0.0  # > 0: Gaussian-kernel weights over all candidates; 0: k nearest
    kernel_block: bool = False  # kernel mode: overlap-aware effective sample size (blocks of h bars)
    min_hist: int = 8760  # bars of history before the first forecast
    k: int = 20
    k_min: int = 5
    excl: int | None = None  # default: h (non-overlapping outcome windows)
    regime: bool = True
    regime_len: int = 1000
    har: tuple = (1, 24, 168)
    jump_c: float = 3.0  # raw jump: |r_t| > c * long-run sigma_{t-1}
    gate_q: float = 0.9

    @property
    def seg(self):
        return self.h

    @property
    def warmup(self):
        return max(self.min_hist, self.regime_len, self.har[-1], self.m * self.seg) + self.h + 50

    def __post_init__(self):
        if self.bandwidth == 0:
            check_exclusion(
                self.k,
                self.regime,
                self.h if self.excl is None else self.excl,
                min(self.lookback, self.min_hist),
            )


HOURLY = VolConfig()
DAILY_W = VolConfig(h=5, min_hist=1000, regime_len=500, long_halflife=250, har=(1, 5, 22))
DAILY_M = VolConfig(h=22, min_hist=2000, k=10, regime_len=500, long_halflife=250, har=(1, 5, 22))

# Recommended for new work: the Gaussian kernel won the pre-registered comparison with kNN (H4);
# kernel_block makes its effective sample size account for overlapping outcome windows. The
# presets above are kept unchanged so the published results reproduce exactly.
RECOMMENDED = {
    name: replace(cfg, bandwidth=0.6, kernel_block=True)
    for name, cfg in (("hourly", HOURLY), ("daily_w", DAILY_W), ("daily_m", DAILY_M))
}


def intraday_rv(bars, fine):
    """Realised variance of each bar of `bars` from the finer bars `fine` (e.g. hourly from
    5-minute): the sum of squared log returns of the fine closes falling inside the bar, the
    first return taken from the previous bar's last fine close. Bars without fine data fall back
    to their squared bar return."""
    lr = np.log(fine.close).diff()
    rv = (lr**2).groupby(fine.index.floor(pd.infer_freq(bars.index[:50]) or "h")).sum(min_count=1)
    rv = rv.reindex(bars.index)
    sq = np.log(bars.close).diff() ** 2
    return rv.fillna(sq).fillna(0.0).to_numpy()


def _roll_mean(x, n):
    """Mean of x over (t-n, t], NaN before n values."""
    cs = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def _fwd_mean(x, n):
    """Mean of x over (t, t+n], NaN where incomplete."""
    out = np.full(len(x), np.nan)
    out[: len(x) - n] = _roll_mean(x, n)[n:]
    return out


class VolData:
    """Causal inputs and realised targets for one series."""

    def __init__(self, mkt, vc, r2=None):
        """r2: optional per-bar variance measure replacing squared bar returns, e.g. realised
        variance from intraday data (then the target, the realised series, the analogue embedding
        and HAR all use it; GARCH and EWMA remain return-based)."""
        self.mkt, self.vc = mkt, vc
        r, _, _, state = core.features(mkt.h, mkt.l, mkt.c, 50.0, 14, vc.regime_len)
        _, sig_long, _, _ = core.features(mkt.h, mkt.l, mkt.c, vc.long_halflife, 14, vc.regime_len)
        self.r, self.state = r, state
        self.r2 = r * r if r2 is None else np.asarray(r2, float)
        self.long = sig_long**2
        fl = FLOOR * self.long
        self.target = np.log(np.maximum(_fwd_mean(self.r2, vc.h), fl))  # log future variance
        self.realised = _fwd_mean(self.r2, vc.h)
        prev_sig = np.r_[np.nan, sig_long[:-1]]
        self.jumps = np.abs(r) > vc.jump_c * prev_sig
        j = self.jumps.astype(float)
        self.tail = np.where(np.isfinite(self.realised), _fwd_mean(j, vc.h) > 0, np.nan)

        # volatility-shape embedding: log variance of m past segments relative to the long-run level
        segs = [
            np.log(
                np.maximum(
                    np.r_[np.full(q * vc.seg, np.nan), _roll_mean(self.r2, vc.seg)[: len(r) - q * vc.seg]], fl
                )
            )
            for q in range(vc.m - 1, -1, -1)
        ]
        self.emb = np.ascontiguousarray(np.column_stack(segs) - np.log(self.long)[:, None])

    def fit_rows(self, a):
        """Rows usable for fitting before fold start a (targets observed)."""
        return slice(self.vc.warmup, max(self.vc.warmup, a - self.vc.h - 1))


# ---------------------------------------------------------------- forecasters (log variance)


def analogue(d, vc=None, **over):
    """Returns (log-variance forecast, tail probability, forecast output) for every bar."""
    vc = vc or d.vc
    prm = (
        dict(
            k=vc.k,
            k_min=vc.k_min,
            excl=vc.h if vc.excl is None else vc.excl,
            regime=vc.regime,
            lookback=vc.lookback,
            bandwidth=vc.bandwidth,
            kernel_block=vc.kernel_block,
        )
        | over
    )
    y = np.ascontiguousarray(np.column_stack([d.target - np.log(d.long), d.tail]))
    F = core.forecast(
        d.emb,
        d.state,
        y,
        vc.m * vc.seg,
        vc.h,
        vc.h,
        prm["lookback"],
        prm["k"],
        vc.k_min,
        prm["excl"],
        prm["regime"],
        vc.warmup,
        prm["bandwidth"],
        vc.h if prm["kernel_block"] else 0,
    )
    return F[:, 0] + np.log(d.long), F[:, 1], F


def ewma(d, halflife):
    _, sig, _, _ = core.features(d.mkt.h, d.mkt.l, d.mkt.c, float(halflife), 14, d.vc.regime_len)
    return np.log(np.maximum(sig**2, FLOOR * d.long))


def garch(d, fit):
    """GARCH(1,1) with variance targeting fitted on `fit`; returns the h-step mean-variance forecast."""
    r = d.r
    var = r[fit].var()

    def nll(x):
        a, b = 1 / (1 + np.exp(-x))
        if a + b >= 0.9999:
            return 1e12
        _, ll = core.garch_filter(r[fit.start : fit.stop], (1 - a - b) * var, a, b, var, 0)
        return -ll

    x = minimize(nll, [-2.5, 2.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6)).x
    a, b = 1 / (1 + np.exp(-x))
    s2, _ = core.garch_filter(r, (1 - a - b) * var, a, b, var, 0)  # s2[t] = forecast for t+1
    p = a + b
    h = d.vc.h
    mean_mult = (1 - p**h) / (1 - p) / h  # average of p^(k-1), k = 1..h
    return np.log(np.maximum(var + (s2 - var) * mean_mult, FLOOR * d.long)), (a, b)


def har(d, fit):
    fl = FLOOR * d.long
    X = np.column_stack([np.ones(len(d.r))] + [np.log(np.maximum(_roll_mean(d.r2, L), fl)) for L in d.vc.har])
    rows = np.arange(len(d.r))[fit]
    ok = rows[np.isfinite(X[rows]).all(1) & np.isfinite(d.target[rows])]
    beta = np.linalg.lstsq(X[ok], d.target[ok], rcond=None)[0]
    return X @ beta


def har_rolling(d, window):
    """Log-HAR re-estimated at every bar by OLS on the most recent `window` rows whose targets are
    observed (rows j <= t - h - 1), following the fitting scheme of Chassot & Audrino (2026).
    Uses running sums of x x' and x y, so the cost is O(n) regardless of the window."""
    fl = FLOOR * d.long
    X = np.column_stack([np.ones(len(d.r))] + [np.log(np.maximum(_roll_mean(d.r2, L), fl)) for L in d.vc.har])
    y = d.target
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo = np.where(ok[:, None], X, 0.0)
    yo = np.where(ok, y, 0.0)
    cxx = np.cumsum(Xo[:, :, None] * Xo[:, None, :], axis=0)
    cxy = np.cumsum(Xo * yo[:, None], axis=0)
    cnt = np.cumsum(ok)
    n, h = len(y), d.vc.h
    out = np.full(n, np.nan)
    t = np.arange(n)
    hi = t - h - 1  # newest usable row
    lo = hi - window  # exclusive lower bound
    use = (hi >= 0) & np.isfinite(X).all(1)
    hi_, lo_ = hi[use], lo[use]
    sxx = cxx[hi_] - np.where(lo_[:, None, None] >= 0, cxx[np.maximum(lo_, 0)], 0.0)
    sxy = cxy[hi_] - np.where(lo_[:, None] >= 0, cxy[np.maximum(lo_, 0)], 0.0)
    m = cnt[hi_] - np.where(lo_ >= 0, cnt[np.maximum(lo_, 0)], 0)
    good = m >= max(50, X.shape[1] * 10)
    beta = np.full((len(hi_), X.shape[1]), np.nan)
    beta[good] = np.linalg.solve(sxx[good] + 1e-10 * np.eye(X.shape[1]), sxy[good][:, :, None])[:, :, 0]
    out[use] = np.einsum("ij,ij->i", X[use], beta)
    return out


def gbm(d, fit, seed=0):
    """Gradient-boosted trees on the union of HAR inputs, the volatility-shape embedding, the regime
    state and the long-run level (fixed, untuned hyperparameters)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    fl = FLOOR * d.long
    X = np.column_stack(
        [np.log(np.maximum(_roll_mean(d.r2, L), fl)) for L in d.vc.har]
        + [d.emb, d.state.astype(float), np.log(d.long)]
    )
    rows = np.arange(len(d.r))[fit]
    ok = rows[np.isfinite(X[rows]).all(1) & np.isfinite(d.target[rows])]
    if len(ok) > 60000:
        ok = np.sort(np.random.default_rng(seed).choice(ok, 60000, replace=False))
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=seed
    )
    model.fit(X[ok], d.target[ok])
    out = np.full(len(d.r), np.nan)
    good = np.isfinite(X).all(1)
    out[good] = model.predict(X[good])
    return out


def qlike(realised, log_f):
    ratio = realised / np.exp(log_f)
    return ratio - np.log(ratio) - 1


def walk_forward(d, bounds, halflives=(5, 10, 20, 50, 100, 200)):
    """OOS log-variance forecasts from every model, refitted per fold. Log models (HAR, analogue)
    are bias-corrected to the variance scale with the mean exp(residual) of the fit window."""
    n = len(d.r)
    names = ["EWMA", "GARCH", "HAR", "GBM", "Analogue", "Analogue+HAR"]
    out = {k: np.full(n, np.nan) for k in names}
    ana, _, _ = analogue(d)
    ew = {hl: ewma(d, hl) for hl in halflives}
    ok_r = d.realised > 0
    info = []
    for a, b in bounds:
        fit = d.fit_rows(a)
        v = np.zeros(n, bool)
        v[fit] = True
        v &= ok_r & np.isfinite(d.target)
        best_hl = min(halflives, key=lambda hl: np.nanmean(qlike(d.realised[v], ew[hl][v])))
        g, ab = garch(d, fit)
        hr = har(d, fit)
        f = {
            "EWMA": ew[best_hl],
            "GARCH": g,
            "HAR": hr,
            "GBM": gbm(d, fit),
            "Analogue": ana,
            "Analogue+HAR": (ana + hr) / 2,
        }
        for k in ("HAR", "GBM", "Analogue", "Analogue+HAR"):
            vv = v & np.isfinite(f[k])
            f[k] = f[k] + np.log(np.mean(np.exp(d.target[vv] - f[k][vv])))
        for k in names:
            out[k][a:b] = f[k][a:b]
        info.append(dict(fold=(a, b), ewma_halflife=best_hl, garch_alpha=ab[0], garch_beta=ab[1]))
    return out, info


# ---------------------------------------------------------------- tail events


def logistic_fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z)

    return minimize(nll, np.zeros(2), method="BFGS").x


def tail_forecasts(d, bounds, ew_log):
    """OOS P(raw jump within h bars) from: constant, static Poisson (v1), Hawkes, logistic on
    log EWMA variance, analogue."""
    n = len(d.r)
    names = ["Constant", "Poisson (v1)", "Hawkes", "Logistic-EWMA", "Analogue"]
    out = {k: np.full(n, np.nan) for k in names}
    pois = poisson_prob(d.jumps, d.vc.h)
    _, ana_p, _ = analogue(d)
    x = ew_log - np.log(d.long)
    for a, b in bounds:
        fit = d.fit_rows(a)
        y = d.tail[fit]
        ok = np.isfinite(y) & np.isfinite(x[fit])
        mu, al, be = fit_hawkes(d.jumps[d.vc.warmup : a])
        w = logistic_fit(x[fit][ok], y[ok])
        f = {
            "Constant": np.full(n, np.nanmean(y)),
            "Poisson (v1)": pois,
            "Hawkes": core.hawkes_prob(d.jumps, mu, al, be, float(d.vc.h)),
            "Logistic-EWMA": 1 / (1 + np.exp(-(w[0] + w[1] * x))),
            "Analogue": ana_p,
        }
        for k in names:
            out[k][a:b] = f[k][a:b]
    return {k: np.clip(v, 0.01, 0.99) for k, v in out.items()}


# ---------------------------------------------------------------- economic overlay


def overlay(d, log_f, a, cap=1.0, band=0.1, cost_bps=5.0, gate=None, kind="vol"):
    """Long exposure scaled by the variance forecast, decided at close t, earning bar t+1.
    kind='vol': w = sigma_target / sigma_hat;  kind='kelly': w = sigma_target^2 / sigma_hat^2
    (variance-managed, the Kelly-optimal scaling for a constant expected return).
    sigma_target is the asset's average per-bar variance before `a`. Weights move only when the
    target drifts more than `band` from the current weight. `gate` (bool array) forces w = 0."""
    tgt = np.nanmean(d.r2[d.vc.warmup : a])
    raw = tgt / np.exp(log_f)
    raw = np.sqrt(raw) if kind == "vol" else raw
    raw = np.clip(np.nan_to_num(raw), 0, cap)
    if gate is not None:
        raw = np.where(gate, 0.0, raw)
    w = np.zeros(len(raw))
    cur = 0.0
    for t in range(a, len(raw)):  # ponytail: python loop over bars; fine at 1e5 bars
        if abs(raw[t] - cur) > band:
            cur = raw[t]
        w[t] = cur
    turnover = np.abs(np.diff(np.r_[0.0, w]))
    # weight set at close t earns bar t+1; the rebalancing cost is paid in bar t+1 as well
    ret = np.r_[0.0, w[:-1] * np.diff(d.mkt.c) / d.mkt.c[:-1] - turnover[:-1] * cost_bps / 1e4]
    return ret, w


def buy_hold(d):
    return np.r_[0.0, np.diff(d.mkt.c) / d.mkt.c[:-1]]
