"""One-day Value-at-Risk and Expected Shortfall forecasts, walk-forward.

All models except historical simulation share the same GARCH(1,1) volatility forecast, so the
comparison isolates the *shape* of the conditional return distribution:
  HS       - empirical quantile of the last 250 returns
  GARCH-N  - Gaussian innovations
  GARCH-t  - standardised Student-t innovations, nu by MLE
  FHS      - empirical distribution of all past standardised residuals (Barone-Adesi et al. 1999)
  Analogue - weighted empirical distribution of the standardised residuals that followed the
             k most similar past volatility states (regime-conditioned kNN, this paper)
Returns are daily log returns; VaR and ES are reported as (negative) return quantiles.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats as st
from scipy.optimize import minimize_scalar

from . import _core as core
from .vol import VolConfig, VolData, garch

ALPHAS = (0.01, 0.025, 0.05)


@dataclass(frozen=True)
class VarConfig:
    seg: int = 5             # days per segment of the volatility-shape embedding
    m: int = 4
    k: int = 250             # analogues per regime (tail quantiles need hundreds)
    regime: bool = True
    hs_window: int = 250
    min_hist: int = 2000

    def vol(self):
        # reuse the volatility-shape embedding; h = seg sets the segment length, the target is next-day
        return VolConfig(h=self.seg, m=self.m, k=self.k, excl=1, regime=self.regime, min_hist=self.min_hist,
                         regime_len=500, long_halflife=250, har=(1, 5, 22))


def weighted_tail(vals, w, alpha):
    """Row-wise weighted alpha-quantile and expected shortfall (mean below the quantile)."""
    vals = np.where(w > 0, vals, np.inf)
    order = np.argsort(vals, axis=1)
    v = np.take_along_axis(vals, order, 1)
    ww = np.take_along_axis(w, order, 1)
    cw = np.cumsum(ww, 1)
    k = np.argmax(cw >= alpha * cw[:, -1:], axis=1)
    q = v[np.arange(len(v)), k]
    below = (np.arange(v.shape[1])[None, :] <= k[:, None]) & np.isfinite(v)
    es = np.where(below, np.where(np.isfinite(v), v, 0.0) * ww, 0).sum(1) / np.where(below, ww, 0).sum(1)
    empty = cw[:, -1] <= 0
    q[empty], es[empty] = np.nan, np.nan
    return q, es


def _t_nu(z):
    z = z[np.isfinite(z)]

    def nll(nu):
        s = np.sqrt((nu - 2) / nu)
        return -np.sum(st.t.logpdf(z / s, nu) - np.log(s))

    return minimize_scalar(nll, bounds=(2.1, 60), method="bounded").x


def forecasts(d, cfg, bounds):
    """d: VolData built with cfg.vol(). Returns {model: {alpha: (VaR, ES)}} for r[t+1] made at t."""
    n = len(d.r)
    r = d.r
    r_next = np.r_[r[1:], np.nan]
    out = {m: {a: (np.full(n, np.nan), np.full(n, np.nan)) for a in ALPHAS}
           for m in ("HS", "GARCH-N", "GARCH-t", "FHS", "Analogue")}

    # analogue sets depend only on the embedding, not on the fold's GARCH fit
    vc = cfg.vol()
    y = np.ascontiguousarray(np.column_stack([r_next]))  # finiteness mask for anchors
    idx, w = core.neighbours(d.emb, d.state, y, vc.m * vc.seg, 1, vc.seg, vc.lookback, vc.k, vc.k_min,
                             1, vc.regime, vc.warmup)

    # historical simulation (no fitting)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(r, cfg.hs_window)                       # row i covers r[i : i+W]
    hs = {}
    for a in ALPHAS:
        q = np.quantile(win, a, axis=1)
        es = np.array([row[row <= qq].mean() for row, qq in zip(win, q)])
        v, e = np.full(n, np.nan), np.full(n, np.nan)
        v[cfg.hs_window - 1:], e[cfg.hs_window - 1:] = q, es
        hs[a] = (v, e)

    for a_fold, b_fold in bounds:
        fit = d.fit_rows(a_fold)
        _, (ga, gb) = garch(d, fit)
        var = r[fit].var()
        s2_next, _ = core.garch_filter(r, (1 - ga - gb) * var, ga, gb, var, 0)   # forecast for t+1
        sig_next = np.sqrt(s2_next)
        sig_now = np.r_[np.sqrt(var), sig_next[:-1]]                            # forecast made at t-1 for t
        z = r / sig_now                                                         # standardised residual at t
        z_next = np.r_[z[1:], np.nan]
        nu = _t_nu(z[fit])
        sl = slice(a_fold, b_fold)
        for a in ALPHAS:
            qn = st.norm.ppf(a)
            out["GARCH-N"][a][0][sl] = sig_next[sl] * qn
            out["GARCH-N"][a][1][sl] = -sig_next[sl] * st.norm.pdf(qn) / a
            s = np.sqrt((nu - 2) / nu)
            qt = st.t.ppf(a, nu)
            es_t = -st.t.pdf(qt, nu) / a * (nu + qt ** 2) / (nu - 1)
            out["GARCH-t"][a][0][sl] = sig_next[sl] * s * qt
            out["GARCH-t"][a][1][sl] = sig_next[sl] * s * es_t
            out["HS"][a][0][sl], out["HS"][a][1][sl] = hs[a][0][sl], hs[a][1][sl]
            # FHS: all standardised residuals observed up to t (expanding)
            zz = z[vc.warmup // 2: b_fold]
            for t in range(a_fold, b_fold):  # ponytail: O(n * window) quantiles; fine for daily data
                past = zz[: t - vc.warmup // 2 + 1]
                past = past[np.isfinite(past)]
                qf = np.quantile(past, a)
                out["FHS"][a][0][t] = sig_next[t] * qf
                out["FHS"][a][1][t] = sig_next[t] * past[past <= qf].mean()
            # Analogue: residuals that followed the analogue states
            vals = np.where(idx[sl] >= 0, z_next[np.maximum(idx[sl], 0)], np.nan)
            ww = np.where(np.isfinite(vals), w[sl], 0.0)
            qa, ea = weighted_tail(vals, ww, a)
            out["Analogue"][a][0][sl] = sig_next[sl] * qa
            out["Analogue"][a][1][sl] = sig_next[sl] * ea
    return out


# ---------------------------------------------------------------- backtests

def kupiec(hits, alpha):
    n, x = len(hits), hits.sum()
    ph = x / n
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - ph) + x * np.log(ph) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return lr, st.chi2.sf(lr, 1)


def christoffersen(hits, alpha):
    """Independence and conditional-coverage LR tests. Returns (p_ind, p_cc)."""
    h0, h1 = hits[:-1], hits[1:]
    n00, n01 = np.sum(~h0 & ~h1), np.sum(~h0 & h1)
    n10, n11 = np.sum(h0 & ~h1), np.sum(h0 & h1)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(q, a, b):
        return (a * np.log(1 - q) if a else 0) + (b * np.log(q) if b else 0)

    lr_ind = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    lr_uc, _ = kupiec(hits, alpha)
    return st.chi2.sf(lr_ind, 1), st.chi2.sf(lr_uc + lr_ind, 2)


def tick_loss(y, v, alpha):
    return (alpha - (y <= v)) * (y - v)


def fz0_loss(y, v, e, alpha):
    """Fissler-Ziegel FZ0 loss (Patton, Ziegel & Chen 2019) for (VaR, ES) with v, e < 0."""
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1
