"""StochQuant v2 strategy, the v1 (Pine) baseline and simple benchmarks, on top of the Rust core."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from . import _core as core


@dataclass(frozen=True)
class Config:
    # analogue matcher
    w: int = 20  # pattern window (bars)
    m: int = 4  # segments in the window; m=1 is the v1 single-return feature
    normalize: bool = True  # scale segment returns by EWMA volatility
    lookback: int = 10**9  # candidate history (bars); default: all past bars
    min_hist: int = 4000  # bars of history before the first forecast
    k: int = 20  # analogues per regime
    k_min: int = 5
    excl: int = 20  # exclusion zone between analogues (bars); 0 disables
    regime: bool = True  # Markov regime mixture; False pools all analogues
    regime_len: int = 1000  # ATR moving average defining the regime
    vol_halflife: float = 50
    # trade rule (identical to v1 defaults)
    h: int = 20  # max holding period (bars)
    sl_mult: float = 2.5  # stop = sl_mult * ATR(14)
    rr: float = 1.5  # target = rr * stop
    # decision
    z: float = 1.0  # enter when E[R] - z*se > 0
    gate: str = "hawkes"  # hawkes | poisson | none
    jump_c: float = 2.5  # jump = |r_t| > c * sigma_{t-1}
    gate_q: float = 0.9  # suppress when P(jump) exceeds this quantile of the calibration period
    sizing: str = "kelly"  # kelly | fixed
    kelly_frac: float = 0.5
    max_risk: float = 0.02  # max fraction of equity risked per trade
    fixed_risk: float = 0.01
    lev_cap: float = 1.0
    cost_bps: float = 5.0  # per side

    @property
    def warmup(self):
        return max(self.min_hist, self.regime_len) + self.w + self.h + 50

    def __post_init__(self):
        check_exclusion(self.k, self.regime, self.excl, min(self.lookback, self.min_hist))


DAILY = dict(min_hist=2000, k=10, regime_len=500, vol_halflife=20)


def check_exclusion(k, regime, excl, history):
    """The exclusion zone must not bind: if (groups * k) analogues spaced `excl` apart fill more than
    a quarter of the candidate history, the selected 'neighbours' degenerate into a near-uniform
    sample of the history instead of close matches."""
    need = (2 if regime else 1) * k * max(excl, 1)
    if need > history / 4:
        raise ValueError(
            f"{need} bars needed for {k} analogues/regime at spacing {excl}, "
            f"but only {history} bars of history; lengthen history or reduce k/excl"
        )


class Market:
    """One price series plus cached, causal intermediate results."""

    def __init__(self, df):
        self.df = df
        self.o, self.h, self.l, self.c = (
            np.ascontiguousarray(df[k].to_numpy(float)) for k in ("open", "high", "low", "close")
        )
        self.n = len(df)
        years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
        self.ppy = self.n / years  # bars per year, empirical
        self._cache = {}

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def features(self, cfg):
        return self._memo(
            ("f", cfg.vol_halflife, cfg.regime_len),
            lambda: core.features(self.h, self.l, self.c, cfg.vol_halflife, 14, cfg.regime_len),
        )

    def forecast(self, cfg):
        key = (
            "F",
            cfg.w,
            cfg.m,
            cfg.normalize,
            cfg.lookback,
            cfg.k,
            cfg.k_min,
            cfg.excl,
            cfg.regime,
            cfg.regime_len,
            cfg.vol_halflife,
            cfg.h,
            cfg.sl_mult,
            cfg.rr,
            cfg.cost_bps,
        )

        def run():
            r, sig, atr, state = self.features(cfg)
            rl, rs, up = self.labels(cfg)
            emb = core.embed(r, sig, cfg.w, cfg.m, cfg.normalize)
            y = np.column_stack([rl, rs, up, rl * rl, rs * rs])
            return core.forecast(
                emb,
                state,
                y,
                cfg.w,
                cfg.h + 1,
                cfg.h,
                cfg.lookback,
                cfg.k,
                cfg.k_min,
                cfg.excl,
                cfg.regime,
                cfg.warmup,
            )

        return self._memo(key, run)

    def labels(self, cfg):
        _, _, atr, _ = self.features(cfg)
        return self._memo(
            ("L", cfg.sl_mult, cfg.rr, cfg.h, cfg.cost_bps, cfg.vol_halflife, cfg.regime_len),
            lambda: core.labels(
                self.o, self.h, self.l, self.c, atr, cfg.sl_mult, cfg.rr, cfg.h, cfg.cost_bps / 1e4
            ),
        )

    def jumps(self, cfg):
        r, sig, _, _ = self.features(cfg)
        prev = np.r_[np.nan, sig[:-1]]
        return np.abs(r) > cfg.jump_c * prev  # NaN -> False

    def backtest(self, dir_, stop, size, cfg, **kw):
        args = (
            dict(
                rr=cfg.rr,
                max_hold=cfg.h,
                cost=cfg.cost_bps / 1e4,
                protect_entry=True,
                close_on_opposite=False,
                risk_sizing=True,
                lev_cap=cfg.lev_cap,
            )
            | kw
        )
        return core.backtest(
            self.o,
            self.h,
            self.l,
            self.c,
            np.ascontiguousarray(dir_, np.int8),
            np.ascontiguousarray(stop, float),
            np.ascontiguousarray(size, float),
            **args,
        )


# ---------------------------------------------------------------- tail-risk gate


def fit_hawkes(events):
    """MLE of (mu, alpha, beta) for the exponential Hawkes process on a boolean event series."""
    times = np.flatnonzero(events).astype(float)
    T = float(len(events))
    if len(times) < 10:
        return len(times) / T, 0.0, 1.0

    def nll(x):
        mu, alpha, beta = np.exp(x[0]), 1 / (1 + np.exp(-x[1])), np.exp(x[2])
        return -core.hawkes_loglik(times, T, mu, alpha, beta)

    x0 = [np.log(0.5 * len(times) / T), 0.0, np.log(0.1)]
    best = minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    x = best.x
    return float(np.exp(x[0])), float(1 / (1 + np.exp(-x[1]))), float(np.exp(x[2]))


def poisson_prob(events, h, window=1000):
    """v1-style static Poisson: rolling jump rate over `window` bars, P(>=1 jump in h bars)."""
    rate = np.convolve(events.astype(float), np.ones(window), "full")[: len(events)] / window
    p = 1 - np.exp(-rate * h)
    p[: window - 1] = np.nan
    return p


def jump_prob(mkt, cfg, fit_end, kind=None):
    """Causal P(jump within h bars) for every bar, with parameters fitted on [warmup, fit_end)."""
    ev = mkt.jumps(cfg)
    kind = kind or cfg.gate
    if kind == "hawkes":
        mu, a, b = fit_hawkes(ev[cfg.warmup : fit_end])
        return core.hawkes_prob(ev, mu, a, b, float(cfg.h))
    if kind == "poisson":
        return poisson_prob(ev, cfg.h)
    raise ValueError(kind)


def gate_allow(mkt, cfg, bounds):
    """Walk-forward gate: for each fold [a, b) refit on data before a and threshold at the
    calibration quantile. Bars before the first fold use the first fold's fit."""
    allow = np.ones(mkt.n, bool)
    if cfg.gate == "none":
        return allow
    for i, (a, b) in enumerate(bounds):
        p = jump_prob(mkt, cfg, a)
        thr = np.nanquantile(p[cfg.warmup : a], cfg.gate_q)
        lo = 0 if i == 0 else a
        allow[lo:b] = ~(p[lo:b] > thr)
    return allow


# ---------------------------------------------------------------- strategies

# forecast columns for the direction model
EL, ES, PUP, M2L, M2S, SEL, SES = 0, 1, 2, 3, 4, 5, 6


def decide(F, cfg, allow, atr):
    eL, eS, m2L, m2S, seL, seS = F[:, EL], F[:, ES], F[:, M2L], F[:, M2S], F[:, SEL], F[:, SES]
    lcbL, lcbS = eL - cfg.z * seL, eS - cfg.z * seS
    long = (lcbL > 0) & (lcbL >= lcbS) & allow
    short = (lcbS > 0) & (lcbS > lcbL) & allow
    d = np.where(long, 1, np.where(short, -1, 0))
    if cfg.sizing == "kelly":
        e, m2 = np.where(long, eL, eS), np.where(long, m2L, m2S)
        with np.errstate(invalid="ignore", divide="ignore"):
            size = np.clip(np.nan_to_num(cfg.kelly_frac * e / m2), 0, cfg.max_risk)
    else:
        size = np.full(len(d), cfg.fixed_risk)
    return d, cfg.sl_mult * atr, size


def run_v2(mkt, cfg, allow):
    F = mkt.forecast(cfg)
    _, _, atr, _ = mkt.features(cfg)
    d, stop, size = decide(F, cfg, allow, atr)
    eq, trades = mkt.backtest(d, stop, size, cfg)
    return dict(eq=eq, trades=trades, dir=d, size=size, stop=stop, F=F)


def run_v1(mkt, cost_bps):
    d, stop, size, adj, pj = core.legacy_v1(mkt.h, mkt.l, mkt.c)
    eq, trades = core.backtest(
        mkt.o,
        mkt.h,
        mkt.l,
        mkt.c,
        d,
        stop,
        size,
        rr=1.5,
        max_hold=21,
        cost=cost_bps / 1e4,
        protect_entry=False,
        close_on_opposite=True,
        risk_sizing=False,
        lev_cap=1.0,
    )
    return dict(eq=eq, trades=trades, dir=d, adj=adj, jump_prob=pj)


def run_momentum(mkt, cfg):
    """Time-series momentum with the same exits and fixed 1% risk."""
    _, _, atr, _ = mkt.features(cfg)
    d = np.sign(mkt.c - np.r_[np.full(cfg.w, np.nan), mkt.c[: -cfg.w]])
    d = np.nan_to_num(d).astype(np.int8)
    d[: cfg.warmup] = 0
    eq, trades = mkt.backtest(d, cfg.sl_mult * atr, np.full(mkt.n, cfg.fixed_risk), cfg)
    return dict(eq=eq, trades=trades)


def buy_hold(mkt):
    return dict(eq=mkt.c / mkt.c[0], trades=None)


def returns(eq):
    return np.r_[0.0, eq[1:] / eq[:-1] - 1]


def folds(n, start, k=5):
    edges = np.linspace(start, n, k + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def walk_forward(mkt, base, grid, allow, bounds, score):
    """Per fold, pick the grid config with the best in-sample `score` on [warmup, fold start),
    then take its returns over the fold. Returns (OOS per-bar returns, chosen configs, all runs)."""
    runs = {g: run_v2(mkt, replace(base, **dict(g)), allow) for g in grid}
    rets = {g: returns(r["eq"]) for g, r in runs.items()}
    out = np.zeros(mkt.n)
    chosen = []
    for a, b in bounds:
        best = max(grid, key=lambda g: score(rets[g][base.warmup : a]))
        chosen.append(best)
        out[a:b] = rets[best][a:b]
    return out, chosen, runs
