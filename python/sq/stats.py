"""Evaluation statistics: forecast comparison, performance metrics and bootstrap inference."""
import numpy as np
from scipy.stats import norm, rankdata


# ---------------------------------------------------------------- bootstrap

def stationary_indices(T, B, block, rng):
    """B x T indices of the Politis-Romano stationary bootstrap (geometric blocks, mean `block`)."""
    new = rng.random((B, T)) < 1.0 / block
    new[:, 0] = True
    starts = rng.integers(0, T, (B, T))
    t = np.arange(T)
    gstart = np.maximum.accumulate(np.where(new, t, 0), axis=1)          # time the current block began
    first = np.take_along_axis(starts, gstart, axis=1)                    # its random origin
    return (first + t - gstart) % T


def politis_white_block(x):
    """Optimal mean block length for the stationary bootstrap (Politis & White 2004, with the
    correction of Patton, Politis & White 2009), using the flat-top lag window."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    n = len(x)
    kn = max(5, int(np.ceil(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = np.ceil(min(3 * np.sqrt(n), n / 3))
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(mmax + 1)])
    rho = acov[1:] / acov[0]
    crit = 2 * np.sqrt(np.log10(n) / n)
    # smallest m whose next kn autocorrelations (lags m+1..m+kn) are all insignificant
    M = mmax
    for m in range(0, mmax - kn + 1):
        if np.all(np.abs(rho[m: m + kn]) < crit):
            M = min(2 * max(m, 1), mmax)
            break
    k = np.arange(-M, M + 1)
    lam = np.clip(2 * (1 - np.abs(k) / M), 0, 1)  # flat-top: 1 for |k|/M <= 0.5, linear to 0 at 1
    r = acov[np.abs(k)]
    g = np.sum(lam * np.abs(k) * r)
    d = 2 * np.sum(lam * r) ** 2
    b = (2 * g ** 2 / d) ** (1 / 3) * n ** (1 / 3) if d > 0 else 1.0
    return float(min(max(b, 1.0), bmax))


def boot_means(X, B=1000, block=20, seed=0, chunk=50):
    """Bootstrap distribution (B x M) of the column means of X (T x M)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    out = []
    for i in range(0, B, chunk):
        idx = stationary_indices(len(X), min(chunk, B - i), block, rng)
        out.append(X[idx].mean(axis=1))
    return np.vstack(out)


# ---------------------------------------------------------------- forecast comparison

def newey_west_var(d, lag):
    d = d - d.mean()
    T = len(d)
    v = d @ d / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / T
    return v / T


def diebold_mariano(loss_a, loss_b, lag):
    """DM test of equal predictive accuracy. Positive stat: model b has lower loss.
    Returns (stat, two-sided p-value)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    stat = d.mean() / np.sqrt(newey_west_var(d, lag))
    return stat, 2 * norm.sf(abs(stat))


def model_confidence_set(losses, B=1000, block=20, seed=0):
    """Hansen, Lunde & Nason (2011) MCS with the T_max statistic.
    losses: dict name -> loss array (same length). Returns dict name -> MCS p-value;
    a model is in the (1 - alpha) MCS iff its p-value >= alpha."""
    names = list(losses)
    L = np.column_stack([losses[k] for k in names])
    bm = boot_means(L, B, block, seed)                  # B x M
    mean = L.mean(0)
    alive = list(range(len(names)))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = mean[a] - mean[a].mean()                 # loss relative to the average of survivors
        dboot = bm[:, a] - bm[:, a].mean(1, keepdims=True)
        var = ((dboot - dbar) ** 2).mean(0)
        t = dbar / np.sqrt(var)
        tmax_boot = ((dboot - dbar) / np.sqrt(var)).max(1)
        p = (tmax_boot >= t.max()).mean()
        running = max(running, p)
        worst = a[np.argmax(t)]
        pvals[names[worst]] = running
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    return pvals


# ---------------------------------------------------------------- probability forecasts

def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y):
    return np.mean((p - y) ** 2)


def auc(p, y):
    y = y.astype(bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- performance

def sharpe(r, ppy):
    s = r.std(ddof=1)
    return r.mean() / s * np.sqrt(ppy) if s > 0 else 0.0


def max_drawdown(r):
    eq = np.cumprod(1 + r)
    return (eq / np.maximum.accumulate(eq) - 1).min()


def cvar(r, q=0.05):
    cut = np.quantile(r, q)
    return r[r <= cut].mean()


def summary(r, ppy, w=None):
    eq = np.cumprod(1 + r)
    years = len(r) / ppy
    out = dict(sharpe=sharpe(r, ppy), ann_ret=eq[-1] ** (1 / years) - 1, ann_vol=r.std() * np.sqrt(ppy),
               max_dd=max_drawdown(r), cvar5=cvar(r))
    out["calmar"] = out["ann_ret"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    if w is not None:
        out["avg_exposure"] = np.mean(np.abs(w))
        out["turnover_yr"] = np.abs(np.diff(w)).sum() / years
    return out


def psr(r, sr_star=0.0):
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012), per-period SR."""
    r = np.asarray(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    g4 = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    return norm.cdf((sr - sr_star) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))


def dsr(r, trial_srs):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): PSR against the expected maximum
    per-period SR of len(trial_srs) unskilled trials."""
    n = len(trial_srs)
    if n < 2:
        return psr(r)
    gamma = 0.5772156649
    sr0 = np.std(trial_srs, ddof=1) * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    return psr(r, sr0)


def sharpe_diff_test(ra, rb, ppy, B=1000, block=20, seed=0):
    """Paired stationary-bootstrap test of SR(a) - SR(b). Returns (difference, two-sided p, 95% CI)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([ra, rb])
    diffs = []
    for i in range(0, B, 50):
        idx = stationary_indices(len(X), min(50, B - i), block, rng)
        s = X[idx]
        m, sd = s.mean(1), s.std(1, ddof=1)
        diffs.append((m[:, 0] / sd[:, 0] - m[:, 1] / sd[:, 1]) * np.sqrt(ppy))
    diffs = np.concatenate(diffs)
    obs = sharpe(ra, ppy) - sharpe(rb, ppy)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, min(p, 1.0), tuple(np.quantile(diffs, [0.025, 0.975]))
