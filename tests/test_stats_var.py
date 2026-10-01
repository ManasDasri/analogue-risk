"""Known-answer tests for the evaluation statistics, VaR/ES backtests, the overlay and the
pre-registered hypothesis script."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sq import hypotheses as Hy, stats as S, var as R  # noqa: E402


# ---------------------------------------------------------------- VaR / ES backtests

def test_kupiec_known_values():
    hits = np.zeros(1000, bool)
    hits[:10] = True                                   # exactly alpha * n violations
    lr, p = R.kupiec(hits, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    hits[:30] = True                                   # 3% violations vs 1%
    lr, p = R.kupiec(hits, 0.01)
    n, x = 1000, 30
    expect = -2 * ((n - x) * np.log(0.99) + x * np.log(0.01) - (n - x) * np.log(0.97) - x * np.log(0.03))
    assert lr == pytest.approx(expect) and p < 1e-5


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    iid = rng.random(5000) < 0.05
    clustered = np.zeros(5000, bool)
    for s in rng.choice(4900, 50, replace=False):
        clustered[s:s + 5] = True                      # same rate, violations in runs
    assert R.christoffersen(iid, 0.05)[0] > 0.01
    assert R.christoffersen(clustered, clustered.mean())[0] < 1e-6


def test_tick_and_fz0_losses_are_minimised_by_the_truth():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(200_000)
    a = 0.025
    q, es = st.norm.ppf(a), -st.norm.pdf(st.norm.ppf(a)) / a
    for shift in (-0.3, 0.3):                          # moving away from the true (VaR, ES) costs loss
        assert R.tick_loss(y, q, a).mean() < R.tick_loss(y, q + shift, a).mean()
        assert R.fz0_loss(y, q, es, a).mean() < R.fz0_loss(y, q + shift, es + shift, a).mean()


def test_weighted_tail_matches_unweighted_definition():
    v = np.array([[5.0, 1.0, 3.0, 2.0, 4.0, np.nan]])
    w = np.array([[1, 1, 1, 1, 1, 0]], float) / 5
    q, es = R.weighted_tail(v, w, 0.4)                 # 40% of weight: values {1, 2}
    assert q[0] == 2.0 and es[0] == pytest.approx(1.5)
    w2 = np.array([[0.1, 0.5, 0.1, 0.1, 0.2, 0.0]])    # value 1.0 alone carries 50% of weight
    q2, es2 = R.weighted_tail(v, w2, 0.4)
    assert q2[0] == 1.0 and es2[0] == 1.0


# ---------------------------------------------------------------- forecast comparison / performance

def test_diebold_mariano_sign_and_size():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(5000)
    stat, p = S.diebold_mariano(e ** 2 + 0.5, e ** 2, lag=1)  # model b uniformly better
    assert stat > 0 and p < 1e-10
    _, p0 = S.diebold_mariano(rng.standard_normal(5000) ** 2, rng.standard_normal(5000) ** 2, lag=1)
    assert p0 > 0.001


def test_newey_west_reduces_to_variance_of_mean_without_lags():
    x = np.random.default_rng(3).standard_normal(1000)
    assert S.newey_west_var(x, 0) == pytest.approx(x.var() / len(x))


def test_sharpe_drawdown_cvar_known_values():
    r = np.array([0.1, -0.05, 0.1, -0.05])
    assert S.sharpe(r, 1) == pytest.approx(r.mean() / r.std(ddof=1))
    assert S.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert S.cvar(np.arange(1, 101, dtype=float) / 100, 0.05) == pytest.approx(0.03)


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(4)
    good = rng.normal(0.1, 1, 5000)
    assert S.psr(good) > 0.99
    assert S.dsr(good, rng.normal(0, 0.05, 100)) < S.psr(good)  # deflation can only lower it


def test_holm_adjustment():
    adj = Hy.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


# ---------------------------------------------------------------- regression: published tables

@pytest.mark.parametrize("res", ["results", "results/confirmatory"])
def test_hypotheses_reproduce_committed_tables(res, tmp_path):
    src = ROOT / res
    committed = pd.read_csv(src / "hypotheses.csv")
    for f in src.glob("e*.csv"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    Hy.run(tmp_path)
    fresh = pd.read_csv(tmp_path / "hypotheses.csv")
    pd.testing.assert_frame_equal(fresh, committed)


# ---------------------------------------------------------------- overlay

def test_overlay_weights_returns_and_costs():
    from sq import model as M, vol as V
    from test_core import synthetic
    d = V.VolData(M.Market(synthetic(12000)), replace_cfg())
    a = 10000
    flat = np.log(np.full(len(d.r), np.nanmean(d.r2[d.vc.warmup:a])))  # forecast = target level
    r, w = V.overlay(d, flat, a, cost_bps=0.0)
    assert np.allclose(w[a:], 1.0) and np.allclose(w[:a], 0.0)            # full exposure, capped at 1
    c = d.mkt.c
    assert np.allclose(r[a + 1:], np.diff(c)[a:] / c[a:-1])               # earns the asset's return
    r_cost, _ = V.overlay(d, flat, a, cost_bps=10.0)
    assert r_cost[a + 1] == pytest.approx(r[a + 1] - 10e-4)               # one unit of turnover at entry
    high = flat + np.log(4.0)                                             # variance 4x target
    _, w2 = V.overlay(d, high, a)
    assert np.allclose(w2[a:], 0.5)                                       # vol target halves exposure


def replace_cfg():
    from dataclasses import replace
    from sq import vol as V
    return replace(V.HOURLY, min_hist=4000, regime_len=300, k=10)


def test_rust_tail_quantiles_match_numpy():
    from sq import _core as core
    rng = np.random.default_rng(5)
    z = rng.standard_t(4, 3000)
    z[[100, 2000]] = np.nan
    for a in (0.01, 0.025, 0.05):
        q, es = core.expanding_tail(z, 50, a)
        for t in (60, 999, 2999):
            past = z[50:t + 1]
            past = past[np.isfinite(past)]
            assert q[t] == np.quantile(past, a)                       # bit-identical quantile
            assert es[t] == pytest.approx(past[past <= q[t]].mean(), rel=1e-12)
        qr, er = core.rolling_tail(np.nan_to_num(z), 250, a)
        for t in (249, 1234, 2999):
            win = np.nan_to_num(z)[t - 249:t + 1]
            assert qr[t] == np.quantile(win, a)
            assert er[t] == pytest.approx(win[win <= qr[t]].mean(), rel=1e-12)
