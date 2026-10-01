"""Correctness tests: causality (no look-ahead), likelihoods against brute force, label/backtest
consistency, planted-signal recovery and the evaluation statistics."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sq import _core as core  # noqa: E402
from sq import model as M, stats as S, vol as V  # noqa: E402


def synthetic(n=6000, seed=0, drift_block=0, vol_regimes=True):
    rng = np.random.default_rng(seed)
    sig = np.where((np.arange(n) // 700) % 2 == 1, 0.012, 0.006) if vol_regimes else np.full(n, 0.008)
    r = rng.normal(0, 1, n) * sig
    if drift_block:
        r += np.repeat(rng.choice([-1, 1], n // drift_block + 1), drift_block)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=1.0),
                        index=pd.date_range("2000", periods=n, freq="h"))


SMALL_DIR = M.Config(min_hist=1800, k=10, regime_len=300, cost_bps=0)
SMALL_VOL = replace(V.HOURLY, min_hist=4000, regime_len=300, har=(1, 24, 168), k=10)


@pytest.mark.parametrize("vc", [SMALL_VOL, replace(SMALL_VOL, bandwidth=0.3), replace(SMALL_VOL, regime=False)])
def test_vol_forecast_is_causal(vc):
    df = synthetic()
    t0 = 4500
    full, _, _ = V.analogue(V.VolData(M.Market(df), vc))
    cut, _, _ = V.analogue(V.VolData(M.Market(df.iloc[:t0]), vc))
    np.testing.assert_array_equal(full[:t0], cut)  # identical, NaNs included


def test_direction_forecast_and_labels_are_causal():
    df = synthetic()
    t0 = 4500
    a, b = M.Market(df), M.Market(df.iloc[:t0])
    np.testing.assert_array_equal(a.forecast(SMALL_DIR)[:t0], b.forecast(SMALL_DIR))
    for x, y in zip(a.labels(SMALL_DIR), b.labels(SMALL_DIR)):
        ok = np.isfinite(y)
        np.testing.assert_array_equal(x[:t0][ok], y[ok])


def test_legacy_v1_is_causal():
    df = synthetic(4000)
    t0 = 3000
    a = core.legacy_v1(*(df[k].to_numpy() for k in ("high", "low", "close")))
    b = core.legacy_v1(*(df[k].to_numpy()[:t0] for k in ("high", "low", "close")))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x[:t0], y)


def test_label_equals_executed_trade():
    """The R-multiple a label assigns to anchor j is exactly what the backtester realises
    for a single signal at j (risk sizing, costs included)."""
    df = synthetic()
    mkt = M.Market(df)
    cfg = replace(SMALL_DIR, cost_bps=5)
    rl, rs, _ = mkt.labels(cfg)
    _, _, atr, _ = mkt.features(cfg)
    for j, d, lab in [(1200, 1, rl), (2500, -1, rs), (4100, 1, rl)]:
        sig = np.zeros(mkt.n, np.int8)
        sig[j] = d
        _, tr = mkt.backtest(sig, cfg.sl_mult * atr, np.full(mkt.n, 0.01), cfg)
        assert tr["r"][0] == pytest.approx(lab[j], rel=1e-12)


def test_hawkes_loglik_matches_brute_force():
    rng = np.random.default_rng(1)
    times = np.sort(rng.choice(2000, 60, replace=False)).astype(float)
    T, mu, a, b = 2000.0, 0.02, 0.4, 0.15
    lam = [mu + sum(a * b * np.exp(-b * (t - s)) for s in times if s < t) for t in times]
    brute = np.sum(np.log(lam)) - mu * T - a * np.sum(1 - np.exp(-b * (T - times)))
    assert core.hawkes_loglik(times, T, mu, a, b) == pytest.approx(brute, rel=1e-10)


def test_hawkes_recovers_parameters():
    rng = np.random.default_rng(2)
    mu, a, b, T = 0.01, 0.5, 0.2, 200000
    ev = np.zeros(T, bool)  # discrete-time simulation by thinning at unit steps
    s = 0.0
    for t in range(T):
        ev[t] = rng.random() < 1 - np.exp(-(mu + a * b * s))
        s = s * np.exp(-b) + ev[t]
    m_hat, a_hat, _ = M.fit_hawkes(ev)
    assert m_hat == pytest.approx(mu, rel=0.25) and a_hat == pytest.approx(a, abs=0.12)


def test_garch_filter_matches_python():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 500)
    om, al, be, s0 = 1e-6, 0.08, 0.9, 1e-4
    s2, ll = core.garch_filter(r, om, al, be, s0, 0)
    s, lp, out = s0, 0.0, []
    for x in r:
        lp -= 0.5 * (np.log(2 * np.pi) + np.log(s) + x * x / s)
        s = om + al * x * x + be * s
        out.append(s)
    np.testing.assert_allclose(s2, out, rtol=1e-12)
    assert ll == pytest.approx(lp, rel=1e-12)


def test_planted_signal_is_found_and_random_walk_is_not():
    from scipy.stats import spearmanr

    def ic(df):
        mkt = M.Market(df)
        F = mkt.forecast(SMALL_DIR)
        rl, _, _ = mkt.labels(SMALL_DIR)
        ok = np.isfinite(F[:, 0]) & np.isfinite(rl)
        return spearmanr(F[ok, 0], rl[ok])[0]

    assert ic(synthetic(8000, drift_block=300)) > 0.15
    assert abs(ic(synthetic(8000))) < 0.05


def test_vol_forecast_tracks_planted_regimes():
    d = V.VolData(M.Market(synthetic(8000)), SMALL_VOL)
    f, _, _ = V.analogue(d)
    ok = np.isfinite(f) & np.isfinite(d.target)
    assert np.corrcoef(f[ok], d.target[ok])[0, 1] > 0.5


def test_binding_exclusion_zone_is_rejected():
    with pytest.raises(ValueError):
        M.Config(min_hist=800)  # 2 regimes x 20 analogues x 20 bars = 800 > 800/4


def test_mcs_keeps_best_and_drops_worst():
    rng = np.random.default_rng(4)
    base = rng.normal(1, 0.5, 3000)
    losses = {"good": base, "same": base + rng.normal(0, 0.05, 3000), "bad": base + 0.3}
    p = S.model_confidence_set(losses, B=300, block=5)
    assert p["good"] > 0.1 and p["same"] > 0.1 and p["bad"] < 0.05


def test_stationary_bootstrap_indices():
    idx = S.stationary_indices(1000, 20, 10, np.random.default_rng(5))
    assert idx.shape == (20, 1000) and idx.min() >= 0 and idx.max() < 1000
    runs = (np.diff(idx, axis=1) == 1).mean()
    assert 0.85 < runs < 0.95  # continuation probability 1 - 1/block


def test_neighbours_reproduce_forecast_means():
    """The index/weight sets from neighbours() give exactly the means that forecast() reports."""
    d = V.VolData(M.Market(synthetic()), SMALL_VOL)
    vc = SMALL_VOL
    y = np.ascontiguousarray(np.column_stack([d.target - np.log(d.long), d.tail]))
    args = (d.emb, d.state, y, vc.m * vc.seg, vc.h, vc.h, vc.lookback, vc.k, vc.k_min, vc.h, vc.regime, vc.warmup)
    F = core.forecast(*args)
    idx, w = core.neighbours(*args)
    vals = np.where(idx >= 0, y[np.maximum(idx, 0), 0], 0.0)
    ok = np.isfinite(F[:, 0])
    np.testing.assert_allclose((w * vals).sum(1)[ok], F[ok, 0], rtol=1e-10, atol=1e-12)
    assert np.allclose(w[ok].sum(1), 1.0)


def test_pinned_snapshot_truncates_and_detects_changes(tmp_path, monkeypatch):
    from sq import data as D
    df = synthetic(300)
    pinned = D._clean(df.iloc[:250])
    monkeypatch.setattr(D, "DATA", tmp_path)
    monkeypatch.setattr(D, "MANIFEST", {"x": {"end": pinned.index[-1].isoformat(), "rows": len(pinned),
                                              "sha256": D.digest(pinned)}})
    df.to_csv(tmp_path / "x.csv.gz")  # the source now has 50 more bars than the snapshot
    out = D._cached("x", None)
    assert len(out) == 250 and D.digest(out) == D.digest(pinned)
    changed = df.copy()
    changed.iloc[10, changed.columns.get_loc("close")] *= 1.01  # a revised historical bar
    changed.to_csv(tmp_path / "x.csv.gz")
    with pytest.warns(UserWarning, match="differs from the pinned snapshot"):
        D._cached("x", None)


def test_klines_timestamps_ms_and_us():
    from sq import data as D
    rows = [[1551398400000, "1", "2", "0.5", "1.5", "10"],          # 2019, milliseconds
            [1740787200000000, "1", "2", "0.5", "1.5", "10"]]       # 2025, microseconds (archive)
    df = D._klines_frame(rows)
    assert list(df.index) == [pd.Timestamp("2019-03-01"), pd.Timestamp("2025-03-01")]


def test_binance_falls_back_to_archive_when_geo_blocked(tmp_path, monkeypatch):
    import urllib.error
    from sq import data as D
    monkeypatch.setattr(D, "DATA", tmp_path)

    def blocked(*a):
        raise urllib.error.HTTPError("u", 451, "restricted location", None, None)

    archive = D._klines_frame([[1551398400000 + 3600000 * i, "1", "2", "0.5", "1.5", "10"] for i in range(5)])
    monkeypatch.setattr(D, "_binance_api", blocked)
    monkeypatch.setattr(D, "_binance_archive", lambda *a: archive)
    with pytest.warns(UserWarning, match="HTTP 451"):
        out = D.binance("TESTUSDT", "1h")
    assert len(out) == 4  # unpinned series drop the still-forming final bar
