"""Reproduces every table in the paper: `python -m analogue_risk.experiments [--quick]`.

Protocol (fixed before any out-of-sample evaluation):
  * each series is split into a calibration period (first 30%, at least warmup + 500 bars) and an
    out-of-sample (OOS) period divided into 5 walk-forward folds;
  * every fitted quantity (EWMA half-life, GARCH, HAR, bias corrections, Hawkes, logistic gate,
    grid selection) uses only data observed before the fold it is applied to;
  * all results below are computed on the OOS period only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import _core as core
from . import data
from . import model as M
from . import stats as S
from . import var as R
from . import vol as V

_REPO = Path(__file__).resolve().parents[2]
# results/ of the source checkout; an installed package writes to ./results
ROOT = _REPO / "results" if (_REPO / "pyproject.toml").exists() else Path.cwd() / "results"
RES = ROOT  # set by main(): results/ for development, results/confirmatory/ for the pre-registered run
AUTO_BLOCK = False  # --auto-block: Politis-White block lengths instead of the fixed defaults


def block_len(default, *series):
    """Bootstrap mean block length: the fixed default, or with --auto-block the mean Politis-White
    estimate over the given series (loss differentials or return differences)."""
    if not AUTO_BLOCK:
        return default
    return max(1, int(round(np.mean([S.politis_white_block(x) for x in series]))))


warnings.filterwarnings("ignore", category=RuntimeWarning)

UNIVERSES = {
    # development set: every design decision was made on these
    "development": dict(
        hourly={"BTC": "BTCUSDT", "ETH": "ETHUSDT", "PAXG": "PAXGUSDT"},
        daily={
            "S&P 500": "^GSPC",
            "Nasdaq-100": "QQQ",
            "Gold": "GLD",
            "Treasuries": "TLT",
            "EUR/USD": "EURUSD=X",
            "NIFTY 50": "^NSEI",
        },
    ),
    # confirmatory set: fixed in PREREGISTRATION.md before any of it was downloaded
    "confirmatory": dict(
        hourly={"SOL": "SOLUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT", "ADA": "ADAUSDT", "LTC": "LTCUSDT"},
        daily={
            "DAX": "^GDAXI",
            "Nikkei 225": "^N225",
            "FTSE 100": "^FTSE",
            "Hang Seng": "^HSI",
            "Russell 2000": "IWM",
            "Emerging mkts": "EEM",
            "Silver": "SLV",
            "Oil": "USO",
            "GBP/USD": "GBPUSD=X",
            "USD/JPY": "JPY=X",
        },
    ),
    # robustness: the distribution-paying funds above, with dividend-adjusted prices
    "adjusted_etfs": dict(
        hourly={},
        daily={"Nasdaq-100": "QQQ", "Treasuries": "TLT", "Russell 2000": "IWM", "Emerging mkts": "EEM"},
        adjusted=True,
    ),
}
MIN_BARS = {True: 20000, False: 3000}  # pre-registered exclusion rule (after cleaning)


def load(universe="development", quick=False):
    """name -> (DataFrame, is_hourly). Series shorter than MIN_BARS are excluded (and logged)."""
    u = UNIVERSES[universe]
    out = {k: (data.binance(v, "1h"), True) for k, v in u["hourly"].items()}
    for k, v in u["daily"].items():
        out[k] = (
            data.yahoo(
                v, start="1927-12-30" if v == "^GSPC" else "1950-01-01", adjusted=u.get("adjusted", False)
            ),
            False,
        )
    for k, (df, hourly) in list(out.items()):
        if len(df) < MIN_BARS[hourly]:
            print(f"  excluded {k}: {len(df)} bars < {MIN_BARS[hourly]}", flush=True)
            del out[k]
    if quick:
        out = dict(list(out.items())[:1] + [x for x in out.items() if not x[1][1]][:1])
    return out


def split(n, warmup):
    cal = max(int(0.3 * n), warmup + 500)
    return cal, M.folds(n, cal)


def md(df, path, floatfmt=".3f", note=""):
    RES.mkdir(exist_ok=True)
    df.to_csv(path.with_suffix(".csv"))
    text = df.to_markdown(floatfmt=floatfmt)
    path.with_suffix(".md").write_text((note + "\n\n" if note else "") + text + "\n")
    print(f"\n## {path.stem}\n{note}\n{text}", flush=True)


# ---------------------------------------------------------------- E1: direction (motivating study)


def e1_direction(sets):
    from scipy.stats import spearmanr

    skill, perf = [], []
    for name, (df, hourly) in sets.items():
        mkt = M.Market(df)
        base = M.Config() if hourly else M.Config(**M.DAILY)
        grid = [
            tuple(sorted(dict(k=k, m=m, z=z).items()))
            for k, m, z in product((base.k // 2, base.k), (1, 2, 4), (0.5, 1.0, 1.5))
        ]
        cal, bounds = split(mkt.n, base.warmup)
        oos = slice(cal, mkt.n)
        for m in (1, 4):
            c = replace(base, m=m, cost_bps=0)
            F = mkt.forecast(c)
            rl, rs, up = mkt.labels(c)
            ok = np.zeros(mkt.n, bool)
            ok[oos] = True
            ok &= np.isfinite(F[:, 0]) & np.isfinite(rl)
            bss = 1 - S.brier(F[ok, M.PUP], up[ok]) / S.brier(
                np.full(ok.sum(), up[base.warmup : cal][np.isfinite(up[base.warmup : cal])].mean()), up[ok]
            )
            skill.append(
                {
                    "series": name,
                    "embedding": f"m={m}",
                    "IC long": spearmanr(F[ok, M.EL], rl[ok])[0],
                    "IC short": spearmanr(F[ok, M.ES], rs[ok])[0],
                    "Brier skill": bss,
                }
            )
        rows = {}
        rows["Buy & hold"] = M.returns(M.buy_hold(mkt)["eq"]), None
        for bps in (1, 5):
            v1 = M.run_v1(mkt, bps)
            rows[f"v1 (Pine), {bps} bp"] = M.returns(v1["eq"]), v1["trades"]
        cfg = replace(base, cost_bps=5)
        allow = M.gate_allow(mkt, cfg, bounds)
        v2 = M.run_v2(mkt, cfg, allow)
        rows["v2 defaults, 5 bp"] = M.returns(v2["eq"]), v2["trades"]
        mo = M.run_momentum(mkt, cfg)
        rows["Momentum, 5 bp"] = M.returns(mo["eq"]), mo["trades"]
        wf, chosen, runs = M.walk_forward(mkt, cfg, grid, allow, bounds, lambda r: S.sharpe(r, mkt.ppy))
        rows["v2 walk-forward grid, 5 bp"] = wf, None
        trial_srs = [M.returns(r["eq"])[oos].mean() / M.returns(r["eq"])[oos].std() for r in runs.values()]
        for label, (r, tr) in rows.items():
            s = S.summary(r[oos], mkt.ppy)
            n_tr = int(np.sum(tr["entry_bar"] >= cal)) if tr is not None else np.nan
            perf.append(
                {
                    "series": name,
                    "strategy": label,
                    "Sharpe": s["sharpe"],
                    "ann. return": s["ann_ret"],
                    "max DD": s["max_dd"],
                    "trades": n_tr,
                    "DSR": S.dsr(r[oos], trial_srs) if "grid" in label else np.nan,
                }
            )
        print(f"  E1 {name} done", flush=True)
    md(
        pd.DataFrame(skill).set_index(["series", "embedding"]),
        RES / "e1_direction_skill",
        note="Out-of-sample directional skill of the analogue forecaster. IC: Spearman correlation of forecast "
        "and realised trade R-multiple. Brier skill: vs. calibration-period base rate (>0 = better).",
    )
    md(
        pd.DataFrame(perf).set_index(["series", "strategy"]),
        RES / "e1_direction_backtests",
        note="Out-of-sample backtests. DSR: deflated Sharpe ratio over the 18-configuration grid "
        "(probability that true Sharpe > 0 after the multiple-testing correction).",
    )


# ---------------------------------------------------------------- E2-E5: risk


def vol_series(sets):
    """(label, VolData, cal, bounds, h) for every series x horizon."""
    out = []
    for name, (df, hourly) in sets.items():
        mkt = M.Market(df)
        for vc in [V.HOURLY] if hourly else [V.DAILY_W, V.DAILY_M]:
            d = V.VolData(mkt, vc)
            cal, bounds = split(mkt.n, vc.warmup)
            out.append((f"{name} (h={vc.h})", d, cal, bounds))
    return out


def valid_rows(d, cal):
    ok = np.zeros(len(d.r), bool)
    ok[cal:] = True
    return ok & (d.realised > 0) & np.isfinite(d.target)


def e2_volatility(series, B):
    rows, tests, forecasts = [], [], {}
    for label, d, cal, bounds in series:
        F, info = V.walk_forward(d, bounds)
        ok = valid_rows(d, cal)
        for f in F.values():
            ok &= np.isfinite(f)
        q = {k: V.qlike(d.realised[ok], f[ok]) for k, f in F.items()}
        mse = {k: (d.target[ok] - f[ok]) ** 2 for k, f in F.items()}
        avg = np.mean(list(q.values()), axis=0)
        lag, block = d.vc.h, block_len(2 * d.vc.h, *[v - avg for v in q.values()])
        mcs = S.model_confidence_set(q, B=B, block=block)
        for k in F:
            rows.append(
                {
                    "series": label,
                    "model": k,
                    "QLIKE": q[k].mean(),
                    "QLIKE / HAR": q[k].mean() / q["HAR"].mean(),
                    "log-MSE": mse[k].mean(),
                    "MCS p (QLIKE)": mcs[k],
                }
            )
        for k in ("Analogue", "Analogue+HAR"):
            for ref in ("EWMA", "GARCH", "HAR"):
                stat, p = S.diebold_mariano(q[ref], q[k], lag)
                tests.append({"series": label, "model": k, "vs": ref, "DM stat": stat, "p": p})
        forecasts[label] = (F, ok)
        print(f"  E2 {label}: n_oos={ok.sum()} folds={[i['ewma_halflife'] for i in info]}", flush=True)
    t = pd.DataFrame(rows).set_index(["series", "model"])
    md(
        t,
        RES / "e2_volatility",
        note="Out-of-sample volatility forecasts (target: mean squared return over the "
        "next h bars). MCS p >= 0.10 means the model is in the 90% model confidence set.",
    )
    md(
        pd.DataFrame(tests).set_index(["series", "model", "vs"]),
        RES / "e2_dm_tests",
        note="Diebold-Mariano tests on QLIKE with Newey-West variance (lag = h). Positive stat: the analogue "
        "model has lower loss than the reference.",
    )
    return forecasts


def e3_ablation(series):
    variants = {
        "full": {},
        "m=1 (level only)": dict(m=1),
        "m=2": dict(m=2),
        "m=8": dict(m=8),
        "no regime mixture": dict(regime=False),
        "no exclusion zone": dict(excl=0),
        "Gaussian kernel (bw 0.6)": dict(bandwidth=0.6),
        "history capped at 5000": dict(lookback=5000),
    }
    rows = []
    for label, d, cal, bounds in series:
        ok = valid_rows(d, cal)
        fit = d.fit_rows(cal)
        res = {}
        for vn, ov in variants.items():
            try:
                vc = replace(d.vc, **ov)
            except ValueError:
                continue
            dd = d if ov.get("m", d.vc.m) == d.vc.m else V.VolData(d.mkt, vc)
            f, _, _ = V.analogue(dd, vc)
            v = np.isfinite(f[fit]) & np.isfinite(dd.target[fit])
            f = f + np.log(np.mean(np.exp(dd.target[fit][v] - f[fit][v])))
            o = ok & np.isfinite(f)
            res[vn] = V.qlike(d.realised[o], f[o]).mean()
        rows.append({"series": label} | {k: v / res["full"] for k, v in res.items()})
        print(f"  E3 {label} done", flush=True)
    t = pd.DataFrame(rows).set_index("series")
    t.loc["mean"] = t.mean()
    md(
        t,
        RES / "e3_ablation",
        note="Analogue ablations: OOS QLIKE relative to the full model (<1 = variant better). "
        "Bias correction fitted on the calibration period.",
    )


def e4_tail(series):
    rows = []
    for label, d, cal, bounds in series:
        ew = V.ewma(d, 20)
        P = V.tail_forecasts(d, bounds, ew)
        ok = np.zeros(len(d.r), bool)
        ok[cal:] = True
        ok &= np.isfinite(d.tail)
        for p in P.values():
            ok &= np.isfinite(p)
        y = d.tail[ok]
        for k, p in P.items():
            rows.append(
                {
                    "series": label,
                    "model": k,
                    "base rate": y.mean(),
                    "log-loss": S.log_loss(p[ok], y),
                    "Brier": S.brier(p[ok], y),
                    "AUC": S.auc(p[ok], y),
                }
            )
        print(f"  E4 {label} done", flush=True)
    md(
        pd.DataFrame(rows).set_index(["series", "model"]),
        RES / "e4_tail",
        note="Out-of-sample probability of at least one raw jump (|r| > 3 long-run sigma) in the next h bars.",
    )


def e5_overlay(series, forecasts, B):
    rows = []
    for label, d, cal, bounds in series:
        if label.endswith("(h=22)"):
            continue  # one horizon per series for the trading overlay: hourly h=24, daily h=5
        F, _ = forecasts[label]
        oos = slice(cal, len(d.r))
        bh = V.buy_hold(d)
        # gate: suppress exposure when the Hawkes P(jump) exceeds its calibration-period quantile
        mu, al, be = M.fit_hawkes(d.jumps[d.vc.warmup : cal])
        ph = core.hawkes_prob(d.jumps, mu, al, be, float(d.vc.h))
        gate = ph > np.nanquantile(ph[d.vc.warmup : cal], d.vc.gate_q)
        strategies = {"Buy & hold": (bh, np.ones(len(bh)))}
        for k in ("EWMA", "GARCH", "HAR", "Analogue", "Analogue+HAR"):
            strategies[f"Vol target: {k}"] = V.overlay(d, F[k], cal)
        strategies["Kelly (1/var): Analogue+HAR"] = V.overlay(d, F["Analogue+HAR"], cal, kind="kelly")
        strategies["Vol target: Analogue+HAR + Hawkes gate"] = V.overlay(d, F["Analogue+HAR"], cal, gate=gate)
        for k, (r, w) in strategies.items():
            block = block_len(2 * d.vc.h, r[oos] - bh[oos])
            s = S.summary(r[oos], d.mkt.ppy, w[oos])
            diff, p, _ = (
                S.sharpe_diff_test(r[oos], bh[oos], d.mkt.ppy, B=B, block=block)
                if k != "Buy & hold"
                else (np.nan, np.nan, None)
            )
            rows.append(
                {
                    "series": label.split(" (")[0],
                    "strategy": k,
                    "Sharpe": s["sharpe"],
                    "ΔSharpe vs B&H": diff,
                    "p": p,
                    "ann. vol": s["ann_vol"],
                    "max DD": s["max_dd"],
                    "CVaR 5%": s["cvar5"],
                    "avg exposure": s.get("avg_exposure", 1.0),
                    "turnover / yr": s.get("turnover_yr", 0.0),
                }
            )
        print(f"  E5 {label} done", flush=True)
    md(
        pd.DataFrame(rows).set_index(["series", "strategy"]),
        RES / "e5_overlay",
        note="Out-of-sample volatility-managed long exposure (no leverage, 5 bp per unit turnover, 0.1 rebalance "
        "band). p: paired stationary-bootstrap test of the Sharpe difference.",
    )


# ---------------------------------------------------------------- E8: VaR / ES


def var_config(n):
    """Pre-registered rule: shorter daily series use fewer analogues and a shorter warm-up."""
    return R.VarConfig() if n >= 6000 else R.VarConfig(k=125, min_hist=1000)


def e8_var(sets, B):
    rows, mcs_rows = [], []
    for name, (df, hourly) in sets.items():
        daily = data.resample(df, "1D") if hourly else df
        mkt = M.Market(daily)
        cfg = var_config(mkt.n)
        vc = cfg.vol()
        d = V.VolData(mkt, vc)
        cal, bounds = split(mkt.n, vc.warmup)
        out = R.forecasts(d, cfg, bounds)
        out["Blend"] = {
            a: tuple(0.5 * out["Analogue"][a][i] + 0.5 * out["FHS"][a][i] for i in (0, 1)) for a in R.ALPHAS
        }
        y = np.r_[d.r[1:], np.nan]
        for a in R.ALPHAS:
            ok = np.zeros(mkt.n, bool)
            ok[cal:] = True
            ok &= np.isfinite(y)
            for f in out.values():
                ok &= np.isfinite(f[a][0]) & np.isfinite(f[a][1]) & (f[a][1] < 0)
            losses = {}
            for m_, f in out.items():
                v, e = f[a][0][ok], f[a][1][ok]
                hits = y[ok] <= v
                losses[m_] = R.fz0_loss(y[ok], v, e, a)
                p_ind, p_cc = R.christoffersen(hits, a)
                rows.append(
                    {
                        "series": name,
                        "alpha": a,
                        "model": m_,
                        "n": int(ok.sum()),
                        "hit rate / alpha": hits.mean() / a,
                        "Kupiec p": R.kupiec(hits, a)[1],
                        "Christoffersen CC p": p_cc,
                        "tick loss": R.tick_loss(y[ok], v, a).mean(),
                        "FZ0": losses[m_].mean(),
                    }
                )
            avg = np.mean(list(losses.values()), axis=0)
            mcs = S.model_confidence_set(
                losses, B=B, block=block_len(10, *[v - avg for v in losses.values()])
            )
            for m_ in losses:
                mcs_rows.append({"series": name, "alpha": a, "model": m_, "MCS p (FZ0)": mcs[m_]})
        print(f"  E8 {name} done", flush=True)
    t = pd.DataFrame(rows).merge(pd.DataFrame(mcs_rows), on=["series", "alpha", "model"])
    md(
        t.set_index(["series", "alpha", "model"]),
        RES / "e8_var",
        note="Out-of-sample one-day VaR/ES. hit rate / alpha = 1 is perfect coverage; Kupiec and Christoffersen "
        "p < 0.05 reject correct coverage; FZ0 is the Fissler-Ziegel joint loss (lower is better).",
    )


# ---------------------------------------------------------------- robustness: adjusted prices


def compare_adjusted():
    """Side-by-side volatility and overlay results for unadjusted vs dividend-adjusted funds."""
    adj = ROOT / "adjusted_etfs"
    names = list(UNIVERSES["adjusted_etfs"]["daily"])
    base = {
        k: pd.concat([pd.read_csv(ROOT / f"{k}.csv"), pd.read_csv(ROOT / "confirmatory" / f"{k}.csv")])
        for k in ("e2_volatility", "e5_overlay")
    }
    new = {k: pd.read_csv(adj / f"{k}.csv") for k in base}
    rows = []
    for k_series in sorted(new["e2_volatility"].series.unique()):
        if k_series.split(" (")[0] not in names:
            continue
        for model in ("HAR", "Analogue", "Analogue+HAR"):
            pick = lambda df: df[(df.series == k_series) & (df.model == model)].iloc[0]
            b, a = pick(base["e2_volatility"]), pick(new["e2_volatility"])
            rows.append(
                {
                    "series": k_series,
                    "result": f"QLIKE / HAR: {model}",
                    "unadjusted": b["QLIKE / HAR"],
                    "adjusted": a["QLIKE / HAR"],
                }
            )
    for name in names:
        for strat in ("Buy & hold", "Vol target: Analogue+HAR"):
            pick = lambda df: df[(df.series == name) & (df.strategy == strat)].iloc[0]
            b, a = pick(base["e5_overlay"]), pick(new["e5_overlay"])
            for col in ("Sharpe", "max DD"):
                rows.append(
                    {"series": name, "result": f"{strat}: {col}", "unadjusted": b[col], "adjusted": a[col]}
                )
    t = pd.DataFrame(rows)
    t["change"] = t.adjusted - t.unadjusted
    global RES
    RES = adj
    md(
        t.set_index(["series", "result"]),
        adj / "comparison",
        note="Robustness: distribution-paying funds with dividend-adjusted vs unadjusted prices "
        "(Yahoo price indices have no adjusted series and are unaffected).",
    )


# ---------------------------------------------------------------- robustness: block lengths


def compare_blocks():
    """How many model-confidence-set memberships and Sharpe-test conclusions change when the fixed
    block lengths are replaced by Politis-White estimates."""
    rows = []
    for label, base in (("Development", ROOT), ("Confirmatory", ROOT / "confirmatory")):
        auto = base / "auto_block"
        if not auto.exists():
            continue
        for name, key, col, cut in (
            ("Volatility MCS (e2)", ["series", "model"], "MCS p (QLIKE)", 0.10),
            ("VaR/ES MCS (e8)", ["series", "alpha", "model"], "MCS p (FZ0)", 0.10),
            ("Overlay Sharpe test (e5)", ["series", "strategy"], "p", 0.05),
        ):
            f = {"e2": "e2_volatility", "e8": "e8_var", "e5": "e5_overlay"}[name.split("(")[1][:2]]
            m = pd.read_csv(base / f"{f}.csv").merge(
                pd.read_csv(auto / f"{f}.csv"), on=key, suffixes=("", "_auto")
            )
            m = m[m[col].notna() & m[col + "_auto"].notna()]
            if "MCS" in name:
                fixed, pw = m[col] >= cut, m[col + "_auto"] >= cut
                verdict = "in 90% MCS"
            else:
                fixed, pw = m[col] < cut, m[col + "_auto"] < cut
                verdict = "significant at 5%"
            rows.append(
                {
                    "set": label,
                    "test": name,
                    "cases": len(m),
                    f"{verdict}: fixed blocks": int(fixed.sum()),
                    f"{verdict}: Politis-White": int(pw.sum()),
                    "verdicts changed": int((fixed != pw).sum()),
                }
            )
    t = pd.DataFrame(rows)
    global RES
    RES = ROOT
    md(
        t.set_index(["set", "test"]),
        ROOT / "block_length_comparison",
        note="Robustness: fixed bootstrap block lengths (2h for volatility and overlay, 10 days for VaR) "
        "vs Politis-White (2004) automatic block lengths.",
    )


# ---------------------------------------------------------------- E6: timeframe claim, E7: speed


def e6_timeframe():
    from scipy.stats import spearmanr

    base = data.binance("BTCUSDT", "15m")
    rows = []
    for tf in ("15min", "1h", "4h"):
        df = base if tf == "15min" else data.resample(base, tf)
        mkt = M.Market(df)
        cfg = M.Config(
            m=1,
            cost_bps=0,
            min_hist=4000 if tf != "4h" else 2000,
            k=20 if tf != "4h" else 10,
            regime_len=1000 if tf != "4h" else 500,
        )
        cal, _ = split(mkt.n, cfg.warmup)
        F = mkt.forecast(cfg)
        rl, rs, up = mkt.labels(cfg)
        ok = np.zeros(mkt.n, bool)
        ok[cal:] = True
        ok &= np.isfinite(F[:, 0]) & np.isfinite(rl)
        v1 = M.run_v1(mkt, 1)
        r = M.returns(v1["eq"])[cal:]
        rows.append(
            {
                "timeframe": tf,
                "bars": mkt.n,
                "IC long (v1 feature)": spearmanr(F[ok, 0], rl[ok])[0],
                "IC short": spearmanr(F[ok, 1], rs[ok])[0],
                "v1 Sharpe (1 bp)": S.sharpe(r, mkt.ppy),
                "v1 trades": int(np.sum(v1["trades"]["entry_bar"] >= cal)),
            }
        )
        print(f"  E6 {tf} done", flush=True)
    md(
        pd.DataFrame(rows).set_index("timeframe"),
        RES / "e6_timeframe",
        note="Does aggregating BTC to higher timeframes create directional predictability (the v1 'low-pass "
        "filter' claim)? OOS, 1D analogue feature.",
    )


SPEED_SNIPPET = """
import sys, time, json; sys.path.insert(0, {py!r})
import numpy as np
from dataclasses import replace
from analogue_risk import data, model as M, vol as V
df = data.binance("BTCUSDT", "15m")
out = []
for n in {sizes}:
    d = V.VolData(M.Market(df.iloc[:n]), replace(V.HOURLY, lookback={lookback}))
    t = time.perf_counter(); V.analogue(d, bandwidth={bw}); out.append((n, time.perf_counter() - t))
print(json.dumps(out))
"""


def e7_speed():
    rows = []
    py = str(Path(__file__).resolve().parents[1])
    for mode, bw, lookback in (
        ("kNN, history 5000", 0.0, 5000),
        ("kNN, full history", 0.0, 10**9),
        ("kernel, full history", 0.25, 10**9),
    ):
        for threads in (1, os.cpu_count()):
            code = SPEED_SNIPPET.format(
                py=py, sizes=[25_000, 50_000, 100_000, 200_000], lookback=lookback, bw=bw
            )
            env = os.environ | {"RAYON_NUM_THREADS": str(threads)}
            res = json.loads(
                subprocess.run(
                    [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
                )
                .stdout.strip()
                .splitlines()[-1]
            )
            for n, sec in res:
                rows.append(
                    {"mode": mode, "threads": threads, "bars": n, "seconds": sec, "bars / s": n / sec}
                )
        print(f"  E7 {mode} done", flush=True)
    md(
        pd.DataFrame(rows).set_index(["mode", "threads", "bars"]),
        RES / "e7_speed",
        floatfmt=".2f",
        note=f"Analogue forecast wall time on {os.cpu_count()} cores (includes all O(n^2) distance work).",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two series, fewer bootstrap draws")
    ap.add_argument("--only", nargs="*", default=None, help="subset of e1..e8")
    ap.add_argument("--universe", default="development", choices=list(UNIVERSES))
    ap.add_argument("--auto-block", action="store_true", help="Politis-White bootstrap block lengths")
    a = ap.parse_args()
    global RES, AUTO_BLOCK
    RES = ROOT if a.universe == "development" else ROOT / a.universe
    if a.auto_block:
        AUTO_BLOCK = True
        RES = RES / "auto_block"
    RES.mkdir(parents=True, exist_ok=True)
    B = 200 if a.quick else 1000
    want = lambda e: a.only is None or e in a.only
    t0 = time.time()
    sets = load(a.universe, a.quick)
    (RES / "config.json").write_text(
        json.dumps(
            {
                "hourly": asdict(V.HOURLY),
                "daily_w": asdict(V.DAILY_W),
                "daily_m": asdict(V.DAILY_M),
                "direction": asdict(M.Config()),
            },
            indent=1,
            default=str,
        )
    )
    if want("e1"):
        e1_direction(sets)
    if any(want(e) for e in ("e2", "e3", "e4", "e5")):
        series = vol_series(sets)
        forecasts = e2_volatility(series, B) if (want("e2") or want("e5")) else None
        if want("e3"):
            e3_ablation(series)
        if want("e4"):
            e4_tail(series)
        if want("e5"):
            e5_overlay(series, forecasts, B)
    if want("e8"):
        e8_var(sets, B)
    if a.universe == "adjusted_etfs" and want("e5"):
        compare_adjusted()
    if want("e6") and not a.quick and a.universe == "development":
        e6_timeframe()
    if want("e7") and not a.quick and a.universe == "development":
        e7_speed()
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
