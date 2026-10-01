"""Paper figures: `python -m analogue_risk.figures`. Writes PDF + PNG to results/figures/."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from . import model as M
from . import vol as V
from .experiments import RES, split  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = RES / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def label_end(ax, x, y, text, color):
    ax.annotate(
        text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=8
    )
    ax.plot([x[-1]], [y[-1]], "o", ms=3.5, color=color)


def nearest_analogues(d, t, k=10):
    """The analogues the kNN forecaster uses at bar t (pooled over regimes, for display)."""
    excl, hi = d.vc.h, t - d.vc.h
    x = d.emb[t]
    cand = np.arange(d.vc.warmup // 2, hi + 1)
    cand = cand[np.isfinite(d.emb[cand]).all(1) & np.isfinite(d.target[cand])]
    order = cand[np.argsort(((d.emb[cand] - x) ** 2).sum(1), kind="stable")]
    picked = []
    for j in order:
        if all(abs(j - p) >= excl for p in picked):
            picked.append(j)
        if len(picked) == k:
            break
    return picked


def fig_explain(d, when):
    """Interpretability: the current volatility shape, its nearest historical analogues and what
    happened after them, next to what actually happened."""
    t = int(np.searchsorted(d.mkt.df.index, pd.Timestamp(when)))
    js = nearest_analogues(d, t)
    m, h = d.vc.m, d.vc.h
    xs_past = np.arange(-m + 1, 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, j in enumerate(js):
        ax.plot(
            np.r_[xs_past, 1],
            np.r_[d.emb[j], d.target[j] - np.log(d.long[j])],
            color=INK2,
            alpha=0.35,
            lw=0.9,
            label="10 nearest historical analogues" if i == 0 else None,
        )
    now = d.emb[t]
    fut = d.target[t] - np.log(d.long[t])
    f, _, _ = V.analogue(d)
    pred = f[t] - np.log(d.long[t])
    ax.plot(xs_past, now, color=BLUE, lw=2, marker="o", ms=4, label="current volatility shape")
    ax.plot([0, 1], [now[-1], pred], color=BLUE, lw=2, ls="--", label="analogue forecast")
    ax.plot([0, 1], [now[-1], fut], color=ORANGE, lw=2, marker="o", ms=4, label="realised")
    ax.set_xticks(np.r_[xs_past, 1])
    ax.set_xticklabels([f"t{q * h:+d}" if q else "t" for q in xs_past] + [f"next {h}"])
    ax.set_ylabel("log variance vs. long-run level")
    ax.set_title(
        f"{d.mkt.df.index[t]:%Y-%m-%d %H:%M} — why the model expects what it expects",
        fontsize=9,
        loc="left",
        color=INK,
    )
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "fig_analogue_explanation")


def fig_cum_loss(series_list):
    fig, axes = plt.subplots(1, len(series_list), figsize=(6.8, 2.6))
    for ax, (label, d, cal, bounds) in zip(np.atleast_1d(axes), series_list):
        F, _ = V.walk_forward(d, bounds)
        ok = np.zeros(len(d.r), bool)
        ok[cal:] = True
        ok &= (d.realised > 0) & np.isfinite(d.target)
        for f in F.values():
            ok &= np.isfinite(f)
        idx = d.mkt.df.index[ok]
        base = V.qlike(d.realised[ok], F["HAR"][ok])
        for k, col in (("Analogue+HAR", BLUE), ("Analogue", ORANGE), ("GARCH", AQUA)):
            c = np.cumsum(base - V.qlike(d.realised[ok], F[k][ok]))
            ax.plot(idx, c, color=col, lw=1.3, label=k)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        ax.tick_params(axis="x", labelrotation=30)
    np.atleast_1d(axes)[0].set_ylabel("cumulative QLIKE gain over HAR")
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_cumulative_loss")


def fig_overlay(series_list):
    fig, axes = plt.subplots(1, len(series_list), figsize=(6.8, 2.6))
    for ax, (label, d, cal, bounds) in zip(np.atleast_1d(axes), series_list):
        F, _ = V.walk_forward(d, bounds)
        oos = slice(cal, len(d.r))
        idx = d.mkt.df.index[oos]
        runs = {
            "Buy & hold": (V.buy_hold(d), INK2, "-", 1.0),
            "Vol target: HAR": (V.overlay(d, F["HAR"], cal)[0], INK, "--", 0.9),
            "Vol target: Analogue+HAR": (V.overlay(d, F["Analogue+HAR"], cal)[0], BLUE, "-", 1.3),
        }
        for k, (r, col, ls, lw) in runs.items():
            eq = np.cumprod(1 + r[oos])
            ax.plot(idx, eq, color=col, ls=ls, lw=lw, label=k)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        ax.tick_params(axis="x", labelrotation=30)
    np.atleast_1d(axes)[0].set_ylabel("growth of 1 (log scale)")
    fig.tight_layout()
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_overlay_equity")


def fig_reliability(d, cal, bounds):
    P = V.tail_forecasts(d, bounds, V.ewma(d, 20))
    ok = np.zeros(len(d.r), bool)
    ok[cal:] = True
    ok &= np.isfinite(d.tail)
    y = d.tail[ok]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    for k, col in (("Hawkes", BLUE), ("Poisson (v1)", ORANGE), ("Analogue", AQUA)):
        p = P[k][ok]
        bins = np.quantile(p, np.linspace(0, 1, 11))
        b = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xm = [p[b == i].mean() for i in range(10) if (b == i).any()]
        ym = [y[b == i].mean() for i in range(10) if (b == i).any()]
        ax.plot(xm, ym, color=col, marker="o", ms=3.5, lw=1.3, label=k)
    ax.set_xlabel("forecast P(jump within 24 bars)")
    ax.set_ylabel("observed frequency")
    ax.set_title("BTC — tail-risk calibration (OOS)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7.5)
    save(fig, "fig_tail_reliability")


def fig_hypotheses():
    """Per-series QLIKE ratios for the three volatility hypotheses, development vs confirmatory."""
    sets = {
        "development": RES / "e2_volatility.csv",
        "confirmatory": RES / "confirmatory" / "e2_volatility.csv",
    }
    abl = {"development": RES / "e3_ablation.csv", "confirmatory": RES / "confirmatory" / "e3_ablation.csv"}
    panels = [
        ("H2: Analogue+HAR vs HAR", lambda q, a: q["Analogue+HAR"] / q["HAR"]),
        ("H3: Analogue vs HAR", lambda q, a: q["Analogue"] / q["HAR"]),
        ("H4: kernel vs kNN analogue", lambda q, a: a),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    for ax, (title, f) in zip(axes, panels):
        for i, (name, col) in enumerate((("development", INK2), ("confirmatory", BLUE))):
            q = pd.read_csv(sets[name]).pivot(index="series", columns="model", values="QLIKE")
            a = pd.read_csv(abl[name]).set_index("series")["Gaussian kernel (bw 0.6)"].drop("mean")
            r = np.asarray(f(q, a.reindex(q.index)), float)
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(r))
            ax.scatter(np.log(r), i + jitter, s=12, color=col, alpha=0.75, edgecolors="none")
            ax.plot([np.median(np.log(r))] * 2, [i - 0.3, i + 0.3], color=col, lw=2)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("log QLIKE ratio (<0 better)")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["development", "confirmatory"])
    fig.tight_layout()
    save(fig, "fig_hypotheses")


def main():
    btc = M.Market(data.binance("BTCUSDT", "1h"))
    spx = M.Market(data.yahoo("^GSPC", start="1927-12-30"))
    series = []
    for label, mkt, vc in (("BTC (hourly, h=24)", btc, V.HOURLY), ("S&P 500 (daily, h=5)", spx, V.DAILY_W)):
        d = V.VolData(mkt, vc)
        cal, bounds = split(mkt.n, vc.warmup)
        series.append((label, d, cal, bounds))
    fig_explain(series[0][1], "2022-11-08 18:00")
    fig_cum_loss(series)
    fig_overlay(series)
    fig_reliability(*[series[0][i] for i in (1, 2, 3)])
    fig_hypotheses()


if __name__ == "__main__":
    main()
