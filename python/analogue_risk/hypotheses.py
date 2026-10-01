"""Pre-registered confirmatory tests (see PREREGISTRATION.md): `python -m analogue_risk.hypotheses [results_dir]`.

Unit of analysis: one series (asset x horizon for volatility and tails; asset x alpha for VaR;
asset for direction and the overlay). Cross-series tests are two-sided Wilcoxon signed-rank tests
on per-series log loss ratios unless a one-sided alternative is stated. Holm's method controls the
family-wise error rate over the eight primary endpoints at 5%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from .experiments import ROOT


def wilcoxon(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return st.wilcoxon(x, alternative=alternative).pvalue if len(x) >= 5 else np.nan


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def run(res):
    e1s = pd.read_csv(res / "e1_direction_skill.csv")
    e1b = pd.read_csv(res / "e1_direction_backtests.csv")
    e2 = pd.read_csv(res / "e2_volatility.csv")
    e3 = pd.read_csv(res / "e3_ablation.csv")
    e4 = pd.read_csv(res / "e4_tail.csv")
    e5 = pd.read_csv(res / "e5_overlay.csv")
    e8 = pd.read_csv(res / "e8_var.csv")
    q = e2.pivot(index="series", columns="model", values="QLIKE")
    mcs = e2.pivot(index="series", columns="model", values="MCS p (QLIKE)")
    ll = e4.pivot(index="series", columns="model", values="log-loss")
    fz = e8[e8.alpha == 0.025].pivot(index="series", columns="model", values="FZ0")
    dd = e5.pivot(index="series", columns="strategy", values="max DD")
    sr = e5.pivot(index="series", columns="strategy", values="Sharpe")
    ic = e1s[e1s.embedding == "m=4"].set_index("series")["IC long"]
    dsr = e1b[e1b.strategy.str.contains("grid")].set_index("series")["DSR"]
    kern = e3[e3.series != "mean"].set_index("series")["Gaussian kernel (bw 0.6)"]

    H = []

    def add(hid, claim, stat, p, n, note=""):
        H.append(dict(id=hid, hypothesis=claim, statistic=stat, p=p, series=n, note=note))

    add("H1", "Analogue direction forecasts have zero mean IC across assets (null expected)",
        f"mean IC {ic.mean():+.4f}; max DSR {dsr.max():.2f}", st.ttest_1samp(ic, 0).pvalue, len(ic),
        "confirmed if p >= 0.05 and no DSR >= 0.95")
    r = np.log(q["Analogue+HAR"] / q["HAR"])
    add("H2", "Analogue+HAR QLIKE differs from HAR", f"median ratio {np.exp(r.median()):.3f}", wilcoxon(r), len(r),
        f"in 90% MCS: Analogue+HAR {int((mcs['Analogue+HAR'] >= .1).sum())}, HAR {int((mcs['HAR'] >= .1).sum())}")
    r = np.log(q["Analogue"] / q["HAR"])
    add("H3", "Analogue alone QLIKE differs from HAR", f"median ratio {np.exp(r.median()):.3f}", wilcoxon(r), len(r))
    r = np.log(kern)
    add("H4", "Gaussian-kernel analogue beats kNN analogue (one-sided)", f"median ratio {np.exp(r.median()):.3f}",
        wilcoxon(r, "less"), len(r))
    r = np.log(q["GBM"] / q["HAR"])
    add("H5", "Gradient boosting QLIKE differs from HAR", f"median ratio {np.exp(r.median()):.3f}", wilcoxon(r), len(r))
    r = np.log(ll["Hawkes"] / ll["Poisson (v1)"])
    add("H6", "Hawkes tail forecasts beat static Poisson (one-sided log-loss)",
        f"median ratio {np.exp(r.median()):.3f}", wilcoxon(r, "less"), len(r),
        f"Logistic-EWMA vs Hawkes p = {wilcoxon(np.log(ll['Logistic-EWMA'] / ll['Hawkes'])):.3g}")
    r = fz["Blend"] - fz["FHS"]  # FZ0 can be negative: use differences, not ratios
    add("H7", "Analogue-FHS blend FZ0 differs from FHS at alpha = 2.5%", f"median diff {r.median():+.4f}",
        wilcoxon(r), len(r), f"Analogue vs FHS p = {wilcoxon(fz['Analogue'] - fz['FHS']):.3g}; "
                             f"vs GARCH-t p = {wilcoxon(fz['Blend'] - fz['GARCH-t']):.3g}")
    better = (dd["Vol target: Analogue+HAR"] > dd["Buy & hold"]).sum()
    add("H8", "Volatility targeting (Analogue+HAR) reduces max drawdown vs buy & hold (one-sided sign test)",
        f"{better}/{len(dd)} improve", st.binomtest(int(better), len(dd), 0.5, alternative="greater").pvalue, len(dd),
        f"Sharpe difference p = {wilcoxon(sr['Vol target: Analogue+HAR'] - sr['Buy & hold']):.3g}")

    t = pd.DataFrame(H).set_index("id")
    t["Holm p"] = holm(t.p.fillna(1.0))
    t.to_csv(res / "hypotheses.csv")
    text = t.to_markdown(floatfmt=".4g")
    (res / "hypotheses.md").write_text("Pre-registered confirmatory tests (Holm-adjusted over H1-H8).\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "confirmatory")
