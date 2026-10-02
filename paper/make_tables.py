"""Generate every LaTeX table and number macro in the research paper from results/*.csv, so no
figure in the manuscript is typed by hand. Run from the repository root:
    python paper/make_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = {"Development": ROOT / "results", "Confirmatory": ROOT / "results" / "confirmatory"}
OUT = ROOT / "paper" / "research" / "tables"
MACROS = {}


def read(name):
    return {k: pd.read_csv(v / f"{name}.csv") for k, v in RES.items() if (v / f"{name}.csv").exists()}


def esc(s):
    return (
        str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("^", r"\textasciicircum{}")
    )


def fmt(x, d=3):
    return "--" if not np.isfinite(x) else f"{x:.{d}f}"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def macro(name, value):
    MACROS[name] = value


def panel_rows(frames, render, ncols):
    lines = []
    for label, df in frames.items():
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{label} set}}}}\\")
        lines += render(df)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines)


# ---------------------------------------------------------------- direction
def direction():
    sk, bt = read("e1_direction_skill"), read("e1_direction_backtests")

    def render(label):
        s = sk[label].pivot(index="series", columns="embedding", values="IC long")
        bss = sk[label][sk[label].embedding == "m=4"].set_index("series")["Brier skill"]
        b = bt[label].pivot(index="series", columns="strategy", values="Sharpe")
        dsr = bt[label][bt[label].strategy.str.contains("grid")].set_index("series")["DSR"]
        rows = []
        for a in s.index:
            rows.append(
                " & ".join(
                    [
                        esc(a),
                        fmt(s.loc[a, "m=1"]),
                        fmt(s.loc[a, "m=4"]),
                        fmt(bss[a]),
                        fmt(b.loc[a, "Buy & hold"], 2),
                        fmt(b.loc[a, "v1 (Pine), 5 bp"], 2),
                        fmt(b.loc[a, "v2 walk-forward grid, 5 bp"], 2),
                        fmt(dsr[a], 2),
                    ]
                )
                + r"\\"
            )
        macro(f"dirMaxDSR{label[:3]}", fmt(dsr.max(), 2))
        macro(f"dirMeanIC{label[:3]}", fmt(s["m=4"].mean(), 3))
        return rows

    body = panel_rows({k: k for k in sk}, render, 8)
    write(
        "tab_direction",
        r"""\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{2}{c}{IC (long)} & Brier & \multicolumn{3}{c}{Sharpe ratio} & \\
\cmidrule(lr){2-3}\cmidrule(lr){5-7}
Asset & $m{=}1$ & $m{=}4$ & skill & B\&H & v1 & v2 WF & DSR \\
\midrule
"""
        + body
        + "\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- volatility
MODELS = ["EWMA", "GARCH", "HAR", "GBM", "Analogue", "Analogue+HAR"]


def volatility():
    e2 = read("e2_volatility")

    def render(df):
        q = df.pivot(index="series", columns="model", values="QLIKE / HAR")[MODELS]
        m = df.pivot(index="series", columns="model", values="MCS p (QLIKE)")[MODELS]
        rows = []
        for s in q.index:
            cells = [
                (r"\textbf{" + fmt(q.loc[s, k]) + "}") if m.loc[s, k] >= 0.10 else fmt(q.loc[s, k])
                for k in MODELS
            ]
            rows.append(esc(s) + " & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace Mean ratio & " + " & ".join(fmt(q[k].mean()) for k in MODELS) + r"\\")
        rows.append(
            r"In 90\% MCS & " + " & ".join(f"{int((m[k] >= 0.1).sum())}/{len(m)}" for k in MODELS) + r"\\"
        )
        return rows

    body = panel_rows(e2, render, 7)
    write(
        "tab_vol",
        r"""\begin{tabular}{lrrrrrr}
\toprule
Series (horizon) & EWMA & GARCH & HAR & GBM & Analogue & Analogue+HAR \\
\midrule
"""
        + body
        + "\n\\end{tabular}\n",
    )


def vol_summary():
    e2 = read("e2_volatility")
    lines = []
    for label, df in e2.items():
        q = df.pivot(index="series", columns="model", values="QLIKE / HAR")[MODELS]
        m = df.pivot(index="series", columns="model", values="MCS p (QLIKE)")[MODELS]
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label} set ({len(q)} series)}}}}\\")
        lines.append(r"Mean QLIKE / HAR & " + " & ".join(fmt(q[k].mean()) for k in MODELS) + r"\\")
        lines.append(r"Median QLIKE / HAR & " + " & ".join(fmt(q[k].median()) for k in MODELS) + r"\\")
        lines.append(r"In 90\% MCS & " + " & ".join(f"{int((m[k] >= 0.1).sum())}" for k in MODELS) + r"\\")
        lines.append(
            r"Lowest QLIKE & " + " & ".join(str(int((q.idxmin(1) == k).sum())) for k in MODELS) + r"\\"
        )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write(
        "tab_vol_summary",
        r"""\begin{tabular}{lrrrrrr}
\toprule
& EWMA & GARCH & HAR & GBM & Analogue & Analogue+HAR \\
\midrule
"""
        + "\n".join(lines)
        + "\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- ablations
def ablation():
    e3 = read("e3_ablation")
    cols = [c for c in next(iter(e3.values())).columns if c not in ("series", "full")]
    lines = []
    for label, df in e3.items():
        mean = df[df.series == "mean"].iloc[0]
        lines.append(esc(label) + " & " + " & ".join(fmt(mean[c]) for c in cols) + r"\\")
    write(
        "tab_ablation",
        r"\begin{tabular}{l"
        + "r" * len(cols)
        + "}\n\\toprule\n& "
        + " & ".join(r"\rotatebox{60}{" + esc(c) + "}" for c in cols)
        + r"\\"
        + "\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- tails
TAIL = ["Constant", "Poisson (v1)", "Hawkes", "Logistic-EWMA", "Analogue"]


def tails():
    e4 = read("e4_tail")

    def render(df):
        ll = df.pivot(index="series", columns="model", values="log-loss")[TAIL]
        auc = df.pivot(index="series", columns="model", values="AUC")[TAIL]
        best = ll.idxmin(1).value_counts()
        rows = [
            r"Mean log-loss / constant & "
            + " & ".join(fmt((ll[k] / ll["Constant"]).mean()) for k in TAIL)
            + r"\\",
            r"Mean AUC & " + " & ".join(fmt(auc[k].mean()) for k in TAIL) + r"\\",
            r"Best log-loss (series) & " + " & ".join(str(int(best.get(k, 0))) for k in TAIL) + r"\\",
            r"Worse than constant & "
            + " & ".join(str(int((ll[k] > ll["Constant"]).sum())) for k in TAIL)
            + r"\\",
        ]
        return rows

    body = panel_rows(e4, render, 6)
    write(
        "tab_tail",
        r"""\begin{tabular}{lrrrrr}
\toprule
& Constant & Poisson (v1) & Hawkes & Logistic-EWMA & Analogue \\
\midrule
"""
        + body
        + "\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- VaR / ES
VAR = ["HS", "GARCH-N", "GARCH-t", "FHS", "Analogue", "Blend"]


def var():
    e8 = read("e8_var")

    def render(df):
        g = df.groupby("model")
        fz = df.pivot_table(index=["series", "alpha"], columns="model", values="FZ0")[VAR]
        rank = fz.rank(axis=1).mean()
        rows = [
            r"Mean $|$hit rate$/\alpha - 1|$ & "
            + " & ".join(fmt((g.get_group(k)["hit rate / alpha"] - 1).abs().mean()) for k in VAR)
            + r"\\",
            r"Kupiec pass rate & "
            + " & ".join(fmt((g.get_group(k)["Kupiec p"] > 0.05).mean(), 2) for k in VAR)
            + r"\\",
            r"Christoffersen CC pass rate & "
            + " & ".join(fmt((g.get_group(k)["Christoffersen CC p"] > 0.05).mean(), 2) for k in VAR)
            + r"\\",
            r"Mean FZ0 rank (1 = best) & " + " & ".join(fmt(rank[k], 2) for k in VAR) + r"\\",
            r"In 90\% MCS (FZ0) & "
            + " & ".join(fmt((g.get_group(k)["MCS p (FZ0)"] >= 0.1).mean(), 2) for k in VAR)
            + r"\\",
        ]
        return rows

    body = panel_rows(e8, render, 7)
    write(
        "tab_var",
        r"""\begin{tabular}{lrrrrrr}
\toprule
& HS & GARCH-N & GARCH-t & FHS & Analogue & Blend \\
\midrule
"""
        + body
        + "\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- overlay
def overlay():
    e5 = read("e5_overlay")

    def render(df):
        dd = df.pivot(index="series", columns="strategy", values="max DD")
        sr = df.pivot(index="series", columns="strategy", values="Sharpe")
        p = df.pivot(index="series", columns="strategy", values="p")
        rows = []
        for a in dd.index:
            vt, bh, gate = "Vol target: Analogue+HAR", "Buy & hold", "Vol target: Analogue+HAR + Hawkes gate"
            rows.append(
                " & ".join(
                    [
                        esc(a),
                        fmt(sr.loc[a, bh], 2),
                        fmt(sr.loc[a, vt], 2),
                        fmt(p.loc[a, vt], 2),
                        fmt(sr.loc[a, gate], 2),
                        fmt(100 * dd.loc[a, bh], 1),
                        fmt(100 * dd.loc[a, vt], 1),
                    ]
                )
                + r"\\"
            )
        return rows

    body = panel_rows(e5, render, 7)
    write(
        "tab_overlay",
        r"""\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{4}{c}{Sharpe ratio} & \multicolumn{2}{c}{Max drawdown (\%)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Asset & B\&H & Vol target & $p$ & + Hawkes gate & B\&H & Vol target \\
\midrule
"""
        + body
        + "\n\\end{tabular}\n",
    )


# ---------------------------------------------------------------- hypotheses
def hypotheses():
    h = read("hypotheses")
    if "Confirmatory" not in h:
        return
    d, c = h["Development"].set_index("id"), h["Confirmatory"].set_index("id")
    rows = []
    for i in c.index:
        rows.append(
            " & ".join(
                [
                    i,
                    esc(c.loc[i, "hypothesis"]),
                    esc(d.loc[i, "statistic"]),
                    fmt(d.loc[i, "p"], 3),
                    esc(c.loc[i, "statistic"]),
                    fmt(c.loc[i, "p"], 3),
                    fmt(c.loc[i, "Holm p"], 3),
                ]
            )
            + r"\\"
        )
    write(
        "tab_hypotheses",
        r"""\begin{tabular}{l>{\raggedright\arraybackslash}p{5.2cm}>{\raggedright\arraybackslash}p{2.6cm}r>{\raggedright\arraybackslash}p{2.6cm}rr}
\toprule
& & \multicolumn{2}{c}{Development (exploratory)} & \multicolumn{3}{c}{Confirmatory (pre-registered)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
ID & Hypothesis & Statistic & $p$ & Statistic & $p$ & Holm $p$ \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def robustness():
    """Appendix tables: dividend-adjusted prices and Politis-White block lengths."""
    adj = ROOT / "results" / "adjusted_etfs" / "comparison.csv"
    if adj.exists():
        t = pd.read_csv(adj)
        keep = t[
            t.result.isin(
                [
                    "QLIKE / HAR: Analogue+HAR",
                    "Buy & hold: Sharpe",
                    "Vol target: Analogue+HAR: Sharpe",
                    "Buy & hold: max DD",
                    "Vol target: Analogue+HAR: max DD",
                ]
            )
        ]
        rows = []
        for _, r in keep.iterrows():
            rows.append(
                " & ".join(
                    [esc(r.series), esc(r.result), fmt(r.unadjusted), fmt(r.adjusted), f"{r.change:+.3f}"]
                )
                + r"\\"
            )
        write(
            "tab_rob_adjusted",
            r"""\begin{tabular}{llrrr}
\toprule
Series & Quantity & Unadjusted & Adjusted & Change \\
\midrule
"""
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n",
        )
    blk = ROOT / "results" / "block_length_comparison.csv"
    if blk.exists():
        t = pd.read_csv(blk)
        rows = []
        for _, r in t.iterrows():
            mcs = pd.notna(r["in 90% MCS: fixed blocks"])
            fixed = r["in 90% MCS: fixed blocks"] if mcs else r["significant at 5%: fixed blocks"]
            auto = r["in 90% MCS: Politis-White"] if mcs else r["significant at 5%: Politis-White"]
            rows.append(
                " & ".join(
                    [
                        esc(r.set),
                        esc(r.test),
                        str(int(r.cases)),
                        "in MCS" if mcs else "significant",
                        str(int(fixed)),
                        str(int(auto)),
                        str(int(r["verdicts changed"])),
                    ]
                )
                + r"\\"
            )
        write(
            "tab_rob_blocks",
            r"""\begin{tabular}{llrlrrr}
\toprule
Set & Test & Cases & Verdict & Fixed & Politis--White & Changed \\
\midrule
"""
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n",
        )


def robustness_har_rv():
    """Appendix tables: rolling-window HAR (e9) and intraday realised variance for crypto (e10)."""
    from scipy.stats import wilcoxon

    rows = []
    for label, base in RES.items():
        f, g = base / "e9_rolling_har.csv", base / "e9_rolling_har_dm.csv"
        if not f.exists():
            continue
        t, dm = pd.read_csv(f), pd.read_csv(g)
        for freq, wins in (("Hourly", (2190, 8760)), ("Daily", (250, 1000))):
            ser = dm[dm.window.isin(wins)].series.unique()
            for w in wins:
                tt = t[t.series.isin(ser)]
                q = tt.pivot(index="series", columns="model", values="QLIKE / HAR (expanding)")
                m = tt.pivot(index="series", columns="model", values="MCS p (QLIKE)")
                d = dm[dm.window == w]
                r = np.log(d["QLIKE ratio (A+H / HAR), rolling"])
                rows.append(
                    " & ".join(
                        [
                            label,
                            freq,
                            str(w),
                            str(len(ser)),
                            fmt(q[f"HAR (rolling {w})"].median()),
                            str(int((m[f"HAR (rolling {w})"] >= 0.1).sum())),
                            fmt(np.exp(r.median())),
                            f"{int(((d['DM stat'] > 0) & (d.p < 0.05)).sum())}/{int(((d['DM stat'] < 0) & (d.p < 0.05)).sum())}",
                            fmt(wilcoxon(r).pvalue, 3) if len(r) >= 5 else "--",
                        ]
                    )
                    + r"\\"
                )
    if rows:
        write(
            "tab_rob_rolling",
            r"""\begin{tabular}{llrrrrrrr}
\toprule
& & & & \multicolumn{2}{c}{Rolling HAR} & \multicolumn{3}{c}{Analogue+HAR vs rolling HAR} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-9}
Set & Freq. & Window & Series & QLIKE / HAR$_{\text{exp}}$ & In MCS & Median ratio & DM sig.\ (A+H/HAR) & Wilcoxon $p$ \\
\midrule
"""
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n",
        )

    cols = ["EWMA", "GARCH", "HAR", "GBM", "Analogue", "Analogue+HAR", "HAR (rolling 8760)"]
    rows = []
    for label, base in RES.items():
        f = base / "e10_intraday_rv.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        q = t.pivot(index="series", columns="model", values="QLIKE / HAR")
        m = t.pivot(index="series", columns="model", values="MCS p (QLIKE)")
        rows.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{label} set}}}}\\")
        for s_ in q.index:
            cells = [
                (r"\textbf{" + fmt(q.loc[s_, k]) + "}") if m.loc[s_, k] >= 0.10 else fmt(q.loc[s_, k])
                for k in cols
            ]
            rows.append(esc(s_) + " & " + " & ".join(cells) + r"\\")
    if rows:
        write(
            "tab_rob_rv",
            r"""\begin{tabular}{lrrrrrrr}
\toprule
Series & EWMA & GARCH & HAR & GBM & Analogue & Analogue+HAR & HAR (rolling 1y) \\
\midrule
"""
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n",
        )


def data_table():
    import sys

    sys.path.insert(0, str(ROOT / "python"))
    from analogue_risk import vol as V
    from analogue_risk.experiments import UNIVERSES, load, split

    rows = []
    for universe in ("development", "confirmatory"):
        u = UNIVERSES[universe]
        tick = {**u["hourly"], **u["daily"]}
        for name, (df, hourly) in load(universe).items():
            vc = V.HOURLY if hourly else V.DAILY_W
            cal, _ = split(len(df), vc.warmup)
            rows.append(
                " & ".join(
                    [
                        universe.capitalize() if not rows or rows[-1].startswith("\\midrule") else "",
                        esc(name),
                        esc(tick[name]),
                        "Binance" if hourly else "Yahoo",
                        "1h" if hourly else "1d",
                        f"{df.index[0]:%Y-%m-%d}",
                        f"{df.index[-1]:%Y-%m-%d}",
                        f"{len(df):,}",
                        f"{len(df) - cal:,}",
                    ]
                )
                + r"\\"
            )
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    write(
        "tab_data",
        r"""\begin{tabular}{lllllllrr}
\toprule
Set & Series & Ticker & Source & Freq. & Start & End & Bars & OOS bars \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\end{tabular}\n",
    )


if __name__ == "__main__":
    for f in (
        direction,
        volatility,
        vol_summary,
        ablation,
        tails,
        var,
        overlay,
        hypotheses,
        robustness,
        robustness_har_rv,
        data_table,
    ):
        f()
    (OUT / "macros.tex").write_text("".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in MACROS.items()))
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
