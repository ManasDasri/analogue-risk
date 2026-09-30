"""Design study on the CALIBRATION period only: models fitted before the last third of the
calibration window and scored on it (QLIKE ratio to HAR). The out-of-sample period is untouched."""
import sys, time, warnings
sys.path.insert(0, "python"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from dataclasses import replace
from sq import data, model as M, vol as V

sets = [("BTC_1h", data.binance("BTCUSDT", "1h"), V.HOURLY), ("ETH_1h", data.binance("ETHUSDT", "1h"), V.HOURLY),
        ("PAXG_1h", data.binance("PAXGUSDT", "1h"), V.HOURLY), ("GSPC_w", data.yahoo("^GSPC", start="1927-12-30"), V.DAILY_W),
        ("GSPC_m", data.yahoo("^GSPC", start="1927-12-30"), V.DAILY_M), ("EURUSD_w", data.yahoo("EURUSD=X"), V.DAILY_W)]
variants = {"knn20": {}, "knn20 pooled": dict(regime=False), "knn50": dict(k=50), "knn50 pooled": dict(k=50, regime=False)}
for bw in (0.15, 0.25, 0.4, 0.6, 1.0):
    variants[f"kernel {bw}"] = dict(bandwidth=bw)
    variants[f"kernel {bw} pooled"] = dict(bandwidth=bw, regime=False)
table = {}
for name, df, vc in sets:
    mkt = M.Market(df); d = V.VolData(mkt, vc)
    cal = max(int(.3 * mkt.n), vc.warmup + 500); vs = cal - (cal - vc.warmup) // 3
    base, _ = V.walk_forward(d, [(vs, cal)])
    s = slice(vs, cal); ok = (d.realised[s] > 0) & np.isfinite(d.target[s])
    sc = lambda f: np.mean(V.qlike(d.realised[s][ok], f[s][ok]))
    row = {k: sc(f) for k, f in base.items() if k in ("EWMA", "GARCH", "HAR")}
    t0 = time.perf_counter()
    for vn, ov in variants.items():
        f, _, _ = V.analogue(d, replace(vc, **ov))
        fit = d.fit_rows(vs); v = np.isfinite(f[fit]) & np.isfinite(d.target[fit])
        f = f + np.log(np.mean(np.exp(d.target[fit][v] - f[fit][v])))
        row[vn] = sc(f)
        row[vn + " +HAR"] = sc((f + base["HAR"]) / 2 + 0)  # equal-weight combination in logs
    table[name] = row
    print(f"{name}: validation {df.index[vs].date()}..{df.index[cal].date()} ({time.perf_counter()-t0:.0f}s)", flush=True)
T = pd.DataFrame(table)
R = T / T.loc["HAR"]
R["mean"] = R.mean(1)
print(R.sort_values("mean").round(3).to_string())
R.sort_values("mean").to_csv("results/dev/design_study.csv")
