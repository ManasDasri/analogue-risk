"""Check the Rust port of the original Pine strategy against TradingView's own backtest.

Export the trade list from TradingView (Strategy Tester -> List of Trades -> export) for the
original script on BINANCE:BTCUSDT, 1h, default inputs, then run:

    python -m analogue_risk.pine_parity trades.csv [--symbol BTCUSDT] [--tz UTC]

`--tz` is the chart timezone the export was made in (TradingView writes times in it).
Trades are matched by entry bar and direction; exit bar and prices are then compared.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from . import data, model as M


def _col(df, *keys):
    for c in df.columns:
        if all(k in c.lower() for k in keys):
            return c
    raise KeyError(f"no column containing {keys} in {list(df.columns)}")


def read_tradingview(path, tz="UTC"):
    """One row per trade: entry/exit time (UTC, bar open), direction (+1/-1), entry/exit price."""
    raw = pd.read_csv(path)
    num, typ, when = _col(raw, "trade"), _col(raw, "type"), _col(raw, "date")
    price = _col(raw, "price")
    raw[when] = pd.to_datetime(raw[when]).dt.tz_localize(tz).dt.tz_convert("UTC").dt.tz_localize(None)
    entry = raw[raw[typ].str.lower().str.startswith("entry")].set_index(num)
    exit_ = raw[raw[typ].str.lower().str.startswith("exit")].set_index(num)
    out = pd.DataFrame({
        "entry_time": entry[when], "exit_time": exit_[when].reindex(entry.index),
        "dir": np.where(entry[typ].str.lower().str.contains("long"), 1, -1),
        "entry_px": entry[price].astype(float), "exit_px": exit_[price].reindex(entry.index).astype(float)})
    return out.dropna(subset=["exit_time"]).sort_values("entry_time").reset_index(drop=True)


def port_trades(mkt):
    tr = M.run_v1(mkt, cost_bps=1.0)["trades"]
    idx = mkt.df.index
    return pd.DataFrame({"entry_time": idx[tr["entry_bar"]], "exit_time": idx[tr["exit_bar"]], "dir": tr["dir"],
                         "entry_px": tr["entry_px"], "exit_px": tr["exit_px"]})


def compare(tv, ours, price_tol=1e-4):
    """Match TradingView trades to port trades by entry bar and direction, within TradingView's
    window (after the first trade, so both have warmed up)."""
    lo, hi = tv.entry_time.min(), tv.entry_time.max()
    ours = ours[(ours.entry_time >= lo) & (ours.entry_time <= hi)]
    m = tv.merge(ours, on=["entry_time", "dir"], how="outer", suffixes=("_tv", "_port"), indicator=True)
    both = m[m["_merge"] == "both"]
    rel = lambda a, b: (a - b).abs() / b.abs()
    report = {
        "tradingview trades": len(tv), "port trades in window": len(ours), "matched entries": len(both),
        "match rate (of TradingView)": len(both) / max(len(tv), 1),
        "same exit bar": float((both.exit_time_tv == both.exit_time_port).mean()) if len(both) else np.nan,
        "entry price within tol": float((rel(both.entry_px_port, both.entry_px_tv) <= price_tol).mean()) if len(both) else np.nan,
        "exit price within tol": float((rel(both.exit_px_port, both.exit_px_tv) <= price_tol).mean()) if len(both) else np.nan,
    }
    return report, m[m["_merge"] != "both"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trades_csv")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tz", default="UTC")
    a = ap.parse_args()
    tv = read_tradingview(a.trades_csv, a.tz)
    report, unmatched = compare(tv, port_trades(M.Market(data.binance(a.symbol, "1h"))))
    for k, v in report.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    if len(unmatched):
        print("\nunmatched trades (first 20):")
        print(unmatched.head(20).to_string(index=False))
    sys.exit(0 if report["match rate (of TradingView)"] >= 0.95 else 1)


if __name__ == "__main__":
    main()
