"""Market data: Binance spot klines and Yahoo daily bars, cached as CSV under data/."""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
COLS = ["open", "high", "low", "close", "volume"]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def _cached(name, fetch):
    path = DATA / f"{name}.csv.gz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        fetch().to_csv(path)
    return _clean(pd.read_csv(path, index_col=0, parse_dates=True))


def _clean(df):
    df = df[COLS].astype(float).dropna()
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df.high >= df.low)
    # the final bar may still be forming
    return df[ok & ~df.index.duplicated()].sort_index().iloc[:-1]


def binance(symbol, interval="1h", start="2018-01-01"):
    def fetch():
        rows, t = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        while True:
            batch = _get(f"https://api.binance.com/api/v3/klines?symbol={symbol}"
                         f"&interval={interval}&startTime={t}&limit=1000")
            if not batch:
                break
            rows += batch
            t = batch[-1][0] + 1
            if len(batch) < 1000:
                break
        df = pd.DataFrame([r[:6] for r in rows], columns=["time"] + COLS)
        df.index = pd.to_datetime(df.pop("time"), unit="ms")
        return df

    return _cached(f"binance_{symbol}_{interval}", fetch)


def yahoo(symbol, start="1990-01-01"):
    def fetch():
        p1 = int(pd.Timestamp(start).timestamp())
        p2 = int(time.time())
        j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                 f"?period1={p1}&period2={p2}&interval=1d")["chart"]["result"][0]
        q = j["indicators"]["quote"][0]
        df = pd.DataFrame({k: q[k] for k in COLS}, index=pd.to_datetime(j["timestamp"], unit="s"))
        return df

    return drop_bad_ticks(_cached(f"yahoo_{symbol.replace('^', '').replace('=', '')}_1d", fetch))


def drop_bad_ticks(df, k=8.0, reversal=0.7, window=20):
    """Drop isolated bad prints: bars whose log return exceeds k trailing standard deviations AND is
    reversed by more than `reversal` of its size on the next bar. Genuine large moves rarely
    reverse that fast; data errors (e.g. Yahoo EUR/USD on 2008-12-08, +16% then -14%) do."""
    r = np.log(df.close).diff()
    sd = r.rolling(window, min_periods=window).std().shift()
    nxt = r.shift(-1)
    bad = (r.abs() > k * sd) & (r * nxt < 0) & (nxt.abs() > reversal * r.abs())
    return df[~bad]


def resample(df, rule):
    """Aggregate bars to a coarser timeframe (e.g. 1-minute -> '1h')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()
