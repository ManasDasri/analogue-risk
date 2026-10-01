"""Market data: Binance spot klines and Yahoo daily bars, cached as CSV under data/.

Every series used in the paper is pinned in data_manifest.json (last bar, row count, SHA-256 of
the cleaned data). Loading a pinned series truncates it to that bar and checks the hash, so a
fresh download reproduces the published data exactly or warns that the source has changed.
"""
import hashlib
import io
import json
import os
import urllib.error
import zipfile
import warnings
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
MANIFEST = json.loads((Path(__file__).with_name("data_manifest.json")).read_text())
COLS = ["open", "high", "low", "close", "volume"]


def _get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


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


def digest(df):
    """SHA-256 of a frame's CSV representation (10 significant digits)."""
    return hashlib.sha256(df.to_csv(float_format="%.10g").encode()).hexdigest()


def _cached(name, fetch):
    path = DATA / f"{name}.csv.gz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        fetch().to_csv(path, date_format="%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="ISO8601")
    df = _clean(df)
    pin = MANIFEST.get(name)
    if pin is None:
        return df.iloc[:-1]  # unpinned: the final bar may still be forming
    df = df[df.index <= pd.Timestamp(pin["end"])]
    if len(df) != pin["rows"] or digest(df) != pin["sha256"]:
        warnings.warn(f"{name}: data differs from the pinned snapshot ({len(df)} rows vs {pin['rows']}); "
                      "the source may have revised its history, so results can differ from the paper",
                      stacklevel=2)
    return df


def _clean(df):
    df = df[COLS].astype(float).dropna()
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df.high >= df.low)
    return df[ok & ~df.index.duplicated()].sort_index()


def binance(symbol, interval="1h", start="2018-01-01"):
    """Binance spot klines. Uses the REST API, or the public bulk archive (data.binance.vision) when
    the API is unreachable or geo-blocked (HTTP 451/403), or when SQ_BINANCE_SOURCE=archive."""
    def fetch():
        if os.environ.get("SQ_BINANCE_SOURCE") != "archive":
            try:
                return _binance_api(symbol, interval, start)
            except urllib.error.HTTPError as e:
                if e.code not in (403, 451):
                    raise
                warnings.warn(f"Binance API refused the request (HTTP {e.code}); using data.binance.vision")
        return _binance_archive(symbol, interval, start)

    return _cached(f"binance_{symbol}_{interval}", fetch)


def _klines_frame(rows):
    df = pd.DataFrame([r[:6] for r in rows], columns=["time"] + COLS)
    t = df.pop("time").astype("int64")
    t = t.where(t < 10**14, t // 1000)  # the archive switched to microseconds in 2025
    df.index = pd.to_datetime(t, unit="ms").astype("datetime64[ns]").rename("time")
    return df.astype(float)


def _binance_api(symbol, interval, start):
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
    return _klines_frame(rows)


def _binance_archive(symbol, interval, start):
    """Monthly zip files; months not yet archived are assembled from daily files."""
    base = "https://data.binance.vision/data/spot"
    rows = []

    def read(blob):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode()
        return [line.split(",") for line in text.splitlines() if line and line[0].isdigit()]

    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    for month in pd.period_range(pd.Timestamp(start), today, freq="M"):
        blob = _get_bytes(f"{base}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip")
        if blob is not None:
            rows += read(blob)
            continue
        for day in pd.date_range(month.start_time, min(month.end_time, today), freq="D"):
            blob = _get_bytes(f"{base}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip")
            if blob is not None:
                rows += read(blob)
    return _klines_frame(rows)


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
