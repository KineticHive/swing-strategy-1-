"""Load OHLCV data from CSV, Yahoo Finance (optional) or a synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lower().replace("adj close", "adj_close"))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[REQUIRED].astype(float).dropna().sort_index()


def load_csv(path: str) -> pd.DataFrame:
    """CSV with a date/time column plus open, high, low, close, volume."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.strip().lower() in ("date", "datetime", "time", "timestamp")), None)
    if date_col is None:
        raise ValueError("CSV needs a date/datetime/time/timestamp column")
    df[date_col] = pd.to_datetime(df[date_col])
    return normalize(df.set_index(date_col))


def load_yahoo(symbol: str, start: str = "2018-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install yfinance to download data") from e
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return normalize(df)


def synthetic(n: int = 1500, seed: int = 7, start_price: float = 100.0) -> pd.DataFrame:
    """Random-walk OHLCV with regime switches, for demos and tests."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0015, n // 100 + 1), 100)[:n]
    rets = drift + rng.normal(0, 0.015, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    wick = np.abs(rng.normal(0, 0.008, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.008, n)) * close
    volume = rng.lognormal(13, 0.4, n) * (1 + 20 * np.abs(rets))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
