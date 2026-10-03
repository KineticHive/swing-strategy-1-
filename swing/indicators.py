"""Technical indicators: RSI, MACD, ATR and a volume profile (POC / VAH / VAL)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI (matches TradingView's ta.rsi)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI 100; no movement at all -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=close.index)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


@dataclass(frozen=True)
class VolumeProfile:
    poc: float          # price of the row with the most volume (row midpoint)
    vah: float          # top of the value area
    val: float          # bottom of the value area
    row_edges: np.ndarray  # len(rows) + 1 price boundaries
    row_volume: np.ndarray  # volume per row


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rows: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Build a volume profile over a range of bars.

    Each bar's volume is spread across the price rows its high-low range
    overlaps, in proportion to the overlap. The value area grows outward from
    the POC one row at a time, always adding whichever neighbouring row has
    more volume, until it holds ``value_area_pct`` of total volume.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)

    top, bottom = high.max(), low.min()
    if top == bottom:
        return VolumeProfile(top, top, bottom, np.array([bottom, top]), np.array([volume.sum()]))

    edges = np.linspace(bottom, top, rows + 1)
    row_lo, row_hi = edges[:-1], edges[1:]

    # overlap[i, r] = length of bar i's range inside row r
    overlap = np.clip(
        np.minimum(high[:, None], row_hi[None, :]) - np.maximum(low[:, None], row_lo[None, :]),
        0.0,
        None,
    )
    span = high - low
    weights = np.zeros_like(overlap)
    ranged = span > 0
    weights[ranged] = overlap[ranged] / span[ranged, None]
    # Zero-range bars (high == low) drop all their volume into one row.
    for i in np.flatnonzero(~ranged):
        r = min(np.searchsorted(edges, high[i], side="right") - 1, rows - 1)
        weights[i, r] = 1.0
    row_volume = (weights * volume[:, None]).sum(axis=0)

    poc_idx = int(np.argmax(row_volume))
    target = row_volume.sum() * value_area_pct
    lo_idx = hi_idx = poc_idx
    acc = row_volume[poc_idx]
    while acc < target and (lo_idx > 0 or hi_idx < rows - 1):
        above = row_volume[hi_idx + 1] if hi_idx < rows - 1 else -1.0
        below = row_volume[lo_idx - 1] if lo_idx > 0 else -1.0
        if above >= below:
            hi_idx += 1
            acc += above
        else:
            lo_idx -= 1
            acc += below

    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0
    return VolumeProfile(poc, edges[hi_idx + 1], edges[lo_idx], edges, row_volume)


def rolling_volume_profile(
    df: pd.DataFrame,
    lookback: int = 150,
    rows: int = 50,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """POC/VAH/VAL for each bar, using the previous ``lookback`` bars (inclusive).

    This is the backtest-safe stand-in for a *visible range* profile: each bar
    only sees data that existed when that bar closed.
    """
    high, low, vol = df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
    n = len(df)
    out = np.full((n, 3), np.nan)
    for i in range(lookback - 1, n):
        s = slice(i - lookback + 1, i + 1)
        vp = volume_profile(high[s], low[s], vol[s], rows, value_area_pct)
        out[i] = (vp.poc, vp.vah, vp.val)
    return pd.DataFrame(out, index=df.index, columns=["poc", "vah", "val"])
