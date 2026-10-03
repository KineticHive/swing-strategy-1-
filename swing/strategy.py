"""Swing strategy: RSI + MACD timing, volume-profile (POC/VAH/VAL) location.

There are four setups. Signals are evaluated on the bar's close and filled at
the next bar's open.

LONG
  * VAL bounce:     price tagged the value-area low recently, closed back above
                    it (still below POC), RSI was oversold and is turning up,
                    and MACD crossed bullish recently. Target VAH (or POC).
  * VAH breakout:   close breaks above the value-area high with RSI in the
                    bullish band (not overbought) and MACD above signal and zero.
                    Target is a multiple of the risk taken (R-multiple).

SHORT (off by default; mirror images)
  * VAH rejection:  tagged VAH, closed back below it (above POC), RSI was
                    overbought and is turning down, MACD crossed bearish.
  * VAL breakdown:  close breaks below VAL, RSI in the bearish band, MACD below
                    signal and zero.

Every trade has an ATR-buffered stop beyond the relevant level. Positions also
close on a momentum exit (opposite MACD cross while RSI is stretched) or after
``max_hold_bars``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr, macd, rolling_volume_profile, rsi


@dataclass
class StrategyConfig:
    # RSI
    rsi_length: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    rsi_bull_min: float = 50.0     # breakout needs RSI >= this ...
    rsi_bull_max: float = 75.0     # ... and <= this (not already exhausted)
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_cross_lookback: int = 3   # a cross within this many bars counts
    # Volume profile
    vp_lookback: int = 120         # bars in the rolling "visible range"
    vp_rows: int = 50
    value_area_pct: float = 0.70
    touch_lookback: int = 5        # bars within which the VA edge must be tagged
    touch_tolerance_atr: float = 0.25
    # Risk / exits
    atr_length: int = 14
    stop_atr_mult: float = 1.0     # stop buffer beyond the level, in ATRs
    breakout_rr: float = 2.0       # target = entry +/- rr * risk for breakouts
    reversion_target: str = "vah"  # "poc" or "vah" (mirror for shorts)
    max_hold_bars: int = 30
    # Which setups trade
    enable_reversion: bool = True
    enable_breakout: bool = True
    enable_shorts: bool = False


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01   # fraction of equity lost if stopped out
    commission_pct: float = 0.0005  # per side, fraction of notional
    slippage_pct: float = 0.0005    # per side, adverse
    max_position_pct: float = 1.0   # cap notional at this fraction of equity


def _crossed_within(a: pd.Series, b: pd.Series, bars: int) -> pd.Series:
    cross = (a > b) & (a.shift(1) <= b.shift(1))
    return cross.rolling(bars, min_periods=1).max().astype(bool)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(df["close"], cfg.rsi_length)
    out = out.join(macd(df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))
    out["atr"] = atr(df, cfg.atr_length)
    out = out.join(rolling_volume_profile(df, cfg.vp_lookback, cfg.vp_rows, cfg.value_area_pct))
    return out


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """Add indicator columns plus signal, stop and target columns.

    ``signal`` is +1 (long), -1 (short) or 0. ``setup`` names the setup that
    fired. ``stop`` and ``target`` are the price levels to use.
    """
    cfg = cfg or StrategyConfig()
    d = compute_indicators(df, cfg)
    n = cfg.touch_lookback
    tol = cfg.touch_tolerance_atr * d["atr"]

    macd_bull_cross = _crossed_within(d["macd"], d["signal"], cfg.macd_cross_lookback)
    macd_bear_cross = _crossed_within(d["signal"], d["macd"], cfg.macd_cross_lookback)
    rsi_up = d["rsi"] > d["rsi"].shift(1)
    rsi_down = d["rsi"] < d["rsi"].shift(1)
    rsi_was_oversold = d["rsi"].rolling(n, min_periods=1).min() <= cfg.rsi_oversold
    rsi_was_overbought = d["rsi"].rolling(n, min_periods=1).max() >= cfg.rsi_overbought

    touched_val = ((d["low"] <= d["val"] + tol).astype(float).rolling(n, min_periods=1).max() > 0)
    touched_vah = ((d["high"] >= d["vah"] - tol).astype(float).rolling(n, min_periods=1).max() > 0)

    # --- Long setups
    long_rev = (
        cfg.enable_reversion
        & touched_val
        & (d["close"] > d["val"])
        & (d["close"] < d["poc"])
        & rsi_was_oversold
        & rsi_up
        & macd_bull_cross
    )
    long_brk = (
        cfg.enable_breakout
        & (d["close"] > d["vah"])
        & (d["close"].shift(1) <= d["vah"].shift(1))
        & d["rsi"].between(cfg.rsi_bull_min, cfg.rsi_bull_max)
        & (d["macd"] > d["signal"])
        & (d["macd"] > 0)
    )

    # --- Short setups (mirror)
    bear_min, bear_max = 100 - cfg.rsi_bull_max, 100 - cfg.rsi_bull_min
    short_rev = (
        cfg.enable_shorts
        & cfg.enable_reversion
        & touched_vah
        & (d["close"] < d["vah"])
        & (d["close"] > d["poc"])
        & rsi_was_overbought
        & rsi_down
        & macd_bear_cross
    )
    short_brk = (
        cfg.enable_shorts
        & cfg.enable_breakout
        & (d["close"] < d["val"])
        & (d["close"].shift(1) >= d["val"].shift(1))
        & d["rsi"].between(bear_min, bear_max)
        & (d["macd"] < d["signal"])
        & (d["macd"] < 0)
    )

    buf = cfg.stop_atr_mult * d["atr"]
    recent_low = d["low"].rolling(n, min_periods=1).min()
    recent_high = d["high"].rolling(n, min_periods=1).max()
    rev_long_target = d["vah"] if cfg.reversion_target == "vah" else d["poc"]
    rev_short_target = d["val"] if cfg.reversion_target == "vah" else d["poc"]

    signal = pd.Series(0, index=d.index)
    setup = pd.Series("", index=d.index, dtype=object)
    stop = pd.Series(np.nan, index=d.index)
    target = pd.Series(np.nan, index=d.index)

    # Assign in reverse priority so reversion setups win ties.
    def assign(mask, sig, name, stp, tgt):
        mask = mask.fillna(False).astype(bool)
        signal[mask] = sig
        setup[mask] = name
        stop[mask] = stp[mask]
        target[mask] = tgt[mask]

    long_brk_stop = d["vah"] - buf
    short_brk_stop = d["val"] + buf
    assign(short_brk, -1, "val_breakdown", short_brk_stop,
           d["close"] - cfg.breakout_rr * (short_brk_stop - d["close"]))
    assign(long_brk, 1, "vah_breakout", long_brk_stop,
           d["close"] + cfg.breakout_rr * (d["close"] - long_brk_stop))
    assign(short_rev, -1, "vah_rejection", np.maximum(recent_high, d["vah"]) + buf, rev_short_target)
    assign(long_rev, 1, "val_bounce", np.minimum(recent_low, d["val"]) - buf, rev_long_target)

    d["signal"] = signal
    d["setup"] = setup
    d["stop"] = stop
    d["target"] = target
    # Momentum exits: opposite MACD cross while RSI is stretched.
    d["exit_long"] = (d["macd"] < d["signal"]) & (d["macd"].shift(1) >= d["signal"].shift(1)) & (
        d["rsi"] >= cfg.rsi_overbought
    )
    d["exit_short"] = (d["macd"] > d["signal"]) & (d["macd"].shift(1) <= d["signal"].shift(1)) & (
        d["rsi"] <= cfg.rsi_oversold
    )
    return d


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict = field(default_factory=dict)


def backtest(
    df: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    bt: BacktestConfig | None = None,
) -> BacktestResult:
    """Event-driven backtest: one position at a time, next-bar-open fills.

    If a bar's range hits both stop and target, the stop is assumed to fill
    first. Gaps through a level fill at the open.
    """
    cfg = cfg or StrategyConfig()
    bt = bt or BacktestConfig()
    d = generate_signals(df, cfg)

    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    sig = d["signal"].to_numpy()
    stops, targets = d["stop"].to_numpy(), d["target"].to_numpy()
    exit_long, exit_short = d["exit_long"].to_numpy(), d["exit_short"].to_numpy()
    setups = d["setup"].to_numpy()
    idx = d.index

    cash = bt.initial_capital
    pos = 0  # +1 / -1 / 0
    qty = entry_px = stop_px = target_px = 0.0
    entry_i = 0
    setup_name = ""
    trades = []
    equity = np.empty(len(d))

    def close_position(i, px, reason):
        nonlocal cash, pos
        fill = px * (1 - pos * bt.slippage_pct)
        pnl = pos * (fill - entry_px) * qty - fill * qty * bt.commission_pct
        cash += pnl
        risk = abs(entry_px - stop_px) * qty
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[i], "side": "long" if pos > 0 else "short",
            "setup": setup_name, "entry": entry_px, "exit": fill, "stop": stop_px, "target": target_px,
            "qty": qty, "pnl": pnl, "r_multiple": pnl / risk if risk else np.nan,
            "bars_held": i - entry_i, "exit_reason": reason,
        })
        pos = 0

    for i in range(len(d)):
        # 1) Enter at this bar's open on the previous bar's signal.
        if pos == 0 and i > 0 and sig[i - 1] != 0 and not np.isnan(stops[i - 1]):
            side = int(sig[i - 1])
            fill = o[i] * (1 + side * bt.slippage_pct)
            risk_per_unit = side * (fill - stops[i - 1])
            reward_per_unit = side * (targets[i - 1] - fill)
            if risk_per_unit > 0 and reward_per_unit > 0:
                q = (cash * bt.risk_per_trade) / risk_per_unit
                q = min(q, cash * bt.max_position_pct / fill)
                if q > 0:
                    pos, qty, entry_px, entry_i = side, q, fill, i
                    stop_px, target_px, setup_name = stops[i - 1], targets[i - 1], setups[i - 1]
                    cash -= fill * qty * bt.commission_pct

        # 2) Manage an open position on this bar.
        if pos != 0:
            if pos > 0:
                if o[i] <= stop_px:
                    close_position(i, o[i], "stop")
                elif l[i] <= stop_px:
                    close_position(i, stop_px, "stop")
                elif o[i] >= target_px:
                    close_position(i, o[i], "target")
                elif h[i] >= target_px:
                    close_position(i, target_px, "target")
            else:
                if o[i] >= stop_px:
                    close_position(i, o[i], "stop")
                elif h[i] >= stop_px:
                    close_position(i, stop_px, "stop")
                elif o[i] <= target_px:
                    close_position(i, o[i], "target")
                elif l[i] <= target_px:
                    close_position(i, target_px, "target")
        if pos != 0:
            # Close-based exits (filled at this close for simplicity).
            if (pos > 0 and exit_long[i]) or (pos < 0 and exit_short[i]):
                close_position(i, c[i], "momentum")
            elif i - entry_i >= cfg.max_hold_bars:
                close_position(i, c[i], "time")

        equity[i] = cash + (pos * (c[i] - entry_px) * qty if pos else 0.0)

    if pos != 0:
        close_position(len(d) - 1, c[-1], "end_of_data")
        equity[-1] = cash

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity, index=idx, name="equity")
    return BacktestResult(trades_df, eq, _stats(trades_df, eq, bt.initial_capital))


def _stats(trades: pd.DataFrame, equity: pd.Series, initial: float) -> dict:
    dd = equity / equity.cummax() - 1
    stats = {
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round((equity.iloc[-1] / initial - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "trades": int(len(trades)),
    }
    if len(trades):
        wins = trades["pnl"] > 0
        gross_win = trades.loc[wins, "pnl"].sum()
        gross_loss = -trades.loc[~wins, "pnl"].sum()
        stats.update({
            "win_rate_pct": round(wins.mean() * 100, 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "avg_r": round(float(trades["r_multiple"].mean()), 2),
            "avg_bars_held": round(float(trades["bars_held"].mean()), 1),
            "by_setup": trades.groupby("setup")["pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
        })
    return stats
