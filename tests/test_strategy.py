import numpy as np
import pandas as pd
import pytest

from swing import BacktestConfig, StrategyConfig, backtest, generate_signals, macd, rsi, volume_profile
from swing.data import synthetic


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert rsi(up).dropna().eq(100).all()
    s = rsi(synthetic(300)["close"]).dropna()
    assert s.between(0, 100).all()


def test_macd_hist_is_line_minus_signal():
    m = macd(synthetic(200)["close"])
    np.testing.assert_allclose(m["hist"], m["macd"] - m["signal"])


def test_volume_profile_poc_and_value_area():
    # Most of the volume trades between 100 and 101.
    high = np.array([101, 101, 101, 110, 95.5])
    low = np.array([100, 100, 100, 109, 95.0])
    vol = np.array([1000, 1000, 1000, 10, 10], dtype=float)
    vp = volume_profile(high, low, vol, rows=30, value_area_pct=0.7)
    assert 100 <= vp.poc <= 101
    assert vp.val >= 99.5 and vp.vah <= 101.5
    assert vp.val <= vp.poc <= vp.vah
    assert vp.row_volume.sum() == pytest.approx(vol.sum())


def test_value_area_holds_requested_share():
    df = synthetic(200)
    vp = volume_profile(df["high"], df["low"], df["volume"], rows=40, value_area_pct=0.7)
    edges, rv = vp.row_edges, vp.row_volume
    inside = rv[(edges[:-1] >= vp.val - 1e-9) & (edges[1:] <= vp.vah + 1e-9)].sum()
    assert inside / rv.sum() >= 0.7


def test_signals_do_not_use_future_data():
    df = synthetic(600)
    cfg = StrategyConfig(enable_shorts=True)
    full = generate_signals(df, cfg)
    cut = generate_signals(df.iloc[:450], cfg)
    cols = ["poc", "vah", "val", "signal", "stop", "target"]
    pd.testing.assert_frame_equal(full[cols].iloc[:450], cut[cols])


def test_signal_levels_are_consistent():
    d = generate_signals(synthetic(1500), StrategyConfig(enable_shorts=True))
    longs, shorts = d[d["signal"] == 1], d[d["signal"] == -1]
    assert len(longs) > 0
    assert (longs["stop"] < longs["close"]).all()
    assert (shorts["stop"] > shorts["close"]).all()


def test_backtest_risk_and_accounting():
    df = synthetic(1500)
    res = backtest(df, StrategyConfig(), BacktestConfig(commission_pct=0, slippage_pct=0))
    t = res.trades
    assert len(t) > 0
    assert res.equity.iloc[-1] == pytest.approx(100_000 + t["pnl"].sum())
    stopped = t[t["exit_reason"] == "stop"]
    # A stop-out without gaps should lose ~1R (1% of equity).
    assert (stopped["r_multiple"] >= -1.5).all()
    assert (t["entry_time"] <= t["exit_time"]).all()
    assert (t["bars_held"] <= StrategyConfig().max_hold_bars).all()
