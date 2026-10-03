#!/usr/bin/env python3
"""Run the RSI + MACD + volume-profile swing strategy backtest.

Examples:
    python scripts/run_backtest.py --synthetic
    python scripts/run_backtest.py --csv data/SPY.csv
    python scripts/run_backtest.py --symbol AAPL --start 2019-01-01 --shorts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing import BacktestConfig, StrategyConfig, backtest  # noqa: E402
from swing.data import load_csv, load_yahoo, synthetic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV with date, open, high, low, close, volume")
    src.add_argument("--symbol", help="Ticker to download via yfinance")
    src.add_argument("--synthetic", action="store_true", help="Use generated demo data")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end")
    p.add_argument("--interval", default="1d")
    p.add_argument("--shorts", action="store_true", help="Enable short setups")
    p.add_argument("--vp-lookback", type=int, default=120)
    p.add_argument("--vp-rows", type=int, default=50)
    p.add_argument("--target", choices=["poc", "vah"], default="vah", help="Reversion target")
    p.add_argument("--risk", type=float, default=0.01, help="Fraction of equity risked per trade")
    p.add_argument("--trades-out", help="Write the trade list to this CSV")
    args = p.parse_args()

    if args.csv:
        df = load_csv(args.csv)
    elif args.symbol:
        df = load_yahoo(args.symbol, args.start, args.end, args.interval)
    else:
        df = synthetic()

    cfg = StrategyConfig(
        enable_shorts=args.shorts,
        vp_lookback=args.vp_lookback,
        vp_rows=args.vp_rows,
        reversion_target=args.target,
    )
    res = backtest(df, cfg, BacktestConfig(risk_per_trade=args.risk))

    print(json.dumps(res.stats, indent=2, default=str))
    if len(res.trades):
        cols = ["entry_time", "side", "setup", "entry", "exit", "pnl", "r_multiple", "exit_reason"]
        print("\nLast 10 trades:")
        print(res.trades[cols].tail(10).round({"entry": 2, "exit": 2, "pnl": 2, "r_multiple": 2}).to_string(index=False))
    if args.trades_out:
        res.trades.to_csv(args.trades_out, index=False)
        print(f"\nTrades written to {args.trades_out}")


if __name__ == "__main__":
    main()
