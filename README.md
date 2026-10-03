# Swing Strategy: RSI + MACD + Volume Profile

A swing trading strategy that uses a **volume profile** (POC, VAH, VAL) to decide
*where* to trade, and **RSI + MACD** to decide *when*.

It comes in two versions that follow the same rules:

| Path | What it is |
|---|---|
| `pine/swing_rsi_macd_vrvp.pine` | TradingView Pine Script v6 strategy, with a visible-range volume profile histogram drawn on the chart |
| `swing/` | Python indicators, signal generator and backtester |
| `scripts/run_backtest.py` | Command-line backtest runner |
| `tests/` | pytest tests, including a check that signals never use future data |

## The rules

Volume profile levels (rows default 50, value area default 70%):
- **POC**: the price row with the most traded volume.
- **VAH / VAL**: the top and bottom of the value area. The value area starts at the
  POC and grows one row at a time toward the busier neighbouring row until it
  holds 70% of the volume.

| Setup | Side | Conditions (on bar close) | Stop | Target |
|---|---|---|---|---|
| **VAL bounce** | Long | Low tagged VAL (within 0.25 ATR) in the last 5 bars; close is back above VAL and below POC; RSI was ≤ 35 and is now rising; MACD crossed above its signal line in the last 3 bars | Lower of the recent low and VAL, minus 1 ATR | VAH (or POC) |
| **VAH breakout** | Long | Close crosses above VAH; RSI is between 50 and 75; MACD is above its signal line and above 0 | VAH − 1 ATR | 2R |
| **VAH rejection** | Short | The mirror of the VAL bounce | Higher of the recent high and VAH, plus 1 ATR | VAL (or POC) |
| **VAL breakdown** | Short | The mirror of the VAH breakout | VAL + 1 ATR | 2R |

Shorts are off by default. Trades also close when:
- MACD crosses against the position while RSI is stretched (≥ 65 for longs, ≤ 35 for shorts).
- The trade has been open for 30 bars.

Position size is set so a stop-out loses 1% of equity. Entries fill at the
next bar's open.

### "Visible range" and backtesting

TradingView's VRVP builds the profile from whatever bars are on screen. If a
backtest used that range, it would see future bars and give misleading results.
So the strategy offers two level sources:

- **Rolling** (default): the profile of the last 120 bars, recalculated on every
  bar. It never looks ahead, so it is the right choice for backtesting. The
  Python version uses this mode.
- **Visible range** (Pine only): a profile anchored at the left edge of the
  chart that builds up bar by bar toward the right edge. It never looks ahead
  within the range, but the backtest result changes whenever you scroll or zoom.

In both modes, the histogram on the right of the chart shows the true
visible-range profile, the same as TradingView's VRVP.

## Usage

### TradingView
1. Open the Pine Editor, paste `pine/swing_rsi_macd_vrvp.pine`, and click *Add to chart*.
2. Use a daily or 4h chart for swing trading. Adjust the inputs, then check the Strategy Tester.
3. The script has *Long signal* and *Short signal* alert conditions.

### Python
```bash
pip install -r requirements.txt
python scripts/run_backtest.py --synthetic             # demo on random data
python scripts/run_backtest.py --csv data/SPY.csv      # your own OHLCV CSV
pip install yfinance && python scripts/run_backtest.py --symbol SPY --start 2015-01-01 --shorts
python -m pytest
```

```python
from swing import StrategyConfig, BacktestConfig, backtest
from swing.data import load_csv

res = backtest(load_csv("data/SPY.csv"), StrategyConfig(reversion_target="poc"), BacktestConfig(risk_per_trade=0.005))
print(res.stats); res.trades.tail()
```

### Differences between the Pine and Python versions
- Python skips a trade if the next bar's open has already gapped past the stop or target. Pine fills the entry and then exits right away.
- Python fills momentum and time exits at the signal bar's close. Pine fills them at the next bar's open.
- Pine's volume profile uses the chart timeframe's bars. TradingView's built-in VRVP may use lower-timeframe data, so its levels can differ slightly.

> Educational code, not financial advice. The default parameters are reasonable
> starting points, not optimised values. Test them out of sample before trading
> real money.
