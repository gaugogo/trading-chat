"""
backtest.py — Historical Backtesting Module for Trading Signals

Validates signal accuracy by running strategies on historical data.
Reports: win rate, profit factor, Sharpe ratio, max drawdown.

Usage:
  from backtest import BacktestEngine
  engine = BacktestEngine(instrument="xau", strategy="swing")
  results = engine.run()
  print(results.summary())

Or CLI:
  python backtest.py --instrument xau -- strategy swing --period 6m
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import argparse

from core import (
    fmt_price,
    fetch_all_timeframes,
    calculate_indicators,
    determine_trend,
    TF_ORDER,
)
from instruments import INSTRUMENTS


# ─── DATA CLASSES ───

@dataclass
class TradeRecord:
    """Record of a single backtest trade."""
    entry_date: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rr_ratio: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_points: Optional[float] = None
    result: Optional[str] = None  # 'WIN', 'LOSS', 'BREAKEVEN'
    reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Aggregated backtest statistics."""
    instrument: str
    strategy: str
    period_start: datetime
    period_end: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary report."""
        def _fmt_date(d):
            if hasattr(d, 'strftime'):
                return d.strftime('%Y-%m-%d')
            return str(d)[:10]

        lines = []
        lines.append("=" * 60)
        lines.append(f"  📊 BACKTEST RESULTS — {self.instrument.upper()} ({self.strategy})")
        lines.append(f"  Period: {_fmt_date(self.period_start)} → {_fmt_date(self.period_end)}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Total Trades:    {self.total_trades}")
        lines.append(f"  Win Rate:        {self.win_rate:.1f}%")
        lines.append(f"  Profit Factor:   {self.profit_factor:.2f}")
        lines.append(f"  Net P&L:         {self.total_pnl_pct:+.2f}%")
        lines.append(f"  Avg Win:         {self.avg_win_pct:+.2f}%")
        lines.append(f"  Avg Loss:        {self.avg_loss_pct:.2f}%")
        lines.append(f"  Max Drawdown:    {self.max_drawdown_pct:.2f}%")
        lines.append(f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}")
        lines.append(f"  Avg Bars Held:   {self.avg_bars_held:.1f}")
        lines.append("")
        lines.append(f"  Breakdown: {self.winning_trades}W / {self.losing_trades}L / {self.breakeven_trades}BE")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export results as dictionary (for JSON serialization)."""
        return {
            "instrument": self.instrument,
            "strategy": self.strategy,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": self.win_rate,
            "total_pnl_pct": self.total_pnl_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_bars_held": self.avg_bars_held,
        }


# ─── SIGNAL GENERATORS ───

class BaseSignalGenerator:
    """Base class for strategy signal generators."""

    def __init__(self, tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]):
        self.tf_data = tf_data
        self.cfg = cfg
        self.decimals = cfg.get("decimals", 2)

    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate list of signal dicts. Override in subclass."""
        raise NotImplementedError


class SwingSignalGenerator(BaseSignalGenerator):
    """SWING strategy signal generator for backtesting."""

    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate swing signals from 4H data."""
        h4 = self.tf_data.get("4H")
        if h4 is None or h4.empty:
            return []

        signals = []
        df = h4.copy()

        # Calculate indicators if not already present
        if 'RSI_14' not in df.columns:
            df = calculate_indicators(df)

        for i in range(50, len(df)):
            window = df.iloc[:i+1]
            current = window.iloc[-1]
            prev = window.iloc[-2] if len(window) > 1 else current

            # Get trend from higher timeframe (Daily)
            daily = self.tf_data.get("Daily")
            daily_trend = "SIDEWAYS"
            if daily is not None and not daily.empty:
                daily_trend, _ = determine_trend(daily)

            price = float(current["Close"])
            rsi = float(current.get("RSI_14", 50))
            macd = float(current.get("MACD", 0))
            macd_sig = float(current.get("MACD_Signal", 0))

            # Swing BUY signal
            if (daily_trend == "UP" and
                rsi > 40 and rsi < 60 and
                macd > macd_sig and
                price > float(current.get("SMA_20", 0))):

                atr = float(current.get("ATR", price * 0.01))
                sl = price - atr * 2.0
                tp1 = price + atr * 2.5
                tp2 = price + atr * 5.0
                risk = price - sl
                rr1 = (tp1 - price) / risk if risk > 0 else 0

                signals.append({
                    "date": current.name,
                    "price": price,
                    "direction": "LONG",
                    "stop_loss": sl,
                    "take_profit": tp1,
                    "rr": rr1,
                    "reason": f"Daily UP, RSI={rsi:.1f}, MACD bullish",
                    "atr": atr,
                })

            # Swing SELL signal
            elif (daily_trend == "DOWN" and
                  rsi > 40 and rsi < 60 and
                  macd < macd_sig and
                  price < float(current.get("SMA_20", price * 1.01))):

                atr = float(current.get("ATR", price * 0.01))
                sl = price + atr * 2.0
                tp1 = price - atr * 2.5
                rr1 = (price - tp1) / (sl - price) if (sl - price) > 0 else 0

                signals.append({
                    "date": current.name,
                    "price": price,
                    "direction": "SHORT",
                    "stop_loss": sl,
                    "take_profit": tp1,
                    "rr": rr1,
                    "reason": f"Daily DOWN, RSI={rsi:.1f}, MACD bearish",
                    "atr": atr,
                })

        return signals


class PositionSignalGenerator(BaseSignalGenerator):
    """POSITION strategy signal generator for backtesting."""

    def generate_signals(self) -> List[Dict[str, Any]]:
        daily = self.tf_data.get("Daily")
        if daily is None or daily.empty or len(daily) < 60:
            return []

        signals = []
        df = daily.copy()

        if 'RSI_14' not in df.columns:
            df = calculate_indicators(df)

        for i in range(60, len(df)):
            window = df.iloc[:i+1]
            current = window.iloc[-1]
            price = float(current["Close"])

            # Golden Cross (SMA50 > SMA200)
            sma50 = float(current.get("SMA_50", np.nan))
            sma200 = float(current.get("SMA_200", np.nan))
            sma50_prev = float(window["SMA_50"].iloc[-2]) if not pd.isna(window["SMA_50"].iloc[-2]) else sma50
            sma200_prev = float(window["SMA_200"].iloc[-2]) if not pd.isna(window["SMA_200"].iloc[-2]) else sma200

            if (not pd.isna(sma50) and not pd.isna(sma200) and
                sma50_prev <= sma200_prev and sma50 > sma200):
                # Golden Cross detected
                atr = float(current.get("ATR", price * 0.02))
                signals.append({
                    "date": current.name,
                    "price": price,
                    "direction": "LONG",
                    "stop_loss": price - atr * 3.0,
                    "take_profit": price + atr * 8.0,
                    "rr": 8.0 / 3.0,
                    "reason": "Golden Cross (SMA50 > SMA200)",
                    "atr": atr,
                })

            # Death Cross
            if (not pd.isna(sma50) and not pd.isna(sma200) and
                sma50_prev >= sma200_prev and sma50 < sma200):
                atr = float(current.get("ATR", price * 0.02))
                signals.append({
                    "date": current.name,
                    "price": price,
                    "direction": "SHORT",
                    "stop_loss": price + atr * 3.0,
                    "take_profit": price - atr * 8.0,
                    "rr": 8.0 / 3.0,
                    "reason": "Death Cross (SMA50 < SMA200)",
                    "atr": atr,
                })

        return signals


class DaytradeSignalGenerator(BaseSignalGenerator):
    """DAY TRADE strategy signal generator for backtesting."""

    def generate_signals(self) -> List[Dict[str, Any]]:
        h4 = self.tf_data.get("4H")
        h1 = self.tf_data.get("1H")
        m15 = self.tf_data.get("15m")

        if h1 is None or h1.empty:
            return []

        signals = []
        df = h1.copy()

        if 'RSI_14' not in df.columns:
            df = calculate_indicators(df)

        h4_trend, _ = determine_trend(h4) if h4 is not None and not h4.empty else ("SIDEWAYS", 0)

        for i in range(30, len(df)):
            window = df.iloc[:i+1]
            current = window.iloc[-1]
            price = float(current["Close"])
            rsi = float(current.get("RSI_14", 50))
            macd = float(current.get("MACD", 0))
            macd_sig = float(current.get("MACD_Signal", 0))
            bb_lower = float(current.get("BB_Lower", price * 0.99))
            bb_upper = float(current.get("BB_Upper", price * 1.01))

            # EMA ribbon
            ema9 = float(current.get("EMA_9", price))
            ema21 = float(current.get("EMA_21", price))

            atr = float(current.get("ATR", price * 0.003))

            # BUY: H4 bullish + price near BB lower + RSI bounce + MACD turning
            if (h4_trend != "DOWN" and
                price <= bb_lower * 1.01 and
                rsi < 45 and
                macd > macd_sig):
                sl = price - atr * 1.5
                tp = price + atr * 2.0
                risk = price - sl
                rr = (tp - price) / risk if risk > 0 else 0
                if rr >= 1.0:
                    signals.append({
                        "date": current.name,
                        "price": price,
                        "direction": "LONG",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "rr": round(rr, 1),
                        "reason": f"BB lower bounce + RSI={rsi:.1f} + MACD cross",
                        "atr": atr,
                    })

            # SELL: H4 bearish + price near BB upper + RSI overbought
            elif (h4_trend != "UP" and
                  price >= bb_upper * 0.99 and
                  rsi > 55 and
                  macd < macd_sig):
                sl = price + atr * 1.5
                tp = price - atr * 2.0
                risk = sl - price
                rr = (price - tp) / risk if risk > 0 else 0
                if rr >= 1.0:
                    signals.append({
                        "date": current.name,
                        "price": price,
                        "direction": "SHORT",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "rr": round(rr, 1),
                        "reason": f"BB upper rejection + RSI={rsi:.1f} + MACD bearish",
                        "atr": atr,
                    })

        return signals


# ─── BACKTEST ENGINE ───

class BacktestEngine:
    """Main backtesting engine.

    Args:
        instrument: Instrument ID (e.g., 'xau', 'btc', 'gbp')
        strategy: Strategy name ('swing', 'position', 'daytrade')
        initial_capital: Starting capital in % (default: 100.0)
        risk_per_trade: Risk per trade as % of capital (default: 1.0)
        commission_pct: Round-trip commission as % (default: 0.05)
        max_concurrent: Max open positions (default: 1)
    """

    SIGNAL_GENERATORS = {
        "swing": SwingSignalGenerator,
        "position": PositionSignalGenerator,
        "daytrade": DaytradeSignalGenerator,
    }

    def __init__(
        self,
        instrument: str = "xau",
        strategy: str = "swing",
        initial_capital: float = 100.0,
        risk_per_trade: float = 1.0,
        commission_pct: float = 0.05,
        max_concurrent: int = 1,
    ):
        if instrument not in INSTRUMENTS:
            raise ValueError(f"Unknown instrument: {instrument}. Choose: {list(INSTRUMENTS.keys())}")

        self.cfg = INSTRUMENTS[instrument]
        self.instrument = instrument
        self.strategy = strategy.lower()

        if self.strategy not in self.SIGNAL_GENERATORS:
            raise ValueError(f"Unknown strategy: {strategy}. Choose: {list(self.SIGNAL_GENERATORS.keys())}")

        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission_pct = commission_pct
        self.max_concurrent = max_concurrent

        self.tf_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self.result: Optional[BacktestResult] = None

    def run(self, use_cache: bool = True) -> BacktestResult:
        """Run the full backtest.

        Args:
            use_cache: Whether to use cached market data

        Returns:
            BacktestResult with all statistics
        """
        # 1. Fetch data
        self.tf_data = fetch_all_timeframes(self.cfg, use_cache=use_cache)

        if not self.tf_data:
            raise RuntimeError("No data fetched. Check connection.")

        # Determine date range from data
        period_start = None
        period_end = None
        for tf_name in TF_ORDER:
            df = self.tf_data.get(tf_name)
            if df is not None and not df.empty:
                if period_start is None or df.index[0] < period_start:
                    period_start = df.index[0]
                if period_end is None or df.index[-1] > period_end:
                    period_end = df.index[-1]

        if period_start is None:
            raise RuntimeError("Could not determine date range from data.")

        # 2. Generate signals
        generator_class = self.SIGNAL_GENERATORS[self.strategy]
        generator = generator_class(self.tf_data, self.cfg)
        signals = generator.generate_signals()

        # 3. Simulate trades
        self._simulate_trades(signals)

        # 4. Calculate statistics
        result = self._calculate_stats(period_start, period_end)
        self.result = result
        return result

    def _simulate_trades(self, signals: List[Dict[str, Any]]) -> None:
        """Simulate trades from signals using bar-by-bar walk-forward.

        For simplicity, this simulates market-on-close entry/exit.
        A more advanced version would check specific bar levels.
        """
        if not signals:
            self.trades = []
            return

        # Use 4H timeframe for swing, 1H for daytrade, Daily for position
        if self.strategy == "swing":
            tf_name = "4H"
        elif self.strategy == "daytrade":
            tf_name = "1H"
        else:
            tf_name = "Daily"

        df = self.tf_data.get(tf_name)
        if df is None or df.empty:
            return

        self.trades = []
        capital = self.initial_capital
        open_trades: List[TradeRecord] = []

        for idx, (bar_date, bar) in enumerate(df.iterrows()):
            bar_price = float(bar["Close"])

            # Close trades that hit SL/TP or that should exit
            remaining_open = []
            for trade in open_trades:
                closed = False

                # Check stop loss
                if trade.direction == "LONG" and trade.stop_loss is not None:
                    if float(bar["Low"]) <= trade.stop_loss:
                        trade.exit_date = bar_date
                        trade.exit_price = trade.stop_loss
                        trade.result = "LOSS"
                        trade.reason = "Stop loss hit"
                        closed = True

                elif trade.direction == "SHORT" and trade.stop_loss is not None:
                    if float(bar["High"]) >= trade.stop_loss:
                        trade.exit_date = bar_date
                        trade.exit_price = trade.stop_loss
                        trade.result = "LOSS"
                        trade.reason = "Stop loss hit"
                        closed = True

                # Check take profit
                if not closed and trade.take_profit is not None:
                    if trade.direction == "LONG" and float(bar["High"]) >= trade.take_profit:
                        trade.exit_date = bar_date
                        trade.exit_price = trade.take_profit
                        trade.result = "WIN"
                        trade.reason = "Take profit hit"
                        closed = True
                    elif trade.direction == "SHORT" and float(bar["Low"]) <= trade.take_profit:
                        trade.exit_date = bar_date
                        trade.exit_price = trade.take_profit
                        trade.result = "WIN"
                        trade.reason = "Take profit hit"
                        closed = True

                if closed:
                    self._finalize_trade(trade)
                    self.trades.append(trade)
                    capital += trade.pnl_pct or 0
                else:
                    remaining_open.append(trade)

            open_trades = remaining_open

            # Check for new signals at this bar
            for signal in signals:
                signal_date = signal["date"]
                if isinstance(signal_date, pd.Timestamp):
                    signal_date = signal_date.to_pydatetime()

                if isinstance(bar_date, pd.Timestamp):
                    bar_date_py = bar_date.to_pydatetime()
                else:
                    bar_date_py = bar_date

                # Enter trade on close of signal bar
                if signal_date == bar_date_py and len(open_trades) < self.max_concurrent:
                    trade = TradeRecord(
                        entry_date=bar_date,
                        entry_price=bar_price,
                        direction=signal["direction"],
                        stop_loss=signal["stop_loss"],
                        take_profit=signal["take_profit"],
                        rr_ratio=signal.get("rr"),
                        notes=[signal.get("reason", "")],
                    )
                    open_trades.append(trade)

            # Track equity
            self.equity_curve.append(capital)

        # Close any remaining open trades at last price
        for trade in open_trades:
            trade.exit_date = df.index[-1]
            trade.exit_price = float(df["Close"].iloc[-1])
            self._finalize_trade(trade)
            self.trades.append(trade)
            capital += trade.pnl_pct or 0
            self.equity_curve[-1] = capital

    def _finalize_trade(self, trade: TradeRecord) -> None:
        """Calculate PnL for a closed trade."""
        if trade.exit_price is None or trade.entry_price is None:
            return

        if trade.direction == "LONG":
            trade.pnl_points = trade.exit_price - trade.entry_price
            trade.pnl_pct = (trade.pnl_points / trade.entry_price) * 100
        else:  # SHORT
            trade.pnl_points = trade.entry_price - trade.exit_price
            trade.pnl_pct = (trade.pnl_points / trade.entry_price) * 100

        # Apply commission
        comm = self.commission_pct * 2  # entry + exit
        trade.pnl_pct = (trade.pnl_pct or 0) - comm

        # Determine result
        if abs(trade.pnl_pct or 0) < 0.1:
            trade.result = "BREAKEVEN"
        elif (trade.pnl_pct or 0) > 0:
            trade.result = "WIN"
        else:
            trade.result = "LOSS"

    def _calculate_stats(self, period_start: datetime, period_end: datetime) -> BacktestResult:
        """Calculate summary statistics from completed trades."""
        result = BacktestResult(
            instrument=self.instrument,
            strategy=self.strategy,
            period_start=period_start,
            period_end=period_end,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

        if not self.trades:
            return result

        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.result == "WIN")
        result.losing_trades = sum(1 for t in self.trades if t.result == "LOSS")
        result.breakeven_trades = sum(1 for t in self.trades if t.result == "BREAKEVEN")

        if result.total_trades > 0:
            result.win_rate = (result.winning_trades / result.total_trades) * 100

        wins = [t.pnl_pct or 0 for t in self.trades if t.result == "WIN"]
        losses = [t.pnl_pct or 0 for t in self.trades if t.result == "LOSS"]

        if wins:
            result.avg_win_pct = np.mean(wins)
        if losses:
            result.avg_loss_pct = abs(np.mean(losses))

        total_gross_profit = sum(wins)
        total_gross_loss = abs(sum(losses))
        result.total_pnl_pct = sum(t.pnl_pct or 0 for t in self.trades)

        if total_gross_loss > 0:
            result.profit_factor = total_gross_profit / total_gross_loss

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd

        # Sharpe ratio (using daily returns approximation)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Average bars held
        held_bars = []
        for t in self.trades:
            if t.entry_date and t.exit_date:
                td = t.exit_date - t.entry_date
                held_bars.append(td.total_seconds() / 3600)  # hours
        if held_bars:
            result.avg_bars_held = np.mean(held_bars)

        return result

    def export_trades(self, path: Optional[str] = None) -> str:
        """Export trades to JSON file.

        Args:
            path: Output path (default: 'backtest_{instrument}_{strategy}.json')

        Returns:
            Path to exported file
        """
        if not self.trades:
            return "No trades to export."

        data = []
        for t in self.trades:
            data.append({
                "entry_date": t.entry_date.isoformat() if hasattr(t.entry_date, 'isoformat') else str(t.entry_date),
                "entry_price": t.entry_price,
                "direction": t.direction,
                "exit_date": t.exit_date.isoformat() if t.exit_date and hasattr(t.exit_date, 'isoformat') else str(t.exit_date),
                "exit_price": t.exit_price,
                "result": t.result,
                "pnl_pct": t.pnl_pct,
                "reason": t.notes[0] if t.notes else "",
            })

        if path is None:
            path = f"backtest_{self.instrument}_{self.strategy}.json"

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description="Backtest Trading Strategies")
    parser.add_argument("--instrument", "-i", default="xau", choices=list(INSTRUMENTS.keys()),
                        help="Trading instrument")
    parser.add_argument("--strategy", "-s", default="swing",
                        choices=["swing", "position", "daytrade"],
                        help="Trading strategy")
    parser.add_argument("--risk", "-r", type=float, default=1.0,
                        help="Risk per trade (%)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass local cache")
    parser.add_argument("--export", "-o", action="store_true",
                        help="Export trades to JSON")

    args = parser.parse_args()

    print(f"🔄 Backtesting {args.instrument.upper()} using {args.strategy} strategy...")
    print(f"   Risk: {args.risk}% per trade | Cache: {'OFF' if args.no_cache else 'ON'}")

    try:
        engine = BacktestEngine(
            instrument=args.instrument,
            strategy=args.strategy,
            risk_per_trade=args.risk,
        )
        result = engine.run(use_cache=not args.no_cache)
        print()
        print(result.summary())

        if args.export and engine.trades:
            path = engine.export_trades()
            print(f"\n📁 Trades exported to: {path}")

    except (ValueError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
