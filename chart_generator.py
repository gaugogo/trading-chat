"""
chart_generator.py — Chart Visualization với matplotlib

Tạo biểu đồ OHLCV kỹ thuật với:
  - Multi-timeframe chart (Daily, 4H, 1H, 15m, 5m)
  - Indicators: EMA, SMA, Bollinger Bands, RSI, MACD, Volume
  - Entry/SL/TP annotations
  - Support/Resistance levels
  - SMC zones (order blocks, FVG)
  - Divergence markers
  - Export PNG để học tập và kiểm tra

Usage:
  from chart_generator import (
      generate_chart, generate_multi_tf_chart,
      ChartConfig, CHART_DIR,
  )

  # Basic chart
  path = generate_chart(tf_data, "xau", "Daily")

  # Multi-TF overview
  paths = generate_multi_tf_chart(tf_data, "xau")
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from core import get_logger, TF_ORDER, fmt_price

logger = get_logger(__name__)


# ─── CONFIG ────────────────────────────────────────────────────────────────

CHART_DIR = Path(__file__).parent / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Default chart dimensions
CHART_WIDTH = 12
CHART_HEIGHT = 8
DPI = 150

# Color scheme
COLORS = {
    "bullish": "#26a69a",   # Green
    "bearish": "#ef5350",   # Red
    "neutral": "#78909c",   # Grey-blue
    "bg": "#1a1a2e",        # Dark background
    "grid": "#2d2d44",      # Grid lines
    "text": "#e0e0e0",      # Text
    "volume_bull": "#26a69a",
    "volume_bear": "#ef5350",
    "ema9": "#ffeb3b",      # Yellow
    "ema21": "#ff9800",     # Orange
    "sma20": "#42a5f5",     # Blue
    "sma50": "#ab47bc",     # Purple
    "sma200": "#f44336",    # Red
    "bb_upper": "#78909c",
    "bb_lower": "#78909c",
    "bb_mid": "#78909c",
    "entry": "#76ff03",     # Bright green for entry
    "sl": "#ff1744",        # Bright red for SL
    "tp": "#00e5ff",        # Cyan for TP
    "support": "#4caf50",   # Support levels
    "resistance": "#f44336", # Resistance levels
    "ob_bull": "rgba(38, 166, 154, 0.15)",  # Order block bullish
    "ob_bear": "rgba(239, 83, 80, 0.15)",   # Order block bearish
    "fvg_bull": "rgba(38, 166, 154, 0.1)",  # FVG bullish
    "fvg_bear": "rgba(239, 83, 80, 0.1)",   # FVG bearish
    "div_bull": "#76ff03",  # Bullish divergence
    "div_bear": "#ff1744",  # Bearish divergence
}


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: int = CHART_WIDTH
    height: int = CHART_HEIGHT
    dpi: int = DPI
    show_volume: bool = True
    show_rsi: bool = True
    show_macd: bool = True
    show_bb: bool = True
    show_sma: bool = True
    show_ema: bool = True
    show_support_resistance: bool = True
    dark_mode: bool = True
    max_bars: int = 200  # Max data points to display

    def apply_style(self):
        """Apply matplotlib style settings."""
        if self.dark_mode:
            plt.style.use("dark_background")
            plt.rcParams["figure.facecolor"] = "#1a1a2e"
            plt.rcParams["axes.facecolor"] = "#1a1a2e"
            plt.rcParams["axes.edgecolor"] = "#2d2d44"
            plt.rcParams["axes.labelcolor"] = "#e0e0e0"
            plt.rcParams["text.color"] = "#e0e0e0"
            plt.rcParams["xtick.color"] = "#78909c"
            plt.rcParams["ytick.color"] = "#78909c"
            plt.rcParams["grid.color"] = "#2d2d44"
            plt.rcParams["legend.facecolor"] = "#1a1a2e"
            plt.rcParams["legend.edgecolor"] = "#2d2d44"
        else:
            plt.style.use("default")


# ─── CANDLESTICK PLOTTING ────────────────────────────────────────────────

def plot_candlesticks(ax, df: pd.DataFrame, color_bull: str, color_bear: str):
    """Plot OHLC candlesticks on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        df: DataFrame with OHLC columns
        color_bull: Color for bullish candles
        color_bear: Color for bearish candles
    """
    idx = df.index.values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # Convert to matplotlib date numbers
    dates = mdates.date2num(pd.to_datetime(idx))

    # Width of each candlestick (in date units)
    if len(dates) > 1:
        width = (dates[-1] - dates[0]) / len(dates) * 0.6
    else:
        width = 0.5

    bull = closes >= opens
    bear = closes < opens

    # Plot bullish candles (green, hollow)
    ax.bar(
        dates[bull], highs[bull] - lows[bull], width,
        bottom=lows[bull], color=color_bull, edgecolor=color_bull,
        linewidth=0.5,
    )
    ax.bar(
        dates[bull], closes[bull] - opens[bull], width * 0.8,
        bottom=opens[bull], color=color_bull, edgecolor=color_bull,
        linewidth=0.5,
    )

    # Plot bearish candles (red, filled)
    ax.bar(
        dates[bear], highs[bear] - lows[bear], width,
        bottom=lows[bear], color=color_bear, edgecolor=color_bear,
        linewidth=0.5,
    )
    ax.bar(
        dates[bear], opens[bear] - closes[bear], width * 0.8,
        bottom=closes[bear], color=color_bear, edgecolor=color_bear,
        linewidth=0.5,
    )


# ─── INDICATOR PLOTTING ───────────────────────────────────────────────────

def plot_moving_averages(ax, df: pd.DataFrame, config: ChartConfig):
    """Plot SMA and EMA lines on the price axis.

    Args:
        ax: Matplotlib axis
        df: DataFrame with indicator columns
        config: Chart configuration
    """
    if config.show_ema:
        if "EMA_9" in df.columns:
            ax.plot(
                df.index, df["EMA_9"],
                color=COLORS["ema9"], linewidth=1.0, alpha=0.8,
                label="EMA 9",
            )
        if "EMA_21" in df.columns:
            ax.plot(
                df.index, df["EMA_21"],
                color=COLORS["ema21"], linewidth=1.0, alpha=0.8,
                label="EMA 21",
            )

    if config.show_sma:
        if "SMA_20" in df.columns:
            ax.plot(
                df.index, df["SMA_20"],
                color=COLORS["sma20"], linewidth=0.8, alpha=0.7, linestyle="--",
                label="SMA 20",
            )
        if "SMA_50" in df.columns:
            ax.plot(
                df.index, df["SMA_50"],
                color=COLORS["sma50"], linewidth=0.8, alpha=0.7, linestyle="--",
                label="SMA 50",
            )
        if "SMA_200" in df.columns:
            ax.plot(
                df.index, df["SMA_200"],
                color=COLORS["sma200"], linewidth=1.0, alpha=0.8,
                label="SMA 200",
            )


def plot_bollinger_bands(ax, df: pd.DataFrame, config: ChartConfig):
    """Plot Bollinger Bands on the price axis.

    Args:
        ax: Matplotlib axis
        df: DataFrame with BB columns
        config: Chart configuration
    """
    if not config.show_bb:
        return

    if all(c in df.columns for c in ["BB_Upper", "BB_Middle", "BB_Lower"]):
        ax.plot(
            df.index, df["BB_Upper"],
            color=COLORS["bb_upper"], linewidth=0.6, alpha=0.5,
            label="BB Upper",
        )
        ax.plot(
            df.index, df["BB_Middle"],
            color=COLORS["bb_mid"], linewidth=0.6, alpha=0.5,
            label="BB Middle",
        )
        ax.plot(
            df.index, df["BB_Lower"],
            color=COLORS["bb_lower"], linewidth=0.6, alpha=0.5,
            label="BB Lower",
        )
        # Fill between bands
        ax.fill_between(
            df.index, df["BB_Upper"], df["BB_Lower"],
            alpha=0.05, color=COLORS["bb_upper"],
        )


def plot_volume(ax, df: pd.DataFrame, config: ChartConfig):
    """Plot volume bars.

    Args:
        ax: Matplotlib axis
        df: DataFrame with Volume column
        config: Chart configuration
    """
    if not config.show_volume or "Volume" not in df.columns:
        return

    colors = np.where(
        df["Close"] >= df["Open"],
        COLORS["volume_bull"],
        COLORS["volume_bear"],
    )

    ax.bar(
        df.index, df["Volume"],
        color=colors, alpha=0.5, width=0.8,
    )
    ax.set_ylabel("Volume", color=COLORS["text"], alpha=0.7)


def plot_rsi(ax, df: pd.DataFrame, config: ChartConfig):
    """Plot RSI indicator.

    Args:
        ax: Matplotlib axis
        df: DataFrame with RSI column
        config: Chart configuration
    """
    if not config.show_rsi or "RSI" not in df.columns:
        return

    ax.plot(df.index, df["RSI"], color="#42a5f5", linewidth=1.0, label="RSI 14")
    ax.axhline(y=70, color=COLORS["bearish"], linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=30, color=COLORS["bullish"], linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=50, color=COLORS["neutral"], linestyle=":", alpha=0.2, linewidth=0.5)
    ax.fill_between(df.index, 70, 30, alpha=0.05, color=COLORS["neutral"])
    ax.set_ylabel("RSI", color=COLORS["text"], alpha=0.7)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8)


def plot_macd(ax, df: pd.DataFrame, config: ChartConfig):
    """Plot MACD indicator.

    Args:
        ax: Matplotlib axis
        df: DataFrame with MACD columns
        config: Chart configuration
    """
    if not config.show_macd:
        return

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        ax.plot(df.index, df["MACD"], color="#42a5f5", linewidth=1.0, label="MACD")
        ax.plot(
            df.index, df["MACD_Signal"],
            color="#ff9800", linewidth=0.8, alpha=0.8,
            label="Signal",
        )

        if "MACD_Histogram" in df.columns:
            hist = df["MACD_Histogram"]
            colors = np.where(hist >= 0, COLORS["bullish"], COLORS["bearish"])
            ax.bar(df.index, hist, color=colors, alpha=0.4, width=0.8)

        ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_ylabel("MACD", color=COLORS["text"], alpha=0.7)
        ax.legend(loc="upper left", fontsize=8)


# ─── ANNOTATIONS ──────────────────────────────────────────────────────────

def plot_trade_annotations(
    ax,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    label: str = "Plan",
):
    """Plot entry/SL/TP annotations on the chart.

    Args:
        ax: Matplotlib axis
        entry: Entry price
        sl: Stop loss price
        tp: Take profit price
        label: Label prefix for annotations
    """
    if entry is not None:
        ax.axhline(
            y=entry, color=COLORS["entry"],
            linestyle="--", linewidth=1.0, alpha=0.7,
        )
        ax.annotate(
            f"  {label} Entry: ${entry:.2f}",
            xy=(0, entry), xycoords=("axes fraction", "data"),
            color=COLORS["entry"], fontsize=8, alpha=0.9,
            va="center",
        )

    if sl is not None:
        ax.axhline(
            y=sl, color=COLORS["sl"],
            linestyle="--", linewidth=1.0, alpha=0.7,
        )
        ax.annotate(
            f"  {label} SL: ${sl:.2f}",
            xy=(0.02, sl), xycoords=("axes fraction", "data"),
            color=COLORS["sl"], fontsize=8, alpha=0.9,
            va="center",
        )

    if tp is not None:
        ax.axhline(
            y=tp, color=COLORS["tp"],
            linestyle="--", linewidth=1.0, alpha=0.7,
        )
        ax.annotate(
            f"  {label} TP: ${tp:.2f}",
            xy=(0.02, tp), xycoords=("axes fraction", "data"),
            color=COLORS["tp"], fontsize=8, alpha=0.9,
            va="center",
        )


def plot_key_levels(
    ax,
    support: Optional[List[float]] = None,
    resistance: Optional[List[float]] = None,
):
    """Plot support and resistance levels.

    Args:
        ax: Matplotlib axis
        support: List of support prices
        resistance: List of resistance prices
    """
    if support:
        for price in support:
            ax.axhline(
                y=price, color=COLORS["support"],
                linestyle=":", linewidth=0.6, alpha=0.4,
            )

    if resistance:
        for price in resistance:
            ax.axhline(
                y=price, color=COLORS["resistance"],
                linestyle=":", linewidth=0.6, alpha=0.4,
            )


def plot_smc_zones(
    ax,
    df: pd.DataFrame,
    order_blocks: Optional[List[Dict]] = None,
    fvgs: Optional[List[Dict]] = None,
):
    """Plot SMC zones (order blocks, FVGs) on the chart.

    Args:
        ax: Matplotlib axis
        df: Price DataFrame
        order_blocks: List of order block dicts with 'type', 'top', 'bottom', 'index'
        fvgs: List of FVG dicts with 'type', 'top', 'bottom', 'index'
    """
    if order_blocks:
        for ob in order_blocks:
            idx = ob.get("index", 0)
            if idx < 0 or idx >= len(df):
                continue
            top = ob.get("top", 0)
            bottom = ob.get("bottom", 0)
            ob_type = ob.get("type", "bullish")
            color = COLORS["ob_bull"] if ob_type == "bullish" else COLORS["ob_bear"]

            ax.axhspan(
                bottom, top,
                alpha=0.3, color=color,
                xmin=0.85, xmax=1.0,
            )

    if fvgs:
        for fvg in fvgs:
            idx = fvg.get("index", 0)
            if idx < 0 or idx >= len(df):
                continue
            top = fvg.get("top", 0)
            bottom = fvg.get("bottom", 0)
            fvg_type = fvg.get("type", "bullish")
            color = COLORS["fvg_bull"] if fvg_type == "bullish" else COLORS["fvg_bear"]

            ax.axhspan(
                bottom, top,
                alpha=0.4, color=color,
                xmin=0.85, xmax=1.0,
            )


# ─── MAIN CHART GENERATION ────────────────────────────────────────────────

def generate_chart(
    tf_data: Dict[str, pd.DataFrame],
    instrument: str = "xau",
    timeframe: str = "Daily",
    config: Optional[ChartConfig] = None,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
    order_blocks: Optional[List[Dict]] = None,
    fvgs: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate a technical chart with indicators and annotations.

    Args:
        tf_data: Dict of {timeframe: DataFrame}
        instrument: Instrument ID
        timeframe: Timeframe to plot ('Daily', '4H', '1H', '15m', '5m')
        config: Chart configuration (defaults to ChartConfig())
        entry: Optional entry price for annotation
        sl: Optional stop loss for annotation
        tp: Optional take profit for annotation
        support_levels: Optional support price levels
        resistance_levels: Optional resistance price levels
        order_blocks: Optional SMC order blocks
        fvgs: Optional SMC FVGs
        output_path: Optional output path (default: charts/{instrument}_{timeframe}.png)

    Returns:
        Path to generated PNG file, or None on failure
    """
    df = tf_data.get(timeframe)
    if df is None or df.empty:
        return None

    config = config or ChartConfig()
    config.apply_style()

    # Limit data points
    if len(df) > config.max_bars:
        df = df.iloc[-config.max_bars:]

    # Determine price direction for color
    last_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0])
    is_bullish = last_close >= first_close

    # Build figure with subplots
    n_subplots = 3 if (config.show_rsi and config.show_macd) else 2
    fig = plt.figure(figsize=(config.width, config.height))
    gs = fig.add_gridspec(
        n_subplots, 1,
        height_ratios=[3, 1, 1][:n_subplots],
        hspace=0.08,
    )

    # Price axis
    ax_price = fig.add_subplot(gs[0])
    plot_candlesticks(
        ax_price, df,
        COLORS["bullish"], COLORS["bearish"],
    )
    plot_moving_averages(ax_price, df, config)
    plot_bollinger_bands(ax_price, df, config)
    plot_key_levels(ax_price, support_levels, resistance_levels)
    plot_trade_annotations(ax_price, entry, sl, tp)
    plot_smc_zones(ax_price, df, order_blocks, fvgs)

    # Format price axis
    ax_price.set_ylabel("Price ($)", color=COLORS["text"], alpha=0.7)
    ax_price.legend(loc="upper left", fontsize=7, ncol=3)
    ax_price.grid(True, alpha=0.15)
    ax_price.tick_params(axis="x", labelbottom=False)

    # Volume axis (shared with price or separate)
    if config.show_volume and "Volume" in df.columns and n_subplots > 1:
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        plot_volume(ax_vol, df, config)
        ax_vol.grid(True, alpha=0.1)
        ax_vol.tick_params(axis="x", labelbottom=(n_subplots == 2))

    # RSI + MACD
    rsi_pos = 2 if config.show_volume else 1
    if config.show_rsi and config.show_macd and n_subplots == 3:
        # RSI
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        plot_rsi(ax_rsi, df, config)
        ax_rsi.grid(True, alpha=0.1)
        ax_rsi.tick_params(axis="x", labelbottom=False)

        # MACD
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
        plot_macd(ax_macd, df, config)
        ax_macd.grid(True, alpha=0.1)
    elif config.show_rsi and n_subplots == 2:
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        plot_rsi(ax_rsi, df, config)
        ax_rsi.grid(True, alpha=0.1)

    # Format x-axis dates
    ax_last = fig.get_axes()[-1]
    ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_last.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_last.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # Title
    from instruments import INSTRUMENTS
    cfg = INSTRUMENTS.get(instrument, {})
    display_name = cfg.get("display_name", instrument.upper())
    fig.suptitle(
        f"{display_name} — {timeframe} Chart",
        fontsize=14, fontweight="bold",
        color=COLORS["text"], y=0.98,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ha="center", fontsize=7, color=COLORS["neutral"], alpha=0.6,
    )

    # Save
    if output_path is None:
        output_path = CHART_DIR / f"{instrument}_{timeframe.lower()}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        dpi=config.dpi,
        bbox_inches="tight",
        facecolor=COLORS["bg"],
        edgecolor="none",
    )
    plt.close(fig)

    logger.info(f"[Chart] Saved: {output_path}")
    return output_path


def generate_multi_tf_chart(
    tf_data: Dict[str, pd.DataFrame],
    instrument: str = "xau",
    timeframes: Optional[List[str]] = None,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
) -> Dict[str, Path]:
    """Generate charts for multiple timeframes.

    Args:
        tf_data: Dict of {timeframe: DataFrame}
        instrument: Instrument ID
        timeframes: List of timeframes (default: ["Daily", "4H", "1H", "15m"])
        entry: Optional entry price
        sl: Optional stop loss
        tp: Optional take profit

    Returns:
        Dict of {timeframe: Path} with generated chart paths
    """
    timeframes = timeframes or ["Daily", "4H", "1H", "15m"]
    results: Dict[str, Path] = {}

    for tf in timeframes:
        if tf in tf_data and not tf_data[tf].empty:
            path = generate_chart(
                tf_data, instrument, tf,
                entry=entry, sl=sl, tp=tp,
            )
            if path:
                results[tf] = path

    return results


def generate_signal_chart(
    tf_data: Dict[str, pd.DataFrame],
    instrument: str = "xau",
    signal_data: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Generate a signal-focused chart with the best timeframe.

    Picks the timeframe with the clearest signal setup.

    Args:
        tf_data: Dict of {timeframe: DataFrame}
        instrument: Instrument ID
        signal_data: Optional dict with 'entry', 'sl', 'tp', 'bias'

    Returns:
        Path to generated chart, or None
    """
    # Prefer 4H or 1H for signal charts
    for tf in ["4H", "1H", "Daily", "15m"]:
        if tf in tf_data and not tf_data[tf].empty:
            break
    else:
        return None

    entry = None
    sl = None
    tp = None
    if signal_data:
        entry = signal_data.get("entry")
        sl = signal_data.get("sl")
        tp = signal_data.get("tp")

    return generate_chart(
        tf_data, instrument, tf,
        entry=entry, sl=sl, tp=tp,
    )


# ─── CONVENIENCE FUNCTION ─────────────────────────────────────────────────

def generate_chart_report(
    tf_data: Dict[str, pd.DataFrame],
    instrument: str = "xau",
) -> str:
    """Generate charts and return a markdown report with image links.

    Args:
        tf_data: Dict of {timeframe: DataFrame}
        instrument: Instrument ID

    Returns:
        Markdown string with chart references
    """
    from instruments import INSTRUMENTS
    cfg = INSTRUMENTS.get(instrument, {})
    display_name = cfg.get("display_name", instrument.upper())

    paths = generate_multi_tf_chart(tf_data, instrument)

    lines = [f"📈 **{display_name} — Chart Analysis**", ""]
    for tf, path in sorted(paths.items()):
        rel_path = path.relative_to(Path(__file__).parent)
        lines.append(f"  **{tf}**: ![Chart]({rel_path})")

    return "\n".join(lines)


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    instrument = sys.argv[1] if len(sys.argv) > 1 else "xau"

    # Fetch data
    from core import fetch_all_timeframes
    from instruments import INSTRUMENTS

    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        print(f"Unknown instrument: {instrument}")
        sys.exit(1)

    print(f"Generating charts for {cfg['display_name']}...")
    tf_data = fetch_all_timeframes(cfg, use_cache=True)

    if not tf_data:
        print("No data available")
        sys.exit(1)

    paths = generate_multi_tf_chart(tf_data, instrument)
    print(f"\nGenerated {len(paths)} charts:")
    for tf, path in sorted(paths.items()):
        print(f"  {tf}: {path}")
