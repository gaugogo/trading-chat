"""
core.py — Shared utilities for Trading Tools

Centralizes all duplicated functions across modules:
  - fmt_price, determine_trend, calculate_indicators, calculate_atr
  - Shared constants: TIMEFRAMES, TF_WEIGHTS, PRICE_COLS
  - Data fetching: fetch_all_timeframes, fetch_chart_data, fetch_spot_price, adjust_to_spot

Usage:
  from core import fmt_price, determine_trend, calculate_indicators, ...
"""

import os
import re
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import requests
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from instruments import INSTRUMENTS

warnings.filterwarnings('ignore')

# ─── LOGGING SETUP ───

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """Configure structured logging for the trading tools.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        log_format: Format string for log messages
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Setup default logging (can be reconfigured later)
setup_logging()
logger = get_logger(__name__)

# ─── PATHS ───

CACHE_DIR = Path(__file__).parent / "cache"


# ─── CONSTANTS ───

# Timeframe definitions for Yahoo Finance data fetching
TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "Daily": {"interval": "1d", "range": "1y", "cache_hours": 2},
    "4H":    {"interval": "1h", "range": "2mo", "resample": 14400, "cache_hours": 1},
    "1H":    {"interval": "1h", "range": "1mo", "cache_hours": 0.5},
    "15m":   {"interval": "15m", "range": "5d", "cache_hours": 0.25},
    "5m":    {"interval": "5m", "range": "2d", "cache_hours": 0.0833},
}

# Scoring weights per timeframe (used by confluence, scoring engines)
TF_WEIGHTS: Dict[str, float] = {
    "Daily": 5.0,
    "4H":    3.0,
    "1H":    2.0,
    "15m":   1.0,
    "5m":    0.5,
}

# Price columns that get adjusted when doing spot adjustment
PRICE_COLS = [
    'Open', 'High', 'Low', 'Close',
    'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_9', 'EMA_21',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B', 'Chikou',
]

# Default timeframe ordering (highest to lowest)
TF_ORDER = ["Daily", "4H", "1H", "15m", "5m"]


# ─── PRICE FORMATTING ───

def fmt_price(val: Any, decimals: int = 2) -> str:
    """Format a price value with $ sign and decimal places.

    Args:
        val: Price value (numeric, NaN, or None)
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string like '$1234.56' or 'N/A'
    """
    if val is None or (isinstance(val, float) and np.isnan(val)) or (isinstance(val, pd.Series) and val.isna().all()):
        return "N/A"
    if isinstance(val, pd.Series):
        val = float(val.iloc[0])
    return f"${float(val):.{decimals}f}"


def fmt_price_f(val: Any, decimals: int = 2) -> str:
    """Alias for fmt_price with different name for backward compatibility."""
    return fmt_price(val, decimals)


# ─── DATA FETCHING ───

def resample_manual(df: pd.DataFrame, period_seconds: int) -> pd.DataFrame:
    """Resample OHLCV data manually using integer division of timestamps.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        period_seconds: Target period in seconds (e.g., 14400 for 4H)

    Returns:
        Resampled DataFrame
    """
    ts = np.array([t.timestamp() for t in df.index])
    groups = ts.astype(np.int64) // period_seconds
    df2 = df.copy()
    df2["_g"] = groups
    result = (
        df2.groupby("_g")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    result.index = pd.to_datetime(result.index * period_seconds, unit="s")
    return result


def fetch_chart_data(
    symbol_encoded: str,
    interval: str,
    range_: str,
    resample: Optional[int] = None,
    use_cache: bool = True,
    cache_key: Optional[str] = None,
    cache_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data using DataProvider with automatic fallback + adaptive cache.

    Delegates to data_provider module which handles:
      - Multi-provider fallback (Yahoo -> Alpha Vantage)
      - Adaptive cache (reduced TTL during market hours)
      - Circuit breaker (auto-disable failing providers)
      - Retry with exponential backoff

    Args:
        symbol_encoded: URL-encoded Yahoo Finance symbol
        interval: Bar interval ('1d', '1h', '15m', '5m')
        range_: Data range ('1y', '2mo', '1mo', '5d', '2d')
        resample: If set, manually resample to this period (seconds)
        use_cache: Whether to use/save local cache
        cache_key: Unique key for cache file
        cache_hours: Cache expiry in hours (will be adapted by market session)

    Returns:
        DataFrame with OHLCV columns, or empty DataFrame on failure
    """
    logger.debug(f"fetch_chart_data: {cache_key} ({interval}/{range_}), cache={use_cache}")

    # Detect symbol type for adaptive cache
    symbol_type = "forex"
    if cache_key:
        # Try to infer from cache_key prefix (e.g., 'btc_Daily' -> crypto)
        instr_id = cache_key.split("_")[0] if "_" in cache_key else ""
        if instr_id in ("btc", "eth", "sol"):
            symbol_type = "crypto"

    from data_provider import fetch_ohlcv
    return fetch_ohlcv(
        symbol=symbol_encoded,
        interval=interval,
        range_=range_,
        resample_to=resample,
        use_cache=use_cache,
        cache_key=cache_key,
        cache_hours=cache_hours,
        symbol_type=symbol_type,
        use_fallback=True,
    )


def fetch_all_timeframes(
    cfg: Dict[str, Any],
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch data for all standard timeframes using DataProvider.

    Uses adaptive cache: TTL is automatically reduced during market hours
    and extended when markets are closed.

    Args:
        cfg: Instrument config dict from INSTRUMENTS
        use_cache: Whether to use cached data

    Returns:
        Dict mapping TF name -> DataFrame with indicators calculated
    """
    # Detect symbol type for market-aware caching
    instr_id = cfg.get("id", "xau")
    from data_provider import detect_symbol_type
    symbol_type = detect_symbol_type(instr_id)

    tf_data: Dict[str, pd.DataFrame] = {}
    logger.info(f"Fetching all TFs for {cfg.get('id', '?')}, cache={use_cache}")
    for tf_name, config in TIMEFRAMES.items():
        try:
            df = fetch_chart_data(
                cfg["symbol_encoded"],
                config["interval"],
                config["range"],
                config.get("resample"),
                use_cache=use_cache,
                cache_key=f"{cfg['id']}_{tf_name}",
                cache_hours=config.get("cache_hours"),
            )
            if not df.empty:
                tf_data[tf_name] = calculate_indicators(df)
        except Exception:
            continue
    return tf_data


# ─── SPOT PRICE ───

_SPOT_CACHE: Dict[str, Tuple[float, float]] = {}  # url -> (price, timestamp)
_SPOT_CACHE_TTL: float = 60.0  # seconds


def fetch_spot_price(url: str, instrument_id: str = "xau", symbol: str = "") -> Optional[float]:
    """Fetch spot price from multiple sources with fallback.

    Tries:
      1. Investing.com scraping (existing, fast)
      2. Yahoo Finance quote (via DataProvider)

    Uses in-memory cache (60s TTL) to avoid rate limits.

    Args:
        url: Investing.com URL for the instrument
        instrument_id: Instrument ID for cache key (e.g., 'xau')
        symbol: Yahoo Finance symbol for fallback quote

    Returns:
        Spot price as float, or None if unavailable
    """
    now = time.time()
    cached = _SPOT_CACHE.get(url)
    if cached and (now - cached[1]) < _SPOT_CACHE_TTL:
        return cached[0]

    # Source 1: Investing.com scraping
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=15,
        )
        patterns = [
            r'data-test="instrument-price-last">([\d,]+\.?\d*)',
            r'<span class="[^"]*text-[^"]*">([\d,]+\.?\d+)</span>',
            r'"last":\s*"?([\d.]+)',
        ]
        for p in patterns:
            m = re.search(p, r.text)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    _SPOT_CACHE[url] = (val, now)
                    return val
                except ValueError:
                    continue
    except Exception:
        pass

    # Source 2: Yahoo Finance quote (fallback)
    if symbol:
        try:
            from data_provider import get_factory
            factory = get_factory()
            provider = factory.get_provider("yahoo")
            quote = provider.fetch_quote(symbol, use_cache=True, cache_ttl_seconds=_SPOT_CACHE_TTL)
            if quote is not None:
                _SPOT_CACHE[url] = (quote, now)
                return quote
        except Exception:
            pass

    return None


def adjust_to_spot(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Adjust futures prices down to spot price if available.

    Calculates offset between futures close and spot price,
    then subtracts that offset from all price columns.

    Args:
        tf_data: Dict of timeframe DataFrames
        cfg: Instrument config

    Returns:
        Adjusted DataFrames (or originals if no spot available)
    """
    if not cfg.get("has_spot") or not cfg.get("spot_url"):
        return tf_data

    spot_price = fetch_spot_price(
        cfg["spot_url"],
        instrument_id=cfg.get("id", "xau"),
        symbol=cfg.get("symbol", ""),
    )
    if spot_price is None:
        return tf_data

    futures_close = None
    for tf_name in TF_ORDER:
        df = tf_data.get(tf_name)
        if df is not None and not df.empty:
            futures_close = df["Close"].iloc[-1]
            break

    if futures_close is None or pd.isna(futures_close):
        return tf_data

    offset = float(futures_close) - spot_price
    if abs(offset) < 0.01:
        return tf_data

    for tf_name, df in tf_data.items():
        for col in PRICE_COLS:
            if col in df.columns:
                df[col] = df[col] - offset

    return tf_data


# ─── INDICATORS ───

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range.

    Args:
        df: DataFrame with High, Low, Close columns
        window: ATR period (default: 14)

    Returns:
        Series with ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators on a DataFrame.

    Adds columns:
      SMA_20, SMA_50, SMA_200, EMA_9, EMA_21,
      RSI_14, MACD, MACD_Signal, MACD_Hist,
      BB_Upper, BB_Middle, BB_Lower, BB_Width,
      ATR, Volume_SMA_20

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added indicator columns
    """
    if df.empty or len(df) < 20:
        return df

    close = df['Close']

    df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    if len(df) >= 50:
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
    else:
        df['SMA_50'] = np.nan
    if len(df) >= 200:
        df['SMA_200'] = SMAIndicator(close, window=200).sma_indicator()
    else:
        df['SMA_200'] = np.nan

    df['EMA_9'] = EMAIndicator(close, window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(close, window=21).ema_indicator()

    df['RSI_14'] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    df['ATR'] = calculate_atr(df, 14)
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()

    return df


# ─── TREND DETERMINATION ───

def determine_trend(df: pd.DataFrame) -> Tuple[str, int]:
    """Determine trend direction and strength from indicators.

    Scoring scheme:
      +1 if Close > SMA20
      +1 if Close > SMA50
      +1 if RSI > 50
      +1 if MACD > MACD_Signal
      +1 if EMA9 > EMA21

    Result:
      score >= 2  → "UP"
      score <= -2 → "DOWN"
      otherwise   → "SIDEWAYS"

    Args:
        df: DataFrame with indicators calculated

    Returns:
        Tuple of (trend_label: str, score: int)
    """
    if df.empty or len(df) < 20:
        return "WAIT", 0

    last = df.iloc[-1]
    score = 0
    close_val = last['Close']

    sma20 = last.get('SMA_20', np.nan)
    if not pd.isna(sma20):
        score += 1 if close_val > sma20 else -1

    sma50 = last.get('SMA_50', np.nan)
    if not pd.isna(sma50):
        score += 1 if close_val > sma50 else -1

    rsi = last.get('RSI_14', np.nan)
    if not pd.isna(rsi):
        score += 1 if rsi > 50 else -1

    macd_val = last.get('MACD', np.nan)
    macd_sig = last.get('MACD_Signal', np.nan)
    if not pd.isna(macd_val) and not pd.isna(macd_sig):
        score += 1 if macd_val > macd_sig else -1

    ema9 = last.get('EMA_9', np.nan)
    ema21 = last.get('EMA_21', np.nan)
    if not pd.isna(ema9) and not pd.isna(ema21):
        score += 1 if ema9 > ema21 else -1

    if score >= 2:
        return "UP", score
    elif score <= -2:
        return "DOWN", score
    else:
        return "SIDEWAYS", score


# ─── CONFLUENCE ───

def build_confluence_summary(tf_data: Dict[str, pd.DataFrame]) -> str:
    """Build a one-line confluence summary across timeframes.

    Shows UP/DOWN counts, weighted score, and trend sequence.

    Args:
        tf_data: Dict of timeframe DataFrames

    Returns:
        Confluence summary string
    """
    trends: Dict[str, str] = {}
    for tf_name in TF_ORDER:
        if tf_name in tf_data and not tf_data[tf_name].empty:
            trend, _ = determine_trend(tf_data[tf_name])
            trends[tf_name] = trend

    up = sum(1 for v in trends.values() if v == "UP")
    down = sum(1 for v in trends.values() if v == "DOWN")

    trend_dir = {"UP": 1, "DOWN": -1, "SIDEWAYS": 0, "WAIT": 0}
    weighted_score = sum(
        trend_dir.get(trends.get(tf, "WAIT"), 0) * TF_WEIGHTS.get(tf, 1)
        for tf in TF_ORDER
        if tf in trends
    )

    seq = " → ".join(f"{k}({v})" for k, v in trends.items())
    return f"UP:{up} DOWN:{down} Weighted:{weighted_score:+.1f} Sequence:{seq}"
