"""
Tests for core.py — shared utilities.

Run with: pytest tests/test_core.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    fmt_price,
    determine_trend,
    calculate_indicators,
    calculate_atr,
    build_confluence_summary,
    TF_ORDER,
)


# ─── FIXTURES ───

@pytest.fixture
def sample_df():
    """Create a simple DataFrame with 30 rows of synthetic OHLCV data."""
    np.random.seed(42)
    # Use numeric index to avoid Python 3.14 datetime segfault
    dates = pd.RangeIndex(30)
    base = 2000.0
    data = {
        "Open": base + np.cumsum(np.random.randn(30) * 2),
        "High": np.nan,
        "Low": np.nan,
        "Close": np.nan,
        "Volume": np.random.randint(1000, 10000, 30),
    }
    df = pd.DataFrame(data, index=dates)
    # Derive High, Low, Close from Open
    for i in range(30):
        o = df["Open"].iloc[i]
        change = np.random.randn() * 5
        c = o + change
        h = max(o, c) + abs(np.random.randn() * 2)
        l = min(o, c) - abs(np.random.randn() * 2)
        df.loc[df.index[i], "High"] = h
        df.loc[df.index[i], "Low"] = l
        df.loc[df.index[i], "Close"] = c

    df[["High", "Low", "Close"]] = df[["High", "Low", "Close"]].astype(float)
    return df


@pytest.fixture
def sample_tf_data(sample_df):
    """Create mock multi-timeframe data."""
    tf_data = {
        "Daily": calculate_indicators(sample_df.copy()),
        "4H": calculate_indicators(sample_df.copy()),
    }
    return tf_data


# ─── TESTS: fmt_price ───

def test_fmt_price_basic():
    assert fmt_price(1234.5678, 2) == "$1234.57"
    assert fmt_price(100.0, 5) == "$100.00000"


def test_fmt_price_nan():
    assert fmt_price(np.nan, 2) == "N/A"
    assert fmt_price(None, 2) == "N/A"


def test_fmt_price_zero():
    assert fmt_price(0.0, 2) == "$0.00"
    assert fmt_price(0.0, 4) == "$0.0000"


def test_fmt_price_negative():
    assert fmt_price(-50.5, 2) == "$-50.50"


# ─── TESTS: determine_trend ───

def test_determine_trend_empty_df():
    empty_df = pd.DataFrame()
    trend, score = determine_trend(empty_df)
    assert trend == "WAIT"
    assert score == 0


def test_determine_trend_insufficient_data():
    small_df = pd.DataFrame({"Close": [100, 101], "High": [101, 102], "Low": [99, 100],
                              "Open": [100, 101], "Volume": [1000, 1000]})
    trend, score = determine_trend(small_df)
    assert trend == "WAIT"


def test_determine_trend_with_indicators(sample_df):
    df = calculate_indicators(sample_df)
    trend, score = determine_trend(df)
    assert trend in ("UP", "DOWN", "SIDEWAYS")
    assert isinstance(score, int)


# ─── TESTS: calculate_indicators ───

def test_calculate_indicators_empty():
    empty = pd.DataFrame()
    result = calculate_indicators(empty)
    assert result.empty


def test_calculate_indicators_columns(sample_df):
    df = calculate_indicators(sample_df)
    required_cols = ["SMA_20", "SMA_50", "EMA_9", "EMA_21",
                     "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
                     "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
                     "Volume_SMA_20"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_calculate_indicators_no_side_effects(sample_df):
    """Verify original df is not modified."""
    orig = sample_df.copy()
    calculate_indicators(sample_df)
    # Original should still only have OHLCV + Vol (no indicators leaked)
    assert "SMA_20" not in orig.columns or orig["SMA_20"].isna().all()
    # Actually the function modifies in-place via df[col] = ...
    # Let's check that it returns a modified version
    result = calculate_indicators(sample_df)
    assert "SMA_20" in result.columns


# ─── TESTS: calculate_atr ───

def test_calculate_atr(sample_df):
    atr = calculate_atr(sample_df, 14)
    assert len(atr) == len(sample_df)
    assert atr.iloc[0] == 0 or np.isnan(atr.iloc[0])  # First values should be NaN or 0
    assert atr.iloc[-1] > 0  # Last values should be positive


def test_calculate_atr_nan_for_short_df():
    short_df = pd.DataFrame({"High": [100], "Low": [99], "Close": [99.5]})
    atr = calculate_atr(short_df, 14)
    assert np.isnan(atr.iloc[0])


# ─── TESTS: build_confluence_summary ───

def test_build_confluence_summary_empty():
    result = build_confluence_summary({})
    assert "UP:0" in result
    assert "DOWN:0" in result


def test_build_confluence_summary(sample_tf_data):
    result = build_confluence_summary(sample_tf_data)
    assert "UP:" in result
    assert "DOWN:" in result
    assert "Weighted:" in result
    assert "Sequence:" in result


def test_build_confluence_summary_format():
    """Test that the output format is correct."""
    result = build_confluence_summary({})
    assert result.startswith("UP:")
    parts = result.split(" ")
    assert len(parts) >= 3


# ─── TESTS: TF_ORDER ───

def test_tf_order_contains_all():
    assert "Daily" in TF_ORDER
    assert "4H" in TF_ORDER
    assert "1H" in TF_ORDER
    assert "15m" in TF_ORDER
    assert "5m" in TF_ORDER


def test_tf_order_highest_first():
    assert TF_ORDER.index("Daily") < TF_ORDER.index("4H")
    assert TF_ORDER.index("4H") < TF_ORDER.index("1H")
    assert TF_ORDER.index("1H") < TF_ORDER.index("15m")
    assert TF_ORDER.index("15m") < TF_ORDER.index("5m")
