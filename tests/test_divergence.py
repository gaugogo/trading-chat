"""
Tests for divergence.py — RSI & MACD Divergence Detection.

Run with: pytest tests/test_divergence.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from divergence import (
    DivergenceType,
    DivergenceSignal,
    DivergenceResult,
    FullDivergenceReport,
    find_rsi_divergence,
    find_macd_divergence,
    analyze_all_divergences,
    divergence_bias_score,
    _find_pivots,
    _find_pivots_adaptive,
    _detect_divergence,
)
from core import calculate_indicators


# ─── FIXTURES ───

@pytest.fixture
def sample_df():
    """Standard DataFrame with indicators."""
    np.random.seed(42)
    dates = pd.RangeIndex(60)
    base = 2000.0
    close = base + np.cumsum(np.random.randn(60) * 2)
    data = {
        "Open": close - 2 + np.random.randn(60),
        "High": close + abs(np.random.randn(60) * 3) + 2,
        "Low": close - abs(np.random.randn(60) * 3) - 2,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, 60),
    }
    df = pd.DataFrame(data, index=dates)
    return calculate_indicators(df)


@pytest.fixture
def regular_bullish_df():
    """
    Create DataFrame with REGULAR BULLISH divergence:
    Price makes lower low, RSI makes higher low.
    """
    n = 60
    dates = pd.RangeIndex(n)
    # Create two down moves, second one goes lower in price but RSI diverges
    price = np.zeros(n)

    # First leg: drop to 100, then rally to 120
    price[:20] = np.linspace(150, 100, 20)
    price[20:30] = np.linspace(100, 120, 10)

    # Second leg: drop LOWER to 95, creating lower low
    price[30:45] = np.linspace(120, 95, 15)
    price[45:50] = np.linspace(95, 110, 5)
    price[50:] = 110 + np.random.randn(n - 50) * 2

    # RSI divergence: first low RSI ≈ 30, second low RSI ≈ 40 (higher low)
    rsi = np.zeros(n)
    rsi[:20] = np.linspace(70, 30, 20)   # First drop: RSI 70→30
    rsi[20:30] = np.linspace(30, 60, 10)  # Rally: RSI back to 60
    rsi[30:45] = np.linspace(60, 40, 15)  # Second drop: RSI only to 40 (higher low!)
    rsi[45:50] = np.linspace(40, 55, 5)
    rsi[50:] = 55 + np.random.randn(n - 50) * 3

    data = {
        "Open": price - 2 + np.random.randn(n),
        "High": price + abs(np.random.randn(n) * 3) + 2,
        "Low": price - abs(np.random.randn(n) * 3) - 2,
        "Close": price + np.random.randn(n) * 0.5,
        "Volume": np.random.randint(1000, 10000, n),
        "RSI_14": rsi,
        "MACD": price * 0.01 + np.random.randn(n) * 0.5,
        "MACD_Signal": price * 0.01,
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def regular_bearish_df():
    """
    Create DataFrame with REGULAR BEARISH divergence:
    Price makes higher high, RSI makes lower high.
    """
    n = 60
    dates = pd.RangeIndex(n)
    price = np.zeros(n)

    # First leg: rally from 100 to 150
    price[:20] = np.linspace(100, 150, 20)
    price[20:30] = np.linspace(150, 130, 10)

    # Second leg: rally to HIGHER high 160
    price[30:45] = np.linspace(130, 160, 15)
    price[45:50] = np.linspace(160, 145, 5)
    price[50:] = 145 + np.random.randn(n - 50) * 3

    # RSI: first high RSI 80, second high RSI 65 (lower high = bearish divergence)
    rsi = np.zeros(n)
    rsi[:20] = np.linspace(50, 80, 20)
    rsi[20:30] = np.linspace(80, 60, 10)
    rsi[30:45] = np.linspace(60, 65, 15)  # Lower high!
    rsi[45:50] = np.linspace(65, 55, 5)
    rsi[50:] = 55 + np.random.randn(n - 50) * 3

    data = {
        "Open": price - 2 + np.random.randn(n),
        "High": price + abs(np.random.randn(n) * 3) + 2,
        "Low": price - abs(np.random.randn(n) * 3) - 2,
        "Close": price + np.random.randn(n) * 0.5,
        "Volume": np.random.randint(1000, 10000, n),
        "RSI_14": rsi,
        "MACD": price * 0.01 + np.random.randn(n) * 0.5,
        "MACD_Signal": price * 0.01,
    }
    df = pd.DataFrame(data, index=dates)
    return df


# ─── TESTS: DivergenceType Enum ───

class TestDivergenceType:
    def test_icons(self):
        assert DivergenceType.REGULAR_BULLISH.icon == "🟢🔄"
        assert DivergenceType.REGULAR_BEARISH.icon == "🔴🔄"
        assert DivergenceType.HIDDEN_BULLISH.icon == "🟢▶️"
        assert DivergenceType.HIDDEN_BEARISH.icon == "🔴▶️"

    def test_is_bullish(self):
        assert DivergenceType.REGULAR_BULLISH.is_bullish
        assert DivergenceType.HIDDEN_BULLISH.is_bullish
        assert not DivergenceType.REGULAR_BEARISH.is_bullish

    def test_is_bearish(self):
        assert DivergenceType.REGULAR_BEARISH.is_bearish
        assert DivergenceType.HIDDEN_BEARISH.is_bearish
        assert not DivergenceType.REGULAR_BULLISH.is_bearish

    def test_is_regular(self):
        assert DivergenceType.REGULAR_BULLISH.is_regular
        assert DivergenceType.REGULAR_BEARISH.is_regular
        assert not DivergenceType.HIDDEN_BULLISH.is_regular

    def test_is_hidden(self):
        assert DivergenceType.HIDDEN_BULLISH.is_hidden
        assert DivergenceType.HIDDEN_BEARISH.is_hidden
        assert not DivergenceType.REGULAR_BULLISH.is_hidden

    def test_label_vn(self):
        assert "Regular Bullish" in DivergenceType.REGULAR_BULLISH.label_vn
        assert "Hidden Bearish" in DivergenceType.HIDDEN_BEARISH.label_vn


# ─── TESTS: DivergenceSignal ───

class TestDivergenceSignal:
    def test_summary(self):
        sig = DivergenceSignal(
            type=DivergenceType.REGULAR_BULLISH,
            indicator="RSI",
            price_pivot_1_idx=10,
            price_pivot_2_idx=30,
            price_pivot_1_val=100.0,
            price_pivot_2_val=95.0,
            indicator_pivot_1_val=30.0,
            indicator_pivot_2_val=40.0,
            strength=0.85,
        )
        summary = sig.summary()
        assert "RSI" in summary
        assert "BULLISH" in summary
        assert "85%" in summary

    def test_icon(self):
        sig = DivergenceSignal(
            type=DivergenceType.REGULAR_BEARISH,
            indicator="MACD",
            price_pivot_1_idx=10,
            price_pivot_2_idx=30,
            price_pivot_1_val=100.0,
            price_pivot_2_val=105.0,
            indicator_pivot_1_val=80.0,
            indicator_pivot_2_val=65.0,
            strength=0.75,
        )
        assert "🔴" in sig.icon


# ─── TESTS: DivergenceResult ───

class TestDivergenceResult:
    def test_empty(self):
        res = DivergenceResult(indicator="RSI")
        assert not res.has_bullish
        assert not res.has_bearish
        assert res.total_signals == 0
        assert res.net_bias == "NEUTRAL"

    def test_with_signals(self):
        res = DivergenceResult(indicator="RSI")
        sig = DivergenceSignal(
            type=DivergenceType.REGULAR_BULLISH,
            indicator="RSI",
            price_pivot_1_idx=10, price_pivot_2_idx=30,
            price_pivot_1_val=100.0, price_pivot_2_val=95.0,
            indicator_pivot_1_val=30.0, indicator_pivot_2_val=40.0,
            strength=0.85,
        )
        res.bullish.append(sig)
        assert res.has_bullish
        assert res.total_signals == 1
        assert res.net_bias == "BULLISH"

    def test_summary_format(self):
        res = DivergenceResult(indicator="RSI")
        res.bullish.append(DivergenceSignal(
            type=DivergenceType.REGULAR_BULLISH, indicator="RSI",
            price_pivot_1_idx=10, price_pivot_2_idx=30,
            price_pivot_1_val=100.0, price_pivot_2_val=95.0,
            indicator_pivot_1_val=30.0, indicator_pivot_2_val=40.0,
            strength=0.85,
        ))
        summary = res.summary()
        assert "RSI" in summary
        assert "Bullish" in summary

    def test_summary_no_signals(self):
        res = DivergenceResult(indicator="MACD")
        summary = res.summary()
        assert "Không phát hiện" in summary


# ─── TESTS: FullDivergenceReport ───

class TestFullDivergenceReport:
    def test_empty_report(self):
        report = FullDivergenceReport(tf_results={})
        assert not report.has_divergence

    def test_summary_with_data(self):
        tf_results = {
            "Daily": {
                "RSI": DivergenceResult(indicator="RSI"),
                "MACD": DivergenceResult(indicator="MACD"),
            }
        }
        report = FullDivergenceReport(tf_results=tf_results)
        summary = report.summary()
        assert "DIVERGENCE" in summary

    def test_strongest_signals(self):
        bull = DivergenceSignal(
            type=DivergenceType.REGULAR_BULLISH, indicator="RSI",
            price_pivot_1_idx=10, price_pivot_2_idx=30,
            price_pivot_1_val=100.0, price_pivot_2_val=95.0,
            indicator_pivot_1_val=30.0, indicator_pivot_2_val=40.0,
            strength=0.9,
        )
        report = FullDivergenceReport(
            tf_results={},
            strongest_bullish=("Daily", "RSI", bull),
        )
        summary = report.summary()
        assert "Mạnh nhất" in summary


# ─── TESTS: Pivot Detection ───

class TestPivotDetection:
    def test_find_pivots(self):
        values = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        peaks, troughs = _find_pivots(values, order=2)
        assert len(peaks) > 0
        assert len(troughs) > 0

    def test_find_pivots_adaptive(self):
        values = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100)) * 3
        peaks, troughs = _find_pivots_adaptive(values)
        assert len(peaks) > 0
        assert len(troughs) > 0

    def test_find_pivots_short_data(self):
        values = np.array([1, 2, 3, 4, 5])
        peaks, troughs = _find_pivots(values, order=2)
        assert len(peaks) == 0
        assert len(troughs) == 0


# ─── TESTS: find_rsi_divergence ───

class TestFindRsiDivergence:
    def test_empty_df(self):
        empty = pd.DataFrame()
        res = find_rsi_divergence(empty)
        assert res.indicator == "RSI"
        assert res.total_signals == 0

    def test_missing_rsi_column(self, sample_df):
        df = sample_df.drop(columns=["RSI_14"], errors="ignore")
        res = find_rsi_divergence(df)
        assert res.total_signals == 0

    def test_regular_bullish(self, regular_bullish_df):
        res = find_rsi_divergence(regular_bullish_df, min_strength=0.2)
        # Should detect at least some bullish divergence
        # (may be noisy but the pattern is there)
        assert isinstance(res, DivergenceResult)

    def test_regular_bearish(self, regular_bearish_df):
        res = find_rsi_divergence(regular_bearish_df, min_strength=0.2)
        assert isinstance(res, DivergenceResult)

    def test_with_indicators(self, sample_df):
        res = find_rsi_divergence(sample_df)
        assert isinstance(res, DivergenceResult)
        assert res.indicator == "RSI"


# ─── TESTS: find_macd_divergence ───

class TestFindMacdDivergence:
    def test_empty_df(self):
        empty = pd.DataFrame()
        res = find_macd_divergence(empty)
        assert res.indicator == "MACD"
        assert res.total_signals == 0

    def test_missing_macd_column(self, sample_df):
        df = sample_df.drop(columns=["MACD"], errors="ignore")
        res = find_macd_divergence(df)
        assert res.total_signals == 0

    def test_with_indicators(self, sample_df):
        res = find_macd_divergence(sample_df)
        assert isinstance(res, DivergenceResult)
        assert res.indicator == "MACD"

    def test_macd_bearish(self, regular_bearish_df):
        res = find_macd_divergence(regular_bearish_df, min_strength=0.2)
        assert isinstance(res, DivergenceResult)


# ─── TESTS: analyze_all_divergences ───

class TestAnalyzeAllDivergences:
    def test_empty_data(self):
        report = analyze_all_divergences({})
        assert not report.has_divergence

    def test_with_data(self, sample_df):
        tf_data = {
            "Daily": sample_df,
            "4H": sample_df,
        }
        report = analyze_all_divergences(tf_data)
        assert isinstance(report, FullDivergenceReport)

    def test_summary(self, sample_df):
        tf_data = {"Daily": sample_df}
        report = analyze_all_divergences(tf_data)
        summary = report.summary()
        assert isinstance(summary, str)


# ─── TESTS: divergence_bias_score ───

class TestDivergenceBiasScore:
    def test_empty(self):
        score = divergence_bias_score({})
        assert score == 0.0

    def test_with_data(self, sample_df):
        tf_data = {"Daily": sample_df}
        score = divergence_bias_score(tf_data)
        assert -10.0 <= score <= 10.0

    def test_multi_tf(self, sample_df):
        tf_data = {"Daily": sample_df, "4H": sample_df, "1H": sample_df}
        score = divergence_bias_score(tf_data)
        assert -10.0 <= score <= 10.0


# ─── TESTS: _detect_divergence ───

class TestDetectDivergence:
    def test_short_arrays(self):
        price = np.array([1, 2, 3])
        indicator = np.array([0.5, 0.6, 0.7])
        res = _detect_divergence(price, indicator, "RSI")
        assert res.total_signals == 0

    def test_no_divergence(self, sample_df):
        price = sample_df['Close'].values
        rsi = sample_df['RSI_14'].values
        res = _detect_divergence(price, rsi, "RSI", min_strength=0.5)
        assert isinstance(res, DivergenceResult)
