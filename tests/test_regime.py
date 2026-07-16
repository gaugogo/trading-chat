"""
Tests for regime.py — Market Regime Detection.

Run with: pytest tests/test_regime.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from regime import (
    MarketRegime,
    RegimeResult,
    calculate_adx,
    detect_regime,
    detect_regime_simple,
    regime_recommendation,
    DEFAULT_REGIME_CONFIG,
)
from core import calculate_indicators


# ─── FIXTURES ───

@pytest.fixture
def uptrend_df():
    """Create a DataFrame with clear uptrend (higher highs, higher lows)."""
    np.random.seed(42)
    n = 60
    dates = pd.RangeIndex(n)
    base = 2000.0
    trend = np.linspace(0, 100, n)  # strong uptrend
    noise = np.random.randn(n) * 3
    close = base + trend + noise
    data = {
        "Open": close - 2 + np.random.randn(n),
        "High": close + abs(np.random.randn(n) * 3) + 2,
        "Low": close - abs(np.random.randn(n) * 3) - 2,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }
    df = pd.DataFrame(data, index=dates)
    return calculate_indicators(df)


@pytest.fixture
def downtrend_df():
    """Create a DataFrame with clear downtrend (lower highs, lower lows)."""
    np.random.seed(42)
    n = 60
    dates = pd.RangeIndex(n)
    base = 2100.0
    trend = np.linspace(0, -100, n)  # strong downtrend
    noise = np.random.randn(n) * 3
    close = base + trend + noise
    data = {
        "Open": close - 2 + np.random.randn(n),
        "High": close + abs(np.random.randn(n) * 3) + 2,
        "Low": close - abs(np.random.randn(n) * 3) - 2,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }
    df = pd.DataFrame(data, index=dates)
    return calculate_indicators(df)


@pytest.fixture
def ranging_df():
    """Create a DataFrame with ranging/sideways market."""
    np.random.seed(42)
    n = 60
    dates = pd.RangeIndex(n)
    base = 2050.0
    close = base + np.random.randn(n) * 5  # no trend, just noise
    data = {
        "Open": close - 2 + np.random.randn(n),
        "High": close + abs(np.random.randn(n) * 4) + 2,
        "Low": close - abs(np.random.randn(n) * 4) - 2,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }
    df = pd.DataFrame(data, index=dates)
    return calculate_indicators(df)


@pytest.fixture
def volatile_df():
    """Create a DataFrame with high volatility."""
    np.random.seed(42)
    n = 60
    dates = pd.RangeIndex(n)
    base = 2050.0
    # High amplitude swings
    close = base + np.random.randn(n) * 30
    data = {
        "Open": close - 5 + np.random.randn(n),
        "High": close + abs(np.random.randn(n) * 10) + 5,
        "Low": close - abs(np.random.randn(n) * 10) - 5,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }
    df = pd.DataFrame(data, index=dates)
    return calculate_indicators(df)


# ─── TESTS: MarketRegime Enum ───

class TestMarketRegime:
    def test_icons(self):
        assert MarketRegime.STRONG_UPTREND.icon == "🟢🟢"
        assert MarketRegime.RANGING.icon == "🟡↔️"
        assert MarketRegime.VOLATILE.icon == "⚡"
        assert MarketRegime.CHOPPY.icon == "🌀"

    def test_is_trending(self):
        assert MarketRegime.STRONG_UPTREND.is_trending
        assert MarketRegime.DOWNTREND.is_trending
        assert not MarketRegime.RANGING.is_trending
        assert not MarketRegime.VOLATILE.is_trending

    def test_is_bullish_bearish(self):
        assert MarketRegime.UPTREND.is_bullish
        assert not MarketRegime.UPTREND.is_bearish
        assert MarketRegime.DOWNTREND.is_bearish
        assert not MarketRegime.DOWNTREND.is_bullish

    def test_is_ranging(self):
        assert MarketRegime.RANGING.is_ranging
        assert not MarketRegime.VOLATILE.is_ranging

    def test_is_volatile(self):
        assert MarketRegime.VOLATILE.is_volatile
        assert MarketRegime.CHOPPY.is_volatile
        assert not MarketRegime.RANGING.is_volatile


# ─── TESTS: RegimeResult ───

class TestRegimeResult:
    def test_summary_format(self):
        result = RegimeResult(
            regime=MarketRegime.STRONG_UPTREND,
            confidence=0.85,
            adx=35.0,
            adx_strength="strong",
            bb_width=0.02,
            bb_volatility="normal",
            atr_ratio=1.1,
            trend="UP",
            details=["ADX 35: Strong trend", "BB normal"],
        )
        summary = result.summary()
        assert "STRONG_UPTREND" in summary
        assert "ADX: 35.0" in summary
        assert "85%" in summary

    def test_detailed_report(self):
        result = RegimeResult(
            regime=MarketRegime.RANGING,
            confidence=0.6,
            adx=18.0,
            adx_strength="weak",
            bb_width=0.015,
            bb_volatility="low",
            atr_ratio=0.8,
            trend="SIDEWAYS",
            details=["ADX 18: Weak trend", "BB Width low", "ATR stable"],
        )
        report = result.detailed_report()
        assert "RANGING" in report
        assert "ADX: 18.0" in report
        assert "Gợi ý giao dịch" in report

    def test_empty_report(self):
        result = RegimeResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            adx=0.0,
            adx_strength="unknown",
            bb_width=0.0,
            bb_volatility="unknown",
            atr_ratio=1.0,
            trend="UNKNOWN",
            details=["No data"],
        )
        report = result.detailed_report()
        assert "UNKNOWN" in report


# ─── TESTS: calculate_adx ───

class TestCalculateADX:
    def test_adx_uptrend(self, uptrend_df):
        adx_data = calculate_adx(uptrend_df, 14)
        adx_val = float(adx_data['ADX'].iloc[-1])
        assert adx_val > 0
        # In an uptrend, Plus_DI > Minus_DI
        plus_di = float(adx_data['Plus_DI'].iloc[-1])
        minus_di = float(adx_data['Minus_DI'].iloc[-1])
        assert plus_di > minus_di

    def test_adx_downtrend(self, downtrend_df):
        adx_data = calculate_adx(downtrend_df, 14)
        plus_di = float(adx_data['Plus_DI'].iloc[-1])
        minus_di = float(adx_data['Minus_DI'].iloc[-1])
        assert minus_di > plus_di

    def test_adx_columns(self, uptrend_df):
        adx_data = calculate_adx(uptrend_df, 14)
        assert 'ADX' in adx_data.columns
        assert 'Plus_DI' in adx_data.columns
        assert 'Minus_DI' in adx_data.columns
        assert len(adx_data) == len(uptrend_df)


# ─── TESTS: detect_regime ───

class TestDetectRegime:
    def test_empty_data(self):
        result = detect_regime({})
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    def test_uptrend_detection(self, uptrend_df):
        tf_data = {"Daily": uptrend_df}
        result = detect_regime(tf_data)
        assert result.regime.is_trending
        assert result.confidence > 0.3
        assert result.adx > 0
        assert result.trend == "UP"

    def test_downtrend_detection(self, downtrend_df):
        tf_data = {"Daily": downtrend_df}
        result = detect_regime(tf_data)
        assert result.regime.is_trending
        assert result.adx > 0
        assert result.trend == "DOWN"

    def test_ranging_detection(self, ranging_df):
        tf_data = {"Daily": ranging_df}
        result = detect_regime(tf_data)
        # Ranging might still show some trend, but ADX should be relatively low
        assert result.adx < 40  # Not a super strong trend

    def test_detect_regime_simple(self, uptrend_df):
        result = detect_regime_simple(uptrend_df)
        assert isinstance(result, RegimeResult)

    def test_missing_bb_width(self, uptrend_df):
        """Should work even without BB_Width column."""
        df = uptrend_df.copy()
        if 'BB_Width' in df.columns:
            del df['BB_Width']
        tf_data = {"Daily": df}
        result = detect_regime(tf_data)
        assert isinstance(result, RegimeResult)
        assert result.bb_volatility in ("unknown", "normal")


# ─── TESTS: regime_recommendation ───

class TestRegimeRecommendation:
    def test_uptrend_recs(self):
        recs = regime_recommendation(MarketRegime.STRONG_UPTREND)
        assert len(recs) > 0
        assert any("LONG" in r for r in recs)

    def test_downtrend_recs(self):
        recs = regime_recommendation(MarketRegime.DOWNTREND)
        assert len(recs) > 0
        assert any("SHORT" in r for r in recs)

    def test_ranging_recs(self):
        recs = regime_recommendation(MarketRegime.RANGING)
        assert len(recs) > 0
        assert any("mean-reversion" in r for r in recs)

    def test_volatile_recs(self):
        recs = regime_recommendation(MarketRegime.VOLATILE)
        assert len(recs) > 0

    def test_choppy_recs(self):
        recs = regime_recommendation(MarketRegime.CHOPPY)
        assert any("KHÔNG trade" in r for r in recs)

    def test_unknown_recs(self):
        recs = regime_recommendation(MarketRegime.UNKNOWN)
        assert any("không đủ dữ liệu" in r.lower() for r in recs)


# ─── TESTS: Config ───

class TestConfig:
    def test_default_config(self):
        assert DEFAULT_REGIME_CONFIG["adx"]["strong"] == 30.0
        assert DEFAULT_REGIME_CONFIG["adx"]["period"] == 14
        assert DEFAULT_REGIME_CONFIG["bb_width"]["high_percentile"] == 0.80
        assert DEFAULT_REGIME_CONFIG["atr"]["high_ratio"] == 1.5
