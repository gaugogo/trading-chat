"""
Tests for scoring_engine.py — Unified Scoring Engine.

Run with: pytest tests/test_scoring_engine.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring_engine import (
    ScoringEngine,
    Bias,
    ScoringResult,
    TfScore,
    quick_score,
    DEFAULT_SCORING_CONFIG,
)


# ─── FIXTURES ───

@pytest.fixture
def sample_df():
    """Create a simple DataFrame with indicators for testing."""
    np.random.seed(42)
    dates = pd.RangeIndex(60)
    base = 2000.0
    data = {
        "Open": base + np.cumsum(np.random.randn(60) * 2),
        "High": np.nan,
        "Low": np.nan,
        "Close": np.nan,
        "Volume": np.random.randint(1000, 10000, 60),
    }
    df = pd.DataFrame(data, index=dates)
    for i in range(60):
        o = float(df["Open"].iloc[i])
        change = float(np.random.randn() * 5)
        c = o + change
        h = max(o, c) + abs(float(np.random.randn() * 2))
        l = min(o, c) - abs(float(np.random.randn() * 2))
        df.loc[df.index[i], "High"] = h
        df.loc[df.index[i], "Low"] = l
        df.loc[df.index[i], "Close"] = c
    df[["High", "Low", "Close"]] = df[["High", "Low", "Close"]].astype(float)
    return df


@pytest.fixture
def sample_tf_data(sample_df):
    """Create mock multi-timeframe data with indicators."""
    from core import calculate_indicators
    tf_data = {}
    for tf in ["Daily", "4H", "1H", "15m", "5m"]:
        df = calculate_indicators(sample_df.copy())
        tf_data[tf] = df
    return tf_data


@pytest.fixture
def empty_tf_data():
    return {}


# ─── TESTS: Bias Enum ───

class TestBias:
    def test_bias_icons(self):
        assert Bias.STRONG_BUY.icon == "🟢🟢"
        assert Bias.STRONG_SELL.icon == "🔴🔴"
        assert Bias.NEUTRAL.icon == "🟡"

    def test_bias_numeric(self):
        assert Bias.STRONG_BUY.numeric == 2
        assert Bias.STRONG_SELL.numeric == -2
        assert Bias.NEUTRAL.numeric == 0


# ─── TESTS: ScoringResult ───

class TestScoringResult:
    def test_summary_format(self):
        result = ScoringResult(
            strategy="swing",
            bias=Bias.BUY,
            confidence=0.75,
            raw_score=5.0,
            normalized_score=4.5,
        )
        summary = result.summary()
        assert "SWING" in summary
        assert "BUY" in summary
        assert "4.5" in summary

    def test_detailed_report(self):
        result = ScoringResult(
            strategy="swing",
            bias=Bias.NEUTRAL,
            confidence=0.3,
            raw_score=0.5,
            normalized_score=0.5,
            tf_scores=[
                TfScore(tf_name="Daily", trend="UP", score=2.0, weight=5.0, weighted_score=10.0),
                TfScore(tf_name="4H", trend="DOWN", score=-1.0, weight=3.0, weighted_score=-3.0),
            ],
            details=["Test detail"],
            warnings=["Test warning"],
        )
        report = result.detailed_report()
        assert "SWING" in report
        assert "Daily" in report
        assert "Test warning" in report


# ─── TESTS: ScoringEngine ───

class TestScoringEngine:
    def test_init_default_config(self):
        engine = ScoringEngine()
        assert engine.config["thresholds"]["strong_buy"] == 4.0
        assert engine.config["weights"]["Daily"] == 5.0

    def test_init_custom_config(self):
        engine = ScoringEngine(config={
            "thresholds": {"strong_buy": 6.0},
        })
        assert engine.config["thresholds"]["strong_buy"] == 6.0
        # Other defaults preserved
        assert engine.config["weights"]["Daily"] == 5.0

    def test_score_empty_data(self):
        engine = ScoringEngine()
        result = engine.score({}, strategy="swing")
        assert result.bias == Bias.WAIT
        assert result.confidence == 0.0

    def test_score_with_data(self, sample_tf_data):
        engine = ScoringEngine()
        result = engine.score(sample_tf_data, strategy="swing")
        assert isinstance(result.bias, Bias)
        assert 0.0 <= result.confidence <= 1.0
        assert -10.0 <= result.normalized_score <= 10.0
        assert len(result.tf_scores) > 0

    def test_score_different_strategies(self, sample_tf_data):
        engine = ScoringEngine()
        for strategy in ["swing", "position", "daytrade", "scalp", "ichimoku"]:
            result = engine.score(sample_tf_data, strategy=strategy)
            assert isinstance(result.bias, Bias)
            assert result.strategy == strategy

    def test_quick_score(self, sample_tf_data):
        summary = quick_score(sample_tf_data, "swing")
        assert isinstance(summary, str)
        assert "SWING" in summary


# ─── TESTS: TF Scoring ───

class TestTfScoring:
    def test_score_timeframe_up(self, sample_df):
        from core import calculate_indicators
        engine = ScoringEngine()
        df = calculate_indicators(sample_df)

        # Force bullish indicators
        df["SMA_20"] = df["Close"] * 0.98
        df["SMA_50"] = df["Close"] * 0.97
        df["RSI_14"] = 60.0
        df["MACD"] = df["Close"] * 0.01 + 1.0
        df["MACD_Signal"] = df["Close"] * 0.01
        df["EMA_9"] = df["Close"] * 1.01
        df["EMA_21"] = df["Close"]

        ts = engine._score_timeframe(df, "Daily", 5.0, engine.config["indicator_weights"], "swing")
        assert ts.score >= 1.0  # Should be bullish
        assert "Daily" in ts.tf_name

    def test_score_timeframe_down(self, sample_df):
        from core import calculate_indicators
        engine = ScoringEngine()
        df = calculate_indicators(sample_df)

        # Force bearish indicators
        df["SMA_20"] = df["Close"] * 1.02
        df["SMA_50"] = df["Close"] * 1.03
        df["RSI_14"] = 40.0
        df["MACD"] = df["Close"] * 0.01 - 1.0
        df["MACD_Signal"] = df["Close"] * 0.01
        df["EMA_9"] = df["Close"] * 0.99
        df["EMA_21"] = df["Close"]

        ts = engine._score_timeframe(df, "Daily", 5.0, engine.config["indicator_weights"], "swing")
        assert ts.score <= -1.0  # Should be bearish


# ─── TESTS: Bias Mapping ───

class TestBiasMapping:
    def test_strong_buy(self):
        engine = ScoringEngine()
        # Default threshold: strong_buy >= 4.0
        assert engine._bias_from_score(8.0, engine.config["thresholds"]) == Bias.STRONG_BUY
        assert engine._bias_from_score(5.0, engine.config["thresholds"]) == Bias.STRONG_BUY

    def test_strong_sell(self):
        engine = ScoringEngine()
        assert engine._bias_from_score(-8.0, DEFAULT_SCORING_CONFIG["thresholds"]) == Bias.STRONG_SELL
        assert engine._bias_from_score(-5.0, DEFAULT_SCORING_CONFIG["thresholds"]) == Bias.STRONG_SELL

    def test_neutral(self):
        engine = ScoringEngine()
        assert engine._bias_from_score(0.0, DEFAULT_SCORING_CONFIG["thresholds"]) == Bias.NEUTRAL
        assert engine._bias_from_score(0.5, DEFAULT_SCORING_CONFIG["thresholds"]) == Bias.NEUTRAL


# ─── TESTS: Strategy Warnings ───

class TestStrategyWarnings:
    def test_conflicting_tfs(self, sample_tf_data):
        engine = ScoringEngine()
        # Create conflicting trends
        result = engine.score(sample_tf_data, strategy="swing")
        # Should not crash, might have warnings
        assert isinstance(result.warnings, list)

    def test_position_needs_daily(self):
        engine = ScoringEngine()
        # Only 5m data (not enough for position)
        from core import calculate_indicators
        df_5m = calculate_indicators(pd.DataFrame({
            "Open": [100] * 30, "High": [101] * 30,
            "Low": [99] * 30, "Close": [100.5] * 30,
            "Volume": [1000] * 30,
        }))
        tf_data = {"5m": df_5m}
        result = engine.score(tf_data, strategy="position")
        # Should still produce a result without crash
        assert isinstance(result, ScoringResult)
