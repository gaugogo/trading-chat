"""
Tests for volume_profile.py — Volume Profile Analysis.

Run with: pytest tests/test_volume_profile.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from volume_profile import (
    VolumeImbalance,
    VPResult,
    FullVPReport,
    analyze_volume_profile,
    analyze_all_timeframes,
    _calculate_dynamic_bins,
    _calculate_delta,
    _detect_session,
)


# ─── FIXTURES ───

@pytest.fixture
def sample_df():
    """Standard OHLCV DataFrame with indicators."""
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
        "ATR": np.random.rand(60) * 10 + 5,
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def bullish_volume_df():
    """DataFrame với buy volume vượt trội (nhiều up-candle hơn)."""
    n = 60
    dates = pd.RangeIndex(n)
    base = 2000.0
    trend = np.linspace(0, 50, n)  # uptrend
    close = base + trend + np.random.randn(n) * 2
    data = {
        "Open": close - 1 + np.random.randn(n),
        "High": close + abs(np.random.randn(n) * 2) + 1,
        "Low": close - abs(np.random.randn(n) * 2) - 1,
        "Close": close,
        "Volume": np.where(np.random.randn(n) > 0, 5000, 1000),  # More buy volume
        "ATR": np.random.rand(n) * 5 + 3,
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure more up candles
    for i in range(n):
        if df['Close'].iloc[i] < df['Open'].iloc[i]:
            df.loc[df.index[i], 'Close'] = df['Open'].iloc[i] + abs(np.random.randn() * 2)
    return df


# ─── TESTS: VolumeImbalance Enum ───

class TestVolumeImbalance:
    def test_icons(self):
        assert VolumeImbalance.EXTREME_BUY.icon == "🟢🟢"
        assert VolumeImbalance.BEARISH.icon == "🔴"
        assert VolumeImbalance.NEUTRAL.icon == "🟡"


# ─── TESTS: VPResult ───

class TestVPResult:
    @pytest.fixture
    def vp_result(self):
        return VPResult(
            tf_name="Daily",
            poc=2050.0,
            poc_volume=50000,
            vah=2070.0,
            val=2030.0,
            total_volume=500000,
            buy_volume=300000,
            sell_volume=200000,
            delta_volume=100000,
            delta_ratio=1.5,
            imbalance=VolumeImbalance.BULLISH,
            bins_count=15,
            bin_size=2.0,
            high=2100.0,
            low=2000.0,
            current_price=2060.0,
            price_in_value_area=True,
        )

    def test_summary(self, vp_result):
        summary = vp_result.summary()
        assert "POC=" in summary
        assert "VA=" in summary
        assert "Δ=" in summary

    def test_detailed_report(self, vp_result):
        report = vp_result.detailed_report()
        assert "Volume Profile" in report
        assert "POC:" in report
        assert "Value Area:" in report
        assert "Volume Delta" in report
        assert "bullish volume pressure" in report.lower() or "Interpretation" in report

    def test_detailed_report_bearish(self):
        vp = VPResult(
            tf_name="4H",
            poc=2050.0, poc_volume=50000,
            vah=2070.0, val=2030.0,
            total_volume=500000, buy_volume=200000, sell_volume=300000,
            delta_volume=-100000, delta_ratio=0.67,
            imbalance=VolumeImbalance.BEARISH,
            bins_count=15, bin_size=2.0,
            high=2100.0, low=2000.0,
            current_price=2040.0, price_in_value_area=True,
        )
        report = vp.detailed_report()
        assert "bearish volume pressure" in report.lower()

    def test_price_outside_va(self):
        vp = VPResult(
            tf_name="1H",
            poc=2050.0, poc_volume=50000,
            vah=2060.0, val=2040.0,
            total_volume=500000, buy_volume=250000, sell_volume=250000,
            delta_volume=0, delta_ratio=1.0,
            imbalance=VolumeImbalance.NEUTRAL,
            bins_count=10, bin_size=2.0,
            high=2100.0, low=2000.0,
            current_price=2080.0, price_in_value_area=False,
        )
        assert not vp.price_in_value_area
        assert vp.current_price > vp.vah


# ─── TESTS: FullVPReport ───

class TestFullVPReport:
    def test_empty(self):
        report = FullVPReport()
        assert not report.has_data

    def test_with_results(self):
        vp = VPResult(
            tf_name="Daily", poc=2050.0, poc_volume=50000,
            vah=2070.0, val=2030.0,
            total_volume=500000, buy_volume=300000, sell_volume=200000,
            delta_volume=100000, delta_ratio=1.5,
            imbalance=VolumeImbalance.BULLISH,
            bins_count=15, bin_size=2.0,
            high=2100.0, low=2000.0,
            current_price=2060.0, price_in_value_area=True,
        )
        report = FullVPReport(results={"Daily": vp})
        assert report.has_data
        summary = report.summary()
        assert "VOLUME PROFILE" in summary

    def test_strongest_imbalance(self):
        vp = VPResult(
            tf_name="4H", poc=100.0, poc_volume=5000,
            vah=110.0, val=90.0,
            total_volume=50000, buy_volume=40000, sell_volume=10000,
            delta_volume=30000, delta_ratio=4.0,
            imbalance=VolumeImbalance.EXTREME_BUY,
            bins_count=10, bin_size=2.0,
            high=120.0, low=80.0,
            current_price=115.0, price_in_value_area=False,
        )
        report = FullVPReport(
            results={"4H": vp},
            strongest_imbalance=("4H", VolumeImbalance.EXTREME_BUY, 4.0),
        )
        summary = report.summary()
        assert "mạnh nhất" in summary.lower()


# ─── TESTS: Utilities ───

class TestUtilities:
    def test_calculate_dynamic_bins(self, sample_df):
        bins = _calculate_dynamic_bins(sample_df, min_bins=8, max_bins=30)
        assert 8 <= bins <= 30

    def test_calculate_dynamic_bins_no_atr(self):
        df = pd.DataFrame({"High": [100, 101], "Low": [99, 100], "Close": [99.5, 100.5]})
        bins = _calculate_dynamic_bins(df, min_bins=5, max_bins=20)
        assert 5 <= bins <= 20

    def test_calculate_delta(self):
        df = pd.DataFrame({
            "Open": [100, 102, 101, 103, 102],
            "Close": [102, 101, 103, 104, 101],
            "Volume": [1000, 2000, 1500, 2500, 1800],
        })
        buy, sell, delta, ratio = _calculate_delta(df)
        assert buy > 0
        assert sell > 0
        assert isinstance(delta, float)
        assert isinstance(ratio, float)

    def test_calculate_delta_empty(self):
        buy, sell, delta, ratio = _calculate_delta(pd.DataFrame())
        assert buy == 0.0
        assert ratio == 1.0

    def test_detect_session_no_datetime(self):
        df = pd.DataFrame({"Close": [100]}, index=pd.RangeIndex(1))
        session = _detect_session(df)
        assert session == "All"

    def test_detect_session_empty(self):
        # Empty DataFrame with RangeIndex (most common in tests)
        df = pd.DataFrame(index=pd.RangeIndex(0))
        session = _detect_session(df)
        assert session == "All"

    def test_detect_session_with_datetime(self):
        df = pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex(["2024-01-15 14:30:00"]))
        session = _detect_session(df)
        assert session in ("Asian", "London", "NewYork")


# ─── TESTS: analyze_volume_profile ───

class TestAnalyzeVolumeProfile:
    def test_empty_df(self):
        vp = analyze_volume_profile(pd.DataFrame(), tf_name="1H")
        assert vp is None

    def test_insufficient_data(self):
        df = pd.DataFrame({"High": [100], "Low": [99], "Close": [99.5], "Volume": [1000]})
        vp = analyze_volume_profile(df, tf_name="1H")
        assert vp is None

    def test_basic_analysis(self, sample_df):
        vp = analyze_volume_profile(sample_df, tf_name="1H", use_dynamic_bins=True)
        assert vp is not None
        assert vp.tf_name == "1H"
        assert vp.poc > 0
        assert vp.vah >= vp.val
        assert vp.total_volume > 0
        assert vp.bins_count >= 8

    def test_fixed_bins(self, sample_df):
        vp = analyze_volume_profile(sample_df, tf_name="Daily", bins=20, use_dynamic_bins=False)
        assert vp is not None
        assert vp.bins_count == 20

    def test_in_value_area(self, sample_df):
        vp = analyze_volume_profile(sample_df, tf_name="1H")
        assert vp is not None
        # Current price should be somewhere
        assert isinstance(vp.price_in_value_area, bool)

    def test_delta_calculation(self, sample_df):
        vp = analyze_volume_profile(sample_df)
        assert vp is not None
        assert vp.buy_volume + vp.sell_volume == pytest.approx(vp.total_volume, abs=1)
        # delta_volume is rounded; approximate comparison
        assert abs(vp.delta_volume - (vp.buy_volume - vp.sell_volume)) < 1.0

    def test_high_low_same(self, sample_df):
        df = sample_df.copy()
        df['High'] = 100.0
        df['Low'] = 100.0
        vp = analyze_volume_profile(df)
        assert vp is None or vp.high == vp.low


# ─── TESTS: analyze_all_timeframes ───

class TestAnalyzeAllTimeframes:
    def test_empty(self):
        report = analyze_all_timeframes({})
        assert isinstance(report, FullVPReport)
        assert not report.has_data

    def test_multi_tf(self, sample_df):
        tf_data = {"Daily": sample_df, "4H": sample_df, "1H": sample_df}
        report = analyze_all_timeframes(tf_data)
        assert report.has_data
        assert "Daily" in report.results
        assert "4H" in report.results

    def test_summary(self, sample_df):
        tf_data = {"Daily": sample_df}
        report = analyze_all_timeframes(tf_data)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "VOLUME PROFILE" in summary
