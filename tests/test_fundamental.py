"""
Tests for fundamental.py — Economic calendar, DXY, bond yields

Tests focus on:
  - DXYData, BondYieldData, EconomicEvent, FundamentalReport data classes
  - fetch_dxy (mocked)
  - fetch_bond_yields (mocked)
  - fetch_economic_calendar (mocked)
  - fundamental_bias analysis
  - fundamental_report formatting
  - Cross-source correlation logic
"""

import os
import sys
import time
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fundamental import (
    DXYData,
    BondYieldData,
    EconomicEvent,
    FundamentalReport,
    fetch_dxy,
    fetch_bond_yields,
    fetch_economic_calendar,
    fundamental_bias,
    fundamental_report,
    _fetch_yahoo_quote,
    HIGH_IMPACT_EVENTS_XAU,
)


# ─── FIXTURES ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dxy():
    """Sample DXY data."""
    return DXYData(
        price=104.50,
        change_24h_pct=0.35,
        timestamp=time.time(),
        high_24h=104.80,
        low_24h=104.20,
    )


@pytest.fixture
def sample_bond_yields():
    """Sample bond yield data (inverted curve)."""
    return BondYieldData(
        yield_10y=4.25,
        yield_2y=4.65,
        spread_10y_2y=4.25 - 4.65,  # -0.40 (inverted)
        timestamp=time.time(),
        yield_10y_change=-0.05,
        yield_2y_change=0.02,
    )


@pytest.fixture
def sample_bond_normal():
    """Sample bond yield data (normal curve)."""
    return BondYieldData(
        yield_10y=4.10,
        yield_2y=3.80,
        spread_10y_2y=0.30,
        timestamp=time.time(),
        yield_10y_change=0.03,
        yield_2y_change=-0.01,
    )


@pytest.fixture
def sample_event_high():
    """Sample high-impact economic event."""
    return EconomicEvent(
        date="2026-07-17",
        time="08:30",
        currency="USD",
        event="CPI (YoY)",
        importance="High",
        previous="3.3%",
        forecast="3.1%",
    )


@pytest.fixture
def sample_event_low():
    """Sample low-impact economic event."""
    return EconomicEvent(
        date="2026-07-18",
        time="10:00",
        currency="EUR",
        event="German ZEW",
        importance="Low",
        previous="47.5",
        forecast="48.0",
    )


# ─── DXYDATA TESTS ────────────────────────────────────────────────────────

class TestDXYData:
    """Tests for DXYData."""

    def test_create(self, sample_dxy):
        assert sample_dxy.price == 104.50
        assert sample_dxy.change_24h_pct == 0.35
        assert sample_dxy.high_24h == 104.80
        assert sample_dxy.low_24h == 104.20

    def test_trend_bullish(self):
        """Positive change above 0.3 → bullish."""
        dxy = DXYData(price=105.0, change_24h_pct=0.5)
        assert dxy.trend == "bullish"

    def test_trend_bearish(self):
        """Negative change below -0.3 → bearish."""
        dxy = DXYData(price=104.0, change_24h_pct=-0.5)
        assert dxy.trend == "bearish"

    def test_trend_neutral(self):
        """Small change → neutral."""
        dxy = DXYData(price=104.5, change_24h_pct=0.1)
        assert dxy.trend == "neutral"

    def test_trend_neutral_negative_small(self):
        dxy = DXYData(price=104.5, change_24h_pct=-0.2)
        assert dxy.trend == "neutral"

    def test_trend_edge_positive(self):
        """Edge: exactly 0.3 is neutral (threshold is > not >=)."""
        dxy = DXYData(price=105.0, change_24h_pct=0.3)
        assert dxy.trend == "neutral"

    def test_trend_edge_negative(self):
        dxy = DXYData(price=104.0, change_24h_pct=-0.3)
        assert dxy.trend == "neutral"

    def test_trend_just_under_positive(self):
        """Just under 0.3 threshold should be neutral."""
        dxy = DXYData(price=104.5, change_24h_pct=0.29)
        assert dxy.trend == "neutral"

    def test_trend_just_under_negative(self):
        dxy = DXYData(price=104.5, change_24h_pct=-0.29)
        assert dxy.trend == "neutral"

    def test_minimal_init(self):
        """Create with just price and change."""
        dxy = DXYData(price=104.0, change_24h_pct=0.0)
        assert dxy.price == 104.0
        assert dxy.high_24h is None
        assert dxy.low_24h is None
        assert dxy.timestamp == 0.0


# ─── BONDYIELDDATA TESTS ─────────────────────────────────────────────────

class TestBondYieldData:
    """Tests for BondYieldData."""

    def test_create(self, sample_bond_yields):
        assert sample_bond_yields.yield_10y == 4.25
        assert sample_bond_yields.yield_2y == 4.65
        assert sample_bond_yields.spread_10y_2y == pytest.approx(-0.40)

    def test_is_inverted(self, sample_bond_yields):
        assert sample_bond_yields.is_inverted is True

    def test_not_inverted(self, sample_bond_normal):
        assert sample_bond_normal.is_inverted is False

    def test_inversion_depth(self, sample_bond_yields):
        assert sample_bond_yields.inversion_depth_bps == pytest.approx(40.0)

    def test_normal_no_inversion_depth(self, sample_bond_normal):
        assert sample_bond_normal.inversion_depth_bps == 0.0

    def test_minimal_init(self):
        yields = BondYieldData(yield_10y=4.0, yield_2y=3.5, spread_10y_2y=0.5)
        assert yields.yield_10y == 4.0
        assert yields.yield_2y == 3.5
        assert yields.spread_10y_2y == 0.5
        assert yields.is_inverted is False

    def test_flat_curve(self):
        yields = BondYieldData(yield_10y=4.0, yield_2y=4.0, spread_10y_2y=0.0)
        assert yields.is_inverted is False
        assert yields.inversion_depth_bps == 0.0


# ─── ECONOMICEVENT TESTS ─────────────────────────────────────────────────

class TestEconomicEvent:
    """Tests for EconomicEvent."""

    def test_create(self, sample_event_high):
        assert sample_event_high.date == "2026-07-17"
        assert sample_event_high.event == "CPI (YoY)"
        assert sample_event_high.importance == "High"
        assert sample_event_high.currency == "USD"

    def test_is_high_impact_true(self, sample_event_high):
        assert sample_event_high.is_high_impact is True

    def test_is_high_impact_false(self, sample_event_low):
        assert sample_event_low.is_high_impact is False

    def test_affects_xau_cpi(self, sample_event_high):
        """CPI is in HIGH_IMPACT_EVENTS_XAU."""
        assert sample_event_high.affects_xau() is True

    def test_affects_xau_non_usd(self):
        """Non-USD event should NOT affect XAU even if high impact."""
        event = EconomicEvent(
            date="2026-07-17", time="09:00",
            currency="EUR", event="GDP (YoY)",
            importance="High",
        )
        # Currency check comes first → EUR events don't affect XAU
        assert event.affects_xau() is False

    def test_affects_xau_low_impact(self, sample_event_low):
        """Low impact EUR event should not affect XAU."""
        assert sample_event_low.affects_xau() is False

    def test_affects_xau_fomc(self):
        event = EconomicEvent(
            date="2026-07-17", time="14:00",
            currency="USD", event="FOMC Minutes",
            importance="High",
        )
        assert event.affects_xau() is True

    def test_affects_xau_nfp(self):
        """Non-Farm Payrolls."""
        event = EconomicEvent(
            date="2026-07-17", time="08:30",
            currency="USD", event="Non-Farm Payrolls",
            importance="High",
        )
        assert event.affects_xau() is True

    def test_affects_xau_unrelated(self):
        """Unrelated event should not affect XAU."""
        event = EconomicEvent(
            date="2026-07-17", time="10:00",
            currency="USD", event="MBA Mortgage Applications",
            importance="Medium",
        )
        assert event.affects_xau() is False

    def test_high_impact_event_list(self):
        """Check that key events are in the list."""
        assert "CPI" in HIGH_IMPACT_EVENTS_XAU
        assert "FOMC" in HIGH_IMPACT_EVENTS_XAU
        assert "NFP" in HIGH_IMPACT_EVENTS_XAU
        assert "Non-Farm" in HIGH_IMPACT_EVENTS_XAU
        assert "GDP" in HIGH_IMPACT_EVENTS_XAU


# ─── FUNDAMENTALREPORT TESTS ─────────────────────────────────────────────

class TestFundamentalReport:
    """Tests for FundamentalReport."""

    def test_create_minimal(self):
        """Create report with just instrument."""
        report = FundamentalReport(instrument="xau")
        assert report.instrument == "xau"
        assert report.bias == "neutral"
        assert report.bias_score == 0.0
        assert report.confidence == "low"
        assert report.dxy is None
        assert report.bond_yields is None
        assert report.upcoming_events == []

    def test_create_with_data(self, sample_dxy, sample_bond_yields, sample_event_high):
        """Create report with all data."""
        report = FundamentalReport(
            instrument="xau",
            dxy=sample_dxy,
            bond_yields=sample_bond_yields,
            upcoming_events=[sample_event_high],
            bias="bullish",
            bias_score=2.5,
            confidence="high",
            summary="Bullish factors dominate.",
        )
        assert report.dxy.price == 104.50
        assert report.bond_yields.is_inverted is True
        assert len(report.upcoming_events) == 1
        assert report.bias == "bullish"

    def test_detailed_report_format(self, sample_dxy, sample_bond_yields, sample_event_high):
        """Test detailed_report generates proper markdown."""
        report = FundamentalReport(
            instrument="xau",
            dxy=sample_dxy,
            bond_yields=sample_bond_yields,
            upcoming_events=[sample_event_high],
            bias="bullish",
            bias_score=3.0,
            confidence="high",
            summary="Test summary",
        )
        output = report.detailed_report()
        assert "Fundamental Analysis" in output
        assert "XAU" in output
        assert "DXY" in output
        assert "US Treasury" in output
        assert "CPI" in output
        assert "3.0" in output
        assert "bullish" in output.lower()

    def test_detailed_report_minimal(self):
        """Test report with no data."""
        report = FundamentalReport(instrument="btc")
        output = report.detailed_report()
        assert "Fundamental Analysis" in output
        assert "BTC" in output

    def test_xau_correlation_analysis_bearish_dxy(self):
        """Test correlation: bearish DXY → bullish XAU."""
        dxy = DXYData(price=103.0, change_24h_pct=-0.8)
        report = FundamentalReport(instrument="xau", dxy=dxy)
        analysis = report._xau_correlation_analysis()
        assert "DXY giảm" in analysis
        assert "hỗ trợ" in analysis

    def test_xau_correlation_analysis_bullish_dxy(self):
        """Test correlation: bullish DXY → bearish XAU."""
        dxy = DXYData(price=105.0, change_24h_pct=0.6)
        report = FundamentalReport(instrument="xau", dxy=dxy)
        analysis = report._xau_correlation_analysis()
        assert "DXY tăng" in analysis
        assert "áp lực" in analysis

    def test_xau_correlation_high_yield(self):
        """Test high 10Y yield impact."""
        yields = BondYieldData(yield_10y=4.8, yield_2y=4.2, spread_10y_2y=0.6)
        report = FundamentalReport(instrument="xau", bond_yields=yields)
        analysis = report._xau_correlation_analysis()
        assert "chi phí cơ hội" in analysis

    def test_xau_correlation_inverted_curve(self, sample_bond_yields):
        """Test inverted yield curve impact."""
        report = FundamentalReport(instrument="xau", bond_yields=sample_bond_yields)
        analysis = report._xau_correlation_analysis()
        assert "đảo ngược" in analysis
        assert "safe-haven" in analysis


# ─── FETCH DXY TESTS ─────────────────────────────────────────────────────

class TestFetchDXY:
    """Tests for fetch_dxy (mocked)."""

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_success(self, mock_fetch):
        """Test successful DXY fetch."""
        mock_fetch.return_value = {
            "price": 104.50,
            "change_pct": 0.35,
            "timestamp": time.time(),
            "high_24h": 104.80,
            "low_24h": 104.20,
        }
        dxy = fetch_dxy(use_cache=False)
        assert dxy is not None
        assert dxy.price == 104.50
        assert dxy.change_24h_pct == 0.35
        assert dxy.high_24h == 104.80
        assert dxy.low_24h == 104.20

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_failure(self, mock_fetch):
        """Test failed DXY fetch returns None."""
        mock_fetch.return_value = None
        dxy = fetch_dxy(use_cache=False)
        assert dxy is None

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_edge_high_price(self, mock_fetch):
        """Test with extreme DXY value."""
        mock_fetch.return_value = {
            "price": 120.0,
            "change_pct": 2.5,
            "timestamp": time.time(),
        }
        dxy = fetch_dxy(use_cache=False)
        assert dxy is not None
        assert dxy.price == 120.0
        assert dxy.trend == "bullish"

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_dxy_cache(self, mock_fetch):
        """Test that cache is used correctly."""
        mock_fetch.return_value = {
            "price": 104.0,
            "change_pct": 0.1,
            "timestamp": time.time(),
        }
        dxy1 = fetch_dxy(use_cache=True)
        dxy2 = fetch_dxy(use_cache=True)
        assert dxy1.price == dxy2.price


# ─── FETCH BOND YIELDS TESTS ─────────────────────────────────────────────

class TestFetchBondYields:
    """Tests for fetch_bond_yields (mocked)."""

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_success_inverted(self, mock_fetch):
        """Test successful bond yield fetch with inverted curve."""
        # Return different values for 10Y and 2Y
        mock_fetch.side_effect = [
            {"price": 4.25, "change_pct": -0.05, "timestamp": time.time()},  # 10Y
            {"price": 4.65, "change_pct": 0.02, "timestamp": time.time()},   # 2Y
        ]
        yields = fetch_bond_yields(use_cache=False)
        assert yields is not None
        assert yields.yield_10y == 4.25
        assert yields.yield_2y == 4.65
        assert yields.is_inverted is True

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_success_normal(self, mock_fetch):
        """Test normal yield curve."""
        mock_fetch.side_effect = [
            {"price": 4.10, "change_pct": 0.03, "timestamp": time.time()},
            {"price": 3.80, "change_pct": -0.01, "timestamp": time.time()},
        ]
        yields = fetch_bond_yields(use_cache=False)
        assert yields is not None
        assert yields.is_inverted is False
        assert yields.spread_10y_2y == pytest.approx(0.30)

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_failure(self, mock_fetch):
        """Test when one yield fails."""
        mock_fetch.return_value = None
        yields = fetch_bond_yields(use_cache=False)
        assert yields is None

    @patch("fundamental._fetch_yahoo_quote")
    def test_fetch_partial_failure(self, mock_fetch):
        """Test when only one yield fetches."""
        mock_fetch.side_effect = [
            {"price": 4.25, "change_pct": 0.0, "timestamp": time.time()},
            None,  # 2Y fails
        ]
        yields = fetch_bond_yields(use_cache=False)
        assert yields is None


# ─── FETCH ECONOMIC CALENDAR TESTS ──────────────────────────────────────

class TestFetchEconomicCalendar:
    """Tests for fetch_economic_calendar."""

    @patch("fundamental._scrape_forexfactory")
    def test_fetch_with_events(self, mock_scrape):
        """Test calendar with events."""
        mock_scrape.return_value = [
            EconomicEvent(
                date="2026-07-17", time="08:30", currency="USD",
                event="CPI (YoY)", importance="High",
            ),
        ]
        events = fetch_economic_calendar(days_ahead=3, use_cache=False)
        assert len(events) >= 1
        assert events[0].event == "CPI (YoY)"

    @patch("fundamental._scrape_forexfactory")
    def test_fetch_empty_calendar(self, mock_scrape):
        """Test calendar with no events."""
        mock_scrape.return_value = []
        events = fetch_economic_calendar(days_ahead=3, use_cache=False)
        assert events == []

    def test_fetch_calendar_fallback(self):
        """Test fallback when scraper fails by calling the fallback directly."""
        from fundamental import _calendar_fallback
        events = _calendar_fallback(days_ahead=3)
        # Should return events based on day of week
        assert isinstance(events, list)
        if len(events) > 0:
            assert all(isinstance(e, EconomicEvent) for e in events)


# ─── FUNDAMENTAL BIAS TESTS ─────────────────────────────────────────────

class TestFundamentalBias:
    """Tests for fundamental_bias analysis."""

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_bias_bullish(
        self, mock_calendar, mock_bonds, mock_dxy
    ):
        """Test bullish fundamental bias."""
        mock_dxy.return_value = DXYData(price=103.0, change_24h_pct=-0.8)
        mock_bonds.return_value = BondYieldData(
            yield_10y=3.2, yield_2y=3.0, spread_10y_2y=0.2,
        )
        mock_calendar.return_value = []

        report = fundamental_bias("xau")
        assert report.bias == "bullish"
        assert report.bias_score > 0

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_bias_bearish(
        self, mock_calendar, mock_bonds, mock_dxy
    ):
        """Test bearish fundamental bias."""
        mock_dxy.return_value = DXYData(price=106.0, change_24h_pct=0.8)
        mock_bonds.return_value = BondYieldData(
            yield_10y=5.0, yield_2y=4.5, spread_10y_2y=0.5,
        )
        mock_calendar.return_value = []

        report = fundamental_bias("xau")
        assert report.bias == "bearish"
        assert report.bias_score < 0

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_bias_neutral(
        self, mock_calendar, mock_bonds, mock_dxy
    ):
        """Test neutral fundamental bias."""
        mock_dxy.return_value = DXYData(price=104.5, change_24h_pct=0.1)
        mock_bonds.return_value = BondYieldData(
            yield_10y=4.0, yield_2y=3.8, spread_10y_2y=0.2,
        )
        mock_calendar.return_value = []

        report = fundamental_bias("xau")
        assert report.bias == "neutral"

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_bias_with_high_impact_events(
        self, mock_calendar, mock_bonds, mock_dxy
    ):
        """Test that high-impact events reduce confidence."""
        mock_dxy.return_value = DXYData(price=104.0, change_24h_pct=-0.2)
        mock_bonds.return_value = None
        mock_calendar.return_value = [
            EconomicEvent(
                date="2026-07-17", time="08:30", currency="USD",
                event="CPI (YoY)", importance="High",
            ),
            EconomicEvent(
                date="2026-07-18", time="14:00", currency="USD",
                event="FOMC Minutes", importance="High",
            ),
        ]

        report = fundamental_bias("xau")
        assert report.bias_score < 2.0  # Events reduce score

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_bias_all_none(
        self, mock_calendar, mock_bonds, mock_dxy
    ):
        """Test when all data sources return None."""
        mock_dxy.return_value = None
        mock_bonds.return_value = None
        mock_calendar.return_value = []

        report = fundamental_bias("xau")
        assert report.bias == "neutral"
        assert report.bias_score == 0.0
        assert report.dxy is None
        assert report.bond_yields is None
        assert report.upcoming_events == []


# ─── FUNDAMENTAL REPORT TESTS ──────────────────────────────────────────

class TestFundamentalReportFunction:
    """Tests for fundamental_report convenience function."""

    @patch("fundamental.fetch_dxy")
    @patch("fundamental.fetch_bond_yields")
    @patch("fundamental.fetch_economic_calendar")
    def test_report_output(self, mock_calendar, mock_bonds, mock_dxy):
        """Test report produces valid output."""
        mock_dxy.return_value = DXYData(price=104.0, change_24h_pct=-0.3)
        mock_bonds.return_value = BondYieldData(
            yield_10y=4.0, yield_2y=4.3, spread_10y_2y=-0.3,
        )
        mock_calendar.return_value = []

        output = fundamental_report("xau")
        assert isinstance(output, str)
        assert len(output) > 100
        assert "Fundamental Analysis" in output


# ─── EDGE CASES ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases for fundamental module."""

    def test_dxy_zero_change(self):
        """DXY with zero change."""
        dxy = DXYData(price=104.0, change_24h_pct=0.0)
        assert dxy.trend == "neutral"

    def test_bond_yield_equal(self):
        """Bond yields equal (flat curve)."""
        yields = BondYieldData(yield_10y=4.0, yield_2y=4.0, spread_10y_2y=0.0)
        assert yields.is_inverted is False

    def test_event_with_none_fields(self):
        """Event with None forecast/previous."""
        event = EconomicEvent(
            date="2026-07-17", time="08:30", currency="USD",
            event="Test Event", importance="Low",
        )
        assert event.forecast is None
        assert event.previous is None
        assert event.actual is None
        assert event.affects_xau() is False

    def test_report_with_no_upcoming(self):
        """Report with empty upcoming events."""
        report = FundamentalReport(instrument="xau")
        output = report.detailed_report()
        assert "Upcoming" not in output  # Should not have calendar section
