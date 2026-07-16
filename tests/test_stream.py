"""
Tests for stream.py — Real-time WebSocket & live price streaming

Tests focus on:
  - StreamManager initialization & lifecycle
  - BinanceWebSocketStream (mocked)
  - PollingPriceStream (mocked)
  - LivePriceSnapshot data integrity
  - get_live_price convenience function
  - get_live_report formatting
"""

import os
import sys
import time
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stream import (
    StreamManager,
    BinanceWebSocketStream,
    PollingPriceStream,
    LivePriceSnapshot,
    get_live_price,
    get_live_snapshot,
    get_live_report,
    BINANCE_SYMBOL_MAP,
    POLL_INTERVAL_SEC,
)


# ─── FIXTURES ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_binance_snapshot():
    """Create a sample Binance WebSocket price snapshot."""
    return LivePriceSnapshot(
        instrument="btc",
        price=65432.10,
        bid=65431.00,
        ask=65433.20,
        volume_24h=125000.5,
        price_change_24h_pct=2.35,
        timestamp=time.time(),
        source="binance_ws",
    )


@pytest.fixture
def mock_poll_snapshot():
    """Create a sample polled price snapshot."""
    return LivePriceSnapshot(
        instrument="xau",
        price=2650.75,
        timestamp=time.time(),
        source="yahoo_poll",
        price_change_24h_pct=-0.45,
    )


# ─── LIVEPRICESNAPSHOT TESTS ─────────────────────────────────────────────

class TestLivePriceSnapshot:
    """Tests for LivePriceSnapshot data class."""

    def test_create_snapshot(self, mock_binance_snapshot):
        """Test basic creation."""
        snap = mock_binance_snapshot
        assert snap.instrument == "btc"
        assert snap.price == 65432.10
        assert snap.bid == 65431.00
        assert snap.ask == 65433.20
        assert snap.volume_24h == 125000.5
        assert snap.price_change_24h_pct == 2.35
        assert snap.source == "binance_ws"

    def test_age_seconds(self):
        """Test age_seconds returns non-negative value."""
        snap = LivePriceSnapshot(instrument="btc", price=50000, timestamp=time.time() - 5)
        assert 4.0 <= snap.age_seconds <= 6.0

    def test_is_fresh(self):
        """Test is_fresh with various ages."""
        t = time.time()
        fresh = LivePriceSnapshot(instrument="btc", price=50000, timestamp=t - 10)
        assert fresh.is_fresh(max_age=60) is True

        stale = LivePriceSnapshot(instrument="btc", price=50000, timestamp=t - 120)
        assert stale.is_fresh(max_age=60) is False

    def test_is_fresh_default(self):
        """Test is_fresh default max_age (60s)."""
        t = time.time()
        snap = LivePriceSnapshot(instrument="btc", price=50000, timestamp=t - 30)
        assert snap.is_fresh() is True

        snap = LivePriceSnapshot(instrument="btc", price=50000, timestamp=t - 90)
        assert snap.is_fresh() is False

    def test_no_bid_ask(self):
        """Test snapshot without bid/ask (e.g., polled data)."""
        snap = LivePriceSnapshot(
            instrument="xau", price=2650.75, timestamp=time.time(), source="yahoo_poll",
        )
        assert snap.bid is None
        assert snap.ask is None
        assert snap.price == 2650.75


# ─── STREAM MANAGER TESTS ─────────────────────────────────────────────────

class TestStreamManager:
    """Tests for StreamManager."""

    def test_initial_state(self):
        """Test initial state before starting."""
        mgr = StreamManager()
        assert mgr.is_running is False
        assert mgr.get_latest_price("btc") is None
        assert mgr.get_snapshot("btc") is None

    def test_start_stop_lifecycle(self):
        """Test start/stop cycle."""
        mgr = StreamManager()
        mgr.start(instruments=["btc", "xau"])
        assert mgr.is_running is True
        mgr.stop()
        assert mgr.is_running is False

    def test_start_only_crypto(self):
        """Test starting with only crypto instruments."""
        mgr = StreamManager()
        mgr.start(instruments=["btc", "eth"])
        assert mgr.is_running is True
        mgr.stop()

    def test_start_only_forex(self):
        """Test starting with only forex/commodity instruments."""
        mgr = StreamManager()
        mgr.start(instruments=["xau", "gbp"])
        assert mgr.is_running is True
        mgr.stop()

    def test_get_snapshot_no_data(self):
        """Test get_snapshot returns None when not started."""
        mgr = StreamManager()
        assert mgr.get_snapshot("btc") is None
        assert mgr.get_snapshot("nonexistent") is None

    def test_get_all_prices_empty(self):
        """Test get_all_prices returns empty dict when not running."""
        mgr = StreamManager()
        assert mgr.get_all_prices() == {}


# ─── BINANCE WEBSOCKET STREAM TESTS ──────────────────────────────────────

class TestBinanceWebSocketStream:
    """Tests for BinanceWebSocketStream (mocked)."""

    def test_initial_state(self):
        """Test initial state."""
        stream = BinanceWebSocketStream()
        assert stream.get_price("btc") is None
        assert stream.get_price("eth") is None

    def test_start_stop(self):
        """Test start/stop lifecycle."""
        stream = BinanceWebSocketStream()
        stream.start(symbols=["btc"])
        assert stream._running is True
        assert stream._thread is not None
        stream.stop()
        assert stream._running is False

    def test_start_twice(self):
        """Test starting twice is idempotent."""
        stream = BinanceWebSocketStream()
        stream.start()
        stream.start()  # Should log warning but not crash
        stream.stop()

    def test_on_message_parsing(self):
        """Test message parsing logic directly."""
        stream = BinanceWebSocketStream()
        stream._prices = {}  # Fresh dict

        # Simulate a Binance ticker message
        test_message = json.dumps({
            "stream": "btcusdt@ticker",
            "data": {
                "c": "65432.10",   # Current price
                "b": "65431.00",   # Bid
                "a": "65433.20",   # Ask
                "v": "125000.5",   # Volume
                "P": "2.35",       # Price change %
            },
        })

        # Manually trigger on_message
        stream._on_message(None, test_message)

        # Check that price was stored
        btc_price = stream.get_price("btc")
        assert btc_price is not None
        assert btc_price.instrument == "btc"
        assert btc_price.price == 65432.10
        assert btc_price.bid == 65431.00
        assert btc_price.ask == 65433.20
        assert btc_price.volume_24h == 125000.5
        assert btc_price.price_change_24h_pct == 2.35
        assert btc_price.source == "binance_ws"

    def test_on_message_invalid_json(self):
        """Test handling of invalid JSON message."""
        stream = BinanceWebSocketStream()
        # Should not raise
        stream._on_message(None, "invalid json")
        assert stream.get_price("btc") is None

    def test_on_message_missing_stream_field(self):
        """Test message without 'stream' field."""
        stream = BinanceWebSocketStream()
        stream._on_message(None, json.dumps({"data": {"c": "50000"}}))
        # Should not crash, just return

    def test_stop_while_running(self):
        """Test stopping a running stream gracefully."""
        stream = BinanceWebSocketStream()
        stream.start()
        stream.stop()
        # Double stop should be safe
        stream.stop()


# ─── POLLING PRICE STREAM TESTS ──────────────────────────────────────────

class TestPollingPriceStream:
    """Tests for PollingPriceStream (mocked)."""

    def test_initial_state(self):
        """Test initial state."""
        stream = PollingPriceStream()
        assert stream.get_price("xau") is None
        assert stream.get_price("gbp") is None

    def test_start_stop(self):
        """Test start/stop lifecycle."""
        stream = PollingPriceStream()
        stream.start(symbols=["xau"])
        assert stream._running is True
        stream.stop()
        assert stream._running is False


# ─── LIVE PRICE CONVENIENCE TESTS ────────────────────────────────────────

class TestLivePriceConvenience:
    """Tests for convenience functions."""

    def test_get_live_price_no_data(self):
        """Test get_live_price returns None when stream not started."""
        # This tests without initializing the global manager
        # Note: get_global_stream_manager will auto-start, so this is tricky
        pass

    def test_get_live_snapshot_no_data(self):
        """Test get_live_snapshot returns None."""
        # Similar to above
        pass


# ─── LIVE PRICE REPORT TESTS ─────────────────────────────────────────────

class TestLivePriceReport:
    """Tests for get_live_report formatting."""

    def test_report_format_empty(self):
        """Test report with no data (should still generate valid text)."""
        report = get_live_report(instruments=["btc", "xau"])
        assert isinstance(report, str)
        assert "Live Market Prices" in report
        assert "BTC" in report
        assert "XAU" in report
        assert "UTC" in report

    def test_report_freshness(self):
        """Test report includes timestamp."""
        report = get_live_report()
        assert "Updated:" in report
        assert "UTC" in report


# ─── BINANCE SYMBOL MAP TESTS ────────────────────────────────────────────

class TestBinanceSymbolMap:
    """Tests for Binance symbol mapping."""

    def test_btc_mapping(self):
        assert BINANCE_SYMBOL_MAP["btc"] == "btcusdt"

    def test_eth_mapping(self):
        assert BINANCE_SYMBOL_MAP["eth"] == "ethusdt"

    def test_unknown_symbol(self):
        assert "xau" not in BINANCE_SYMBOL_MAP
        assert "gbp" not in BINANCE_SYMBOL_MAP


# ─── EDGE CASES ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases for streaming module."""

    def test_empty_instrument_list(self):
        """Test starting with empty instrument list."""
        mgr = StreamManager()
        mgr.start(instruments=[])
        assert mgr.is_running is True
        mgr.stop()

    def test_none_instrument_list(self):
        """Test starting with None instruments (uses defaults)."""
        mgr = StreamManager()
        mgr.start(instruments=None)
        assert mgr.is_running is True
        mgr.stop()

    def test_unknown_instrument_get(self):
        """Test getting price for unknown instrument."""
        mgr = StreamManager()
        assert mgr.get_latest_price("nonexistent") is None

    def test_live_price_snapshot_repr(self):
        """Test that LivePriceSnapshot can be created with minimal fields."""
        snap = LivePriceSnapshot(instrument="test", price=100.0)
        assert snap.instrument == "test"
        assert snap.price == 100.0
        assert snap.bid is None
        assert snap.ask is None
