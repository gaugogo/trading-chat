"""
stream.py — Real-time & near-real-time market data streaming

Provides:
  - Binance WebSocket for BTC/USDT (real-time 1m candles & tick data)
  - Binance WebSocket for ETH/USDT
  - Polling-based "live" updates for XAU (since OANDA WebSocket is premium)
  - Yahoo Finance 1m poll fallback for all instruments
  - Auto-reconnection with exponential backoff

Usage:
  from stream import get_live_price, stream_forever, StreamManager

  mgr = StreamManager()
  mgr.start(instrument="btc")   # starts Binance WebSocket for BTC
  price = mgr.get_latest_price("btc")
  print(f"BTC live: {price}")
  mgr.stop()
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import get_logger

logger = get_logger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_REST_BASE = "https://api.binance.com/api/v3"

# Which instruments can use Binance WebSocket (crypto pairs)
BINANCE_SYMBOL_MAP = {
    "btc": "btcusdt",
    "eth": "ethusdt",
    "sol": "solusdt",
    "xrp": "xrpusdt",
}

# Poll interval for non-WebSocket instruments (XAU, GBP)
POLL_INTERVAL_SEC = 30  # Check Yahoo every 30 seconds
WS_RECONNECT_DELAY = 5  # Initial reconnect delay (seconds)
WS_MAX_RECONNECT_DELAY = 120  # Max reconnect delay


@dataclass
class LivePriceSnapshot:
    """Latest price snapshot from any source."""
    instrument: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h_pct: Optional[float] = None
    timestamp: float = 0.0
    source: str = "unknown"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_fresh(self, max_age: float = 60.0) -> bool:
        """Check if the snapshot is fresh (within max_age seconds)."""
        return self.age_seconds < max_age


# ─── BINANCE WEBSOCKET STREAM (for crypto) ───────────────────────────────

class BinanceWebSocketStream:
    """WebSocket connection to Binance for real-time crypto prices.

    Uses a background thread to maintain the connection.
    Auto-reconnects with exponential backoff.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._prices: Dict[str, LivePriceSnapshot] = {}
        self._lock = threading.Lock()
        self._reconnect_count = 0

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start the WebSocket stream in a background thread.

        Args:
            symbols: List of instrument IDs ('btc', 'eth', etc.).
                     Defaults to all supported crypto.
        """
        if self._running:
            logger.warning("[BinanceWS] Already running")
            return

        self._running = True
        self._symbols = symbols or list(BINANCE_SYMBOL_MAP.keys())
        self._reconnect_count = 0
        self._thread = threading.Thread(
            target=self._run_loop,
            name="binance-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[BinanceWS] Started for: {', '.join(self._symbols)}")

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        logger.info("[BinanceWS] Stopped")

    def get_price(self, instrument: str) -> Optional[LivePriceSnapshot]:
        """Get the latest price for an instrument.

        Args:
            instrument: Instrument ID ('btc', 'eth', etc.)

        Returns:
            LivePriceSnapshot or None
        """
        with self._lock:
            return self._prices.get(instrument)

    def _run_loop(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                self._connect_and_listen()
            except Exception as e:
                logger.error(f"[BinanceWS] Connection error: {e}")

            if not self._running:
                break

            # Exponential backoff reconnection
            delay = min(
                WS_RECONNECT_DELAY * (2 ** self._reconnect_count),
                WS_MAX_RECONNECT_DELAY,
            )
            self._reconnect_count += 1
            logger.info(f"[BinanceWS] Reconnecting in {delay:.0f}s (attempt {self._reconnect_count})")

            deadline = time.time() + delay
            while self._running and time.time() < deadline:
                time.sleep(0.5)

    def _connect_and_listen(self) -> None:
        """Connect to Binance WebSocket and listen for messages."""
        import websocket

        # Build stream names: btcusdt@ticker, ethusdt@ticker, ...
        streams = []
        for instr_id in self._symbols:
            pair = BINANCE_SYMBOL_MAP.get(instr_id)
            if pair:
                streams.append(f"{pair}@ticker")

        if not streams:
            logger.warning("[BinanceWS] No valid symbols to stream")
            return

        stream_name = "/".join(streams)
        url = f"{BINANCE_WS_BASE}/{stream_name}"

        ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        # Run forever (blocking, runs in thread)
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_message(self, ws, message: str) -> None:
        """Process incoming ticker message from Binance."""
        try:
            data = json.loads(message)
            if "stream" not in data:
                return

            stream_name = data["stream"]  # e.g., "btcusdt@ticker"
            ticker = data.get("data", {})

            # Extract instrument ID from stream name
            for instr_id, pair in BINANCE_SYMBOL_MAP.items():
                if stream_name.startswith(pair):
                    # Parse ticker data
                    price = float(ticker.get("c", 0))  # Current price
                    bid = float(ticker.get("b", 0))   # Best bid
                    ask = float(ticker.get("a", 0))   # Best ask
                    volume = float(ticker.get("v", 0)) # Volume (24h)
                    change_pct = float(ticker.get("P", 0))  # Price change % (24h)

                    snapshot = LivePriceSnapshot(
                        instrument=instr_id,
                        price=price,
                        bid=bid,
                        ask=ask,
                        volume_24h=volume,
                        price_change_24h_pct=change_pct,
                        timestamp=time.time(),
                        source="binance_ws",
                    )

                    with self._lock:
                        self._prices[instr_id] = snapshot
                    break

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"[BinanceWS] Parse error: {e}")

    def _on_error(self, ws, error) -> None:
        logger.error(f"[BinanceWS] Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.info(f"[BinanceWS] Closed (code={close_status_code})")

    def _on_open(self, ws) -> None:
        logger.info("[BinanceWS] Connected")
        self._reconnect_count = 0


# ─── POLLING-BASED LIVE PRICES (for XAU, GBP) ─────────────────────────────

class PollingPriceStream:
    """Polling-based live price updates for non-crypto instruments.

    Uses Yahoo Finance 1m interval for near-real-time prices.
    Falls back to Investing.com spot scraping.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prices: Dict[str, LivePriceSnapshot] = {}
        self._lock = threading.Lock()

        # Yahoo Finance symbol mapping
        self._yahoo_symbols = {
            "xau": "GC=F",
            "gbp": "GBPUSD=X",
            "eur": "EURUSD=X",
            "xag": "SI=F",
        }

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start polling in a background thread.

        Args:
            symbols: List of instrument IDs ('xau', 'gbp', etc.).
                     Defaults to all supported forex/commodities.
        """
        if self._running:
            return

        self._running = True
        self._symbols = symbols or list(self._yahoo_symbols.keys())
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="polling-stream",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[PollStream] Started for: {', '.join(self._symbols)}")

    def stop(self) -> None:
        """Stop polling."""
        self._running = False
        logger.info("[PollStream] Stopped")

    def get_price(self, instrument: str) -> Optional[LivePriceSnapshot]:
        """Get the latest polled price.

        Args:
            instrument: Instrument ID ('xau', 'gbp', etc.)

        Returns:
            LivePriceSnapshot or None
        """
        with self._lock:
            return self._prices.get(instrument)

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            for instr_id in self._symbols:
                if not self._running:
                    break
                try:
                    self._poll_single(instr_id)
                except Exception as e:
                    logger.debug(f"[PollStream] Error polling {instr_id}: {e}")

            # Sleep for poll interval
            deadline = time.time() + POLL_INTERVAL_SEC
            while self._running and time.time() < deadline:
                time.sleep(0.5)

    def _poll_single(self, instr_id: str) -> None:
        """Poll a single instrument from Yahoo Finance.

        Args:
            instr_id: Instrument ID
        """
        yahoo_symbol = self._yahoo_symbols.get(instr_id)
        if not yahoo_symbol:
            return

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {"interval": "1m", "range": "1d"}
            headers = {"User-Agent": "Mozilla/5.0"}

            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()

            result = data.get("chart", {}).get("result", [{}])[0]
            meta = result.get("meta", {})
            quote = result.get("indicators", {}).get("quote", [{}])[0]

            # Get latest close
            closes = quote.get("close", [])
            if closes:
                valid_closes = [c for c in closes if c is not None]
                if valid_closes:
                    price = float(valid_closes[-1])
                    prev_close = meta.get("previousClose", price)
                    change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0

                    snapshot = LivePriceSnapshot(
                        instrument=instr_id,
                        price=price,
                        timestamp=time.time(),
                        source="yahoo_poll",
                        price_change_24h_pct=change_pct,
                    )

                    with self._lock:
                        self._prices[instr_id] = snapshot

        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            logger.debug(f"[PollStream] Yahoo fetch failed for {instr_id}: {e}")

            # Fallback: try Investing.com spot scraping
            self._poll_spot_fallback(instr_id)

    def _poll_spot_fallback(self, instr_id: str) -> None:
        """Fallback: scrape spot price from Investing.com.

        Args:
            instr_id: Instrument ID
        """
        try:
            from instruments import INSTRUMENTS
            from core import fetch_spot_price

            cfg = INSTRUMENTS.get(instr_id)
            if cfg and cfg.get("has_spot"):
                spot_url = cfg.get("spot_url")
                if spot_url:
                    price = fetch_spot_price(spot_url)
                    if price:
                        snapshot = LivePriceSnapshot(
                            instrument=instr_id,
                            price=price,
                            timestamp=time.time(),
                            source="investing_scrape",
                        )
                        with self._lock:
                            self._prices[instr_id] = snapshot
        except Exception as e:
            logger.debug(f"[PollStream] Spot scrape failed for {instr_id}: {e}")


# ─── UNIFIED STREAM MANAGER ───────────────────────────────────────────────

class StreamManager:
    """Unified manager that starts appropriate stream for each instrument.

    - Crypto (BTC, ETH, SOL): Binance WebSocket (real-time)
    - Forex/Commodity (XAU, GBP): Polling from Yahoo + Investing.com fallback
    - Auto-detects instrument type
    """

    def __init__(self):
        self._binance: Optional[BinanceWebSocketStream] = None
        self._poll: Optional[PollingPriceStream] = None
        self._is_running = False

    def start(self, instruments: Optional[List[str]] = None) -> None:
        """Start streaming for specified instruments.

        Args:
            instruments: List of instrument IDs. Defaults to all supported.
        """
        if self._is_running:
            logger.warning("[StreamManager] Already running")
            return

        instruments = instruments or ["btc", "eth", "xau", "gbp"]
        crypto = [i for i in instruments if i in BINANCE_SYMBOL_MAP]
        non_crypto = [i for i in instruments if i not in BINANCE_SYMBOL_MAP]

        if crypto:
            self._binance = BinanceWebSocketStream()
            self._binance.start(crypto)

        if non_crypto:
            self._poll = PollingPriceStream()
            self._poll.start(non_crypto)

        self._is_running = True
        logger.info(f"[StreamManager] Started: crypto={crypto}, poll={non_crypto}")

    def stop(self) -> None:
        """Stop all streams."""
        if self._binance:
            self._binance.stop()
        if self._poll:
            self._poll.stop()
        self._is_running = False
        logger.info("[StreamManager] Stopped")

    def get_latest_price(self, instrument: str) -> Optional[float]:
        """Get the latest price for an instrument from any stream source.

        Args:
            instrument: Instrument ID

        Returns:
            Current price as float, or None
        """
        snapshot = self.get_snapshot(instrument)
        return snapshot.price if snapshot else None

    def get_snapshot(self, instrument: str) -> Optional[LivePriceSnapshot]:
        """Get the latest full snapshot for an instrument.

        Args:
            instrument: Instrument ID

        Returns:
            LivePriceSnapshot or None
        """
        if instrument in BINANCE_SYMBOL_MAP and self._binance:
            return self._binance.get_price(instrument)
        if self._poll:
            return self._poll.get_price(instrument)
        return None

    def get_all_prices(self) -> Dict[str, LivePriceSnapshot]:
        """Get all latest prices as a dict."""
        result: Dict[str, LivePriceSnapshot] = {}

        if self._binance:
            for instr_id in BINANCE_SYMBOL_MAP:
                snapshot = self._binance.get_price(instr_id)
                if snapshot:
                    result[instr_id] = snapshot

        # Also check polling for any non-crypto
        all_ids = set(
            list(BINANCE_SYMBOL_MAP.keys())
            + (["xau", "gbp", "eur", "xag"] if self._poll else [])
        )
        for instr_id in all_ids:
            if instr_id not in result:
                snapshot = self.get_snapshot(instr_id)
                if snapshot:
                    result[instr_id] = snapshot

        return result

    @property
    def is_running(self) -> bool:
        return self._is_running


# ─── CONVENIENCE FUNCTIONS ────────────────────────────────────────────────

_global_manager: Optional[StreamManager] = None


def get_global_stream_manager() -> StreamManager:
    """Get or create the global StreamManager singleton.

    Starts streaming automatically on first call.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = StreamManager()
        _global_manager.start()
    return _global_manager


def get_live_price(instrument: str = "btc") -> Optional[float]:
    """Quick convenience: get live price for any instrument.

    Auto-starts global stream manager if needed.

    Args:
        instrument: Instrument ID ('btc', 'eth', 'xau', 'gbp')

    Returns:
        Current price as float, or None if unavailable
    """
    mgr = get_global_stream_manager()
    return mgr.get_latest_price(instrument)


def get_live_snapshot(instrument: str = "btc") -> Optional[LivePriceSnapshot]:
    """Quick convenience: get full live snapshot.

    Args:
        instrument: Instrument ID

    Returns:
        LivePriceSnapshot or None
    """
    mgr = get_global_stream_manager()
    return mgr.get_snapshot(instrument)


def get_live_report(instruments: Optional[List[str]] = None) -> str:
    """Generate a human-readable live price report.

    Args:
        instruments: List of instrument IDs. Defaults to all.

    Returns:
        Formatted report string
    """
    instruments = instruments or ["btc", "eth", "xau", "gbp"]
    mgr = get_global_stream_manager()

    lines = ["📡 **Live Market Prices**", "=" * 40, ""]
    for instr_id in instruments:
        snapshot = mgr.get_snapshot(instr_id)
        if snapshot:
            change_str = ""
            if snapshot.price_change_24h_pct is not None:
                arrow = "🟢" if snapshot.price_change_24h_pct >= 0 else "🔴"
                change_str = f" {arrow} {snapshot.price_change_24h_pct:+.2f}%"

            lines.append(
                f"**{instr_id.upper()}**: ${snapshot.price:,.2f}{change_str}"
                f"  (via {snapshot.source}, {snapshot.age_seconds:.0f}s ago)"
            )
        else:
            lines.append(f"**{instr_id.upper()}**: ⏳ No data yet")

    lines.append("")
    lines.append("---")
    lines.append(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    return "\n".join(lines)


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Quick test: print live prices once
        print("Fetching live prices...")
        mgr = StreamManager()
        mgr.start()
        time.sleep(3)  # Give it time to connect
        prices = mgr.get_all_prices()
        for instr_id, snap in prices.items():
            print(f"  {instr_id}: ${snap.price:.2f} (source: {snap.source})")
        mgr.stop()
    else:
        # Stream forever
        print("📡 Live market stream started. Press Ctrl+C to stop.")
        mgr = get_global_stream_manager()
        try:
            while True:
                time.sleep(10)
                for instr_id in ["btc", "eth", "xau", "gbp"]:
                    p = mgr.get_latest_price(instr_id)
                    if p:
                        print(f"  {instr_id.upper()}: ${p:,.2f}")
        except KeyboardInterrupt:
            print("\nStopping...")
            mgr.stop()
