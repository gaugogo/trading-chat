"""
data_provider.py — Abstract data provider + multi-source fallback

Design:
  BaseDataProvider (ABC)
    ├── YahooFinanceProvider    (primary)
    ├── AlphaVantageProvider    (fallback)
    └── ... (future: Polygon.io, OANDA, Binance)

Features:
  - Circuit breaker pattern (auto-disable failing providers)
  - Retry with exponential backoff
  - Cross-provider validation
  - Adaptive cache (session-aware TTL)

Usage:
  from data_provider import DataProviderFactory
  provider = DataProviderFactory.create(use_fallback=True)
  df = provider.fetch_ohlcv("BTC-USD", "1d", "1mo")
"""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from core import CACHE_DIR, resample_manual

logger = logging.getLogger(__name__)


# ─── CONFIG ────────────────────────────────────────────────────────────

# Market hours for XAU (forex) — roughly 24h but reduced on weekends
# For session-based adaptive cache
FOREX_SESSION_OPEN = 22  # Sunday 22:00 UTC (Sydney open)
FOREX_SESSION_CLOSE = 21  # Friday 21:00 UTC (NY close)
STOCK_MARKET_OPEN_HOUR = 13  # 13:30 UTC = 9:30 AM ET
STOCK_MARKET_CLOSE_HOUR = 20  # 20:00 UTC = 4:00 PM ET
CRYPTO_247 = True  # Crypto never sleeps

# Circuit breaker defaults
CIRCUIT_BREAKER_THRESHOLD = 3   # Failures before opening circuit
CIRCUIT_BREAKER_RESET_SEC = 60  # Seconds before trying again


# ─── EXCEPTIONS ────────────────────────────────────────────────────────

class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass

class RateLimitError(DataProviderError):
    """API rate limit exceeded."""
    pass

class NoDataError(DataProviderError):
    """No data returned for the requested symbol/timeframe."""
    pass

class CircuitBreakerOpenError(DataProviderError):
    """Provider is temporarily disabled due to too many failures."""
    pass


# ─── CIRCUIT BREAKER ───────────────────────────────────────────────────

@dataclass
class CircuitBreakerState:
    """Tracks failure count and cooldown for a provider."""
    failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    threshold: int = CIRCUIT_BREAKER_THRESHOLD
    reset_seconds: float = CIRCUIT_BREAKER_RESET_SEC

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.is_open = True

    def record_success(self):
        self.failures = 0
        self.is_open = False

    def can_try(self) -> bool:
        if not self.is_open:
            return True
        # Check if enough time has passed to reset
        if time.time() - self.last_failure_time > self.reset_seconds:
            self.is_open = False  # half-open
            return True
        return False

    @property
    def status(self) -> str:
        if self.is_open:
            return f"OPEN (retry in {max(0, self.reset_seconds - (time.time() - self.last_failure_time)):.0f}s)"
        return f"CLOSED ({self.failures}/{self.threshold} failures)"


# ─── ADAPTIVE CACHE ────────────────────────────────────────────────────

def is_market_open(symbol_type: str = "forex") -> bool:
    """Check if the market is currently open for session-based cache.

    Args:
        symbol_type: 'forex', 'crypto', or 'stock'

    Returns:
        True if market is in active session
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour

    if symbol_type == "crypto":
        return True  # 24/7

    if symbol_type == "forex":
        # Forex opens Sunday 22:00 UTC, closes Friday 21:00 UTC
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            return hour >= 22
        if weekday == 4:  # Friday
            return hour < 21
        return True  # Monday-Thursday: 24h

    if symbol_type == "stock":
        # Stock market: Mon-Fri, 9:30-16:00 ET (13:30-20:00 UTC)
        if weekday >= 5:  # Weekend
            return False
        return STOCK_MARKET_OPEN_HOUR <= hour < STOCK_MARKET_CLOSE_HOUR

    return True


def get_adaptive_cache_ttl(base_hours: float, symbol_type: str = "forex") -> float:
    """Get adjusted cache TTL based on market session.

    Reduces TTL when market is open (data changes faster),
    increases when closed (data is stale/won't change).

    Args:
        base_hours: Base cache TTL in hours
        symbol_type: 'forex', 'crypto', or 'stock'

    Returns:
        Adjusted TTL in hours
    """
    if symbol_type == "crypto":
        # Crypto 24/7 — use base TTL but don't cache longer than 30 min for low TFs
        return min(base_hours, 0.5)

    if is_market_open(symbol_type):
        # Market open — reduce TTL by 50%
        return max(base_hours * 0.5, 0.05)  # minimum ~3 min
    else:
        # Market closed — extend TTL by 3x
        return base_hours * 3.0


def get_cache_buster_key(symbol_type: str = "forex") -> str:
    """Generate a cache key suffix that changes with market sessions.

    This prevents serving pre-market data during market hours and vice versa.

    Args:
        symbol_type: 'forex', 'crypto', or 'stock'

    Returns:
        Session key string
    """
    if is_market_open(symbol_type):
        return "open"
    else:
        return "closed"


# ─── RETRY WITH EXPONENTIAL BACKOFF ────────────────────────────────────

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[Tuple] = None,
) -> Any:
    """Execute a function with retry and exponential backoff.

    Args:
        func: Function to call
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry
        retryable_exceptions: Exception types that trigger retry.
            If None, retries on all exceptions.

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

    raise last_exception  # Should not reach here


# ─── INSTRUMENT TYPE DETECTION ────────────────────────────────────────

def detect_symbol_type(instrument_id: str) -> str:
    """Detect symbol type for market session logic.

    Args:
        instrument_id: e.g., 'xau', 'btc', 'gbp'

    Returns:
        'forex', 'crypto', or 'stock'
    """
    crypto_ids = {"btc", "eth", "sol", "xrp", "ada", "doge"}
    if instrument_id in crypto_ids:
        return "crypto"
    # Gold + forex pairs
    forex_ids = {"xau", "gbp", "eur", "jpy", "aud", "nzd", "cad", "chf", "xag"}
    if instrument_id in forex_ids:
        return "forex"
    return "stock"


# ─── BASE DATA PROVIDER ────────────────────────────────────────────────

class BaseDataProvider(ABC):
    """Abstract base class for market data providers.

    Subclasses must implement:
      - provider_name (class attr)
      - fetch_ohlcv_impl()
      - fetch_quote_impl()
    """

    provider_name: str = "base"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get(self._env_key(), "")
        self.circuit_breaker = CircuitBreakerState()
        self._stats = {"calls": 0, "failures": 0, "cache_hits": 0}

    @classmethod
    @abstractmethod
    def _env_key(cls) -> str:
        """Environment variable name for the API key."""
        return ""

    @abstractmethod
    def fetch_ohlcv_impl(
        self,
        symbol: str,
        interval: str,
        range_: str,
    ) -> pd.DataFrame:
        """Actual implementation of OHLCV fetching.

        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD', 'GC=F')
            interval: Bar interval ('1d', '1h', '15m', '5m')
            range_: Data range ('1y', '2mo', '1mo', '5d', '2d')

        Returns:
            DataFrame with OHLCV columns, or empty on failure
        """
        pass

    @abstractmethod
    def fetch_quote_impl(self, symbol: str) -> Optional[float]:
        """Fetch current price quote.

        Args:
            symbol: Ticker symbol

        Returns:
            Current price as float, or None
        """
        pass

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        range_: str,
        resample_to: Optional[int] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        cache_hours: Optional[float] = None,
        symbol_type: str = "forex",
    ) -> pd.DataFrame:
        """Fetch OHLCV with caching, circuit breaker, and retry.

        Args:
            symbol: Ticker symbol
            interval: Bar interval
            range_: Data range
            resample_to: Optional manual resample period (seconds)
            use_cache: Whether to use/save local cache
            cache_key: Unique cache key
            cache_hours: Base cache TTL (will be adapted)
            symbol_type: For adaptive cache ('forex', 'crypto', 'stock')

        Returns:
            DataFrame with OHLCV columns, or empty DataFrame
        """
        self._stats["calls"] += 1

        # Check circuit breaker
        if not self.circuit_breaker.can_try():
            logger.warning(f"[{self.provider_name}] Circuit breaker OPEN — skipping")
            self._stats["failures"] += 1
            return pd.DataFrame()

        # Adaptive cache TTL
        effective_ttl = cache_hours
        if cache_hours is not None:
            effective_ttl = get_adaptive_cache_ttl(cache_hours, symbol_type)

        # Session-based cache key
        actual_cache_key = cache_key
        if use_cache and cache_key and effective_ttl is not None:
            session_suffix = get_cache_buster_key(symbol_type)
            actual_cache_key = f"{cache_key}_{session_suffix}"

        # Check cache
        if use_cache and actual_cache_key and effective_ttl is not None:
            cache_file = CACHE_DIR / f"{actual_cache_key}.json"
            if cache_file.exists():
                age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if age < timedelta(hours=effective_ttl):
                    self._stats["cache_hits"] += 1
                    return pd.read_json(cache_file, orient='split')

        # Fetch with retry
        try:
            df = retry_with_backoff(
                lambda: self.fetch_ohlcv_impl(symbol, interval, range_),
                max_retries=2,
                base_delay=1.0,
            )
        except Exception as e:
            logger.error(f"[{self.provider_name}] Failed to fetch {symbol} {interval}: {e}")
            self.circuit_breaker.record_failure()
            self._stats["failures"] += 1
            return pd.DataFrame()

        if df.empty:
            self.circuit_breaker.record_failure()
            self._stats["failures"] += 1
            return df

        # Manual resample if needed
        if resample_to and not df.empty:
            df = resample_manual(df, resample_to)

        # Save to cache
        if use_cache and actual_cache_key and effective_ttl is not None and not df.empty:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = CACHE_DIR / f"{actual_cache_key}.json"
            df.to_json(cache_file, orient='split', date_format='iso')

        self.circuit_breaker.record_success()
        return df

    def fetch_quote(
        self,
        symbol: str,
        use_cache: bool = True,
        cache_ttl_seconds: float = 60.0,
    ) -> Optional[float]:
        """Fetch current price quote with lightweight caching.

        Args:
            symbol: Ticker symbol
            use_cache: Whether to cache quote
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            Price as float, or None
        """
        # Quote cache in memory (dict, not file)
        _quote_cache: Dict = {}
        now = time.time()
        cache_key = f"quote_{self.provider_name}_{symbol}"

        if use_cache and cache_key in _quote_cache:
            cached_val, cached_time = _quote_cache[cache_key]
            if now - cached_time < cache_ttl_seconds:
                return cached_val

        try:
            price = self.fetch_quote_impl(symbol)
            if price is not None and use_cache:
                _quote_cache[cache_key] = (price, now)
            return price
        except Exception as e:
            logger.error(f"[{self.provider_name}] Quote failed for {symbol}: {e}")
            return None

    def stats(self) -> Dict[str, Any]:
        """Return provider statistics."""
        return {
            "provider": self.provider_name,
            "circuit": self.circuit_breaker.status,
            **self._stats,
        }


# ─── YAHOO FINANCE PROVIDER ───────────────────────────────────────────

class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider (primary).

    Free, no API key needed. Rate-limited but works for moderate usage.
    """

    provider_name = "yahoo"

    @classmethod
    def _env_key(cls) -> str:
        return "YAHOO_API_KEY"  # Not actually used by Yahoo

    def fetch_ohlcv_impl(
        self,
        symbol: str,
        interval: str,
        range_: str,
    ) -> pd.DataFrame:
        """Fetch from Yahoo Finance v8 chart API."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": interval, "range": range_}
            headers = {"User-Agent": "Mozilla/5.0"}

            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()

            data = r.json()
            results = data.get("chart", {}).get("result", [])
            if not results:
                return pd.DataFrame()

            result = results[0]
            timestamps = result.get("timestamp", [])
            if not timestamps:
                return pd.DataFrame()

            quotes = result.get("indicators", {}).get("quote", [{}])[0]
            df = pd.DataFrame(
                {
                    "Open": quotes.get("open", []),
                    "High": quotes.get("high", []),
                    "Low": quotes.get("low", []),
                    "Close": quotes.get("close", []),
                    "Volume": quotes.get("volume", []),
                },
                index=pd.to_datetime(timestamps, unit="s"),
            )
            return df.dropna(how="all")
        except Exception:
            return pd.DataFrame()

    def fetch_quote_impl(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Yahoo Finance quote endpoint."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1m", "range": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()

        data = r.json()
        result = data.get("chart", {}).get("result", [{}])[0]
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        closes = quotes.get("close", [])
        if closes:
            valid = [c for c in closes if c is not None]
            if valid:
                return float(valid[-1])
        return None


# ─── ALPHA VANTAGE PROVIDER ────────────────────────────────────────────

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider (fallback).

    Requires API key from https://www.alphavantage.co/support/#api-key
    Free tier: 5 API calls/min, 500/day.
    """

    provider_name = "alpha_vantage"
    BASE_URL = "https://www.alphavantage.co/query"

    # Alpha Vantage interval mapping
    _INTERVAL_MAP = {
        "1d": "Daily",
        "1h": "60min",
        "15m": "15min",
        "5m": "5min",
        "1m": "1min",
    }

    @classmethod
    def _env_key(cls) -> str:
        return "ALPHA_VANTAGE_API_KEY"

    def fetch_ohlcv_impl(
        self,
        symbol: str,
        interval: str,
        range_: str,
    ) -> pd.DataFrame:
        """Fetch from Alpha Vantage.

        Note: range_ parameter is approximate — Alpha Vantage returns a fixed
        amount of data per call (e.g., 100 data points for intraday).
        """
        if not self.api_key:
            logger.warning("[alpha_vantage] No API key set — skipping")
            return pd.DataFrame()

        av_interval = self._INTERVAL_MAP.get(interval, interval)
        function = "TIME_SERIES_INTRADAY" if interval != "1d" else "TIME_SERIES_DAILY"

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }

        if interval != "1d":
            params["interval"] = av_interval
            if range_ in ("1y", "6mo"):
                params["outputsize"] = "full"
            else:
                params["outputsize"] = "compact"
        else:
            if range_ in ("1y", "max"):
                params["outputsize"] = "full"
            else:
                params["outputsize"] = "compact"

        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Parse response (different key format per interval)
        time_key = None
        for key in data:
            if "Time Series" in key:
                time_key = key
                break

        if not time_key:
            # Check for error message
            if "Note" in data:
                logger.warning(f"[alpha_vantage] API note: {data['Note']}")
                raise RateLimitError(data["Note"])
            return pd.DataFrame()

        records = []
        for date_str, values in data[time_key].items():
            records.append({
                "Open": float(values.get("1. open", 0)),
                "High": float(values.get("2. high", 0)),
                "Low": float(values.get("3. low", 0)),
                "Close": float(values.get("4. close", 0)),
                "Volume": float(values.get("5. volume", 0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(
            records,
            index=pd.to_datetime(list(data[time_key].keys())),
        )
        df.index.name = "Date"
        # Sort ascending
        df.sort_index(inplace=True)
        return df

    def fetch_quote_impl(self, symbol: str) -> Optional[float]:
        """Fetch quote from Alpha Vantage Global Quote endpoint."""
        if not self.api_key:
            return None

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        r = requests.get(self.BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price", "")
        if price_str:
            try:
                return float(price_str)
            except ValueError:
                pass
        return None


# ─── DATA PROVIDER FACTORY ────────────────────────────────────────────

class DataProviderFactory:
    """Creates and manages data providers with fallback chain.

    Usage:
        factory = DataProviderFactory()
        # Primary + auto-fallback
        provider = factory.get_provider()
        df = provider.fetch_ohlcv("GC=F", "1d", "1y")

        # With explicit fallback
        providers = factory.get_all_providers()
        for provider in providers:
            df = provider.fetch_ohlcv(...)
            if not df.empty:
                break
    """

    PROVIDERS = {
        "yahoo": YahooFinanceProvider,
        "alpha_vantage": AlphaVantageProvider,
    }

    PROVIDER_PRIORITY = ["yahoo", "alpha_vantage"]

    def __init__(self):
        self._instances: Dict[str, BaseDataProvider] = {}

    def get_provider(self, name: Optional[str] = None) -> BaseDataProvider:
        """Get a specific provider by name (cached).

        Args:
            name: Provider name ('yahoo', 'alpha_vantage'). Default: primary.

        Returns:
            Provider instance
        """
        name = name or self.PROVIDER_PRIORITY[0]
        if name not in self._instances:
            cls = self.PROVIDERS.get(name)
            if not cls:
                raise ValueError(f"Unknown provider: {name}. Available: {list(self.PROVIDERS.keys())}")
            self._instances[name] = cls()
        return self._instances[name]

    def get_all_providers(self) -> List[BaseDataProvider]:
        """Get all registered providers in priority order.

        Returns:
            List of provider instances
        """
        return [self.get_provider(name) for name in self.PROVIDER_PRIORITY]

    def fetch_with_fallback(
        self,
        symbol: str,
        interval: str,
        range_: str,
        resample_to: Optional[int] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        cache_hours: Optional[float] = None,
        symbol_type: str = "forex",
        validate_cross: bool = False,
    ) -> Tuple[pd.DataFrame, str]:
        """Fetch data with automatic fallback across providers.

        Tries primary provider first. On failure, tries fallback providers.
        Optionally validates cross-provider consistency.

        Args:
            symbol: Ticker symbol
            interval: Bar interval
            range_: Data range
            resample_to: Optional manual resample period
            use_cache: Whether to use cache
            cache_key: Base cache key
            cache_hours: Base cache TTL
            symbol_type: For adaptive cache
            validate_cross: If True, validates results from multiple providers

        Returns:
            Tuple of (DataFrame, provider_name_used)
        """
        results: List[Tuple[pd.DataFrame, str]] = []

        for provider_name in self.PROVIDER_PRIORITY:
            provider = self.get_provider(provider_name)

            df = provider.fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                range_=range_,
                resample_to=resample_to,
                use_cache=use_cache,
                cache_key=cache_key,
                cache_hours=cache_hours,
                symbol_type=symbol_type,
            )

            if not df.empty:
                results.append((df, provider_name))

                # If we got data from the first provider and don't need cross-validation, return it
                if len(results) == 1 and not validate_cross:
                    return df, provider_name

        # If we have multiple results, validate/merge
        if len(results) >= 2 and validate_cross:
            return self._cross_validate(results, symbol_type)

        # Return the first valid result, or empty
        if results:
            return results[0]

        logger.error(f"[DataProvider] All providers failed for {symbol} {interval}")
        return pd.DataFrame(), "none"

    def _cross_validate(
        self,
        results: List[Tuple[pd.DataFrame, str]],
        symbol_type: str,
    ) -> Tuple[pd.DataFrame, str]:
        """Cross-validate results from multiple providers.

        Checks that the latest close prices are within a tolerance threshold.
        If discrepancy is too large, logs a warning.

        Args:
            results: List of (DataFrame, provider_name) tuples
            symbol_type: For tolerance calculation

        Returns:
            Best DataFrame and provider name
        """
        if len(results) < 2:
            return results[0] if results else (pd.DataFrame(), "none")

        # Compare latest closes
        closes = []
        for df, name in results:
            if not df.empty:
                closes.append((float(df["Close"].iloc[-1]), name, df))

        if len(closes) < 2:
            return closes[0][2], closes[0][1] if closes else (pd.DataFrame(), "none")

        # Calculate max discrepancy
        prices = [c[0] for c in closes]
        max_price = max(prices)
        min_price = min(prices)
        discrepancy_pct = abs(max_price - min_price) / ((max_price + min_price) / 2) * 100

        tolerance = 0.5 if symbol_type == "crypto" else 0.1  # 0.5% for crypto, 0.1% for forex

        if discrepancy_pct > tolerance:
            logger.warning(
                f"[DataProvider] Cross-provider price discrepancy {discrepancy_pct:.2f}% "
                f"(tolerance: {tolerance}%): "
                + " vs ".join(f"{name}={price}" for price, name, _ in closes)
            )

        # Return the provider with more data points (likely more reliable)
        best = max(closes, key=lambda c: len(c[2]))
        return best[2], best[1]

    def get_spot_price(
        self,
        symbol: str,
        instrument_id: str = "xau",
    ) -> Optional[float]:
        """Get spot price from Investing.com (cross-provider spot check).

        Args:
            symbol: Yahoo Finance symbol (for backup quote)
            instrument_id: Instrument ID for cache key

        Returns:
            Spot price or None
        """
        # Try investing.com scraping first (existing logic)
        from core import fetch_spot_price as _core_spot
        from instruments import INSTRUMENTS

        cfg = INSTRUMENTS.get(instrument_id)
        spot_url = cfg.get("spot_url") if cfg else None
        if spot_url:
            spot = _core_spot(spot_url)
            if spot:
                return spot

        # Fallback: use Yahoo quote
        provider = self.get_provider("yahoo")
        quote = provider.fetch_quote(symbol)
        return quote

    def stats(self) -> Dict[str, Any]:
        """Return stats for all providers."""
        return {
            name: self.get_provider(name).stats()
            for name in self.PROVIDER_PRIORITY
        }


# ─── CONVENIENCE WRAPPER ──────────────────────────────────────────────

# Global factory instance
_factory: Optional[DataProviderFactory] = None


def get_factory() -> DataProviderFactory:
    """Get or create the global DataProviderFactory singleton."""
    global _factory
    if _factory is None:
        _factory = DataProviderFactory()
    return _factory


def fetch_ohlcv(
    symbol: str,
    interval: str,
    range_: str,
    resample_to: Optional[int] = None,
    use_cache: bool = True,
    cache_key: Optional[str] = None,
    cache_hours: Optional[float] = None,
    symbol_type: str = "forex",
    use_fallback: bool = True,
) -> pd.DataFrame:
    """Convenience function: fetch OHLCV with automatic fallback.

    Args:
        symbol: Ticker symbol
        interval: Bar interval
        range_: Data range
        resample_to: Optional manual resample
        use_cache: Whether to cache
        cache_key: Cache key
        cache_hours: Base TTL
        symbol_type: For adaptive cache
        use_fallback: Whether to try fallback providers on failure

    Returns:
        DataFrame with OHLCV data
    """
    factory = get_factory()

    if use_fallback:
        df, _ = factory.fetch_with_fallback(
            symbol=symbol,
            interval=interval,
            range_=range_,
            resample_to=resample_to,
            use_cache=use_cache,
            cache_key=cache_key,
            cache_hours=cache_hours,
            symbol_type=symbol_type,
        )
        return df
    else:
        provider = factory.get_provider("yahoo")
        return provider.fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            range_=range_,
            resample_to=resample_to,
            use_cache=use_cache,
            cache_key=cache_key,
            cache_hours=cache_hours,
            symbol_type=symbol_type,
        )
