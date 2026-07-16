"""
Tests for data_provider.py — DataProvider, adaptive cache, circuit breaker.

Run with: pytest tests/test_data_provider.py -v
"""

import os
import sys
import time
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_provider import (
    is_market_open,
    get_adaptive_cache_ttl,
    get_cache_buster_key,
    retry_with_backoff,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    DataProviderFactory,
    YahooFinanceProvider,
    AlphaVantageProvider,
    detect_symbol_type,
    get_factory,
)


# ─── TESTS: Market Session Detection ───

class TestMarketSession:
    """Test market open/close detection logic."""

    def test_crypto_always_open(self):
        assert is_market_open("crypto") is True

    def test_detect_symbol_type(self):
        assert detect_symbol_type("btc") == "crypto"
        assert detect_symbol_type("eth") == "crypto"
        assert detect_symbol_type("xau") == "forex"
        assert detect_symbol_type("gbp") == "forex"
        assert detect_symbol_type("unknown") == "stock"

    def test_adaptive_cache_crypto(self):
        ttl = get_adaptive_cache_ttl(2.0, "crypto")
        assert ttl <= 0.5  # Crypto capped at 30 min


# ─── TESTS: Circuit Breaker ───

class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_initial_state(self):
        cb = CircuitBreakerState()
        assert cb.can_try() is True
        assert cb.failures == 0
        assert cb.is_open is False

    def test_opens_after_threshold(self):
        cb = CircuitBreakerState(threshold=2, reset_seconds=60)
        assert cb.can_try() is True

        cb.record_failure()
        assert cb.can_try() is True  # 1/2 failures

        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_try() is False  # Circuit open

    def test_resets_after_timeout(self):
        cb = CircuitBreakerState(threshold=1, reset_seconds=0.1)
        cb.record_failure()
        assert cb.can_try() is False  # Open

        time.sleep(0.15)
        assert cb.can_try() is True  # Half-open after timeout

    def test_success_resets(self):
        cb = CircuitBreakerState(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        cb.record_success()
        assert cb.failures == 0
        assert cb.is_open is False

    def test_status_string(self):
        cb = CircuitBreakerState()
        assert "CLOSED" in cb.status
        assert "0/3" in cb.status

        cb.record_failure()
        assert "1/3" in cb.status


# ─── TESTS: Retry With Backoff ───

class TestRetry:
    """Test retry with exponential backoff."""

    def test_success_first_try(self):
        """Should succeed on first attempt without retries."""
        func = MagicMock(return_value=42)
        result = retry_with_backoff(func, max_retries=3)
        assert result == 42
        assert func.call_count == 1

    def test_retry_on_failure(self):
        """Should retry on failure and eventually succeed."""
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        result = retry_with_backoff(func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert func.call_count == 3

    def test_all_retries_exhausted(self):
        """Should raise last exception after all retries."""
        func = MagicMock(side_effect=Exception("always fail"))
        with pytest.raises(Exception, match="always fail"):
            retry_with_backoff(func, max_retries=2, base_delay=0.01)
        assert func.call_count == 3  # 1 original + 2 retries

    def test_no_retry_on_success(self):
        func = MagicMock(return_value=99)
        result = retry_with_backoff(func, max_retries=5, base_delay=0.5)
        assert result == 99
        assert func.call_count == 1


# ─── TESTS: DataProviderFactory ───

class TestDataProviderFactory:
    """Test factory and provider instantiation."""

    def test_get_provider_yahoo(self):
        factory = DataProviderFactory()
        provider = factory.get_provider("yahoo")
        assert provider.provider_name == "yahoo"
        assert isinstance(provider, YahooFinanceProvider)

    def test_get_provider_alpha_vantage(self):
        factory = DataProviderFactory()
        provider = factory.get_provider("alpha_vantage")
        assert provider.provider_name == "alpha_vantage"
        assert isinstance(provider, AlphaVantageProvider)

    def test_get_provider_invalid(self):
        factory = DataProviderFactory()
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.get_provider("nonexistent")

    def test_get_all_providers(self):
        factory = DataProviderFactory()
        providers = factory.get_all_providers()
        assert len(providers) == 2
        assert providers[0].provider_name == "yahoo"  # Primary first
        assert providers[1].provider_name == "alpha_vantage"

    def test_get_provider_cached(self):
        factory = DataProviderFactory()
        p1 = factory.get_provider("yahoo")
        p2 = factory.get_provider("yahoo")
        assert p1 is p2  # Same instance (cached)

    def test_global_factory(self):
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2  # Singleton


# ─── TESTS: Provider Stats ───

class TestProviderStats:
    """Test provider statistics tracking."""

    def test_yahoo_stats(self):
        factory = DataProviderFactory()
        provider = factory.get_provider("yahoo")
        stats = provider.stats()
        assert stats["provider"] == "yahoo"
        assert "circuit" in stats
        assert "calls" in stats
        assert "failures" in stats
        assert "cache_hits" in stats

    def test_factory_stats(self):
        factory = DataProviderFactory()
        stats = factory.stats()
        assert "yahoo" in stats
        assert "alpha_vantage" in stats


# ─── TESTS: Cross-Validate ───

class TestCrossValidation:
    """Test cross-provider data validation."""

    def test_cross_validate_single_result(self):
        factory = DataProviderFactory()
        # _cross_validate with single result
        import pandas as pd
        df = pd.DataFrame({"Close": [100.0]})
        result_df, name = factory._cross_validate(
            [(df, "yahoo")],
            "forex",
        )
        assert name == "yahoo"
        assert not result_df.empty

    def test_cross_validate_empty(self):
        factory = DataProviderFactory()
        result_df, name = factory._cross_validate([], "forex")
        assert name == "none"
        assert result_df.empty

    def test_cross_validate_discrepancy(self, caplog):
        """Should log warning on high discrepancy."""
        factory = DataProviderFactory()
        import pandas as pd
        import numpy as np

        df1 = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2025-01-01"]),
        )
        df2 = pd.DataFrame(
            {"Close": [101.0]},  # 1% difference (exceeds 0.1% tolerance)
            index=pd.to_datetime(["2025-01-01"]),
        )

        # This should log a warning about discrepancy
        import logging
        with patch.object(logging.Logger, 'warning') as mock_warn:
            result_df, name = factory._cross_validate(
                [(df1, "yahoo"), (df2, "alpha_vantage")],
                "forex",
            )
            assert mock_warn.called


# ─── TESTS: Empty/Error Handling ───

class TestProviderDataHandling:
    """Test that providers handle missing/empty data gracefully."""

    def test_yahoo_empty_dataframe_on_failure(self):
        """Yahoo provider should return empty DF on network failure."""
        provider = YahooFinanceProvider()
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            df = provider.fetch_ohlcv_impl("GC=F", "1d", "1y")
            assert df.empty, "Should return empty on network error"

    def test_yahoo_empty_on_bad_json(self):
        """Should handle empty results gracefully."""
        provider = YahooFinanceProvider()
        mock_response = MagicMock()
        mock_response.json.return_value = {"chart": {"result": []}}
        with patch("requests.get", return_value=mock_response):
            df = provider.fetch_ohlcv_impl("GC=F", "1d", "1y")
            assert df.empty

    def test_yahoo_empty_on_missing_timestamps(self):
        """Should handle missing timestamps gracefully."""
        provider = YahooFinanceProvider()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "chart": {
                "result": [{"timestamp": [], "indicators": {"quote": [{}]}}]
            }
        }
        with patch("requests.get", return_value=mock_response):
            df = provider.fetch_ohlcv_impl("GC=F", "1d", "1y")
            assert df.empty

    def test_alpha_vantage_no_api_key(self):
        """Should return empty if no API key set."""
        provider = AlphaVantageProvider(api_key="")
        df = provider.fetch_ohlcv_impl("GC=F", "1d", "1y")
        assert df.empty

    def test_alpha_vantage_quote_no_key(self):
        prov = AlphaVantageProvider(api_key="")
        price = prov.fetch_quote_impl("GC=F")
        assert price is None
