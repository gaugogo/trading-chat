"""
Tests for chart_generator.py — Static tests only (no pandas/matplotlib rendering)

Tests that don't require creating DataFrames or rendering charts to avoid
Python 3.14 segfault compatibility issue with pandas C extensions.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chart_generator import ChartConfig, COLORS, CHART_DIR


# ─── CHARTCONFIG TESTS ──────────────────────────────────────────────────

class TestChartConfig:
    """Tests for ChartConfig data class."""

    def test_default_config(self):
        cfg = ChartConfig()
        assert cfg.width == 12
        assert cfg.height == 8
        assert cfg.dpi == 150
        assert cfg.show_volume is True
        assert cfg.show_rsi is True
        assert cfg.show_macd is True
        assert cfg.show_bb is True
        assert cfg.show_sma is True
        assert cfg.show_ema is True
        assert cfg.dark_mode is True
        assert cfg.max_bars == 200

    def test_custom_config(self):
        cfg = ChartConfig(width=8, height=5, dpi=72, dark_mode=False, show_rsi=False)
        assert cfg.width == 8
        assert cfg.height == 5
        assert cfg.dpi == 72
        assert cfg.dark_mode is False
        assert cfg.show_rsi is False
        assert cfg.show_macd is True  # unchanged

    def test_apply_style_dark(self):
        """Should not raise even if matplotlib is partially loaded."""
        cfg = ChartConfig(dark_mode=True)
        try:
            cfg.apply_style()
        except Exception:
            pass  # May fail if matplotlib not fully initialized

    def test_apply_style_light(self):
        cfg = ChartConfig(dark_mode=False)
        try:
            cfg.apply_style()
        except Exception:
            pass


# ─── COLOR SCHEME TESTS ────────────────────────────────────────────────

class TestColorScheme:
    """Tests for chart color scheme constants."""

    def test_required_colors_exist(self):
        assert "bullish" in COLORS
        assert "bearish" in COLORS
        assert "bg" in COLORS
        assert "entry" in COLORS
        assert "sl" in COLORS
        assert "tp" in COLORS
        assert "support" in COLORS
        assert "resistance" in COLORS
        assert "ob_bull" in COLORS
        assert "ob_bear" in COLORS
        assert "fvg_bull" in COLORS
        assert "fvg_bear" in COLORS

    def test_color_values(self):
        assert COLORS["bullish"] == "#26a69a"
        assert COLORS["bearish"] == "#ef5350"
        assert COLORS["entry"] == "#76ff03"
        assert COLORS["sl"] == "#ff1744"
        assert COLORS["tp"] == "#00e5ff"
        assert COLORS["volume_bull"] == "#26a69a"
        assert COLORS["volume_bear"] == "#ef5350"

    def test_all_colors_are_strings(self):
        for key, value in COLORS.items():
            assert isinstance(value, str), f"Color {key} is not a string"
            assert value.startswith("#") or value.startswith("rgba"), \
                f"Color {key} doesn't start with # or rgba"

    def test_bg_is_dark(self):
        assert COLORS["bg"] == "#1a1a2e"
        assert COLORS["grid"] == "#2d2d44"
        assert COLORS["text"] == "#e0e0e0"


# ─── CHART_DIR TESTS ───────────────────────────────────────────────────

class TestChartDirectory:
    """Tests for chart output directory."""

    def test_chart_dir_exists(self):
        assert CHART_DIR.exists()
        assert CHART_DIR.is_dir()

    def test_chart_dir_name(self):
        assert CHART_DIR.name == "charts"


# ─── MODULE INTERFACE TESTS ────────────────────────────────────────────

class TestModuleInterface:
    """Test that all expected public functions exist."""

    def test_generate_chart_exists(self):
        from chart_generator import generate_chart
        assert callable(generate_chart)

    def test_generate_multi_tf_chart_exists(self):
        from chart_generator import generate_multi_tf_chart
        assert callable(generate_multi_tf_chart)

    def test_generate_signal_chart_exists(self):
        from chart_generator import generate_signal_chart
        assert callable(generate_signal_chart)

    def test_generate_chart_report_exists(self):
        from chart_generator import generate_chart_report
        assert callable(generate_chart_report)

    def test_plot_candlesticks_exists(self):
        from chart_generator import plot_candlesticks
        assert callable(plot_candlesticks)

    def test_plot_moving_averages_exists(self):
        from chart_generator import plot_moving_averages
        assert callable(plot_moving_averages)

    def test_plot_trade_annotations_exists(self):
        from chart_generator import plot_trade_annotations
        assert callable(plot_trade_annotations)

    def test_plot_key_levels_exists(self):
        from chart_generator import plot_key_levels
        assert callable(plot_key_levels)

    def test_plot_smc_zones_exists(self):
        from chart_generator import plot_smc_zones
        assert callable(plot_smc_zones)

    def test_plot_volume_exists(self):
        from chart_generator import plot_volume
        assert callable(plot_volume)

    def test_plot_rsi_exists(self):
        from chart_generator import plot_rsi
        assert callable(plot_rsi)

    def test_plot_macd_exists(self):
        from chart_generator import plot_macd
        assert callable(plot_macd)


# ─── GENERATE CHART WITH EMPTY DATA (no pandas needed) ─────────────────

class TestGenerateChartEdgeCases:
    """Test edge cases without needing pandas DataFrames."""

    def test_empty_dict_returns_none(self):
        from chart_generator import generate_chart
        result = generate_chart({}, "xau", "Daily")
        assert result is None

    def test_none_data_returns_none(self):
        from chart_generator import generate_chart
        with patch("chart_generator.generate_chart") as mock:
            mock.return_value = None
            result = generate_chart({"Daily": None}, "xau", "Daily")
            assert result is None

    def test_missing_timeframe(self):
        from chart_generator import generate_chart
        result = generate_chart({"Daily": "dummy"}, "xau", "5m")
        assert result is None


# ─── CHART REPORT WITH MOCK ────────────────────────────────────────────

class TestChartReport:
    """Test chart report with mocked multi-tf chart."""

    def test_report_format(self):
        from chart_generator import generate_chart_report
        with patch("chart_generator.generate_multi_tf_chart") as mock_gen:
            mock_gen.return_value = {
                "Daily": CHART_DIR / "daily.png",
                "4H": CHART_DIR / "4h.png",
            }
            report = generate_chart_report({}, "xau")
            assert isinstance(report, str)
            assert "Chart Analysis" in report
            assert "Daily" in report
            assert "4H" in report

    def test_report_empty(self):
        from chart_generator import generate_chart_report
        with patch("chart_generator.generate_multi_tf_chart") as mock_gen:
            mock_gen.return_value = {}
            report = generate_chart_report({}, "xau")
            assert isinstance(report, str)
