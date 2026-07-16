"""
Tests for risk_calculator.py — Risk Management Calculator.

Run with: pytest tests/test_risk_calculator.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_calculator import (
    RiskCalculator,
    RiskResult,
    RiskProfile,
    RiskLevel,
    quick_risk_calc,
    INSTRUMENT_SPECS,
)


# ─── FIXTURES ───

@pytest.fixture
def calculator():
    return RiskCalculator()


@pytest.fixture
def sample_result(calculator):
    return calculator.calculate(
        account_size=10000,
        risk_percent=1.0,
        entry_price=2650.00,
        stop_loss=2640.00,
        take_profit=2670.00,
        instrument="xau",
    )


# ─── TESTS: RiskLevel Enum ───

class TestRiskLevel:
    def test_icons(self):
        assert RiskLevel.CONSERVATIVE.icon == "🟢"
        assert RiskLevel.EXTREME.icon == "🔴"

    def test_default_risk_pct(self):
        assert RiskLevel.CONSERVATIVE.default_risk_pct == 0.5
        assert RiskLevel.MODERATE.default_risk_pct == 1.0
        assert RiskLevel.AGGRESSIVE.default_risk_pct == 2.0


# ─── TESTS: RiskProfile ───

class TestRiskProfile:
    def test_max_daily_loss(self):
        profile = RiskProfile(
            account_size=10000,
            risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0,
            max_concurrent_trades=3,
            risk_level=RiskLevel.MODERATE,
        )
        assert profile.max_daily_loss() == 300.0
        assert profile.max_risk_per_trade() == 100.0

    def test_summary(self):
        profile = RiskProfile(
            account_size=50000,
            risk_per_trade_pct=0.5,
            max_daily_loss_pct=2.0,
            max_concurrent_trades=2,
            risk_level=RiskLevel.CONSERVATIVE,
        )
        summary = profile.summary()
        assert "50000" in summary
        assert "0.5%" in summary
        assert "2%" in summary


# ─── TESTS: RiskCalculator ───

class TestRiskCalculator:
    def test_init_defaults(self, calculator):
        assert "xau" in calculator.specs
        assert calculator.specs["xau"]["pip_value_per_lot"] == 10.0

    def test_init_custom_specs(self):
        calc = RiskCalculator(specs={
            "custom": {"pip_value_per_lot": 5.0, "pip_size": 0.01, "min_lot": 0.1,
                        "lot_step": 0.05, "margin_per_lot": 1000, "name": "Custom"}
        })
        assert "xau" in calc.specs  # defaults still present
        assert calc.specs["custom"]["pip_value_per_lot"] == 5.0

    def test_calculate_basic(self, sample_result):
        assert isinstance(sample_result, RiskResult)
        assert sample_result.position_size_lots > 0
        assert sample_result.dollar_risk > 0
        assert sample_result.sl_pips > 0

    def test_calculate_xau(self, calculator):
        result = calculator.calculate(
            account_size=10000,
            risk_percent=1.0,
            entry_price=2650.00,
            stop_loss=2640.00,
            take_profit=2670.00,
            instrument="xau",
        )
        # Gold: pip_size=0.01, SL=10.0 points = 1000 pips
        # 1 lot risk = 1000 pips * $10/pip = $10,000
        # $100 risk / $10,000 = 0.01 lots
        assert result.position_size_lots == pytest.approx(0.01, abs=0.005)
        assert result.sl_pips == pytest.approx(1000.0, abs=10.0)
        assert result.risk_reward_ratio == pytest.approx(2.0, abs=0.1)

    def test_calculate_btc(self, calculator):
        result = calculator.calculate(
            account_size=50000,
            risk_percent=1.0,
            entry_price=65000.00,
            stop_loss=64000.00,
            take_profit=67000.00,
            instrument="btc",
        )
        # BTC: 1 lot = 1 BTC = $1/pip, SL = 1000 pips, risk = $500
        # position_size = 500 / (1000 * 1) = 0.5 BTC
        assert result.position_size_lots > 0
        assert result.risk_reward_ratio == pytest.approx(2.0, abs=0.1)

    def test_calculate_gbp(self, calculator):
        result = calculator.calculate(
            account_size=10000,
            risk_percent=1.0,
            entry_price=1.26500,
            stop_loss=1.26000,
            take_profit=1.27500,
            instrument="gbp",
        )
        # GBP: 1 lot = $10/pip, SL = 50 pips (0.0001 * 50 = 0.005)
        # risk = $100, risk per lot = 50 * 10 = $500
        # position_size = 100 / 500 = 0.2 lot
        assert result.position_size_lots == pytest.approx(0.2, abs=0.05)
        assert result.sl_pips == pytest.approx(50.0, abs=1.0)

    def test_no_tp(self, calculator):
        result = calculator.calculate(
            account_size=10000,
            risk_percent=1.0,
            entry_price=2650.00,
            stop_loss=2640.00,
            instrument="xau",
        )
        assert result.take_profit is None
        assert result.tp_pips is None
        assert result.reward_if_hit is None
        assert result.risk_reward_ratio is None

    def test_zero_sl_distance(self, calculator):
        result = calculator.calculate(
            account_size=10000,
            risk_percent=1.0,
            entry_price=2650.00,
            stop_loss=2650.00,  # zero distance
            instrument="xau",
        )
        assert result.position_size_lots == 0
        assert len(result.warnings) > 0

    def test_risk_level_mapping(self, calculator):
        # Conservative
        r = calculator.calculate(10000, 0.3, 2650, 2640, instrument="xau")
        assert r.profile.risk_level == RiskLevel.CONSERVATIVE

        # Moderate
        r = calculator.calculate(10000, 1.0, 2650, 2640, instrument="xau")
        assert r.profile.risk_level == RiskLevel.MODERATE

        # Aggressive
        r = calculator.calculate(10000, 1.5, 2650, 2640, instrument="xau")
        assert r.profile.risk_level == RiskLevel.AGGRESSIVE

        # Extreme
        r = calculator.calculate(10000, 3.0, 2650, 2640, instrument="xau")
        assert r.profile.risk_level == RiskLevel.EXTREME
        assert any("EXTREME" in w for w in r.warnings)


# ─── TESTS: RiskResult ───

class TestRiskResult:
    def test_summary(self, sample_result):
        summary = sample_result.summary()
        assert "XAU" in summary
        assert "lot" in summary
        assert "Risk:" in summary

    def test_detailed_report(self, sample_result):
        report = sample_result.detailed_report()
        assert "RISK MANAGEMENT" in report
        assert "Entry:" in report
        assert "Stop Loss:" in report
        assert "Position size:" in report
        assert "R:R" in report

    def test_detailed_report_no_tp(self, calculator):
        result = calculator.calculate(
            account_size=10000,
            risk_percent=1.0,
            entry_price=2650.00,
            stop_loss=2640.00,
            instrument="xau",
        )
        report = result.detailed_report()
        assert "Take Profit" in report
        assert "Không đặt" in report


# ─── TESTS: Convenience Functions ───

class TestConvenience:
    def test_quick_risk_calc(self):
        result = quick_risk_calc(
            account_size=10000,
            risk_percent=1.0,
            entry_price=2650.00,
            stop_loss=2640.00,
            take_profit=2670.00,
            instrument="xau",
        )
        assert isinstance(result, str)
        assert "RISK MANAGEMENT" in result

    def test_quick_risk_calc_defaults(self):
        result = quick_risk_calc()
        assert isinstance(result, str)

    def test_quick_risk(self, calculator):
        summary = calculator.quick_risk(
            account_size=10000,
            entry_price=2650.00,
            stop_loss=2640.00,
            risk_percent=1.0,
            instrument="xau",
        )
        assert isinstance(summary, str)
        assert "XAU" in summary

    def test_position_size_for_risk(self, calculator):
        lots = calculator.position_size_for_risk(
            account_size=10000,
            risk_percent=1.0,
            sl_pips=10.0,
            instrument="xau",
        )
        # XAU: pip_value=10, risk_per_lot=10*10=100, risk=100, lots=100/100=1.0
        assert lots == pytest.approx(1.0, abs=0.1)

    def test_position_size_for_risk_zero(self, calculator):
        lots = calculator.position_size_for_risk(
            account_size=10000,
            risk_percent=1.0,
            sl_pips=0.0,
            instrument="xau",
        )
        assert lots == 0.0


# ─── TESTS: Instrument Specs ───

class TestInstrumentSpecs:
    def test_xau_specs(self):
        spec = INSTRUMENT_SPECS["xau"]
        assert spec["pip_value_per_lot"] == 10.0
        assert spec["pip_size"] == 0.01
        assert spec["min_lot"] == 0.01

    def test_btc_specs(self):
        spec = INSTRUMENT_SPECS["btc"]
        assert spec["pip_value_per_lot"] == 1.0
        assert spec["pip_size"] == 1.0

    def test_gbp_specs(self):
        spec = INSTRUMENT_SPECS["gbp"]
        assert spec["pip_value_per_lot"] == 10.0
        assert spec["pip_size"] == 0.0001

    def test_all_specs_have_required_keys(self):
        required = ["pip_value_per_lot", "pip_size", "min_lot", "lot_step",
                      "margin_per_lot", "name"]
        for instr, spec in INSTRUMENT_SPECS.items():
            for key in required:
                assert key in spec, f"{instr} missing {key}"
