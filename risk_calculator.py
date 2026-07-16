"""
risk_calculator.py — Risk Management Calculator

Tính toán position size, max contracts, daily loss limit, và R:R-based recommendations.

Công thức:
  Position Size = (Account × Risk%) / (SL Distance × Pip/Contract Value)
  Max Risk/Day = Account × Daily Risk%
  R:R check: TP distance / SL distance >= Min R:R

Usage:
  from risk_calculator import RiskCalculator, RiskResult
  calc = RiskCalculator()
  result = calc.calculate(
      account_size=10000,
      risk_percent=1.0,
      entry_price=2650.00,
      stop_loss=2640.00,
      take_profit=2670.00,
      instrument="xau",
  )
  print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from core import fmt_price


# ─── TYPICAL CONTRACT SPECS ───

# Pip values, contract sizes, margin requirements per instrument
INSTRUMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "xau": {  # XAUUSD (Gold) — 1 lot = 100 oz
        "name": "XAUUSD (Gold)",
        "pip_value_per_lot": 10.0,         # $10 per $0.01 pip movement
        "pip_decimal": 2,                   # Price decimal for pip
        "pip_size": 0.01,                   # Pip size
        "contract_size": 100,               # oz per lot
        "margin_per_lot": 5000.0,           # Approx margin for 1 lot
        "min_lot": 0.01,                    # Minimum lot size
        "lot_step": 0.01,                   # Lot size increment
        "typical_spread": 0.30,             # Average spread in pips
        "pip_label": "pip ($0.01)",
        "price_decimals": 2,
    },
    "btc": {  # BTCUSD (Bitcoin)
        "name": "BTC/USD (Bitcoin)",
        "pip_value_per_lot": 1.0,           # $1 per $1 movement for 1 BTC
        "pip_decimal": 1,                   # Price decimal for pip
        "pip_size": 1.0,                    # 1 USD = 1 pip for BTC
        "contract_size": 1,                 # BTC per lot
        "margin_per_lot": 50000.0,          # Approx margin for 1 BTC
        "min_lot": 0.001,                   # Minimum lot size
        "lot_step": 0.001,                  # Lot size increment
        "typical_spread": 10.0,             # Average spread in pips ($10)
        "pip_label": "pip ($1)",
        "price_decimals": 1,
    },
    "gbp": {  # GBPUSD (Cable)
        "name": "GBP/USD (Cable)",
        "pip_value_per_lot": 10.0,          # $10 per pip for standard lot
        "pip_decimal": 4,                   # Price decimal for pip
        "pip_size": 0.0001,                 # Pip size
        "contract_size": 100000,            # Units per lot
        "margin_per_lot": 2000.0,           # Approx margin
        "min_lot": 0.01,                    # Minimum lot size
        "lot_step": 0.01,                   # Lot size increment
        "typical_spread": 1.2,              # Average spread in pips
        "pip_label": "pip ($0.0001)",
        "price_decimals": 5,
    },
}


# ─── RISK LEVELS ───

class RiskLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"     # 0.5% per trade
    MODERATE = "MODERATE"             # 1.0% per trade
    AGGRESSIVE = "AGGRESSIVE"         # 2.0% per trade
    EXTREME = "EXTREME"               # 3%+ per trade (not recommended)

    @property
    def icon(self) -> str:
        return {
            RiskLevel.CONSERVATIVE: "🟢",
            RiskLevel.MODERATE: "🟡",
            RiskLevel.AGGRESSIVE: "🟠",
            RiskLevel.EXTREME: "🔴",
        }.get(self, "❓")

    @property
    def default_risk_pct(self) -> float:
        return {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 2.0,
            RiskLevel.EXTREME: 3.0,
        }.get(self, 1.0)


# ─── RISK PROFILE ───

@dataclass
class RiskProfile:
    """Hồ sơ rủi ro cá nhân."""
    account_size: float
    risk_per_trade_pct: float        # % risk per trade (default: 1.0)
    max_daily_loss_pct: float        # % daily loss limit (default: 3.0)
    max_concurrent_trades: int       # Max positions at once (default: 3)
    risk_level: RiskLevel

    def max_daily_loss(self) -> float:
        return self.account_size * self.max_daily_loss_pct / 100.0

    def max_risk_per_trade(self) -> float:
        return self.account_size * self.risk_per_trade_pct / 100.0

    def summary(self) -> str:
        return (
            f"📋 Hồ sơ rủi ro ({self.risk_level.icon} {self.risk_level.value}):\n"
            f"  • Tài khoản: {fmt_price(self.account_size, 0)}\n"
            f"  • Rủi ro/lệnh: {self.risk_per_trade_pct:.1f}% = {fmt_price(self.max_risk_per_trade(), 0)}\n"
            f"  • Giới hạn/ngày: {self.max_daily_loss_pct:.0f}% = {fmt_price(self.max_daily_loss(), 0)}\n"
            f"  • Lệnh đồng thời tối đa: {self.max_concurrent_trades}"
        )


# ─── RESULT ───

@dataclass
class RiskResult:
    """Kết quả tính toán risk management."""
    profile: RiskProfile
    instrument: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    sl_distance: float               # Absolute SL distance
    sl_pips: float                   # SL distance in pips
    tp_distance: Optional[float]     # Absolute TP distance
    tp_pips: Optional[float]         # TP distance in pips
    risk_per_lot: float              # $ risk per 1 lot
    position_size_lots: float        # Recommended lot size
    dollar_risk: float               # $ amount at risk
    reward_if_hit: Optional[float]   # $ reward if TP hit
    risk_reward_ratio: Optional[float]  # R:R
    margin_required: float           # Margin needed
    margin_used_pct: float           # % of account used as margin
    daily_loss_remaining: float      # Remaining daily loss budget
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary."""
        rr_str = f"R:R {self.risk_reward_ratio:.2f}" if self.risk_reward_ratio else "No TP"
        return (
            f"{self.profile.risk_level.icon} "
            f"{self.instrument.upper()}: {self.position_size_lots:.2f} lot(s) "
            f"| Risk: {fmt_price(self.dollar_risk)} ({self.profile.risk_per_trade_pct:.1f}%) "
            f"| SL: {self.sl_pips:.1f} pips | {rr_str}"
        )

    def detailed_report(self) -> str:
        """Full detailed report."""
        lines = [f"{'='*60}"]
        lines.append(f"  📊 RISK MANAGEMENT CALCULATOR")
        lines.append(f"{'='*60}")
        lines.append("")
        lines.append(f"  📋 Hồ sơ rủi ro:")
        lines.append(f"    • Tài khoản: {fmt_price(self.profile.account_size, 0)}")
        lines.append(f"    • Rủi ro/lệnh: {self.profile.risk_per_trade_pct:.1f}% → {fmt_price(self.dollar_risk, 0)}")
        lines.append(f"    • Giới hạn/ngày: {self.profile.max_daily_loss_pct:.0f}% → còn {fmt_price(self.daily_loss_remaining, 0)}")
        lines.append(f"")
        lines.append(f"  📐 Thông số lệnh:")
        lines.append(f"    • Công cụ: {INSTRUMENT_SPECS.get(self.instrument, {}).get('name', self.instrument.upper())}")
        lines.append(f"    • Entry: {fmt_price(self.entry_price)}")
        lines.append(f"    • Stop Loss: {fmt_price(self.stop_loss)} ({self.sl_pips:.1f} pips)")
        if self.take_profit:
            lines.append(f"    • Take Profit: {fmt_price(self.take_profit)} ({self.tp_pips:.1f} pips)")
        else:
            lines.append(f"    • Take Profit: Không đặt (trailing stop?)")
        lines.append(f"")
        lines.append(f"  💰 Kết quả:")

        # Position sizing
        lines.append(f"    • Rủi ro 1 lot: {fmt_price(self.risk_per_lot, 0)}")
        lines.append(f"    • Position size: {self.position_size_lots:.2f} lot(s)")
        lines.append(f"    • Dollar risk: {fmt_price(self.dollar_risk, 0)}")
        if self.reward_if_hit is not None:
            lines.append(f"    • Reward nếu TP: {fmt_price(self.reward_if_hit, 0)}")
        if self.risk_reward_ratio is not None:
            rr_icon = "✅" if self.risk_reward_ratio >= 2.0 else ("⚠️" if self.risk_reward_ratio >= 1.0 else "❌")
            lines.append(f"    • R:R: {rr_icon} {self.risk_reward_ratio:.2f}")
        lines.append(f"")
        lines.append(f"  🏦 Margin:")
        lines.append(f"    • Margin yêu cầu: {fmt_price(self.margin_required, 0)}")
        lines.append(f"    • % tài khoản: {self.margin_used_pct:.1f}%")
        lines.append(f"")
        if self.warnings:
            lines.append(f"  ⚠️ Cảnh báo:")
            for w in self.warnings:
                lines.append(f"    • {w}")
            lines.append("")
        lines.append(f"  💡 Gợi ý điều chỉnh:")
        lines.append(f"    • {self.profile.risk_level.icon} {self.profile.risk_level.value}: {self.profile.risk_per_trade_pct:.1f}%/lệnh")
        lines.append(f"    • Điều chỉnh lot size: giảm risk% nếu không thoải mái")
        lines.append(f"    • Kiểm tra R:R tối thiểu (khuyến nghị ≥ 1:2)")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ─── RISK CALCULATOR ───

class RiskCalculator:
    """Risk management calculator for position sizing.

    Args:
        specs: Override default instrument specs
    """

    def __init__(self, specs: Optional[Dict[str, Dict]] = None):
        self.specs = INSTRUMENT_SPECS.copy()
        if specs:
            self.specs.update(specs)

    def calculate(
        self,
        account_size: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        instrument: str = "xau",
        max_concurrent: int = 3,
        max_daily_loss_pct: float = 3.0,
    ) -> RiskResult:
        """Calculate position size and risk metrics.

        Args:
            account_size: Account balance in USD
            risk_percent: Percent of account to risk per trade (e.g., 1.0 for 1%)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            instrument: Instrument ID ('xau', 'btc', 'gbp')
            max_concurrent: Max concurrent positions
            max_daily_loss_pct: Max daily loss %

        Returns:
            RiskResult with all calculations
        """
        # Determine risk level
        if risk_percent <= 0.5:
            risk_level = RiskLevel.CONSERVATIVE
        elif risk_percent <= 1.0:
            risk_level = RiskLevel.MODERATE
        elif risk_percent <= 2.0:
            risk_level = RiskLevel.AGGRESSIVE
        else:
            risk_level = RiskLevel.EXTREME

        profile = RiskProfile(
            account_size=account_size,
            risk_per_trade_pct=risk_percent,
            max_daily_loss_pct=max_daily_loss_pct,
            max_concurrent_trades=max_concurrent,
            risk_level=risk_level,
        )

        # Get instrument specs
        spec = self.specs.get(instrument, self.specs["xau"])
        pip_size = spec["pip_size"]
        pip_value = spec["pip_value_per_lot"]
        min_lot = spec["min_lot"]
        lot_step = spec["lot_step"]
        margin_per_lot = spec["margin_per_lot"]

        warnings: List[str] = []

        # SL distance
        sl_distance = abs(entry_price - stop_loss)
        sl_pips = sl_distance / pip_size

        # TP distance
        tp_distance = None
        tp_pips = None
        reward_if_hit = None
        risk_reward = None
        if take_profit is not None:
            tp_distance = abs(take_profit - entry_price)
            tp_pips = tp_distance / pip_size
            reward_if_hit = tp_distance / pip_size * pip_value
            risk_reward = tp_distance / sl_distance if sl_distance > 0 else None

        # Risk per lot
        risk_per_lot = sl_pips * pip_value

        if risk_per_lot <= 0:
            return RiskResult(
                profile=profile,
                instrument=instrument,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                sl_distance=sl_distance,
                sl_pips=sl_pips,
                tp_distance=tp_distance,
                tp_pips=tp_pips,
                risk_per_lot=0,
                position_size_lots=0,
                dollar_risk=0,
                reward_if_hit=reward_if_hit,
                risk_reward_ratio=risk_reward,
                margin_required=0,
                margin_used_pct=0,
                daily_loss_remaining=profile.max_daily_loss(),
                warnings=["❌ SL distance quá nhỏ, không thể tính position size"],
            )

        # Dollar risk
        dollar_risk = account_size * risk_percent / 100.0

        # Position size (lots)
        position_size_lots = dollar_risk / risk_per_lot
        # Round down to nearest lot_step
        position_size_lots = max(min_lot, (position_size_lots // lot_step) * lot_step)
        position_size_lots = round(position_size_lots, 4)

        # Actual dollar risk (after rounding)
        actual_risk = position_size_lots * risk_per_lot
        actual_risk_pct = (actual_risk / account_size) * 100.0 if account_size > 0 else 0

        # Reward if TP hit
        if reward_if_hit is not None:
            reward_if_hit = position_size_lots * (tp_pips / pip_size * pip_value / (1/pip_size) if False else risk_per_lot / sl_pips * tp_distance / pip_size * pip_value)
            # Simpler: reward = position_size * (tp_pips * pip_value)
            reward_if_hit = position_size_lots * tp_pips * pip_value

        # Margin
        margin_required = position_size_lots * margin_per_lot
        margin_used_pct = (margin_required / account_size) * 100.0 if account_size > 0 else 0

        # Daily loss remaining
        daily_loss_remaining = profile.max_daily_loss()

        # Warnings
        if risk_level == RiskLevel.EXTREME:
            warnings.append(f"⚠️ Risk {risk_percent:.1f}%/lệnh là EXTREME! Khuyến nghị ≤ 2%")
        if margin_used_pct > 50:
            warnings.append(f"⚠️ Margin {margin_used_pct:.0f}% > 50% tài khoản — rủi ro margin call!")
        if risk_reward is not None and risk_reward < 1.0:
            warnings.append(f"❌ R:R {risk_reward:.2f} < 1.0 — không nên vào lệnh này")
        elif risk_reward is not None and risk_reward < 2.0:
            warnings.append(f"⚠️ R:R {risk_reward:.2f} < 2.0 — cần win rate > 50% để có lợi")
        if position_size_lots < min_lot:
            warnings.append(f"ℹ️ Position size {position_size_lots} lot < min {min_lot} lot")
        if actual_risk > dollar_risk * 1.1:
            warnings.append(f"⚠️ Actual risk {fmt_price(actual_risk)} > target {fmt_price(dollar_risk)}")

        return RiskResult(
            profile=profile,
            instrument=instrument,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_distance=sl_distance,
            sl_pips=round(sl_pips, 1),
            tp_distance=tp_distance,
            tp_pips=round(tp_pips, 1) if tp_pips else None,
            risk_per_lot=round(risk_per_lot, 2),
            position_size_lots=position_size_lots,
            dollar_risk=round(actual_risk, 2),
            reward_if_hit=round(reward_if_hit, 2) if reward_if_hit else None,
            risk_reward_ratio=round(risk_reward, 2) if risk_reward else None,
            margin_required=round(margin_required, 2),
            margin_used_pct=round(margin_used_pct, 1),
            daily_loss_remaining=round(daily_loss_remaining, 2),
            warnings=warnings,
        )

    def quick_risk(
        self,
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 1.0,
        instrument: str = "xau",
    ) -> str:
        """Quick one-line risk assessment.

        Args:
            account_size: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk % per trade
            instrument: Instrument ID

        Returns:
            One-line summary
        """
        result = self.calculate(
            account_size=account_size,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
            instrument=instrument,
        )
        return result.summary()

    def position_size_for_risk(
        self,
        account_size: float,
        risk_percent: float,
        sl_pips: float,
        instrument: str = "xau",
    ) -> float:
        """Calculate position size directly from pip distance.

        Args:
            account_size: Account balance
            risk_percent: Risk % per trade
            sl_pips: Stop loss distance in pips
            instrument: Instrument ID

        Returns:
            Position size in lots
        """
        spec = self.specs.get(instrument, self.specs["xau"])
        pip_value = spec["pip_value_per_lot"]
        min_lot = spec["min_lot"]
        lot_step = spec["lot_step"]

        dollar_risk = account_size * risk_percent / 100.0
        risk_per_lot = sl_pips * pip_value

        if risk_per_lot <= 0:
            return 0.0

        lots = dollar_risk / risk_per_lot
        lots = max(min_lot, (lots // lot_step) * lot_step)
        return round(lots, 4)


# ─── CONVENIENCE ───

def quick_risk_calc(
    account_size: float = 10000.0,
    risk_percent: float = 1.0,
    entry_price: float = 2650.00,
    stop_loss: float = 2640.00,
    take_profit: Optional[float] = 2670.00,
    instrument: str = "xau",
) -> str:
    """Quick risk calculation — entry point for CLI.

    Args:
        account_size: Account balance
        risk_percent: Risk % per trade
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price (optional)
        instrument: Instrument ID

    Returns:
        Detailed report
    """
    calc = RiskCalculator()
    result = calc.calculate(
        account_size=account_size,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        instrument=instrument,
    )
    return result.detailed_report()
