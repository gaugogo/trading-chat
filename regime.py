"""
regime.py — Market Regime Detection

Xác định trạng thái thị trường dựa trên ADX, Bollinger Bands Width, và ATR ratio.

Output:
  - Trending (UP/DOWN): thị trường có xu hướng rõ ràng
  - Ranging: thị trường đi ngang, sideway
  - Volatile: biến động mạnh, không rõ hướng

Usage:
  from regime import detect_regime, RegimeResult, add_regime_to_config
  result = detect_regime(df_daily)
  print(result.regime, result.confidence)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from core import determine_trend, fmt_price

logger = logging.getLogger(__name__)


# ─── ENUMS ───

class MarketRegime(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGING = "RANGING"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"

    @property
    def icon(self) -> str:
        return {
            MarketRegime.STRONG_UPTREND: "🟢🟢",
            MarketRegime.UPTREND: "🟢",
            MarketRegime.WEAK_UPTREND: "🟢⬇️",
            MarketRegime.RANGING: "🟡↔️",
            MarketRegime.WEAK_DOWNTREND: "🔴⬆️",
            MarketRegime.DOWNTREND: "🔴",
            MarketRegime.STRONG_DOWNTREND: "🔴🔴",
            MarketRegime.VOLATILE: "⚡",
            MarketRegime.CHOPPY: "🌀",
            MarketRegime.UNKNOWN: "❓",
        }.get(self, "❓")

    @property
    def is_trending(self) -> bool:
        return self in (
            MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND,
            MarketRegime.WEAK_UPTREND,
            MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND,
            MarketRegime.WEAK_DOWNTREND,
        )

    @property
    def is_bullish(self) -> bool:
        return self in (
            MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND,
            MarketRegime.WEAK_UPTREND,
        )

    @property
    def is_bearish(self) -> bool:
        return self in (
            MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND,
            MarketRegime.WEAK_DOWNTREND,
        )

    @property
    def is_ranging(self) -> bool:
        return self == MarketRegime.RANGING

    @property
    def is_volatile(self) -> bool:
        return self in (MarketRegime.VOLATILE, MarketRegime.CHOPPY)


@dataclass
class RegimeResult:
    """Kết quả phát hiện market regime."""
    regime: MarketRegime
    confidence: float           # 0.0 to 1.0
    adx: float                  # ADX value
    adx_strength: str           # 'strong', 'moderate', 'weak'
    bb_width: float             # Bollinger Band Width (normalized)
    bb_volatility: str          # 'high', 'normal', 'low'
    atr_ratio: float            # Current ATR / SMA_50 ATR
    trend: str                  # 'UP', 'DOWN', 'SIDEWAYS'
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"{self.regime.icon} Regime: {self.regime.value} "
            f"(ADX: {self.adx:.1f}, BB: {self.bb_width:.4f}, "
            f"conf: {self.confidence:.0%})"
        )

    def detailed_report(self) -> str:
        """Full detailed report."""
        lines = [f"{'='*60}"]
        lines.append(f"  {self.regime.icon} MARKET REGIME ANALYSIS")
        lines.append(f"{'='*60}")
        lines.append(f"")
        lines.append(f"  Regime:      {self.regime.value} ({self.regime.icon})")
        lines.append(f"  Confidence:  {self.confidence:.0%}")
        lines.append(f"  Trend:       {self.trend}")
        lines.append(f"")
        lines.append(f"  ┌─ ADX: {self.adx:.1f} ({self.adx_strength})")
        lines.append(f"  ├─ BB Width: {self.bb_width:.4f} ({self.bb_volatility})")
        lines.append(f"  └─ ATR Ratio: {self.atr_ratio:.2f}")
        lines.append(f"")
        lines.append(f"  📋 Chi tiết:")
        for d in self.details:
            lines.append(f"    • {d}")
        lines.append("")
        lines.append("  💡 Gợi ý giao dịch theo regime:")
        if self.regime.is_trending:
            direction = "LONG" if self.regime.is_bullish else "SHORT"
            lines.append(f"    → Trend-follow: ưu tiên {direction}")
            lines.append(f"    → Tránh counter-trend, chờ pullback để entry")
        elif self.regime == MarketRegime.RANGING:
            lines.append(f"    → Mean-reversion: mua đáy, bán đỉnh range")
            lines.append(f"    → Giảm position size, range sẽ breakout")
        elif self.regime == MarketRegime.VOLATILE:
            lines.append(f"    → Chờ hết volatile rồi mới trade")
            lines.append(f"    → Tăng SL, giảm position size nếu vẫn trade")
        elif self.regime == MarketRegime.CHOPPY:
            lines.append(f"    → Không trade, chờ thị trường clear hướng")
        return "\n".join(lines)


# ─── DEFAULT CONFIG ───

DEFAULT_REGIME_CONFIG: Dict[str, Any] = {
    # ADX thresholds
    "adx": {
        "strong": 30.0,      # ADX >= 30 → strong trend
        "moderate": 20.0,    # ADX 20-30 → moderate trend
        "weak": 15.0,        # ADX 15-20 → weak trend / ranging
        "period": 14,        # ADX period
    },
    # BB Width thresholds (percentile-based)
    "bb_width": {
        "high_percentile": 0.80,   # Above 80th percentile → volatile
        "low_percentile": 0.20,    # Below 20th percentile → low volatility
        "period": 20,
    },
    # ATR ratio threshold
    "atr": {
        "high_ratio": 1.5,         # Current ATR > 1.5x SMA50 ATR → volatile
        "low_ratio": 0.5,          # Current ATR < 0.5x SMA50 ATR → very quiet
        "sma_period": 50,          # SMA period for ATR baseline
    },
    # Trend strength (from determine_trend)
    "trend": {
        "strong_score": 4,         # Score >= 4 → strong trend
    },
    # Confidence weights
    "confidence_weights": {
        "adx": 0.3,
        "bb_width": 0.2,
        "atr_ratio": 0.2,
        "trend_alignment": 0.3,
    },
}


# ─── ADX CALCULATION ───

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    ADX measures trend strength (not direction):
      - ADX >= 30 → strong trend
      - ADX 20-30 → moderate trend
      - ADX < 20 → weak trend / ranging

    Also returns +DI and -DI for direction.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ADX period (default: 14)

    Returns:
        DataFrame with ADX, Plus_DI, Minus_DI columns
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    # Smoothed ATR and DM (Wilder's method: SMA then EMA-like)
    atr_smooth = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_smooth)

    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)

    # ADX = EMA of DX
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    result = pd.DataFrame({
        'ADX': adx,
        'Plus_DI': plus_di,
        'Minus_DI': minus_di,
    }, index=df.index)
    return result


# ─── DETECT REGIME ───

def detect_regime(
    tf_data: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
) -> RegimeResult:
    """Detect market regime from multi-timeframe data.

    Analyzes Daily timeframe for regime, uses lower TFs for confirmation.

    Args:
        tf_data: Dict of timeframe DataFrames with indicators
        config: Override default regime config

    Returns:
        RegimeResult with regime type and confidence
    """
    cfg = DEFAULT_REGIME_CONFIG.copy()
    if config:
        _deep_merge(cfg, config)

    logger.debug(f"Detecting regime from {len(tf_data)} TFs")
    # Need at least Daily data
    df = tf_data.get("Daily")
    if df is None or df.empty or len(df) < 30:
        # Fallback: try 4H
        df = tf_data.get("4H")
        if df is None or df.empty or len(df) < 30:
            return RegimeResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx=0.0,
                adx_strength="unknown",
                bb_width=0.0,
                bb_volatility="unknown",
                atr_ratio=1.0,
                trend="UNKNOWN",
                details=["Không đủ dữ liệu để phân tích regime (cần >= 30 nến)"] * 3,
            )

    details: List[str] = []
    close = df['Close']

    # ── 1. ADX ──────────────────────────────────────────────────────
    adx_data = calculate_adx(df, cfg["adx"]["period"])
    adx_val = float(adx_data['ADX'].iloc[-1])
    plus_di = float(adx_data['Plus_DI'].iloc[-1])
    minus_di = float(adx_data['Minus_DI'].iloc[-1])

    if adx_val >= cfg["adx"]["strong"]:
        adx_strength = "strong"
        details.append(f"ADX {adx_val:.1f} ≥ {cfg['adx']['strong']}: Trend mạnh")
    elif adx_val >= cfg["adx"]["moderate"]:
        adx_strength = "moderate"
        details.append(f"ADX {adx_val:.1f} ({cfg['adx']['moderate']}-{cfg['adx']['strong']}): Trend vừa")
    elif adx_val >= cfg["adx"]["weak"]:
        adx_strength = "weak"
        details.append(f"ADX {adx_val:.1f} ({cfg['adx']['weak']}-{cfg['adx']['moderate']}): Trend yếu, range")
    else:
        adx_strength = "very_weak"
        details.append(f"ADX {adx_val:.1f} < {cfg['adx']['weak']}: Không trend, sideway")

    # +DI / -DI direction
    di_direction = "UP" if plus_di > minus_di else "DOWN"
    details.append(f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f} → {di_direction} bias")

    # ── 2. BB Width ────────────────────────────────────────────────
    bb_width_col = df.get('BB_Width', pd.Series(index=df.index, dtype=float))
    if bb_width_col.isna().all() or len(bb_width_col.dropna()) < 10:
        bb_width_val = 0.0
        bb_volatility = "unknown"
    else:
        bb_width_val = float(bb_width_col.iloc[-1])
        bb_width_history = bb_width_col.dropna()
        if len(bb_width_history) >= 10:
            percentile = (bb_width_history < bb_width_val).mean()
            if percentile >= cfg["bb_width"]["high_percentile"]:
                bb_volatility = "high"
                details.append(f"BB Width {bb_width_val:.4f} (P{percentile:.0%}): Biến động cao ⚡")
            elif percentile <= cfg["bb_width"]["low_percentile"]:
                bb_volatility = "low"
                details.append(f"BB Width {bb_width_val:.4f} (P{percentile:.0%}): Biến động thấp")
            else:
                bb_volatility = "normal"
                details.append(f"BB Width {bb_width_val:.4f} (P{percentile:.0%}): Biến động bình thường")
        else:
            bb_volatility = "normal"
            details.append(f"BB Width {bb_width_val:.4f}: Không đủ history để so sánh")

    # ── 3. ATR Ratio ────────────────────────────────────────────────
    atr_col = df.get('ATR', pd.Series(index=df.index, dtype=float))
    if atr_col.isna().all() or len(atr_col.dropna()) < cfg["atr"]["sma_period"]:
        atr_ratio = 1.0
        details.append("ATR ratio: Không đủ dữ liệu")
    else:
        current_atr = float(atr_col.iloc[-1])
        baseline_atr = float(atr_col.tail(cfg["atr"]["sma_period"]).mean())
        atr_ratio = current_atr / baseline_atr if baseline_atr > 0 else 1.0
        if atr_ratio >= cfg["atr"]["high_ratio"]:
            details.append(f"ATR ratio {atr_ratio:.2f}x: Biến động tăng mạnh ⚡")
        elif atr_ratio <= cfg["atr"]["low_ratio"]:
            details.append(f"ATR ratio {atr_ratio:.2f}x: Biến động giảm mạnh (nén)")
        else:
            details.append(f"ATR ratio {atr_ratio:.2f}x: Biến động ổn định")

    # ── 4. Trend ────────────────────────────────────────────────────
    trend, trend_score = determine_trend(df)
    details.append(f"Trend: {trend} (score: {trend_score:+d})")

    # ── 5. Determine Regime ──────────────────────────────────────────
    regime, confidence = _classify_regime(
        adx_val=adx_val,
        adx_strength=adx_strength,
        di_direction=di_direction,
        bb_width_val=bb_width_val,
        bb_volatility=bb_volatility,
        atr_ratio=atr_ratio,
        trend=trend,
        trend_score=trend_score,
        config=cfg,
    )

    return RegimeResult(
        regime=regime,
        confidence=round(confidence, 2),
        adx=round(adx_val, 1),
        adx_strength=adx_strength,
        bb_width=round(bb_width_val, 4),
        bb_volatility=bb_volatility,
        atr_ratio=round(atr_ratio, 2),
        trend=trend,
        details=details,
    )


def _classify_regime(
    adx_val: float,
    adx_strength: str,
    di_direction: str,
    bb_width_val: float,
    bb_volatility: str,
    atr_ratio: float,
    trend: str,
    trend_score: int,
    config: Dict[str, Any],
) -> Tuple[MarketRegime, float]:
    """Classify market regime based on all signals."""
    cfg = config["adx"]
    bb_cfg = config["bb_width"]
    atr_cfg = config["atr"]
    trend_cfg = config["trend"]
    conf_w = config["confidence_weights"]

    logger.debug(f"Classifying regime: ADX={adx_val:.1f}, BB={bb_width_val:.4f}, ATR={atr_ratio:.2f}")
    # Score components for confidence
    adx_conf = min(1.0, adx_val / cfg["strong"])
    bb_conf = 1.0 if bb_volatility == "high" else (0.5 if bb_volatility == "normal" else 0.3)
    atr_conf = min(1.0, max(0.0, (atr_ratio - 0.5) / 1.0))

    # Base regime logic
    is_volatile = bb_volatility == "high" and atr_ratio >= atr_cfg["high_ratio"]
    is_choppy = adx_strength == "very_weak" and bb_volatility == "high"
    is_ranging = adx_strength in ("weak", "very_weak") and bb_volatility in ("normal", "low")

    if is_choppy:
        regime = MarketRegime.CHOPPY
        trend_alignment = 0.0

    elif is_volatile and adx_strength in ("very_weak", "weak"):
        regime = MarketRegime.VOLATILE
        trend_alignment = 0.2

    elif is_volatile and adx_strength in ("strong", "moderate"):
        # Volatile trending → still trending, just wider ranges
        if di_direction == "UP" and trend in ("UP", "SIDEWAYS"):
            regime = MarketRegime.UPTREND if adx_strength == "moderate" else MarketRegime.STRONG_UPTREND
        elif di_direction == "DOWN" and trend in ("DOWN", "SIDEWAYS"):
            regime = MarketRegime.DOWNTREND if adx_strength == "moderate" else MarketRegime.STRONG_DOWNTREND
        else:
            regime = MarketRegime.VOLATILE
        trend_alignment = 0.7 if regime.is_trending else 0.3

    elif is_ranging:
        regime = MarketRegime.RANGING
        trend_alignment = 0.0

    else:
        # Trending based on ADX + DI + trend
        trend_conf = abs(trend_score) / (trend_cfg["strong_score"] * 2)

        if di_direction == "UP":
            if adx_strength in ("strong", "moderate") and trend in ("UP",):
                regime = MarketRegime.STRONG_UPTREND if adx_strength == "strong" else MarketRegime.UPTREND
            elif trend == "UP":
                regime = MarketRegime.WEAK_UPTREND
            elif trend == "DOWN":
                regime = MarketRegime.RANGING  # conflicting signals
            else:
                regime = MarketRegime.WEAK_UPTREND
        else:  # DOWN
            if adx_strength in ("strong", "moderate") and trend in ("DOWN",):
                regime = MarketRegime.STRONG_DOWNTREND if adx_strength == "strong" else MarketRegime.DOWNTREND
            elif trend == "DOWN":
                regime = MarketRegime.WEAK_DOWNTREND
            elif trend == "UP":
                regime = MarketRegime.RANGING
            else:
                regime = MarketRegime.WEAK_DOWNTREND

        trend_alignment = trend_conf

    # Calculate confidence
    confidence = (
        adx_conf * conf_w.get("adx", 0.3) +
        bb_conf * conf_w.get("bb_width", 0.2) +
        atr_conf * conf_w.get("atr_ratio", 0.2) +
        trend_alignment * conf_w.get("trend_alignment", 0.3)
    )
    confidence = max(0.0, min(1.0, confidence))

    return regime, confidence


def detect_regime_simple(df: pd.DataFrame, config: Optional[Dict] = None) -> RegimeResult:
    """Detect regime from a single DataFrame (convenience wrapper).

    Args:
        df: DataFrame with indicators calculated
        config: Override config

    Returns:
        RegimeResult
    """
    tf_data = {"Daily": df}
    return detect_regime(tf_data, config)


def regime_recommendation(regime: MarketRegime) -> List[str]:
    """Get trading recommendations based on regime.

    Args:
        regime: Detected market regime

    Returns:
        List of recommendation strings
    """
    recs = {
        MarketRegime.STRONG_UPTREND: [
            "✅ Xu hướng tăng mạnh — ưu tiên LONG",
            "→ Entry: chờ pullback về EMA21/SMA20",
            "→ SL dưới swing low gần nhất",
            "→ TP: sử dụng trailing stop, để lợi nhuận chạy",
            "→ Chiến lược: position/swing (giữ lâu)",
        ],
        MarketRegime.UPTREND: [
            "✅ Xu hướng tăng — ưu tiên LONG",
            "→ Entry: chờ pullback về hỗ trợ (SMA20, EMA21)",
            "→ SL dưới đáy pullback",
            "→ Chiến lược: swing/daytrade",
        ],
        MarketRegime.WEAK_UPTREND: [
            "⚠️ Xu hướng tăng yếu — cẩn trọng LONG",
            "→ Chỉ entry khi có xác nhận từ lower TF (15m/5m)",
            "→ Giảm position size, SL chặt",
        ],
        MarketRegime.RANGING: [
            "🟡 Thị trường đi ngang — ưu tiên mean-reversion",
            "→ Mua gần hỗ trợ (BB Lower, S/R), bán gần kháng cự",
            "→ Giảm position size, đặt SL chặt",
            "→ Chờ breakout để trade trend",
        ],
        MarketRegime.WEAK_DOWNTREND: [
            "⚠️ Xu hướng giảm yếu — cẩn trọng SHORT",
            "→ Chỉ entry khi có xác nhận từ lower TF",
            "→ Giảm position size, SL chặt",
        ],
        MarketRegime.DOWNTREND: [
            "✅ Xu hướng giảm — ưu tiên SHORT",
            "→ Entry: chờ pullback lên kháng cự (SMA20, EMA21)",
            "→ SL trên đỉnh pullback",
            "→ Chiến lược: swing/daytrade",
        ],
        MarketRegime.STRONG_DOWNTREND: [
            "✅ Xu hướng giảm mạnh — ưu tiên SHORT",
            "→ Entry: chờ pullback lên EMA21/SMA20",
            "→ SL trên swing high gần nhất",
            "→ TP: trailing stop, để lợi nhuận chạy",
            "→ Chiến lược: position/swing (giữ lâu)",
        ],
        MarketRegime.VOLATILE: [
            "⚡ Biến động cao — thận trọng",
            "→ Tăng SL (2x ATR bình thường)",
            "→ Giảm position size (50% bình thường)",
            "→ Chờ hết volatile rồi mới trade",
        ],
        MarketRegime.CHOPPY: [
            "🌀 Thị trường chop — KHÔNG trade",
            "→ Tỉ lệ thua cao khi chop",
            "→ Chờ clear hướng (ADX > 25) mới vào lệnh",
        ],
        MarketRegime.UNKNOWN: [
            "❓ Không đủ dữ liệu xác định regime",
        ],
    }
    return recs.get(regime, ["Không có khuyến nghị cho regime này"])


# ─── HELPER ───

def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
