"""
divergence.py — RSI & MACD Divergence Detection

Phát hiện divergence (regular & hidden) giữa price action và indicator.

Các loại divergence:
  - Regular Bullish: price tạo đáy thấp hơn, indicator tạo đáy cao hơn → đảo chiều UP
  - Regular Bearish: price tạo đỉnh cao hơn, indicator tạo đỉnh thấp hơn → đảo chiều DOWN
  - Hidden Bullish: price tạo đáy cao hơn, indicator tạo đáy thấp hơn → tiếp diễn UP
  - Hidden Bearish: price tạo đỉnh thấp hơn, indicator tạo đỉnh cao hơn → tiếp diễn DOWN

Usage:
  from divergence import (
      find_rsi_divergence, find_macd_divergence,
      DivergenceResult, DivergenceType,
      analyze_all_divergences,
  )
  result = find_rsi_divergence(df)
  print(result.bullish, result.bearish)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ─── ENUMS ───

class DivergenceType(Enum):
    REGULAR_BULLISH = "REGULAR_BULLISH"     # Price lower low, RSI higher low → reversal UP
    REGULAR_BEARISH = "REGULAR_BEARISH"     # Price higher high, RSI lower high → reversal DOWN
    HIDDEN_BULLISH = "HIDDEN_BULLISH"       # Price higher low, RSI lower low → continuation UP
    HIDDEN_BEARISH = "HIDDEN_BEARISH"       # Price lower high, RSI higher high → continuation DOWN

    @property
    def icon(self) -> str:
        return {
            DivergenceType.REGULAR_BULLISH: "🟢🔄",
            DivergenceType.REGULAR_BEARISH: "🔴🔄",
            DivergenceType.HIDDEN_BULLISH: "🟢▶️",
            DivergenceType.HIDDEN_BEARISH: "🔴▶️",
        }.get(self, "❓")

    @property
    def is_bullish(self) -> bool:
        return self in (DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH)

    @property
    def is_bearish(self) -> bool:
        return self in (DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH)

    @property
    def is_regular(self) -> bool:
        return self in (DivergenceType.REGULAR_BULLISH, DivergenceType.REGULAR_BEARISH)

    @property
    def is_hidden(self) -> bool:
        return self in (DivergenceType.HIDDEN_BULLISH, DivergenceType.HIDDEN_BEARISH)

    @property
    def label_vn(self) -> str:
        return {
            DivergenceType.REGULAR_BULLISH: "Regular Bullish (đảo chiều lên)",
            DivergenceType.REGULAR_BEARISH: "Regular Bearish (đảo chiều xuống)",
            DivergenceType.HIDDEN_BULLISH: "Hidden Bullish (tiếp diễn lên)",
            DivergenceType.HIDDEN_BEARISH: "Hidden Bearish (tiếp diễn xuống)",
        }.get(self, "Unknown")


@dataclass
class DivergenceSignal:
    """Một tín hiệu divergence cụ thể."""
    type: DivergenceType
    indicator: str                  # 'RSI' or 'MACD'
    price_pivot_1_idx: int          # First price pivot index
    price_pivot_2_idx: int          # Second price pivot index
    price_pivot_1_val: float        # First price pivot value
    price_pivot_2_val: float        # Second price pivot value
    indicator_pivot_1_val: float    # Indicator value at first pivot
    indicator_pivot_2_val: float    # Indicator value at second pivot
    strength: float                 # 0.0 to 1.0 (based on slope difference)
    age_bars: int = 0               # How many bars since detected

    def summary(self) -> str:
        """One-line summary."""
        dir_str = "BULLISH ↗️" if self.type.is_bullish else "BEARISH ↘️"
        reg_str = "Regular" if self.type.is_regular else "Hidden"
        return (
            f"{self.icon} {self.indicator} {reg_str} {dir_str} "
            f"(price: {self.price_pivot_1_val:.2f}→{self.price_pivot_2_val:.2f}, "
            f"{self.indicator}: {self.indicator_pivot_1_val:.1f}→{self.indicator_pivot_2_val:.1f}, "
            f"strength: {self.strength:.0%})"
        )

    @property
    def icon(self) -> str:
        return self.type.icon


@dataclass
class DivergenceResult:
    """Kết quả phân tích divergence cho 1 indicator."""
    indicator: str
    bullish: List[DivergenceSignal] = field(default_factory=list)
    bearish: List[DivergenceSignal] = field(default_factory=list)

    @property
    def has_bullish(self) -> bool:
        return len(self.bullish) > 0

    @property
    def has_bearish(self) -> bool:
        return len(self.bearish) > 0

    @property
    def total_signals(self) -> int:
        return len(self.bullish) + len(self.bearish)

    @property
    def net_bias(self) -> str:
        """Net divergence bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'."""
        bull_strength = sum(s.strength for s in self.bullish)
        bear_strength = sum(s.strength for s in self.bearish)
        if bull_strength > bear_strength + 0.3:
            return "BULLISH"
        elif bear_strength > bull_strength + 0.3:
            return "BEARISH"
        return "NEUTRAL"

    def summary(self) -> str:
        """Multi-line summary."""
        lines = [f"📊 {self.indicator} Divergence Analysis:"]
        if not self.bullish and not self.bearish:
            lines.append("  Không phát hiện divergence ⏸️")
            return "\n".join(lines)

        if self.bullish:
            lines.append(f"  🟢 Bullish ({len(self.bullish)}):")
            for d in self.bullish:
                lines.append(f"    • {d.summary()}")
        if self.bearish:
            lines.append(f"  🔴 Bearish ({len(self.bearish)}):")
            for d in self.bearish:
                lines.append(f"    • {d.summary()}")

        lines.append(f"  → Net bias: {self.net_bias}")
        return "\n".join(lines)


@dataclass
class FullDivergenceReport:
    """Báo cáo đầy đủ tất cả divergence trên tất cả TF."""
    tf_results: Dict[str, Dict[str, DivergenceResult]]  # tf_name -> {indicator: result}
    strongest_bullish: Optional[Tuple[str, str, DivergenceSignal]] = None   # (tf, indicator, signal)
    strongest_bearish: Optional[Tuple[str, str, DivergenceSignal]] = None   # (tf, indicator, signal)

    @property
    def has_divergence(self) -> bool:
        """Whether ANY divergence was found."""
        for tf_res in self.tf_results.values():
            for res in tf_res.values():
                if res.total_signals > 0:
                    return True
        return False

    def summary(self) -> str:
        """Full summary across all timeframes."""
        lines = [f"{'='*60}"]
        lines.append("  🔄 DIVERGENCE ANALYSIS — TẤT CẢ KHUNG THỜI GIAN")
        lines.append(f"{'='*60}")

        has_any = False
        for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
            if tf_name not in self.tf_results:
                continue
            lines.append(f"\n── [{tf_name}] ──")
            for ind_name in ["RSI", "MACD"]:
                res = self.tf_results[tf_name].get(ind_name)
                if res and res.total_signals > 0:
                    has_any = True
                    for d in res.bullish:
                        strength = "⚠️" if d.strength >= 0.7 else "📌"
                        lines.append(
                            f"  {strength} {d.type.icon} {d.indicator} {d.type.label_vn} "
                            f"(strength: {d.strength:.0%})"
                        )
                    for d in res.bearish:
                        strength = "⚠️" if d.strength >= 0.7 else "📌"
                        lines.append(
                            f"  {strength} {d.type.icon} {d.indicator} {d.type.label_vn} "
                            f"(strength: {d.strength:.0%})"
                        )
                elif res:
                    lines.append(f"  {ind_name}: không có divergence")

        if not has_any:
            lines.append("\n  ✅ Không phát hiện divergence đáng kể.")

        # Strongest signals
        if self.strongest_bullish:
            tf, ind, sig = self.strongest_bullish
            lines.append(f"\n  🟢🔺 Mạnh nhất: [{tf}] {sig.summary()}")
        if self.strongest_bearish:
            tf, ind, sig = self.strongest_bearish
            lines.append(f"\n  🔴🔺 Mạnh nhất: [{tf}] {sig.summary()}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)


# ─── PIVOT DETECTION ───

def _find_pivots(
    values: np.ndarray,
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find swing highs and lows in a price/indicator series.

    Args:
        values: 1D array of values
        order: Number of bars on each side to confirm a pivot (default: 5)

    Returns:
        Tuple of (peaks_idx, troughs_idx) — arrays of indices
    """
    peaks = []
    troughs = []

    for i in range(order, len(values) - order):
        window = values[i - order:i + order + 1]
        center = values[i]

        # Check if center is a peak (highest in window)
        if center == np.max(window) and center > np.median(window):
            # Make sure it's not a flat area
            if np.std(window) > np.std(values) * 0.05:
                peaks.append(i)

        # Check if center is a trough (lowest in window)
        if center == np.min(window) and center < np.median(window):
            if np.std(window) > np.std(values) * 0.05:
                troughs.append(i)

    return np.array(peaks), np.array(troughs)


def _find_pivots_adaptive(
    values: np.ndarray,
    min_order: int = 3,
    max_order: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find pivots with adaptive order based on data length.

    Args:
        values: 1D array of values
        min_order: Minimum pivot order
        max_order: Maximum pivot order

    Returns:
        Tuple of (peaks_idx, troughs_idx)
    """
    # Use larger order for longer data
    n = len(values)
    order = min(max_order, max(min_order, n // 20))
    return _find_pivots(values, order)


# ─── DIVERGENCE DETECTION ───

def _detect_divergence(
    price: np.ndarray,
    indicator: np.ndarray,
    indicator_name: str,
    min_strength: float = 0.3,
) -> DivergenceResult:
    """Detect all divergences between price and an indicator.

    Args:
        price: Array of close prices
        indicator: Array of indicator values (same length)
        indicator_name: 'RSI' or 'MACD'
        min_strength: Minimum strength to report (0.0 to 1.0)

    Returns:
        DivergenceResult with bullish and bearish signals
    """
    result = DivergenceResult(indicator=indicator_name)
    if len(price) < 20 or len(indicator) < 20:
        return result

    peaks, troughs = _find_pivots_adaptive(price)
    ind_peaks, ind_troughs = _find_pivots_adaptive(indicator)

    logger.debug(f"Detecting divergence: {len(peaks)} peaks, {len(troughs)} troughs")

    # Convert to sets for quick lookup
    peak_set = set(peaks)
    trough_set = set(troughs)
    ind_peak_set = set(ind_peaks)
    ind_trough_set = set(ind_troughs)

    # ── Regular Bearish Divergence ──
    # Price makes higher high, indicator makes lower high
    for i in range(len(peaks) - 1):
        p1_idx = peaks[i]
        p2_idx = peaks[i + 1]
        p1_price = price[p1_idx]
        p2_price = price[p2_idx]

        if p2_price <= p1_price:
            continue  # Need higher high

        # Find corresponding indicator peaks near these price peaks
        nearby_ind_peaks_1 = ind_peaks[
            (ind_peaks >= p1_idx - 3) & (ind_peaks <= p1_idx + 3)
        ]
        nearby_ind_peaks_2 = ind_peaks[
            (ind_peaks >= p2_idx - 3) & (ind_peaks <= p2_idx + 3)
        ]

        if len(nearby_ind_peaks_1) == 0 or len(nearby_ind_peaks_2) == 0:
            continue

        i1_val = float(indicator[nearby_ind_peaks_1[-1]])
        i2_val = float(indicator[nearby_ind_peaks_2[0]])

        if i2_val >= i1_val:
            continue  # Need lower high on indicator

        # Calculate strength
        price_change = (p2_price - p1_price) / p1_price
        ind_change = (i1_val - i2_val) / max(abs(i1_val), 0.01)
        strength = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)

        if strength >= min_strength:
            result.bearish.append(DivergenceSignal(
                type=DivergenceType.REGULAR_BEARISH,
                indicator=indicator_name,
                price_pivot_1_idx=p1_idx,
                price_pivot_2_idx=p2_idx,
                price_pivot_1_val=float(p1_price),
                price_pivot_2_val=float(p2_price),
                indicator_pivot_1_val=i1_val,
                indicator_pivot_2_val=i2_val,
                strength=round(strength, 2),
                age_bars=len(price) - p2_idx,
            ))

    # ── Regular Bullish Divergence ──
    # Price makes lower low, indicator makes higher low
    for i in range(len(troughs) - 1):
        t1_idx = troughs[i]
        t2_idx = troughs[i + 1]
        t1_price = price[t1_idx]
        t2_price = price[t2_idx]

        if t2_price >= t1_price:
            continue  # Need lower low

        # Find corresponding indicator troughs
        nearby_ind_troughs_1 = ind_troughs[
            (ind_troughs >= t1_idx - 3) & (ind_troughs <= t1_idx + 3)
        ]
        nearby_ind_troughs_2 = ind_troughs[
            (ind_troughs >= t2_idx - 3) & (ind_troughs <= t2_idx + 3)
        ]

        if len(nearby_ind_troughs_1) == 0 or len(nearby_ind_troughs_2) == 0:
            continue

        i1_val = float(indicator[nearby_ind_troughs_1[-1]])
        i2_val = float(indicator[nearby_ind_troughs_2[0]])

        if i2_val <= i1_val:
            continue  # Need higher low on indicator

        price_change = (t1_price - t2_price) / t1_price
        ind_change = (i2_val - i1_val) / max(abs(i1_val), 0.01)
        strength = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)

        if strength >= min_strength:
            result.bullish.append(DivergenceSignal(
                type=DivergenceType.REGULAR_BULLISH,
                indicator=indicator_name,
                price_pivot_1_idx=t1_idx,
                price_pivot_2_idx=t2_idx,
                price_pivot_1_val=float(t1_price),
                price_pivot_2_val=float(t2_price),
                indicator_pivot_1_val=i1_val,
                indicator_pivot_2_val=i2_val,
                strength=round(strength, 2),
                age_bars=len(price) - t2_idx,
            ))

    # ── Hidden Bearish Divergence ──
    # Price makes lower high, indicator makes higher high
    for i in range(len(peaks) - 1):
        p1_idx = peaks[i]
        p2_idx = peaks[i + 1]
        p1_price = price[p1_idx]
        p2_price = price[p2_idx]

        if p2_price >= p1_price:
            continue  # Need lower high

        nearby_ind_peaks_1 = ind_peaks[
            (ind_peaks >= p1_idx - 3) & (ind_peaks <= p1_idx + 3)
        ]
        nearby_ind_peaks_2 = ind_peaks[
            (ind_peaks >= p2_idx - 3) & (ind_peaks <= p2_idx + 3)
        ]

        if len(nearby_ind_peaks_1) == 0 or len(nearby_ind_peaks_2) == 0:
            continue

        i1_val = float(indicator[nearby_ind_peaks_1[-1]])
        i2_val = float(indicator[nearby_ind_peaks_2[0]])

        if i2_val <= i1_val:
            continue  # Need higher high on indicator

        price_change = (p1_price - p2_price) / p1_price
        ind_change = (i2_val - i1_val) / max(abs(i1_val), 0.01)
        strength = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)

        if strength >= min_strength:
            result.bearish.append(DivergenceSignal(
                type=DivergenceType.HIDDEN_BEARISH,
                indicator=indicator_name,
                price_pivot_1_idx=p1_idx,
                price_pivot_2_idx=p2_idx,
                price_pivot_1_val=float(p1_price),
                price_pivot_2_val=float(p2_price),
                indicator_pivot_1_val=i1_val,
                indicator_pivot_2_val=i2_val,
                strength=round(strength, 2),
                age_bars=len(price) - p2_idx,
            ))

    # ── Hidden Bullish Divergence ──
    # Price makes higher low, indicator makes lower low
    for i in range(len(troughs) - 1):
        t1_idx = troughs[i]
        t2_idx = troughs[i + 1]
        t1_price = price[t1_idx]
        t2_price = price[t2_idx]

        if t2_price <= t1_price:
            continue  # Need higher low

        nearby_ind_troughs_1 = ind_troughs[
            (ind_troughs >= t1_idx - 3) & (ind_troughs <= t1_idx + 3)
        ]
        nearby_ind_troughs_2 = ind_troughs[
            (ind_troughs >= t2_idx - 3) & (ind_troughs <= t2_idx + 3)
        ]

        if len(nearby_ind_troughs_1) == 0 or len(nearby_ind_troughs_2) == 0:
            continue

        i1_val = float(indicator[nearby_ind_troughs_1[-1]])
        i2_val = float(indicator[nearby_ind_troughs_2[0]])

        if i2_val >= i1_val:
            continue  # Need lower low on indicator

        price_change = (t2_price - t1_price) / t1_price
        ind_change = (i1_val - i2_val) / max(abs(i1_val), 0.01)
        strength = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)

        if strength >= min_strength:
            result.bullish.append(DivergenceSignal(
                type=DivergenceType.HIDDEN_BULLISH,
                indicator=indicator_name,
                price_pivot_1_idx=t1_idx,
                price_pivot_2_idx=t2_idx,
                price_pivot_1_val=float(t1_price),
                price_pivot_2_val=float(t2_price),
                indicator_pivot_1_val=i1_val,
                indicator_pivot_2_val=i2_val,
                strength=round(strength, 2),
                age_bars=len(price) - t2_idx,
            ))

    return result


def find_rsi_divergence(
    df: pd.DataFrame,
    rsi_col: str = 'RSI_14',
    min_strength: float = 0.3,
) -> DivergenceResult:
    """Find RSI divergences in a DataFrame.

    Args:
        df: DataFrame with Close and RSI_14 columns
        rsi_col: RSI column name
        min_strength: Minimum strength to report (0.0 to 1.0)

    Returns:
        DivergenceResult
    """
    if df.empty or len(df) < 20:
        return DivergenceResult(indicator="RSI")

    if rsi_col not in df.columns:
        return DivergenceResult(indicator="RSI")

    price = df['Close'].values.astype(float)
    rsi = df[rsi_col].values.astype(float)

    return _detect_divergence(price, rsi, "RSI", min_strength)


def find_macd_divergence(
    df: pd.DataFrame,
    macd_col: str = 'MACD',
    hist_col: str = 'MACD_Hist',
    min_strength: float = 0.3,
) -> DivergenceResult:
    """Find MACD divergences in a DataFrame.

    Uses MACD line (not histogram) for divergence detection.

    Args:
        df: DataFrame with Close and MACD columns
        macd_col: MACD line column name
        hist_col: MACD histogram column (unused, for API consistency)
        min_strength: Minimum strength to report (0.0 to 1.0)

    Returns:
        DivergenceResult
    """
    if df.empty or len(df) < 20:
        return DivergenceResult(indicator="MACD")

    if macd_col not in df.columns:
        return DivergenceResult(indicator="MACD")

    price = df['Close'].values.astype(float)
    macd = df[macd_col].values.astype(float)

    return _detect_divergence(price, macd, "MACD", min_strength)


# ─── MULTI-TF ANALYSIS ───

def analyze_all_divergences(
    tf_data: Dict[str, pd.DataFrame],
    min_strength: float = 0.3,
) -> FullDivergenceReport:
    """Analyze RSI + MACD divergences across all timeframes.

    Args:
        tf_data: Dict of timeframe DataFrames with indicators
        min_strength: Minimum strength to report

    Returns:
        FullDivergenceReport
    """
    from core import TF_ORDER

    tf_results: Dict[str, Dict[str, DivergenceResult]] = {}
    strongest_bullish: Optional[Tuple[str, str, DivergenceSignal]] = None
    strongest_bearish: Optional[Tuple[str, str, DivergenceSignal]] = None

    for tf_name in TF_ORDER:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue

        rsi_res = find_rsi_divergence(df, min_strength=min_strength)
        macd_res = find_macd_divergence(df, min_strength=min_strength)
        tf_results[tf_name] = {"RSI": rsi_res, "MACD": macd_res}

        # Track strongest signals
        for res in [rsi_res, macd_res]:
            for sig in res.bullish:
                if strongest_bullish is None or sig.strength > strongest_bullish[2].strength:
                    strongest_bullish = (tf_name, res.indicator, sig)
            for sig in res.bearish:
                if strongest_bearish is None or sig.strength > strongest_bearish[2].strength:
                    strongest_bearish = (tf_name, res.indicator, sig)

    return FullDivergenceReport(
        tf_results=tf_results,
        strongest_bullish=strongest_bullish,
        strongest_bearish=strongest_bearish,
    )


# ─── CONVENIENCE ───

def divergence_bias_score(
    tf_data: Dict[str, pd.DataFrame],
) -> float:
    """Calculate a divergence bias score (-10 to +10).

    Positive = bullish divergence bias
    Negative = bearish divergence bias
    Zero = neutral / no divergence

    Args:
        tf_data: Dict of timeframe DataFrames

    Returns:
        Bias score
    """
    from core import TF_WEIGHTS

    report = analyze_all_divergences(tf_data)
    score = 0.0

    for tf_name, tf_res in report.tf_results.items():
        w = TF_WEIGHTS.get(tf_name, 1.0)
        for ind_res in tf_res.values():
            for sig in ind_res.bullish:
                score += sig.strength * w * (2.0 if sig.type.is_regular else 1.0)
            for sig in ind_res.bearish:
                score -= sig.strength * w * (2.0 if sig.type.is_regular else 1.0)

    # Normalize to -10 to +10
    max_possible = sum(TF_WEIGHTS.get(tf, 1.0) * 4.0 for tf in tf_data.keys())
    if max_possible > 0:
        score = max(-10.0, min(10.0, (score / max_possible) * 10.0))

    return round(score, 1)
