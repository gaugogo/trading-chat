#!/usr/bin/env python3
"""
Price Action Analysis Module
=============================
Phân tích Price Action thuần — không dùng indicator, chỉ dựa trên:
  - Nến (candlestick patterns)
  - Cấu trúc thị trường (market structure)
  - Hỗ trợ / Kháng cự (support & resistance)
  - Hành động giá (price action signals)

Usage:
  from price_action import (
      analyze_price_action,
      format_pa_summary,
      detect_candlestick_patterns,
      find_support_resistance,
      analyze_market_structure,
      get_pa_signal,
  )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from core import TF_ORDER, TF_WEIGHTS


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandlePattern:
    """A detected candlestick pattern."""
    name: str               # e.g. "Pin Bar", "Engulfing Bullish"
    type: str               # "bullish", "bearish", "neutral"
    timeframe: str
    strength: float         # 0.0 - 1.0
    description: str
    price: float = 0.0
    confirmation: bool = False  # whether it's confirmed (closed) or forming


@dataclass
class SupportResistance:
    """A key support or resistance level."""
    level_type: str         # "support" or "resistance"
    price: float
    strength: float         # 0.0 - 1.0 (touches count / significance)
    timeframe: str
    touches: int = 1
    description: str = ""


@dataclass
class MarketStructure:
    """Market structure analysis result."""
    trend: str              # "UP", "DOWN", "SIDEWAYS"
    timeframe: str
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    structure_broken: bool = False
    description: str = ""


@dataclass
class PriceActionSignal:
    """Overall price action signal."""
    direction: str          # "buy", "sell", "neutral"
    confidence: float       # 0.0 - 1.0
    timeframe: str
    patterns: List[CandlePattern] = field(default_factory=list)
    key_levels: List[SupportResistance] = field(default_factory=list)
    structure: Optional[MarketStructure] = None
    reasoning: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════════════════════

def _body_size(candle: pd.Series) -> float:
    return abs(candle['Close'] - candle['Open'])


def _upper_wick(candle: pd.Series) -> float:
    return candle['High'] - max(candle['Close'], candle['Open'])


def _lower_wick(candle: pd.Series) -> float:
    return min(candle['Close'], candle['Open']) - candle['Low']


def _total_range(candle: pd.Series) -> float:
    return candle['High'] - candle['Low']


def _is_bullish(candle: pd.Series) -> bool:
    return candle['Close'] > candle['Open']


def _avg_body(df: pd.DataFrame, window: int = 20) -> float:
    """Average body size over last N candles."""
    bodies = df['Close'] - df['Open']
    return float(bodies.tail(window).abs().mean())


def _avg_range(df: pd.DataFrame, window: int = 20) -> float:
    """Average total range over last N candles."""
    ranges = df['High'] - df['Low']
    return float(ranges.tail(window).mean())


def detect_candlestick_patterns(df: pd.DataFrame, tf_name: str) -> List[CandlePattern]:
    """
    Detect candlestick patterns on the last 3-5 candles.

    Patterns detected:
      - Pin Bar (Hammer / Shooting Star)
      - Engulfing (Bullish / Bearish)
      - Inside Bar
      - Doji
      - Bullish / Bearish Harami
      - Morning Star / Evening Star (3-candle)
      - Three White Soldiers / Three Black Crows (3-candle)
    """
    if df.empty or len(df) < 6:
        return []

    patterns: List[CandlePattern] = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    prev3 = df.iloc[-4] if len(df) >= 4 else None

    avg_b = _avg_body(df)
    avg_r = _avg_range(df)
    last_body = _body_size(last)
    last_range = _total_range(last)
    last_upper = _upper_wick(last)
    last_lower = _lower_wick(last)
    bullish = _is_bullish(last)
    prev_bullish = _is_bullish(prev)

    # ── Pin Bar (Hammer / Shooting Star) ──
    # A candle with a small body and a long wick (>= 2x body) on one side
    if last_range > 0 and last_body > 0 and last_body < avg_b * 0.6:
        wick_ratio_upper = last_upper / last_body
        wick_ratio_lower = last_lower / last_body

        if wick_ratio_lower >= 2.0 and last_upper < last_body * 0.5:
            # Long lower wick — Hammer (bullish) / Hanging Man (bearish if in downtrend)
            patterns.append(CandlePattern(
                name="Hammer" if bullish else "Hanging Man",
                type="bullish" if bullish else "bearish",
                timeframe=tf_name,
                strength=min(1.0, wick_ratio_lower / 4.0),
                description=f"Long lower wick ({last_lower:.2f}) vs body ({last_body:.2f}) — rejection of lows",
                price=float(last['Close']),
                confirmation=True,
            ))

        if wick_ratio_upper >= 2.0 and last_lower < last_body * 0.5:
            # Long upper wick — Shooting Star (bearish) / Inverted Hammer (bullish if in uptrend)
            patterns.append(CandlePattern(
                name="Shooting Star" if not bullish else "Inverted Hammer",
                type="bearish" if not bullish else "bullish",
                timeframe=tf_name,
                strength=min(1.0, wick_ratio_upper / 4.0),
                description=f"Long upper wick ({last_upper:.2f}) vs body ({last_body:.2f}) — rejection of highs",
                price=float(last['Close']),
                confirmation=True,
            ))

    # ── Doji ──
    # Very small body (<= 10% of range) — indecision
    if last_range > 0 and last_body <= last_range * 0.1:
        patterns.append(CandlePattern(
            name="Doji",
            type="neutral",
            timeframe=tf_name,
            strength=0.5,
            description="Indecision candle — very small body relative to range",
            price=float(last['Close']),
            confirmation=True,
        ))

    # ── Engulfing ──
    # Bullish Engulfing: prev bearish, current bullish, body fully engulfs prev body
    if not prev_bullish and bullish and last_body > _body_size(prev) * 1.1:
        if last['Open'] < prev['Close'] and last['Close'] > prev['Open']:
            patterns.append(CandlePattern(
                name="Engulfing Bullish",
                type="bullish",
                timeframe=tf_name,
                strength=min(1.0, last_body / _body_size(prev)),
                description=f"Bullish candle engulfs previous bearish body ({last_body:.2f} vs {_body_size(prev):.2f})",
                price=float(last['Close']),
                confirmation=True,
            ))

    # Bearish Engulfing: prev bullish, current bearish, body fully engulfs prev body
    if prev_bullish and not bullish and last_body > _body_size(prev) * 1.1:
        if last['Open'] > prev['Close'] and last['Close'] < prev['Open']:
            patterns.append(CandlePattern(
                name="Engulfing Bearish",
                type="bearish",
                timeframe=tf_name,
                strength=min(1.0, last_body / _body_size(prev)),
                description=f"Bearish candle engulfs previous bullish body ({last_body:.2f} vs {_body_size(prev):.2f})",
                price=float(last['Close']),
                confirmation=True,
            ))

    # ── Inside Bar ──
    # Current candle's range is inside previous candle's range
    if last['High'] <= prev['High'] and last['Low'] >= prev['Low']:
        inside_ratio = last_range / _total_range(prev) if _total_range(prev) > 0 else 1
        if inside_ratio <= 0.85:
            patterns.append(CandlePattern(
                name="Inside Bar",
                type="neutral",
                timeframe=tf_name,
                strength=1.0 - inside_ratio,
                description=f"Price contracting — inside bar ({last_range:.2f} within {_total_range(prev):.2f})",
                price=float(last['Close']),
                confirmation=True,
            ))

    # ── Harami ──
    # Opposite of engulfing: small body inside previous large body
    if prev2 is not None:
        prev_body = _body_size(prev)
        if prev_body > avg_b * 1.2 and last_body < prev_body * 0.6:
            if last['High'] <= prev['High'] and last['Low'] >= prev['Low']:
                patterns.append(CandlePattern(
                    name="Bullish Harami" if not bullish else "Bearish Harami",
                    type="bullish" if not bullish else "bearish",
                    timeframe=tf_name,
                    strength=0.4,
                    description=f"Small candle inside previous large body — potential reversal",
                    price=float(last['Close']),
                    confirmation=True,
                ))

    # ── 3-Candle Patterns ──
    if prev2 is not None and prev3 is not None:
        c3 = prev3  # -4
        c2 = prev2  # -3
        c1 = prev   # -2
        c0 = last   # -1

        # Morning Star: long bearish, small indecision, long bullish
        if (not _is_bullish(c3) and _body_size(c3) > avg_b * 1.2 and
            _body_size(c2) < avg_b * 0.6 and
            bullish and _body_size(c0) > avg_b * 1.1 and
            c0['Close'] > c3['Open'] * 0.99):
            patterns.append(CandlePattern(
                name="Morning Star",
                type="bullish",
                timeframe=tf_name,
                strength=0.8,
                description="3-candle reversal: long bearish, indecision, long bullish",
                price=float(c0['Close']),
                confirmation=True,
            ))

        # Evening Star: long bullish, small indecision, long bearish
        if (_is_bullish(c3) and _body_size(c3) > avg_b * 1.2 and
            _body_size(c2) < avg_b * 0.6 and
            not bullish and _body_size(c0) > avg_b * 1.1 and
            c0['Close'] < c3['Open'] * 1.01):
            patterns.append(CandlePattern(
                name="Evening Star",
                type="bearish",
                timeframe=tf_name,
                strength=0.8,
                description="3-candle reversal: long bullish, indecision, long bearish",
                price=float(c0['Close']),
                confirmation=True,
            ))

        # Three White Soldiers: 3 consecutive long bullish candles
        if (bullish and _is_bullish(c1) and _is_bullish(c2) and
            _body_size(c0) > avg_b * 0.8 and
            _body_size(c1) > avg_b * 0.8 and
            _body_size(c2) > avg_b * 0.8):
            patterns.append(CandlePattern(
                name="Three White Soldiers",
                type="bullish",
                timeframe=tf_name,
                strength=0.7,
                description="3 consecutive strong bullish candles — sustained buying pressure",
                price=float(c0['Close']),
                confirmation=True,
            ))

        # Three Black Crows: 3 consecutive long bearish candles
        if (not bullish and not _is_bullish(c1) and not _is_bullish(c2) and
            _body_size(c0) > avg_b * 0.8 and
            _body_size(c1) > avg_b * 0.8 and
            _body_size(c2) > avg_b * 0.8):
            patterns.append(CandlePattern(
                name="Three Black Crows",
                type="bearish",
                timeframe=tf_name,
                strength=0.7,
                description="3 consecutive strong bearish candles — sustained selling pressure",
                price=float(c0['Close']),
                confirmation=True,
            ))

    # ── Long Wick Rejection ──
    if last_range > 0 and last_body > 0:
        upper_pct = last_upper / last_range * 100
        lower_pct = last_lower / last_range * 100

        if upper_pct >= 65 and lower_pct <= 20:
            patterns.append(CandlePattern(
                name="Rejection at High",
                type="bearish",
                timeframe=tf_name,
                strength=min(1.0, upper_pct / 80),
                description=f"Price rejected at high — upper wick {upper_pct:.0f}% of range",
                price=float(last['Close']),
                confirmation=True,
            ))

        if lower_pct >= 65 and upper_pct <= 20:
            patterns.append(CandlePattern(
                name="Rejection at Low",
                type="bullish",
                timeframe=tf_name,
                strength=min(1.0, lower_pct / 80),
                description=f"Price rejected at low — lower wick {lower_pct:.0f}% of range",
                price=float(last['Close']),
                confirmation=True,
            ))

    # ── Marubozu (strong trend candle) ──
    if last_range > 0 and last_body / last_range >= 0.9:
        patterns.append(CandlePattern(
            name="Bullish Marubozu" if bullish else "Bearish Marubozu",
            type="bullish" if bullish else "bearish",
            timeframe=tf_name,
            strength=0.6,
            description=f"Strong {'bullish' if bullish else 'bearish'} candle with minimal wicks — momentum",
            price=float(last['Close']),
            confirmation=True,
        ))

    return patterns


# ═══════════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE
# ═══════════════════════════════════════════════════════════════════

def _find_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[float], List[float]]:
    """Find swing high and swing low prices."""
    highs: List[float] = []
    lows: List[float] = []

    for i in range(lookback, len(df) - lookback):
        if all(df['High'].iloc[i] >= df['High'].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df['High'].iloc[i] >= df['High'].iloc[i + j] for j in range(1, lookback + 1)):
            highs.append(float(df['High'].iloc[i]))
        if all(df['Low'].iloc[i] <= df['Low'].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df['Low'].iloc[i] <= df['Low'].iloc[i + j] for j in range(1, lookback + 1)):
            lows.append(float(df['Low'].iloc[i]))

    return highs, lows


def _cluster_levels(levels: List[float], tolerance: float) -> List[Tuple[float, int]]:
    """Cluster nearby levels together."""
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters: List[Tuple[float, int]] = []  # (avg_price, count)

    current_cluster = [sorted_levels[0]]
    for price in sorted_levels[1:]:
        if abs(price - sum(current_cluster) / len(current_cluster)) <= tolerance:
            current_cluster.append(price)
        else:
            avg = sum(current_cluster) / len(current_cluster)
            clusters.append((avg, len(current_cluster)))
            current_cluster = [price]

    if current_cluster:
        avg = sum(current_cluster) / len(current_cluster)
        clusters.append((avg, len(current_cluster)))

    return clusters


def find_support_resistance(df: pd.DataFrame, tf_name: str, decimals: int = 2) -> List[SupportResistance]:
    """
    Find key support and resistance levels using swing points and clustering.

    Returns:
        List of SupportResistance objects, sorted by strength.
    """
    if df.empty or len(df) < 30:
        return []

    levels: List[SupportResistance] = []
    last_price = float(df['Close'].iloc[-1])
    avg_range = float((df['High'] - df['Low']).tail(20).mean())
    tolerance = avg_range * 0.5  # Cluster levels within half an ATR

    # Get swing points
    highs, lows = _find_swing_highs_lows(df, lookback=5)

    # Also check 20/50 period highs/lows
    if len(df) >= 20:
        highs.append(float(df['High'].tail(20).max()))
        lows.append(float(df['Low'].tail(20).min()))
    if len(df) >= 50:
        highs.append(float(df['High'].tail(50).max()))
        lows.append(float(df['Low'].tail(50).min()))
    if len(df) >= 200:
        highs.append(float(df['High'].tail(200).max()))
        lows.append(float(df['Low'].tail(200).min()))

    # Round number levels (psychological levels)
    if avg_range > 0:
        # Determine step size based on price range
        if last_price < 10:
            step = round(avg_range, 2)
        elif last_price < 100:
            step = round(avg_range, 1)
        elif last_price < 1000:
            step = round(avg_range / 5, -1)  # round to nearest 10
            if step == 0:
                step = 50
        else:
            step = round(avg_range / 10, -2)  # round to nearest 100
            if step == 0:
                step = 500

        if step > 0:
            nearest = round(last_price / step) * step
            for offset in [-3, -2, -1, 1, 2, 3]:
                rnd_level = nearest + offset * step
                if rnd_level > 0:
                    if rnd_level > last_price:
                        highs.append(rnd_level)
                    else:
                        lows.append(rnd_level)

    # Cluster highs as resistance
    res_clusters = _cluster_levels(highs, tolerance)
    for avg_price, count in res_clusters:
        if avg_price > last_price * 0.99:  # Only above or near current price
            strength = min(1.0, count / 5.0)
            label = "Major" if strength >= 0.6 else "Minor"
            levels.append(SupportResistance(
                level_type="resistance",
                price=round(avg_price, decimals),
                strength=strength,
                timeframe=tf_name,
                touches=count,
                description=f"{label} resistance — touched {count}x",
            ))

    # Cluster lows as support
    sup_clusters = _cluster_levels(lows, tolerance)
    for avg_price, count in sup_clusters:
        if avg_price <= last_price * 1.01:  # Only below or near current price
            strength = min(1.0, count / 5.0)
            label = "Major" if strength >= 0.6 else "Minor"
            levels.append(SupportResistance(
                level_type="support",
                price=round(avg_price, decimals),
                strength=strength,
                timeframe=tf_name,
                touches=count,
                description=f"{label} support — touched {count}x",
            ))

    # Sort by strength descending, keep top 4 resistance and 4 support
    res = sorted([l for l in levels if l.level_type == "resistance"], key=lambda x: x.strength, reverse=True)[:4]
    sup = sorted([l for l in levels if l.level_type == "support"], key=lambda x: x.strength, reverse=True)[:4]
    res.sort(key=lambda x: x.price)  # nearest first
    sup.sort(key=lambda x: x.price, reverse=True)  # nearest first

    return sup + res


# ═══════════════════════════════════════════════════════════════════
#  MARKET STRUCTURE
# ═══════════════════════════════════════════════════════════════════

def analyze_market_structure(df: pd.DataFrame, tf_name: str) -> MarketStructure:
    """
    Analyze market structure using HH/HL/LH/LL logic.

    Uptrend: Higher Highs (HH) + Higher Lows (HL)
    Downtrend: Lower Highs (LH) + Lower Lows (LL)
    """
    if df.empty or len(df) < 20:
        return MarketStructure(trend="WAIT", timeframe=tf_name)

    last_price = float(df['Close'].iloc[-1])
    lookback = min(30, len(df) // 3)

    # Find recent swing points
    highs, lows = _find_swing_highs_lows(df.tail(lookback), lookback=3)

    if len(highs) < 2 or len(lows) < 2:
        return MarketStructure(trend="SIDEWAYS", timeframe=tf_name)

    # Classify recent structure
    recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
    recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]

    hh = True  # higher highs
    lh = True  # lower highs
    hl = True  # higher lows
    ll = True  # lower lows

    for i in range(1, len(recent_highs)):
        if not (recent_highs[i] > recent_highs[i-1]):
            hh = False
        if not (recent_highs[i] < recent_highs[i-1]):
            lh = False

    for i in range(1, len(recent_lows)):
        if not (recent_lows[i] > recent_lows[i-1]):
            hl = False
        if not (recent_lows[i] < recent_lows[i-1]):
            ll = False

    # Determine trend
    trend = "SIDEWAYS"
    if hh and hl:
        trend = "UP"
    elif lh and ll:
        trend = "DOWN"
    elif hh and not hl:
        trend = "UP_WEAK"
    elif lh and not ll:
        trend = "DOWN_WEAK"

    # Check for structure break
    structure_broken = False
    if trend == "UP":
        # Structure broken if price breaks below last swing low
        last_swing_low = lows[-1] if lows else 0
        if last_price < last_swing_low:
            structure_broken = True
    elif trend == "DOWN":
        # Structure broken if price breaks above last swing high
        last_swing_high = highs[-1] if highs else 0
        if last_price > last_swing_high:
            structure_broken = True

    last_h = highs[-1] if highs else None
    last_l = lows[-1] if lows else None
    prev_h = highs[-2] if len(highs) >= 2 else None
    prev_l = lows[-2] if len(lows) >= 2 else None

    # Build description
    desc_parts: List[str] = []
    if trend == "UP":
        desc_parts.append("📈 Uptrend — Higher Highs + Higher Lows")
    elif trend == "DOWN":
        desc_parts.append("📉 Downtrend — Lower Highs + Lower Lows")
    elif trend == "UP_WEAK":
        desc_parts.append("⚠️ Weak uptrend — higher highs but not higher lows")
    elif trend == "DOWN_WEAK":
        desc_parts.append("⚠️ Weak downtrend — lower lows but not lower highs")
    else:
        desc_parts.append("➡️ Sideways — no clear structure")

    if last_h and prev_h:
        desc_parts.append(f"Highs: {prev_h:.2f} → {last_h:.2f} ({'HH' if last_h > prev_h else 'LH'})")
    if last_l and prev_l:
        desc_parts.append(f"Lows: {prev_l:.2f} → {last_l:.2f} ({'HL' if last_l > prev_l else 'LL'})")

    if structure_broken:
        desc_parts.append("🚨 Structure broken! Potential reversal.")

    return MarketStructure(
        trend=trend,
        timeframe=tf_name,
        last_high=last_h,
        last_low=last_l,
        higher_highs=hh,
        higher_lows=hl,
        lower_highs=lh,
        lower_lows=ll,
        structure_broken=structure_broken,
        description=" | ".join(desc_parts),
    )


# ═══════════════════════════════════════════════════════════════════
#  PRICE ACTION SIGNAL
# ═══════════════════════════════════════════════════════════════════

def get_pa_signal(
    patterns: List[CandlePattern],
    structure: MarketStructure,
    levels: List[SupportResistance],
) -> PriceActionSignal:
    """
    Generate a consolidated Price Action signal from patterns, structure, and levels.
    """
    if not patterns and structure.trend == "SIDEWAYS":
        return PriceActionSignal(
            direction="neutral",
            confidence=0.0,
            timeframe=structure.timeframe,
            patterns=patterns,
            key_levels=levels,
            structure=structure,
            reasoning=["No clear price action signals detected."],
        )

    reasoning: List[str] = []
    bullish_score = 0.0
    bearish_score = 0.0
    weights = {"pattern": 0.4, "structure": 0.4, "levels": 0.2}

    # ── Evaluate patterns ──
    for p in patterns:
        if p.type == "bullish":
            bullish_score += weights["pattern"] * p.strength
            reasoning.append(f"🟢 {p.name}: {p.description}")
        elif p.type == "bearish":
            bearish_score += weights["pattern"] * p.strength
            reasoning.append(f"🔴 {p.name}: {p.description}")

    # ── Evaluate structure ──
    if structure.trend == "UP":
        bullish_score += weights["structure"] * 0.8
        reasoning.append(f"📈 Structure: {structure.description}")
    elif structure.trend == "DOWN":
        bearish_score += weights["structure"] * 0.8
        reasoning.append(f"📉 Structure: {structure.description}")
    elif structure.trend == "UP_WEAK":
        bullish_score += weights["structure"] * 0.3
        reasoning.append(f"⚠️ Structure: {structure.description}")
    elif structure.trend == "DOWN_WEAK":
        bearish_score += weights["structure"] * 0.3
        reasoning.append(f"⚠️ Structure: {structure.description}")

    if structure.structure_broken:
        if structure.trend in ("UP", "UP_WEAK"):
            bearish_score += weights["structure"] * 0.5
            reasoning.append("🚨 Uptrend structure broken — potential reversal down")
        else:
            bullish_score += weights["structure"] * 0.5
            reasoning.append("🚨 Downtrend structure broken — potential reversal up")

    # ── Evaluate levels ──
    last_price = patterns[0].price if patterns else 0
    for lvl in levels:
        if lvl.level_type == "resistance" and abs(lvl.price - last_price) < (last_price * 0.005):
            reasoning.append(f"🔴 Price near resistance: {lvl.price:.2f} ({lvl.description})")
            bearish_score += weights["levels"] * lvl.strength * 0.5
        elif lvl.level_type == "support" and abs(lvl.price - last_price) < (last_price * 0.005):
            reasoning.append(f"🟢 Price near support: {lvl.price:.2f} ({lvl.description})")
            bullish_score += weights["levels"] * lvl.strength * 0.5

    # Normalize scores
    max_score = 1.0
    bullish_score = min(bullish_score, max_score)
    bearish_score = min(bearish_score, max_score)

    # Determine direction
    diff = bullish_score - bearish_score
    if diff > 0.15:
        direction = "buy"
        confidence = bullish_score
        reasoning.append(f"✅ Overall: BUY signal (confidence: {confidence:.0%})")
    elif diff < -0.15:
        direction = "sell"
        confidence = bearish_score
        reasoning.append(f"✅ Overall: SELL signal (confidence: {confidence:.0%})")
    else:
        direction = "neutral"
        confidence = max(bullish_score, bearish_score)
        reasoning.append(f"➡️ Overall: NEUTRAL — conflicting signals")

    return PriceActionSignal(
        direction=direction,
        confidence=round(confidence, 2),
        timeframe=structure.timeframe,
        patterns=patterns,
        key_levels=levels,
        structure=structure,
        reasoning=reasoning,
    )


# ═══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_price_action(
    tf_data: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    target_tf: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full Price Action analysis across all timeframes (or a specific one).

    Args:
        tf_data: Dict of timeframe -> OHLCV DataFrame
        cfg: Instrument config
        target_tf: Optional specific timeframe (e.g. "1H"), else all

    Returns:
        {
            "results": { tf_name: { patterns, structure, levels, signal } },
            "multi_tf_signal": consolidated signal across TFs
        }
    """
    results: Dict[str, Any] = {}
    tfs = [target_tf] if target_tf else TF_ORDER

    for tf_name in tfs:
        if tf_name not in tf_data or tf_data[tf_name].empty or len(tf_data[tf_name]) < 10:
            continue

        df = tf_data[tf_name]
        decimals = cfg.get("decimals", 2)

        # 1. Detect candlestick patterns
        patterns = detect_candlestick_patterns(df, tf_name)

        # 2. Find support/resistance
        levels = find_support_resistance(df, tf_name, decimals)

        # 3. Analyze market structure
        structure = analyze_market_structure(df, tf_name)

        # 4. Generate signal
        signal = get_pa_signal(patterns, structure, levels)

        results[tf_name] = {
            "patterns": patterns,
            "levels": levels,
            "structure": structure,
            "signal": signal,
        }

    # Build multi-TF consensus
    multi_tf_signal = _build_multi_tf_signal(results, tfs)

    return {
        "results": results,
        "multi_tf_signal": multi_tf_signal,
    }


def _build_multi_tf_signal(
    results: Dict[str, Any],
    tfs: List[str],
) -> PriceActionSignal:
    """Build a consolidated signal across multiple timeframes."""
    buy_weight = 0.0
    sell_weight = 0.0
    all_patterns: List[CandlePattern] = []
    all_levels: List[SupportResistance] = []
    all_structures: List[MarketStructure] = []
    all_reasoning: List[str] = []

    for tf_name in tfs:
        if tf_name not in results:
            continue
        r = results[tf_name]
        signal = r.get("signal")
        if not signal:
            continue

        weight = TF_WEIGHTS.get(tf_name, 1.0)

        if signal.direction == "buy":
            buy_weight += weight * signal.confidence
            all_reasoning.append(f"[{tf_name}] BUY (confidence: {signal.confidence:.0%}, weight: {weight:.1f})")
        elif signal.direction == "sell":
            sell_weight += weight * signal.confidence
            all_reasoning.append(f"[{tf_name}] SELL (confidence: {signal.confidence:.0%}, weight: {weight:.1f})")
        else:
            all_reasoning.append(f"[{tf_name}] Neutral")

        all_patterns.extend(signal.patterns)
        all_levels.extend(signal.key_levels)
        if signal.structure:
            all_structures.append(signal.structure)

    total_buy = buy_weight
    total_sell = sell_weight
    max_possible = sum(TF_WEIGHTS.get(tf, 1.0) for tf in tfs if tf in results)

    if max_possible > 0:
        buy_pct = total_buy / max_possible
        sell_pct = total_sell / max_possible
    else:
        buy_pct = sell_pct = 0

    if buy_pct > sell_pct + 0.1:
        direction = "buy"
        confidence = buy_pct
    elif sell_pct > buy_pct + 0.1:
        direction = "sell"
        confidence = sell_pct
    else:
        direction = "neutral"
        confidence = max(buy_pct, sell_pct)

    all_reasoning.append(f"\n📊 Multi-TF Consensus:")
    all_reasoning.append(f"   Buy alignment: {buy_pct:.0%} | Sell alignment: {sell_pct:.0%}")
    all_reasoning.append(f"   Final: {direction.upper()} (confidence: {confidence:.0%})")

    return PriceActionSignal(
        direction=direction,
        confidence=round(confidence, 2),
        timeframe="MULTI",
        patterns=all_patterns,
        key_levels=all_levels,
        reasoning=all_reasoning,
    )


# ═══════════════════════════════════════════════════════════════════
#  FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_pa_summary(pa_data: Dict[str, Any], decimals: int = 2) -> str:
    """Format Price Action analysis into a readable string."""
    lines: List[str] = []
    results = pa_data.get("results", {})
    multi_tf = pa_data.get("multi_tf_signal")

    lines.append("=" * 72)
    lines.append("  📊 PRICE ACTION ANALYSIS")
    lines.append("=" * 72)

    # Multi-TF header
    if multi_tf:
        dir_emoji = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}
        arrow = dir_emoji.get(multi_tf.direction, "⚪")
        lines.append(f"\n{arrow} SIGNAL: {multi_tf.direction.upper()} (confidence: {multi_tf.confidence:.0%})")
        lines.append("")

    # Per-timeframe details
    for tf_name in TF_ORDER:
        if tf_name not in results:
            continue
        r = results[tf_name]
        signal = r.get("signal")
        structure = r.get("structure")
        patterns = r.get("patterns", [])
        levels = r.get("levels", [])

        lines.append(f"\n── [{tf_name}] ──{'─' * 40}")

        # Structure
        if structure:
            lines.append(f"  {structure.description}")

        # Patterns
        if patterns:
            lines.append(f"  📍 Candlestick Patterns ({len(patterns)}):")
            for p in patterns:
                emoji = "🟢" if p.type == "bullish" else ("🔴" if p.type == "bearish" else "⚪")
                lines.append(f"     {emoji} {p.name} ({p.strength:.0%}): {p.description}")

        # Key levels
        if levels:
            lines.append(f"  📌 Key Levels:")
            for lvl in levels:
                star = "⭐" if lvl.strength >= 0.6 else "▪️"
                lines.append(f"     {star} {lvl.level_type.title()}: {lvl.price:.{decimals}f} ({lvl.description})")

        # Signal
        if signal:
            dir_emoji = "🟢" if signal.direction == "buy" else ("🔴" if signal.direction == "sell" else "⚪")
            lines.append(f"  → {dir_emoji} {signal.direction.upper()} (conf: {signal.confidence:.0%})")

        lines.append("")

    # Multi-TF reasoning
    if multi_tf and multi_tf.reasoning:
        lines.append("── [Multi-TF Reasoning] ──" + "─" * 35)
        for r in multi_tf.reasoning:
            lines.append(f"  {r}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def format_pa_compact(pa_data: Dict[str, Any], decimals: int = 2) -> str:
    """Compact one-line summary per timeframe."""
    lines: List[str] = []
    results = pa_data.get("results", {})
    multi_tf = pa_data.get("multi_tf_signal")

    if multi_tf:
        dir_emoji = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}
        lines.append(f"PA Signal: {dir_emoji.get(multi_tf.direction, '⚪')} {multi_tf.direction.upper()} ({multi_tf.confidence:.0%})")

    for tf_name in TF_ORDER:
        if tf_name not in results:
            continue
        r = results[tf_name]
        signal = r.get("signal")
        structure = r.get("structure")
        patterns = r.get("patterns", [])

        pa_str = f"[{tf_name}] "
        if structure:
            trend_emoji = {"UP": "📈", "DOWN": "📉", "SIDEWAYS": "➡️", "UP_WEAK": "📈⚠️", "DOWN_WEAK": "📉⚠️"}
            pa_str += f"{trend_emoji.get(structure.trend, '➡️')} "
        if patterns:
            pa_str += f"{len(patterns)}P "
        if signal:
            s = signal.direction.upper()
            pa_str += f"→ {s} ({signal.confidence:.0%})"
        lines.append(pa_str)

    return "\n".join(lines)
