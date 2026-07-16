"""
volume_profile.py — Volume Profile Analysis

Cải thiện từ bản cũ (daytrade.py: volume_profile_poc):
  - Dynamic bins dựa trên ATR (tự động điều chỉnh số lượng bins theo biến động)
  - Delta volume: phân biệt buy volume (candle xanh) vs sell volume (candle đỏ)
  - Session-based VP: phân tích riêng từng phiên (Asian/London/US)
  - Value Area tính chính xác hơn với 70% volume rule
  - Imbalance detection: vùng giá mất cân bằng volume

Usage:
  from volume_profile import VolumeProfile, VPResult
  vp = VolumeProfile()
  result = vp.analyze(df_1h)
  print(result.poc, result.vah, result.val, result.delta_ratio)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from core import fmt_price

logger = logging.getLogger(__name__)


# ─── ENUMS ───

class VolumeImbalance(Enum):
    BULLISH = "BULLISH"       # Buy volume dominates at key levels
    BEARISH = "BEARISH"       # Sell volume dominates at key levels
    NEUTRAL = "NEUTRAL"       # Balanced volume
    EXTREME_BUY = "EXTREME_BUY"   # Extreme buy imbalance
    EXTREME_SELL = "EXTREME_SELL" # Extreme sell imbalance

    @property
    def icon(self) -> str:
        return {
            VolumeImbalance.EXTREME_BUY: "🟢🟢",
            VolumeImbalance.BULLISH: "🟢",
            VolumeImbalance.NEUTRAL: "🟡",
            VolumeImbalance.BEARISH: "🔴",
            VolumeImbalance.EXTREME_SELL: "🔴🔴",
        }.get(self, "🟡")


@dataclass
class VPResult:
    """Kết quả Volume Profile cho 1 khung thời gian/session."""
    tf_name: str                    # Timeframe or session name
    poc: float                      # Point of Control (giá nhiều volume nhất)
    poc_volume: float               # Volume tại POC
    vah: float                      # Value Area High (top 70% volume)
    val: float                      # Value Area Low (bottom 70% volume)
    total_volume: float             # Tổng volume
    buy_volume: float               # Buy volume (up candles)
    sell_volume: float              # Sell volume (down candles)
    delta_volume: float             # buy - sell
    delta_ratio: float              # buy/sell ratio (1.0 = balanced)
    imbalance: VolumeImbalance      # Volume imbalance status
    bins_count: int                 # Number of bins used
    bin_size: float                 # Price per bin
    high: float                     # Price high
    low: float                      # Price low
    current_price: float            # Current price relative to VP
    price_in_value_area: bool       # Is current price in value area?
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary."""
        dir_str = f"POC={fmt_price(self.poc)}, VA={fmt_price(self.val)}-{fmt_price(self.vah)}"
        delta_str = f"Δ={self.delta_volume:.0f} ({self.imbalance.icon})"
        return f"VP [{self.tf_name}]: {dir_str} | {delta_str}"

    def detailed_report(self) -> str:
        """Full detailed report."""
        lines = [f"📊 Volume Profile — {self.tf_name}"]
        lines.append(f"  POC: {fmt_price(self.poc)} (vol: {self.poc_volume:.0f})")
        lines.append(f"  Value Area: {fmt_price(self.val)} - {fmt_price(self.vah)}")
        lines.append(f"  Range: {fmt_price(self.low)} - {fmt_price(self.high)}")
        lines.append(f"  Current: {fmt_price(self.current_price)}")
        lines.append(f"  In Value Area: {'✅ Có' if self.price_in_value_area else '❌ Không'}")

        # Delta analysis
        pct_buy = (self.buy_volume / self.total_volume * 100) if self.total_volume > 0 else 0
        lines.append(f"")
        lines.append(f"  📈 Volume Delta:")
        lines.append(f"    Buy:  {self.buy_volume:.0f} ({pct_buy:.1f}%)")
        lines.append(f"    Sell: {self.sell_volume:.0f} ({100-pct_buy:.1f}%)")
        lines.append(f"    Δ:    {self.delta_volume:+.0f} ({self.delta_ratio:.2f}x)")
        lines.append(f"    Status: {self.imbalance.icon} {self.imbalance.value}")

        lines.append(f"")
        lines.append(f"  ⚙️ Parameters:")
        lines.append(f"    Bins: {self.bins_count} (size: ${self.bin_size:.2f})")
        lines.append(f"    Total Volume: {self.total_volume:.0f}")

        for d in self.details:
            lines.append(f"  {d}")

        # Interpretation
        lines.append(f"")
        lines.append(f"  💡 Interpretation:")
        if self.imbalance in (VolumeImbalance.EXTREME_BUY, VolumeImbalance.BULLISH):
            lines.append(f"    → Bullish volume pressure. Giá có xu hướng tăng.")
            lines.append(f"    → Nếu giá trên VA, kỳ vọng tiếp diễn xu hướng.")
            lines.append(f"    → Nếu giá dưới VA, có thể là accumulation.")
        elif self.imbalance in (VolumeImbalance.EXTREME_SELL, VolumeImbalance.BEARISH):
            lines.append(f"    → Bearish volume pressure. Giá có xu hướng giảm.")
            lines.append(f"    → Nếu giá dưới VA, kỳ vọng tiếp diễn xu hướng.")
            lines.append(f"    → Nếu giá trên VA, có thể là distribution.")
        else:
            lines.append(f"    → Volume cân bằng. Thị trường đang do dự.")
            lines.append(f"    → Chờ volume breakout để xác định hướng.")

        return "\n".join(lines)


@dataclass
class FullVPReport:
    """Báo cáo Volume Profile cho nhiều khung thời gian."""
    results: Dict[str, VPResult] = field(default_factory=dict)
    strongest_imbalance: Optional[Tuple[str, VolumeImbalance, float]] = None  # (tf, imbalance, delta_ratio)

    @property
    def has_data(self) -> bool:
        return len(self.results) > 0

    def summary(self) -> str:
        """Multi-VP summary."""
        lines = [f"{'='*60}"]
        lines.append("  📊 VOLUME PROFILE ANALYSIS — TẤT CẢ KHUNG THỜI GIAN")
        lines.append(f"{'='*60}")

        for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
            vp = self.results.get(tf_name)
            if vp:
                lines.append(f"\n── [{tf_name}] ──")
                lines.append(f"  {vp.summary()}")
                lines.append(f"  Delta: {vp.delta_volume:+.0f} ({vp.imbalance.icon})")
                if vp.price_in_value_area:
                    lines.append(f"  Giá trong VA: ✅")
                else:
                    lines.append(f"  Giá ngoài VA: {'📈 Trên' if vp.current_price > vp.vah else '📉 Dưới'} VA")

        if self.strongest_imbalance:
            tf, imb, ratio = self.strongest_imbalance
            lines.append(f"\n  🔺 Mạnh nhất: [{tf}] {imb.icon} {imb.value} (Δ ratio: {ratio:.2f})")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)


# ─── DYNAMIC BINS ───

def _calculate_dynamic_bins(df: pd.DataFrame, min_bins: int = 8, max_bins: int = 30) -> int:
    """Calculate optimal number of bins based on ATR and data length.

    More volatile markets → fewer bins (wider range per bin)
    More data → more bins (finer granularity)

    Args:
        df: DataFrame with ATR column
        min_bins: Minimum number of bins
        max_bins: Maximum number of bins

    Returns:
        Number of bins
    """
    n = len(df)
    available = df.get('ATR', pd.Series([0] * n))

    if not available.isna().all() and available.iloc[-1] > 0:
        atr = float(available.iloc[-1])
        price_range = float(df['High'].max()) - float(df['Low'].min())
        if price_range > 0 and atr > 0:
            suggested = int(price_range / (atr * 0.5))  # ~2 bins per ATR
            return max(min_bins, min(max_bins, suggested))

    # Fallback: scale with data length
    return max(min_bins, min(max_bins, n // 5))


# ─── DELTA VOLUME ───

def _calculate_delta(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Calculate buy/sell volume from OHLCV data.

    - Buy volume: volume on up-close candles (Close > Open)
    - Sell volume: volume on down-close candles (Close < Open)
    - Neutral: Close == Open → split 50/50

    Args:
        df: DataFrame with Open, Close, Volume columns

    Returns:
        Tuple of (buy_volume, sell_volume, delta, delta_ratio)
    """
    if df.empty or 'Volume' not in df.columns:
        return 0.0, 0.0, 0.0, 1.0

    volume = df['Volume'].fillna(0).values
    close = df['Close'].values
    open_ = df['Open'].values

    buy_mask = close >= open_
    sell_mask = close < open_

    buy_vol = float(np.sum(volume[buy_mask]))
    sell_vol = float(np.sum(volume[sell_mask]))

    delta = buy_vol - sell_vol
    delta_ratio = buy_vol / sell_vol if sell_vol > 0 else (2.0 if buy_vol > 0 else 1.0)

    return buy_vol, sell_vol, delta, delta_ratio


# ─── SESSION DETECTION ───

def _detect_session(df: pd.DataFrame) -> str:
    """Detect trading session based on time of day.

    Only works if DataFrame has DatetimeIndex.

    Returns:
        Session name: 'Asian', 'London', 'NewYork', 'All', 'Unknown'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return "All"

    if df.empty:
        return "Unknown"

    last_time = df.index[-1]
    hour = last_time.hour

    # Rough session times in UTC
    if 0 <= hour < 8:
        return "Asian"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 22:
        return "NewYork"
    else:
        return "Asian"


# ─── MAIN ANALYSIS ───

def analyze_volume_profile(
    df: pd.DataFrame,
    tf_name: str = "1H",
    bins: Optional[int] = None,
    use_dynamic_bins: bool = True,
    min_bins: int = 8,
    max_bins: int = 30,
) -> Optional[VPResult]:
    """Analyze Volume Profile for a single timeframe.

    Args:
        df: DataFrame with OHLCV data (can have indicators)
        tf_name: Timeframe name for display
        bins: Fixed number of bins (None = dynamic)
        use_dynamic_bins: Auto-calculate bins from ATR
        min_bins: Minimum bins for dynamic calculation
        max_bins: Maximum bins for dynamic calculation

    Returns:
        VPResult or None if insufficient data
    """
    if df.empty or len(df) < 10:
        return None

    logger.debug(f"Volume Profile for {tf_name}: {len(df)} candles")
    high = float(df['High'].max())
    low = float(df['Low'].min())
    current_price = float(df['Close'].iloc[-1])

    if high == low or np.isnan(high) or np.isnan(low):
        return None

    # Calculate bins
    if use_dynamic_bins and bins is None:
        bins_count = _calculate_dynamic_bins(df, min_bins, max_bins)
    else:
        bins_count = bins or 10

    bin_size = (high - low) / bins_count

    # Initialize bins
    volume_bins = [0.0] * bins_count
    buy_vol_bins = [0.0] * bins_count
    sell_vol_bins = [0.0] * bins_count
    bin_centers = [low + (i + 0.5) * bin_size for i in range(bins_count)]

    # Aggregate volume into bins
    for _, row in df.iterrows():
        r_high = float(row['High']) if not pd.isna(row['High']) else 0
        r_low = float(row['Low']) if not pd.isna(row['Low']) else 0
        r_vol = float(row.get('Volume', 0)) if not pd.isna(row.get('Volume', np.nan)) else 0

        if r_vol <= 0:
            continue

        # Distribute volume across price range of the candle
        candle_range = r_high - r_low
        if candle_range <= 0:
            continue

        # Determine if this candle is buy or sell
        r_close = float(row['Close']) if not pd.isna(row['Close']) else 0
        r_open = float(row['Open']) if not pd.isna(row['Open']) else 0
        is_buy = r_close >= r_open

        # Distribute volume across bins the candle spans
        for b in range(bins_count):
            bin_low = low + b * bin_size
            bin_high = bin_low + bin_size

            overlap_start = max(r_low, bin_low)
            overlap_end = min(r_high, bin_high)

            if overlap_end > overlap_start:
                overlap_pct = (overlap_end - overlap_start) / candle_range
                vol_share = r_vol * overlap_pct
                volume_bins[b] += vol_share
                if is_buy:
                    buy_vol_bins[b] += vol_share
                else:
                    sell_vol_bins[b] += vol_share

    # Total volumes
    total_volume = sum(volume_bins)
    total_buy = sum(buy_vol_bins)
    total_sell = sum(sell_vol_bins)
    delta = total_buy - total_sell
    delta_ratio = total_buy / total_sell if total_sell > 0 else (2.0 if total_buy > 0 else 1.0)

    # Determine imbalance
    if delta_ratio >= 1.8:
        imbalance = VolumeImbalance.EXTREME_BUY
    elif delta_ratio >= 1.3:
        imbalance = VolumeImbalance.BULLISH
    elif delta_ratio <= 0.55:
        imbalance = VolumeImbalance.EXTREME_SELL
    elif delta_ratio <= 0.75:
        imbalance = VolumeImbalance.BEARISH
    else:
        imbalance = VolumeImbalance.NEUTRAL

    # Find POC
    max_vol = max(volume_bins) if volume_bins else 0
    if max_vol == 0:
        return None

    poc_idx = volume_bins.index(max_vol)
    poc_price = round(bin_centers[poc_idx], 2)
    poc_volume = max_vol

    # Value Area (70% of total volume around POC)
    target_vol = total_volume * 0.7
    accumulated = volume_bins[poc_idx]
    vah_idx = poc_idx
    val_idx = poc_idx

    while accumulated < target_vol:
        if vah_idx + 1 < bins_count and val_idx > 0:
            if volume_bins[vah_idx + 1] >= volume_bins[val_idx - 1]:
                vah_idx += 1
                accumulated += volume_bins[vah_idx]
            else:
                val_idx -= 1
                accumulated += volume_bins[val_idx]
        elif vah_idx + 1 < bins_count:
            vah_idx += 1
            accumulated += volume_bins[vah_idx]
        elif val_idx > 0:
            val_idx -= 1
            accumulated += volume_bins[val_idx]
        else:
            break

    vah = round(bin_centers[vah_idx], 2)
    val = round(bin_centers[val_idx], 2)

    # Is current price in value area?
    price_in_va = val <= current_price <= vah

    # Details
    details: List[str] = []
    session = _detect_session(df)
    if session != "All":
        details.append(f"Session: {session}")
    details.append(f"Bins: {bins_count} (size=${bin_size:.2f})")
    details.append(f"POC volume: {max_vol:.0f} ({max_vol/total_volume*100:.1f}% of total)")

    # Check for volume gaps (bins with very low volume)
    avg_bin_vol = total_volume / bins_count
    gaps = [i for i in range(bins_count) if volume_bins[i] < avg_bin_vol * 0.1]
    if gaps:
        gap_ranges = []
        for g in gaps:
            gap_ranges.append(f"${bin_centers[g]:.2f}")
        if len(gap_ranges) <= 5:
            details.append(f"Volume gaps at: {', '.join(gap_ranges)}")

    return VPResult(
        tf_name=tf_name,
        poc=poc_price,
        poc_volume=poc_volume,
        vah=vah,
        val=val,
        total_volume=total_volume,
        buy_volume=total_buy,
        sell_volume=total_sell,
        delta_volume=round(delta, 0),
        delta_ratio=round(delta_ratio, 2),
        imbalance=imbalance,
        bins_count=bins_count,
        bin_size=round(bin_size, 2),
        high=round(high, 2),
        low=round(low, 2),
        current_price=current_price,
        price_in_value_area=price_in_va,
        details=details,
    )


def analyze_all_timeframes(
    tf_data: Dict[str, pd.DataFrame],
) -> FullVPReport:
    """Analyze Volume Profile across all timeframes.

    Args:
        tf_data: Dict of timeframe DataFrames

    Returns:
        FullVPReport
    """
    from core import TF_ORDER

    results: Dict[str, VPResult] = {}
    strongest: Optional[Tuple[str, VolumeImbalance, float]] = None
    max_delta_abs = 0.0

    for tf_name in TF_ORDER:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue

        vp = analyze_volume_profile(df, tf_name=tf_name)
        if vp:
            results[tf_name] = vp

            # Track strongest imbalance (by delta ratio magnitude)
            ratio_mag = abs(vp.delta_ratio - 1.0)
            if ratio_mag > max_delta_abs:
                max_delta_abs = ratio_mag
                strongest = (tf_name, vp.imbalance, vp.delta_ratio)

    return FullVPReport(
        results=results,
        strongest_imbalance=strongest,
    )


def volume_profile_bias(
    tf_data: Dict[str, pd.DataFrame],
) -> float:
    """Calculate volume profile bias score (-10 to +10).

    Positive = bullish volume pressure
    Negative = bearish volume pressure

    Args:
        tf_data: Dict of timeframe DataFrames

    Returns:
        Bias score
    """
    from core import TF_WEIGHTS

    report = analyze_all_timeframes(tf_data)
    score = 0.0

    for tf_name, vp in report.results.items():
        w = TF_WEIGHTS.get(tf_name, 1.0)
        # Delta ratio > 1 = bullish, < 1 = bearish
        delta_factor = (vp.delta_ratio - 1.0) * 3.0  # Scale to ~ -3 to +3
        score += delta_factor * w

    # Normalize
    max_possible = sum(TF_WEIGHTS.get(tf, 1.0) * 3.0 for tf in report.results.keys())
    if max_possible > 0:
        score = max(-10.0, min(10.0, (score / max_possible) * 10.0))

    return round(score, 1)
