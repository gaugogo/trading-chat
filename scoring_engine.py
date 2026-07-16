"""
scoring_engine.py — Unified Scoring Engine cho mọi strategy

Centralizes scoring logic that was duplicated across:
  - position.py  (macro_trend_score, position_confluence)
  - swing.py     (swing_score)
  - daytrade.py  (daytrade_signal_15m)
  - scalp.py     (scalp_signal_5m)
  - ichimoku.py  (ichimoku_signal_df, ichimoku_score)
  - smc.py       (_smc_signal)

Design:
  ScoringEngine (base)
    ├── TrendScorer       : xác định trend + strength từ indicators
    ├── ConfluenceScorer  : multi-TF alignment + weighted score
    ├── SignalScorer      : tổng hợp bias + entry/exit quality
    └── RiskScorer        : R:R, ATR-based position sizing

Usage:
  from scoring_engine import ScoringEngine
  engine = ScoringEngine(config={
      "weights": {"Daily": 5.0, "4H": 3.0, "1H": 2.0, "15m": 1.0, "5m": 0.5},
      "thresholds": {"strong_buy": 4.0, "buy_bias": 1.5, ...},
  })
  result = engine.score(tf_data, strategy="swing")
  print(result.bias, result.confidence)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from core import (
    determine_trend,
    fmt_price,
    TF_ORDER,
    TF_WEIGHTS as DEFAULT_WEIGHTS,
)

from regime import detect_regime, MarketRegime, regime_recommendation
from divergence import divergence_bias_score
from volume_profile import volume_profile_bias


# ─── ENUMS ──────────────────────────────────────────────────────────────

class Bias(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    BUY_BIAS = "BUY_BIAS"
    NEUTRAL = "NEUTRAL"
    SELL_BIAS = "SELL_BIAS"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    WAIT = "WAIT"

    @property
    def icon(self) -> str:
        return {
            Bias.STRONG_BUY: "🟢🟢",
            Bias.BUY: "🟢",
            Bias.BUY_BIAS: "🟢",
            Bias.NEUTRAL: "🟡",
            Bias.SELL_BIAS: "🔴",
            Bias.SELL: "🔴",
            Bias.STRONG_SELL: "🔴🔴",
            Bias.WAIT: "⏸️",
        }.get(self, "🟡")

    @property
    def numeric(self) -> int:
        return {
            Bias.STRONG_BUY: 2,
            Bias.BUY: 1,
            Bias.BUY_BIAS: 1,
            Bias.NEUTRAL: 0,
            Bias.SELL_BIAS: -1,
            Bias.SELL: -1,
            Bias.STRONG_SELL: -2,
            Bias.WAIT: 0,
        }.get(self, 0)


# ─── DATA CLASSES ───────────────────────────────────────────────────────

@dataclass
class TfScore:
    """Score for a single timeframe."""
    tf_name: str
    trend: str            # 'UP', 'DOWN', 'SIDEWAYS', 'WAIT'
    score: float          # Raw score from indicators
    weight: float         # TF weight multiplier
    weighted_score: float # score * weight
    details: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    """Result from the scoring engine."""
    strategy: str
    bias: Bias
    confidence: float      # 0.0 to 1.0
    raw_score: float       # Unnormalized total score
    normalized_score: float  # -10 to +10
    tf_scores: List[TfScore] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"{self.bias.icon} {self.strategy.upper()}: {self.bias.value} "
            f"(score: {self.normalized_score:+.1f}, confidence: {self.confidence:.0%})"
        )

    def detailed_report(self) -> str:
        """Full detailed report."""
        lines = [f"{'='*60}", f"  {self.bias.icon} {self.strategy.upper()} SCORING REPORT"]
        lines.append(f"  Bias: {self.bias.value} | Score: {self.normalized_score:+.1f}/10 | Confidence: {self.confidence:.0%}")
        lines.append(f"{'='*60}")
        lines.append("")
        lines.append(f"{'TF':<8} {'Trend':<10} {'Score':<8} {'Weight':<8} {'Weighted':<8}")
        lines.append("-" * 45)
        for ts in self.tf_scores:
            lines.append(f"{ts.tf_name:<8} {ts.trend:<10} {ts.score:+.1f}    {ts.weight:<8.1f} {ts.weighted_score:+.1f}")
        lines.append("")
        if self.details:
            lines.append("📋 Chi tiết:")
            for d in self.details:
                lines.append(f"  • {d}")
        if self.warnings:
            lines.append("")
            lines.append("⚠️ Cảnh báo:")
            for w in self.warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


# ─── DEFAULT CONFIG ─────────────────────────────────────────────────────

DEFAULT_SCORING_CONFIG: Dict[str, Any] = {
    # TF weights (mặc định, có thể override)
    "weights": {
        "Daily": 5.0,
        "4H": 3.0,
        "1H": 2.0,
        "15m": 1.0,
        "5m": 0.5,
    },
    # Ngưỡng bias
    "thresholds": {
        "strong_buy": 4.0,
        "buy": 2.5,
        "buy_bias": 1.0,
        "sell_bias": -1.0,
        "sell": -2.5,
        "strong_sell": -4.0,
    },
    # Indicator weights trong mỗi TF
    "indicator_weights": {
        "sma_position": 1.0,     # Price vs SMA20/50
        "rsi_zone": 1.0,         # RSI > 50 or < 50
        "macd_cross": 1.0,       # MACD > Signal
        "ema_alignment": 1.0,    # EMA9 > EMA21
        "bb_position": 0.5,      # Bollinger position
        "volume_confirmation": 0.5,  # Volume spike confirming
    },
    # ATR-based risk parameters
    "risk": {
        "position_atr_sl": 3.0,
        "position_atr_tp": 8.0,
        "swing_atr_sl": 2.0,
        "swing_atr_tp": 5.0,
        "daytrade_atr_sl": 1.5,
        "daytrade_atr_tp": 3.5,
        "scalp_atr_sl": 1.0,
        "scalp_atr_tp": 2.5,
    },
}


# ─── SCORING ENGINE ─────────────────────────────────────────────────────

class ScoringEngine:
    """Unified scoring engine for all trading strategies.

    Args:
        config: Override default scoring config (weights, thresholds, indicator weights)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._deep_copy(DEFAULT_SCORING_CONFIG)
        if config:
            self._deep_merge(self.config, config)

    @staticmethod
    def _deep_copy(source: Dict) -> Dict:
        """Deep copy a nested dict."""
        result = {}
        for key, value in source.items():
            if isinstance(value, dict):
                result[key] = ScoringEngine._deep_copy(value)
            elif isinstance(value, list):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """Deep merge override into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ScoringEngine._deep_merge(base[key], value)
            else:
                base[key] = value

    def score(
        self,
        tf_data: Dict[str, pd.DataFrame],
        strategy: str = "swing",
        decimals: int = 2,
    ) -> ScoringResult:
        """Score multi-timeframe data for a given strategy.

        Args:
            tf_data: Dict of timeframe DataFrames with indicators
            strategy: 'swing', 'position', 'daytrade', 'scalp', 'ichimoku'
            decimals: Price decimal places

        Returns:
            ScoringResult with bias, confidence, and detailed breakdown
        """
        if not tf_data:
            return ScoringResult(
                strategy=strategy,
                bias=Bias.WAIT,
                confidence=0.0,
                raw_score=0.0,
                normalized_score=0.0,
                details=["No data available"],
                warnings=["Empty timeframe data"],
            )

        weights = self.config["weights"]
        thresholds = self.config["thresholds"]
        indicator_weights = self.config["indicator_weights"]

        # Score each timeframe
        tf_scores: List[TfScore] = []
        total_weighted = 0.0
        total_weight = 0.0
        all_details: List[str] = []

        for tf_name in TF_ORDER:
            df = tf_data.get(tf_name)
            if df is None or df.empty or len(df) < 20:
                continue

            w = weights.get(tf_name, 1.0)
            ts = self._score_timeframe(df, tf_name, w, indicator_weights, strategy)
            tf_scores.append(ts)
            total_weighted += ts.weighted_score
            total_weight += w
            if ts.details:
                all_details.append(f"[{tf_name}] " + "; ".join(ts.details))

        if total_weight == 0:
            return ScoringResult(
                strategy=strategy,
                bias=Bias.WAIT,
                confidence=0.0,
                raw_score=0.0,
                normalized_score=0.0,
                details=["Insufficient data"],
                warnings=["No timeframe had enough data"],
            )

        raw_score = total_weighted
        max_possible = 5.0 * total_weight  # max ~5 per TF
        normalized = max(-10.0, min(10.0, (raw_score / max_possible) * 10)) if max_possible > 0 else 0.0

        # Determine bias
        bias = self._bias_from_score(normalized, thresholds)

        # Calculate confidence (0.0 to 1.0)
        # Based on: alignment across TFs, distance from neutral, data freshness
        confidence = self._calculate_confidence(tf_scores, normalized, tf_data)

        # Strategy-specific warnings
        warnings = self._strategy_warnings(tf_scores, strategy, tf_data)

        # ── Divergence adjustment ──
        div_score = divergence_bias_score(tf_data)
        if abs(div_score) > 1.0:
            # Apply divergence bias to score
            normalized += div_score * 0.15  # 15% weight
            normalized = max(-10.0, min(10.0, normalized))
            bias = self._bias_from_score(normalized, thresholds)
            div_dir = "BULLISH" if div_score > 0 else "BEARISH"
            all_details.append(f"🔄 Divergence bias: {div_dir} ({div_score:+.1f})")

        # ── Volume Profile adjustment ──
        vp_score = volume_profile_bias(tf_data)
        if abs(vp_score) > 1.0:
            normalized += vp_score * 0.1  # 10% weight
            normalized = max(-10.0, min(10.0, normalized))
            bias = self._bias_from_score(normalized, thresholds)
            vp_dir = "BULLISH" if vp_score > 0 else "BEARISH"
            all_details.append(f"📊 Volume Profile bias: {vp_dir} ({vp_score:+.1f})")

        # ── Regime-based adjustment ──
        regime_result = detect_regime(tf_data, self.config.get("regime"))
        regime_adj = self._apply_regime_factor(
            bias, normalized, confidence, regime_result.regime, strategy
        )
        if regime_adj:
            adjusted_bias, adjusted_confidence = regime_adj
            if adjusted_bias != bias or abs(adjusted_confidence - confidence) > 0.05:
                warnings.append(
                    f"🔄 Regime-adjusted: {regime_result.regime.value} "
                    f"({regime_result.regime.icon}) ảnh hưởng độ tin cậy"
                )
            bias = adjusted_bias
            confidence = adjusted_confidence

        return ScoringResult(
            strategy=strategy,
            bias=bias,
            confidence=confidence,
            raw_score=raw_score,
            normalized_score=round(normalized, 1),
            tf_scores=tf_scores,
            details=all_details,
            warnings=warnings,
        )

    def _score_timeframe(
        self,
        df: pd.DataFrame,
        tf_name: str,
        weight: float,
        indicator_weights: Dict[str, float],
        strategy: str,
    ) -> TfScore:
        """Score a single timeframe using indicators."""
        last = df.iloc[-1]
        close = float(last["Close"])

        score = 0.0
        details: List[str] = []
        iw = indicator_weights

        # 1. SMA position (price vs SMA20, SMA50)
        sma20 = last.get("SMA_20", np.nan)
        sma50 = last.get("SMA_50", np.nan)
        sma_score = 0.0
        if not pd.isna(sma20):
            sma_score += 1.0 if close > float(sma20) else -1.0
        if not pd.isna(sma50):
            sma_score += 1.0 if close > float(sma50) else -1.0
        if sma_score != 0:
            score += sma_score * iw.get("sma_position", 1.0)
            details.append(f"SMA:{sma_score:+.0f}")

        # 2. RSI zone
        rsi = last.get("RSI_14", np.nan)
        if not pd.isna(rsi):
            rsi_val = float(rsi)
            if rsi_val > 50:
                score += 1.0 * iw.get("rsi_zone", 1.0)
                details.append(f"RSI:{rsi_val:.0f} (bull)")
            else:
                score -= 1.0 * iw.get("rsi_zone", 1.0)
                details.append(f"RSI:{rsi_val:.0f} (bear)")

            # Overbought/oversold warning
            if rsi_val > 75:
                details.append("⚠️ Overbought")
            elif rsi_val < 25:
                details.append("⚠️ Oversold")

        # 3. MACD cross
        macd = last.get("MACD", np.nan)
        macd_sig = last.get("MACD_Signal", np.nan)
        if not pd.isna(macd) and not pd.isna(macd_sig):
            macd_cross = float(macd) > float(macd_sig)
            if macd_cross:
                score += 1.0 * iw.get("macd_cross", 1.0)
                details.append("MACD>Sig (bull)")
            else:
                score -= 1.0 * iw.get("macd_cross", 1.0)
                details.append("MACD<Sig (bear)")

        # 4. EMA alignment (EMA9 vs EMA21)
        ema9 = last.get("EMA_9", np.nan)
        ema21 = last.get("EMA_21", np.nan)
        if not pd.isna(ema9) and not pd.isna(ema21):
            if float(ema9) > float(ema21):
                score += 1.0 * iw.get("ema_alignment", 1.0)
                details.append("EMA9>21 (bull)")
            else:
                score -= 1.0 * iw.get("ema_alignment", 1.0)
                details.append("EMA9<21 (bear)")

        # 5. Bollinger Band position (strategy-dependent)
        bb_up = last.get("BB_Upper", np.nan)
        bb_low = last.get("BB_Lower", np.nan)
        if not pd.isna(bb_up) and not pd.isna(bb_low):
            bb_pos = (close - float(bb_low)) / (float(bb_up) - float(bb_low)) if (float(bb_up) - float(bb_low)) > 0 else 0.5
            bb_weight = iw.get("bb_position", 0.5)

            if strategy in ("scalp", "daytrade"):
                # Mean reversion: extreme = fade
                if bb_pos < 0.2:
                    score += 1.0 * bb_weight
                    details.append(f"BB lower ({bb_pos:.0%})")
                elif bb_pos > 0.8:
                    score -= 1.0 * bb_weight
                    details.append(f"BB upper ({bb_pos:.0%})")
            else:
                # Trend following: same side as trend
                if bb_pos > 0.8:
                    score += 0.5 * bb_weight
                    details.append("BB upper (momentum)")
                elif bb_pos < 0.2:
                    score -= 0.5 * bb_weight
                    details.append("BB lower (weakness)")

        return TfScore(
            tf_name=tf_name,
            trend=determine_trend(df)[0],
            score=round(score, 1),
            weight=weight,
            weighted_score=round(score * weight, 1),
            details=details,
        )

    def _bias_from_score(self, normalized: float, thresholds: Dict[str, float]) -> Bias:
        """Map normalized score to bias enum."""
        if normalized >= thresholds.get("strong_buy", 4.0):
            return Bias.STRONG_BUY
        elif normalized >= thresholds.get("buy", 2.5):
            return Bias.BUY
        elif normalized >= thresholds.get("buy_bias", 1.0):
            return Bias.BUY_BIAS
        elif normalized <= thresholds.get("strong_sell", -4.0):
            return Bias.STRONG_SELL
        elif normalized <= thresholds.get("sell", -2.5):
            return Bias.SELL
        elif normalized <= thresholds.get("sell_bias", -1.0):
            return Bias.SELL_BIAS
        else:
            return Bias.NEUTRAL

    def _calculate_confidence(
        self,
        tf_scores: List[TfScore],
        normalized: float,
        tf_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Calculate confidence level 0.0-1.0."""
        if not tf_scores:
            return 0.0

        # Factor 1: Alignment — all TFs agree?
        trends = [ts.trend for ts in tf_scores if ts.trend in ("UP", "DOWN")]
        total_directional = len(trends)
        if total_directional == 0:
            alignment = 0.0
        else:
            up_count = sum(1 for t in trends if t == "UP")
            down_count = sum(1 for t in trends if t == "DOWN")
            majority = max(up_count, down_count)
            alignment = majority / total_directional if total_directional > 0 else 0.0

        # Factor 2: Score magnitude (distance from neutral)
        magnitude = min(1.0, abs(normalized) / 8.0)

        # Factor 3: Data freshness (penalize if no low-TF data)
        has_intraday = any(
            tf in tf_data and not tf_data[tf].empty
            for tf in ["15m", "5m"]
        )
        freshness = 1.0 if has_intraday else 0.8

        # Weighted combination
        confidence = alignment * 0.5 + magnitude * 0.3 + freshness * 0.2
        return round(max(0.0, min(1.0, confidence)), 2)

    def _strategy_warnings(
        self,
        tf_scores: List[TfScore],
        strategy: str,
        tf_data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Generate strategy-specific warnings."""
        warnings: List[str] = []

        # Check for conflicting signals
        trends = {ts.tf_name: ts.trend for ts in tf_scores}
        up_tfs = [tf for tf, t in trends.items() if t == "UP"]
        down_tfs = [tf for tf, t in trends.items() if t == "DOWN"]

        if up_tfs and down_tfs:
            warnings.append(
                f"⚠️ Conflicting timeframes: {', '.join(up_tfs)} bullish vs "
                f"{', '.join(down_tfs)} bearish"
            )

        # Check TF alignment for strategy
        if strategy == "position":
            daily_trend = trends.get("Daily", "?")
            if daily_trend in ("SIDEWAYS", "WAIT"):
                warnings.append("⚠️ Position requires clear Daily trend")
            if up_tfs and "Daily" in down_tfs:
                warnings.append("⚠️ Going against Daily trend is high risk for position")

        elif strategy == "scalp":
            # Scalping needs context from 1H
            ctx = tf_data.get("1H")
            if ctx is not None and not ctx.empty:
                ctx_trend, _ = determine_trend(ctx)
                m5_trends = [ts.trend for ts in tf_scores if ts.tf_name in ("5m", "15m")]
                m5_up = all(t == "UP" for t in m5_trends)
                if ctx_trend == "DOWN" and m5_up:
                    warnings.append("⚠️ Scalping LONG against 1H downtrend")

        elif strategy == "ichimoku":
            if len(tf_data.get("Daily", pd.DataFrame())) < 53:
                warnings.append("⚠️ Ichimoku needs ≥53 candles on Daily for Kumo")

        return warnings

    def _apply_regime_factor(
        self,
        bias: Bias,
        normalized: float,
        confidence: float,
        regime: MarketRegime,
        strategy: str,
    ) -> Optional[Tuple[Bias, float]]:
        """Adjust bias/confidence based on market regime.

        - Trending market: trend-follow signals get higher confidence
        - Ranging market: mean-reversion signals get boost, trend gets penalty
        - Volatile market: all signals get reduced confidence
        - Choppy: heavily penalize all signals
        """
        new_bias = bias
        new_confidence = confidence
        modified = False

        if regime == MarketRegime.RANGING:
            # Ranging: mean-reversion strategies get boost
            if strategy in ("scalp", "daytrade"):
                if bias in (Bias.BUY_BIAS, Bias.SELL_BIAS):
                    new_confidence = min(1.0, confidence * 1.3)
                    modified = True
            # Trend strategies get penalty
            if strategy in ("position", "swing"):
                new_confidence *= 0.6
                if normalized > 0 and normalized < 3:
                    new_bias = Bias.NEUTRAL
                modified = True

        elif regime.is_trending:
            # Trending: trend-follow strategies get boost
            if strategy in ("position", "swing"):
                if (regime.is_bullish and bias in (Bias.BUY, Bias.BUY_BIAS, Bias.STRONG_BUY)) or \
                   (regime.is_bearish and bias in (Bias.SELL, Bias.SELL_BIAS, Bias.STRONG_SELL)):
                    new_confidence = min(1.0, confidence * 1.2)
                    modified = True
            # Counter-trend signals get penalty
            if (regime.is_bullish and bias in (Bias.SELL, Bias.SELL_BIAS, Bias.STRONG_SELL)) or \
               (regime.is_bearish and bias in (Bias.BUY, Bias.BUY_BIAS, Bias.STRONG_BUY)):
                new_confidence *= 0.4
                new_bias = Bias.NEUTRAL if abs(normalized) < 3 else (
                    Bias.BUY_BIAS if bias in (Bias.SELL, Bias.SELL_BIAS) else Bias.SELL_BIAS
                )
                modified = True

        elif regime == MarketRegime.VOLATILE:
            new_confidence *= 0.5
            modified = True

        elif regime == MarketRegime.CHOPPY:
            new_confidence *= 0.3
            if abs(normalized) < 5:
                new_bias = Bias.WAIT
            modified = True

        if modified:
            return (new_bias, round(new_confidence, 2))
        return None

    # ─── STRATEGY-SPECIFIC SHORTCUTS ──────────────────────────────────

    def score_position(self, tf_data: Dict[str, pd.DataFrame]) -> ScoringResult:
        """Quick scoring for position trading."""
        return self.score(tf_data, strategy="position")

    def score_swing(self, tf_data: Dict[str, pd.DataFrame]) -> ScoringResult:
        """Quick scoring for swing trading."""
        return self.score(tf_data, strategy="swing")

    def score_daytrade(self, tf_data: Dict[str, pd.DataFrame]) -> ScoringResult:
        """Quick scoring for day trading."""
        return self.score(tf_data, strategy="daytrade")

    def score_scalp(self, tf_data: Dict[str, pd.DataFrame]) -> ScoringResult:
        """Quick scoring for scalping."""
        return self.score(tf_data, strategy="scalp")

    def score_ichimoku(self, tf_data: Dict[str, pd.DataFrame]) -> ScoringResult:
        """Quick scoring for ichimoku."""
        return self.score(tf_data, strategy="ichimoku")


# ─── CONVENIENCE ────────────────────────────────────────────────────────

def quick_score(
    tf_data: Dict[str, pd.DataFrame],
    strategy: str = "swing",
) -> str:
    """Quick one-line score summary.

    Args:
        tf_data: Multi-timeframe data
        strategy: Trading strategy name

    Returns:
        One-line summary string
    """
    engine = ScoringEngine()
    result = engine.score(tf_data, strategy=strategy)
    return result.summary()
