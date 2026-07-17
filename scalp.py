#!/usr/bin/env python3
"""
Scalping Analysis Module
Focus: 1m, 5m, 15m khung thời gian thấp
Tín hiệu: momentum break, pullback entry, tight SL/TP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from core import (
    adjust_to_spot,
    fetch_all_timeframes,
    fetch_spot_price,
    calculate_indicators,
    fmt_price,
    TF_ORDER,
)
from instruments import INSTRUMENTS

# ─── Scalping-specific config ────────────────────────────────────────

SCALP_TIMEFRAMES = ["15m", "5m"]  # Khung TF chính của scalping
CONTEXT_TIMEFRAME = "1H"           # TF để xác định bias chung

# Indicators nhanh hơn cho scalping
SCALP_EMA_SHORT = 5
SCALP_EMA_LONG = 9
SCALP_RSI = 7                       # RSI nhanh
SCALP_ATR = 7                       # ATR nhanh

# ─── Scalping signal scoring ──────────────────────────────────────────

def scalp_signal_5m(df: pd.DataFrame) -> Dict[str, Any]:
    """Tính tín hiệu scalping cho một khung thời gian."""
    if df.empty or len(df) < 20:
        return {"trend": "WAIT", "score": 0, "details": []}

    close = df["Close"]
    last = df.iloc[-1]
    score = 0
    details: List[str] = []

    # 1. EMA cross nhanh (5/9)
    ema5 = close.ewm(span=SCALP_EMA_SHORT, adjust=False).mean()
    ema9 = close.ewm(span=SCALP_EMA_LONG, adjust=False).mean()
    ema5_val = float(ema5.iloc[-1])
    ema9_val = float(ema9.iloc[-1])

    if ema5_val > ema9_val:
        score += 1
        details.append(f"EMA5({ema5_val:.2f}) > EMA9({ema9_val:.2f}) [+1]")
    else:
        score -= 1
        details.append(f"EMA5({ema5_val:.2f}) < EMA9({ema9_val:.2f}) [-1]")

    # 2. EMA slope (dốc lên/xuống)
    ema5_slope = float(ema5.iloc[-1] - ema5.iloc[-5]) if len(ema5) >= 5 else 0
    if ema5_slope > 0:
        score += 0.5
        details.append(f"EMA5 slope UP [+0.5]")
    elif ema5_slope < 0:
        score -= 0.5
        details.append(f"EMA5 slope DOWN [-0.5]")

    # 3. RSI nhanh (7)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / SCALP_RSI, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / SCALP_RSI, adjust=False).mean()
    rs_val = avg_gain / avg_loss.replace(0, np.nan)
    rsi7 = 100.0 - (100.0 / (1.0 + rs_val))
    rsi7_val = float(rsi7.iloc[-1])

    if rsi7_val > 60:
        score += 0.5
        details.append(f"RSI7={rsi7_val:.1f} (bullish) [+0.5]")
    elif rsi7_val < 40:
        score -= 0.5
        details.append(f"RSI7={rsi7_val:.1f} (bearish) [-0.5]")
    else:
        details.append(f"RSI7={rsi7_val:.1f} (neutral)")

    # 4. ATR volatility check
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr7 = tr.ewm(alpha=1 / SCALP_ATR, adjust=False).mean()
    atr_val = float(atr7.iloc[-1])
    atr_pct = (atr_val / float(close.iloc[-1])) * 100
    details.append(f"ATR7={atr_val:.2f} ({atr_pct:.3f}%)")

    # 5. Momentum (3 nến gần nhất)
    last3 = close.iloc[-3:]
    if len(last3) >= 3 and last3.iloc[-1] > last3.iloc[-2] > last3.iloc[-3]:
        score += 1
        details.append("3 nến tăng liên tiếp [+1]")
    elif len(last3) >= 3 and last3.iloc[-1] < last3.iloc[-2] < last3.iloc[-3]:
        score -= 1
        details.append("3 nến giảm liên tiếp [-1]")

    # 6. Vị trí so với range gần nhất (10 nến)
    h10 = df["High"].iloc[-10:].max()
    l10 = df["Low"].iloc[-10:].min()
    range10 = h10 - l10
    pos_pct = (float(last["Close"]) - l10) / range10 * 100 if range10 > 0 else 50

    if pos_pct > 80:
        score += 0.5
        details.append(f"Price near high of 10-bar range ({pos_pct:.0f}%) [+0.5]")
    elif pos_pct < 20:
        score -= 0.5
        details.append(f"Price near low of 10-bar range ({pos_pct:.0f}%) [-0.5]")
    else:
        details.append(f"Price in middle of 10-bar range ({pos_pct:.0f}%)")

    trend = "UP" if score >= 1.5 else "DOWN" if score <= -1.5 else "SIDEWAYS"

    return {
        "trend": trend,
        "score": round(score, 1),
        "rsi7": round(rsi7_val, 1),
        "ema5": round(ema5_val, 2),
        "ema9": round(ema9_val, 2),
        "atr": round(atr_val, 2),
        "atr_pct": round(atr_pct, 4),
        "range_pos_pct": round(pos_pct, 1),
        "details": details,
    }


def scalp_entry_zones(
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    bias: str,
    decimals: int,
) -> Dict[str, Any]:
    """Tính vùng entry/exit cho scalping dựa trên 5m + 15m."""
    last_5m = df_5m.iloc[-1]
    last_15m = df_15m.iloc[-1]
    price = float(last_5m["Close"])

    # ATR-based SL/TP
    tr_5m = pd.concat([
        df_5m["High"] - df_5m["Low"],
        (df_5m["High"] - df_5m["Close"].shift(1)).abs(),
        (df_5m["Low"] - df_5m["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_5m = float(tr_5m.ewm(alpha=1/7, adjust=False).mean().iloc[-1])

    # Swing levels từ 5m
    swing_h = float(df_5m["High"].iloc[-20:].max())
    swing_l = float(df_5m["Low"].iloc[-20:].min())

    if bias == "BUY":
        entry_low = max(swing_l, price - atr_5m * 0.5)
        entry_high = price
        sl = round(entry_low - atr_5m * 1.0, decimals)
        tp1 = round(price + atr_5m * 1.5, decimals)
        tp2 = round(price + atr_5m * 2.5, decimals)

        # Risk/Reward
        risk = price - sl
        rr1 = (tp1 - price) / risk if risk > 0 else 0
        rr2 = (tp2 - price) / risk if risk > 0 else 0
    else:
        entry_high = min(swing_h, price + atr_5m * 0.5)
        entry_low = price
        sl = round(entry_high + atr_5m * 1.0, decimals)
        tp1 = round(price - atr_5m * 1.5, decimals)
        tp2 = round(price - atr_5m * 2.5, decimals)

        risk = sl - price
        rr1 = (price - tp1) / risk if risk > 0 else 0
        rr2 = (price - tp2) / risk if risk > 0 else 0

    return {
        "entry_zone": f"{fmt_price(entry_low, decimals)} - {fmt_price(entry_high, decimals)}",
        "stop_loss": fmt_price(sl, decimals),
        "tp1": fmt_price(tp1, decimals),
        "tp2": fmt_price(tp2, decimals),
        "rr1": round(rr1, 1),
        "rr2": round(rr2, 1),
        "atr_5m": fmt_price(atr_5m, decimals),
    }


# ─── Build scalping report ────────────────────────────────────────────

def build_scalp_report(
    tf_data: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> str:
    """Tạo báo cáo scalping đầy đủ."""
    d = cfg["decimals"]
    name = cfg["display_name"]

    # Context bias từ 1H
    ctx_df = tf_data.get(CONTEXT_TIMEFRAME)
    if ctx_df is None or ctx_df.empty:
        return f"ERROR: Need {CONTEXT_TIMEFRAME} data for scalping context."

    from core import determine_trend
    ctx_trend, ctx_score = determine_trend(ctx_df)
    ctx_price = float(ctx_df["Close"].iloc[-1])

    # Scalp signals cho 15m và 5m
    sig_15m = scalp_signal_5m(tf_data.get("15m", pd.DataFrame()))
    sig_5m = scalp_signal_5m(tf_data.get("5m", pd.DataFrame()))

    # Tổng hợp bias
    scalp_score = sig_5m["score"] * 2 + sig_15m["score"] * 1.5
    if ctx_trend == "UP":
        scalp_score += 2
    elif ctx_trend == "DOWN":
        scalp_score -= 2

    # Check context contradiction: lower TFs disagree with 1H bias
    lower_tf_direction = "BUY" if (sig_5m["score"] * 2 + sig_15m["score"] * 1.5) >= 0 else "SELL"
    ctx_direction = "BUY" if ctx_trend == "UP" else "SELL" if ctx_trend == "DOWN" else None
    contradict = ctx_direction and lower_tf_direction != ctx_direction

    if scalp_score >= 3.5:
        scalp_bias = "⚠️ COUNTER-TREND BUY" if contradict else "STRONG BUY"
        bias_icon = "⚠️" if contradict else "🟢"
    elif scalp_score >= 1.5:
        scalp_bias = "⚠️ COUNTER-TREND BUY" if contradict else "BUY"
        bias_icon = "⚠️" if contradict else "🟢"
    elif scalp_score <= -3.5:
        scalp_bias = "⚠️ COUNTER-TREND SELL" if contradict else "STRONG SELL"
        bias_icon = "⚠️" if contradict else "🔴"
    elif scalp_score <= -1.5:
        scalp_bias = "⚠️ COUNTER-TREND SELL" if contradict else "SELL"
        bias_icon = "⚠️" if contradict else "🔴"
    else:
        scalp_bias = "WAIT"
        bias_icon = "🟡"

    # Entry zones
    df_15m = tf_data.get("15m", pd.DataFrame())
    df_5m = tf_data.get("5m", pd.DataFrame())

    if df_15m.empty or df_5m.empty:
        return "ERROR: Need 15m and 5m data for scalping."

    if "BUY" in scalp_bias:
        zones = scalp_entry_zones(df_15m, df_5m, "BUY", d)
    elif "SELL" in scalp_bias:
        zones = scalp_entry_zones(df_15m, df_5m, "SELL", d)
    else:
        zones = None

    # Build report
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"  ⚡ SCALPING ANALYSIS — {name}")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("=" * 64)
    lines.append("")

    # Context
    lines.append("【CONTEXT BIAS — 1H】")
    lines.append(f"  Trend: {ctx_trend} (score: {ctx_score:+d})")
    lines.append(f"  Price: {fmt_price(ctx_price, d)}")
    lines.append(f"  Ý nghĩa: {'Scalp theo hướng LONG' if ctx_trend == 'UP' else 'Scalp theo hướng SHORT' if ctx_trend == 'DOWN' else 'Chờ breakout rõ ràng'}")
    lines.append("")

    # 15m signal
    lines.append("【15M SIGNAL】")
    lines.append(f"  Trend: {sig_15m['trend']} | Score: {sig_15m['score']:+.1f}")
    lines.append(f"  RSI7: {sig_15m['rsi7']} | EMA5: {sig_15m['ema5']} | EMA9: {sig_15m['ema9']}")
    lines.append(f"  ATR: {sig_15m['atr']} ({sig_15m['atr_pct']:.3f}%) | RangePos: {sig_15m['range_pos_pct']:.0f}%")
    lines.append(f"  Details:")
    for detail in sig_15m['details']:
        lines.append(f"    - {detail}")
    lines.append("")

    # 5m signal
    lines.append("【5M SIGNAL — Primary】")
    lines.append(f"  Trend: {sig_5m['trend']} | Score: {sig_5m['score']:+.1f}")
    lines.append(f"  RSI7: {sig_5m['rsi7']} | EMA5: {sig_5m['ema5']} | EMA9: {sig_5m['ema9']}")
    lines.append(f"  ATR: {sig_5m['atr']} ({sig_5m['atr_pct']:.3f}%) | RangePos: {sig_5m['range_pos_pct']:.0f}%")
    lines.append(f"  Details:")
    for detail in sig_5m['details']:
        lines.append(f"    - {detail}")
    lines.append("")

    # Overall signal
    lines.append(f"【SCALPING SIGNAL: {bias_icon} {scalp_bias} (score: {scalp_score:+.1f})】")

    if contradict:
        lines.append("")
        lines.append(f"  ⚠️  CẢNH BÁO: Tín hiệu ngược xu hướng 1H ({ctx_trend})!")
        lines.append(f"  • Giảm 50% khối lượng nếu vẫn muốn vào lệnh")
        lines.append(f"  • TP1 gần hơn, không giữ lệnh qua đêm")

    if zones:
        lines.append("")
        lines.append("【ENTRY & EXIT ZONES】")
        lines.append(f"  Entry:    {zones['entry_zone']}")
        lines.append(f"  Stop Loss: {zones['stop_loss']}")
        lines.append(f"  TP1:      {zones['tp1']} (R:R = 1:{zones['rr1']})")
        lines.append(f"  TP2:      {zones['tp2']} (R:R = 1:{zones['rr2']})")
        lines.append(f"  ATR(5m):  {zones['atr_5m']}")
        lines.append("")
        lines.append("  ⚠️  Quy tắc Scalping:")
        lines.append("  • Vào lệnh trong vùng entry, không chase giá")
        lines.append("  • SL tuyệt đối không dời xa hơn ban đầu")
        lines.append("  • Đóng 50% tại TP1, dời SL về entry")
        lines.append("  • Thời gian giữ lệnh: 5-15 phút")
        lines.append("  • Max 3 lệnh/phiên, nếu thua 2 lệnh liên tiếp thì dừng")

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


# ─── Scalping summary (compact, for quick signal) ─────────────────────

def build_scalp_data_for_ai(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Build raw data dump for DeepSeek — scalping context."""
    d = cfg["decimals"]
    name = cfg["display_name"]
    lines: List[str] = []
    lines.append(f"=== SCALPING RAW DATA: {name} ===")
    lines.append("")

    # Context
    ctx_df = tf_data.get(CONTEXT_TIMEFRAME)
    if ctx_df is not None and not ctx_df.empty:
        from core import determine_trend
        ctx_trend, ctx_score = determine_trend(ctx_df)
        ctx_price = float(ctx_df["Close"].iloc[-1])
        lines.append(f"【CONTEXT — {CONTEXT_TIMEFRAME}】")
        lines.append(f"  Trend: {ctx_trend} (score: {ctx_score:+d})")
        lines.append(f"  Price: {fmt_price(ctx_price, d)}")
        lines.append("")

    for tf_name in ["15m", "5m"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        sig = scalp_signal_5m(df)
        lines.append(f"【{tf_name}】")
        lines.append(f"  O={fmt_price(last['Open'], d)} H={fmt_price(last['High'], d)} L={fmt_price(last['Low'], d)} C={fmt_price(last['Close'], d)}")
        lines.append(f"  EMA5: {sig['ema5']}  EMA9: {sig['ema9']}")
        lines.append(f"  RSI7: {sig['rsi7']}  ATR7: {sig['atr']} ({sig['atr_pct']:.4f}%)")
        lines.append(f"  Range Position: {sig['range_pos_pct']:.0f}%")
        for detail in sig["details"]:
            lines.append(f"  - {detail}")
        lines.append("")

    return "\n".join(lines)


def build_scalp_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Tóm tắt scalping ngắn gọn, dùng cho quick check."""
    d = cfg["decimals"]

    from core import determine_trend
    ctx_trend, _ = determine_trend(tf_data.get(CONTEXT_TIMEFRAME, pd.DataFrame()))

    sig_5m = scalp_signal_5m(tf_data.get("5m", pd.DataFrame()))
    sig_15m = scalp_signal_5m(tf_data.get("15m", pd.DataFrame()))

    scalp_score = sig_5m["score"] * 2 + sig_15m["score"] * 1.5
    if ctx_trend == "UP":
        scalp_score += 2
    elif ctx_trend == "DOWN":
        scalp_score -= 2

    # Check context contradiction
    lower_tf_direction = "BUY" if (sig_5m["score"] * 2 + sig_15m["score"] * 1.5) >= 0 else "SELL"
    ctx_direction = "BUY" if ctx_trend == "UP" else "SELL" if ctx_trend == "DOWN" else None
    contradict = ctx_direction and lower_tf_direction != ctx_direction
    prefix = "⚠️CT " if contradict else ""

    if scalp_score >= 1.5:
        bias = f"{prefix}BUY"
    elif scalp_score <= -1.5:
        bias = f"{prefix}SELL"
    else:
        bias = "WAIT"

    lines = [
        f"SCALP[{cfg['display_name']}] 1H:{ctx_trend} "
        f"15m:{sig_15m['trend']}({sig_15m['score']:+.1f}) "
        f"5m:{sig_5m['trend']}({sig_5m['score']:+.1f}) "
        f"→ {bias} ({scalp_score:+.1f}) "
        f"RSI7:{sig_5m['rsi7']} ATR:{sig_5m['atr']}"
    ]
    return "\n".join(lines)


# ─── Main function (called by trade_cli.py) ───────────────────────────

def run_scalp_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    """Entry point: lấy dữ liệu và tạo báo cáo scalping."""
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}. Choose: {', '.join(INSTRUMENTS)}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    tf_data = adjust_to_spot(tf_data, cfg)

    if not tf_data:
        return "No data. Check connection."

    # Kiểm tra có đủ TF thấp không
    for tf in ["1H", "15m", "5m"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Missing {tf} data — scalping needs 1H, 15m, 5m."

    report = build_scalp_report(tf_data, cfg)
    return report


def run_scalp_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    """Quick scalp signal."""
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    tf_data = adjust_to_spot(tf_data, cfg)

    if not tf_data:
        return "No data."

    summary = build_scalp_summary(tf_data, cfg)
    return summary
