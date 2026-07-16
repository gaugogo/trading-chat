#!/usr/bin/env python3
"""
XAUUSD SMC (Smart Money Concepts) Analysis Module
Market Structure, Order Blocks, Fair Value Gaps, Liquidity Sweeps

Standalone — does not depend on xauusd_analysis.py indicators.
Only needs OHLC DataFrame input.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

SWING_LOOKBACK: Dict[str, int] = {
    "Daily": 3,
    "4H": 4,
    "1H": 5,
    "15m": 5,
    "5m": 5,
}

TF_WEIGHTS_SMC: Dict[str, float] = {
    "Daily": 5.0,
    "4H": 3.0,
    "1H": 2.0,
    "15m": 1.0,
    "5m": 0.5,
}


def find_swing_points(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    high_rows = []
    low_rows = []
    n = len(df)
    for i in range(left, n - right):
        if all(df['High'].iloc[i] > df['High'].iloc[i - j] for j in range(1, left + 1)) and \
           all(df['High'].iloc[i] >= df['High'].iloc[i + j] for j in range(1, right + 1)):
            high_rows.append({'date': df.index[i], 'price': float(df['High'].iloc[i]), 'idx': i})
        if all(df['Low'].iloc[i] < df['Low'].iloc[i - j] for j in range(1, left + 1)) and \
           all(df['Low'].iloc[i] <= df['Low'].iloc[i + j] for j in range(1, right + 1)):
            low_rows.append({'date': df.index[i], 'price': float(df['Low'].iloc[i]), 'idx': i})
    sh = pd.DataFrame(high_rows) if high_rows else pd.DataFrame(columns=['date', 'price', 'idx'])
    sl = pd.DataFrame(low_rows) if low_rows else pd.DataFrame(columns=['date', 'price', 'idx'])
    return sh, sl


def detect_market_structure(
    swing_highs: pd.DataFrame,
    swing_lows: pd.DataFrame,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    swings: List[Dict] = []
    for _, r in swing_highs.iterrows():
        swings.append({'date': r['date'], 'price': float(r['price']), 'type': 'high', 'idx': int(r['idx'])})
    for _, r in swing_lows.iterrows():
        swings.append({'date': r['date'], 'price': float(r['price']), 'type': 'low', 'idx': int(r['idx'])})
    swings.sort(key=lambda x: x['date'])

    result: Dict[str, Any] = {'trend': 'SIDEWAYS', 'highs': [], 'lows': [], 'bos': [], 'choch': None}

    if len(swings) < 4:
        return result

    highs = [s for s in swings if s['type'] == 'high']
    lows = [s for s in swings if s['type'] == 'low']

    for i in range(1, len(highs)):
        highs[i]['label'] = 'HH' if highs[i]['price'] > highs[i - 1]['price'] else 'LH'
    if highs:
        highs[0]['label'] = 'H'

    for i in range(1, len(lows)):
        lows[i]['label'] = 'HL' if lows[i]['price'] > lows[i - 1]['price'] else 'LL'
    if lows:
        lows[0]['label'] = 'L'

    recent_h = [h.get('label', '') for h in highs[-3:]]
    recent_l = [l.get('label', '') for l in lows[-3:]]
    hh = recent_h.count('HH')
    lh = recent_h.count('LH')
    hl = recent_l.count('HL')
    ll = recent_l.count('LL')

    if hh >= lh and hl >= ll:
        trend = 'UP'
    elif lh >= hh and ll >= hl:
        trend = 'DOWN'
    else:
        trend = 'SIDEWAYS'

    last_close = float(df['Close'].iloc[-1])

    bos_list = []
    if highs and trend == 'UP' and last_close > highs[-1]['price']:
        bos_list.append({'type': 'bullish', 'level': highs[-1]['price'], 'date': df.index[-1]})
    if lows and trend == 'DOWN' and last_close < lows[-1]['price']:
        bos_list.append({'type': 'bearish', 'level': lows[-1]['price'], 'date': df.index[-1]})

    choch = None
    if len(highs) >= 2 and len(lows) >= 2:
        last_h_label = highs[-1].get('label', '')
        prev_h_label = highs[-2].get('label', '')
        last_l_label = lows[-1].get('label', '')
        prev_l_label = lows[-2].get('label', '')
        if prev_h_label == 'LH' and last_h_label == 'HH' and prev_l_label == 'LL' and last_l_label == 'HL':
            choch = {'direction': 'bullish', 'date': highs[-1]['date']}
        elif prev_h_label == 'HH' and last_h_label == 'LH' and prev_l_label == 'HL' and last_l_label == 'LL':
            choch = {'direction': 'bearish', 'date': highs[-1]['date']}

    result = {
        'trend': trend,
        'highs': highs,
        'lows': lows,
        'bos': bos_list,
        'choch': choch,
    }
    return result


def find_order_blocks(
    df: pd.DataFrame,
    min_impulse_atr: float = 1.5,
) -> List[Dict[str, Any]]:
    if len(df) < 30:
        return []

    body = (df['Close'] - df['Open']).abs()
    avg_body = body.rolling(20).mean().replace(0, np.nan)
    atr_val = (df['High'] - df['Low']).rolling(14).mean()

    obs: List[Dict] = []
    for i in range(5, len(df)):
        body_i = body.iloc[i]
        avg_i = avg_body.iloc[i]
        atr_i = atr_val.iloc[i]
        if pd.isna(avg_i) or pd.isna(atr_i) or avg_i <= 0:
            continue

        if df['Close'].iloc[i] > df['Open'].iloc[i] and body_i > avg_i * min_impulse_atr:
            for j in range(i - 1, max(i - 6, -1), -1):
                if df['Close'].iloc[j] < df['Open'].iloc[j]:
                    obs.append({
                        'type': 'bullish',
                        'top': float(df['High'].iloc[j]),
                        'bottom': float(df['Low'].iloc[j]),
                        'formed_at': df.index[j],
                        'impulse_idx': i,
                    })
                    break

        if df['Close'].iloc[i] < df['Open'].iloc[i] and body_i > avg_i * min_impulse_atr:
            for j in range(i - 1, max(i - 6, -1), -1):
                if df['Close'].iloc[j] > df['Open'].iloc[j]:
                    obs.append({
                        'type': 'bearish',
                        'top': float(df['High'].iloc[j]),
                        'bottom': float(df['Low'].iloc[j]),
                        'formed_at': df.index[j],
                        'impulse_idx': i,
                    })
                    break

    if not obs:
        return []

    last_price = float(df['Close'].iloc[-1])
    unique: List[Dict] = []
    for ob in reversed(obs):
        top, bot = ob['top'], ob['bottom']
        mitigated = False
        if ob['type'] == 'bullish':
            if last_price < bot - (top - bot) * 0.1:
                mitigated = True
        else:
            if last_price > top + (top - bot) * 0.1:
                mitigated = True
        if not mitigated:
            ob['mitigated'] = False
            unique.append(ob)
            if len(unique) >= 4:
                break

    return unique


# ─── ADVANCED SMC CONCEPTS ──────────────────────────────────────────

def find_breaker_blocks(
    df: pd.DataFrame,
    swing_highs: pd.DataFrame,
    swing_lows: pd.DataFrame,
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find Breaker Blocks — OB that failed and became new resistance/support.

    Breaker Block (bullish):
      - Old bearish OB
      - Price broke below it (failed)
      - Price came back and respected it as resistance → now flipped to support
      - Actually, a Breaker Block is formed when an OB is broken and then
        becomes the opposite type (bearish OB → broken → becomes bullish breaker)

    Breaker Block (bearish):
      - Old bullish OB
      - Price broke above it (failed)
      - Price came back and respected it as support → now flipped to resistance
    """
    if len(df) < 20 or swing_highs.empty or swing_lows.empty:
        return []

    last_price = float(df['Close'].iloc[-1])
    last_high = float(df['High'].iloc[-1])
    last_low = float(df['Low'].iloc[-1])
    breakers: List[Dict] = []

    trend = structure.get('trend', 'SIDEWAYS')

    # Find potential Breaker Blocks from order blocks that were mitigated
    obs = find_order_blocks(df, min_impulse_atr=1.2)

    for ob in obs:
        ob_top = ob['top']
        ob_bot = ob['bottom']
        ob_type = ob['type']

        if ob_type == 'bullish':
            # Bullish OB broken below → becomes bearish breaker
            broken_below = last_low < ob_bot
            if broken_below:
                # Check if price came back and respected OB as resistance
                back_to_ob = last_price >= ob_bot - (ob_top - ob_bot) * 0.3
                if back_to_ob:
                    breakers.append({
                        'type': 'bearish',
                        'top': ob_top,
                        'bottom': ob_bot,
                        'original_type': 'bullish_OB',
                        'formed_at': ob['formed_at'],
                        'strength': 'confirmed' if last_price < ob_bot else 'potential',
                    })
        else:
            # Bearish OB broken above → becomes bullish breaker
            broken_above = last_high > ob_top
            if broken_above:
                back_to_ob = last_price <= ob_top + (ob_top - ob_bot) * 0.3
                if back_to_ob:
                    breakers.append({
                        'type': 'bullish',
                        'top': ob_top,
                        'bottom': ob_bot,
                        'original_type': 'bearish_OB',
                        'formed_at': ob['formed_at'],
                        'strength': 'confirmed' if last_price > ob_top else 'potential',
                    })

    return breakers[:5]


def find_mitigation_blocks(
    df: pd.DataFrame,
    swing_highs: pd.DataFrame,
    swing_lows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Find Mitigation Blocks — FVGs that have been filled (mitigated).

    When a Fair Value Gap gets filled, it becomes a Mitigation Block.
    These act as support/resistance after being filled.
    """
    if len(df) < 20:
        return []

    fvgs = find_fair_value_gaps(df)
    mitigated_blocks: List[Dict] = []
    lookback = min(60, len(df))
    recent_df = df.iloc[-lookback:]

    for fvg in reversed(fvgs):
        fvg_type = fvg['type']
        gap_top = fvg['top']
        gap_bot = fvg['bottom']

        # Check if FVG was filled (mitigated) in recent price action
        if fvg_type == 'bullish':
            # Bullish FVG: gap goes up, filled when price drops back into it
            filled = recent_df['Low'].min() <= gap_top
        else:
            # Bearish FVG: gap goes down, filled when price rises back into it
            filled = recent_df['High'].max() >= gap_bot

        if filled:
            mitigated_blocks.append({
                'type': fvg_type,
                'top': gap_top,
                'bottom': gap_bot,
                'original_type': 'FVG',
                'formed_at': fvg['formed_at'],
                'mitigated': True,
            })

    return mitigated_blocks[:5]


def find_reclaimed_order_blocks(
    df: pd.DataFrame,
    swing_highs: pd.DataFrame,
    swing_lows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Find Reclaimed Order Blocks — OBs that were mitigated then reclaimed.

    A reclaimed OB is when price:
      1. Forms an OB (strong impulse away)
      2. Returns and mitigates the OB (breaks into it)
      3. Reclaims it (closes back outside)
    This is a strong signal that the OB is still valid.
    """
    if len(df) < 30:
        return []

    obs = find_order_blocks(df, min_impulse_atr=1.2)
    reclaimed: List[Dict] = []
    last_close = float(df['Close'].iloc[-1])

    for ob in obs:
        ob_top = ob['top']
        ob_bot = ob['bottom']
        ob_type = ob['type']

        # Check if price entered the OB (mitigation)
        entered_ob = False
        reclaimed_out = False

        # Look at recent candles for entry and exit
        recent = df.iloc[max(0, ob.get('impulse_idx', len(df)-20)):]
        for i in range(len(recent)):
            r_high = float(recent['High'].iloc[i])
            r_low = float(recent['Low'].iloc[i])
            r_close = float(recent['Close'].iloc[i])

            # Check if price entered OB
            if ob_bot <= r_high and r_low <= ob_top:
                entered_ob = True

            # Check if price reclaimed (closed outside)
            if entered_ob:
                if ob_type == 'bullish' and r_close > ob_top:
                    reclaimed_out = True
                    break
                elif ob_type == 'bearish' and r_close < ob_bot:
                    reclaimed_out = True
                    break

        if entered_ob and reclaimed_out:
            reclaimed.append({
                'type': ob_type,
                'top': ob_top,
                'bottom': ob_bot,
                'formed_at': ob['formed_at'],
                'strength': 'strong',
            })

    return reclaimed[:5]


def find_fair_value_gaps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    fvgs: List[Dict] = []
    for i in range(2, len(df)):
        lo_i = float(df['Low'].iloc[i])
        hi_i2 = float(df['High'].iloc[i - 2])
        hi_i = float(df['High'].iloc[i])
        lo_i2 = float(df['Low'].iloc[i - 2])

        if lo_i > hi_i2:
            gap_bot = hi_i2
            gap_top = lo_i
            if gap_top - gap_bot > 0:
                fvgs.append({
                    'type': 'bullish',
                    'top': gap_top,
                    'bottom': gap_bot,
                    'formed_at': df.index[i],
                })

        if hi_i < lo_i2:
            gap_bot = hi_i
            gap_top = lo_i2
            if gap_top - gap_bot > 0:
                fvgs.append({
                    'type': 'bearish',
                    'top': gap_top,
                    'bottom': gap_bot,
                    'formed_at': df.index[i],
                })

    if not fvgs:
        return []

    last_price = float(df['Close'].iloc[-1])
    unmitigated = []
    for fvg in reversed(fvgs):
        mitigated = False
        if fvg['type'] == 'bullish':
            if last_price < fvg['bottom']:
                mitigated = True
        else:
            if last_price > fvg['top']:
                mitigated = True
        if not mitigated:
            fvg['mitigated'] = False
            unmitigated.append(fvg)
            if len(unmitigated) >= 4:
                break

    return unmitigated


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: pd.DataFrame,
    swing_lows: pd.DataFrame,
    lookback: int = 30,
) -> List[Dict[str, Any]]:
    sweeps: List[Dict] = []
    if len(df) < lookback + 5:
        return sweeps

    recent_df = df.iloc[-lookback:]
    last_close = float(df['Close'].iloc[-1])

    for _, sh in swing_highs.iterrows():
        if sh['idx'] < len(df) - lookback:
            continue
        level = float(sh['price'])
        region_high = recent_df['High'].max()
        if region_high > level * 1.001 and last_close < level:
            sweeps.append({
                'type': 'bearish_sweep',
                'level': level,
                'swept_high': float(region_high),
                'date': recent_df['High'].idxmax(),
            })

    for _, sl in swing_lows.iterrows():
        if sl['idx'] < len(df) - lookback:
            continue
        level = float(sl['price'])
        region_low = recent_df['Low'].min()
        if region_low < level * 0.999 and last_close > level:
            sweeps.append({
                'type': 'bullish_sweep',
                'level': level,
                'swept_low': float(region_low),
                'date': recent_df['Low'].idxmin(),
            })

    return sweeps


def analyze_smc(df: pd.DataFrame, tf_name: str) -> Dict[str, Any]:
    if df.empty or len(df) < 20:
        return {'tf': tf_name, 'error': 'insufficient data'}

    left = SWING_LOOKBACK.get(tf_name, 3)
    right = SWING_LOOKBACK.get(tf_name, 3)

    sh, sl = find_swing_points(df, left, right)
    structure = detect_market_structure(sh, sl, df)
    obs = find_order_blocks(df)
    fvgs = find_fair_value_gaps(df)
    sweeps = detect_liquidity_sweeps(df, sh, sl)

    last_close = float(df['Close'].iloc[-1])
    last_high = float(df['High'].iloc[-1])
    last_low = float(df['Low'].iloc[-1])
    last_date = df.index[-1]

    return {
        'tf': tf_name,
        'candles': len(df),
        'last_price': last_close,
        'last_high': last_high,
        'last_low': last_low,
        'last_date': last_date,
        'structure': structure,
        'order_blocks': obs,
        'breaker_blocks': find_breaker_blocks(df, sh, sl, structure),
        'mitigation_blocks': find_mitigation_blocks(df, sh, sl),
        'reclaimed_obs': find_reclaimed_order_blocks(df, sh, sl),
        'fvgs': fvgs,
        'sweeps': sweeps,
    }


def format_smc_footer(all_results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  \U0001f4a1 SMC (SMART MONEY CONCEPTS) ANALYSIS")
    lines.append("=" * 72)

    for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
        if tf_name not in all_results:
            continue
        r = all_results[tf_name]
        if r.get('error'):
            continue

        lines.append(f"")
        lines.append(f"\u3010SMC \u2014 {tf_name}\u3011({r['candles']} candles)")
        lines.append(f"  Price: ${r['last_price']:.2f}")

        s = r['structure']
        trend = s['trend']
        trend_icon = "\U0001f7e2" if trend == 'UP' else "\U0001f534" if trend == 'DOWN' else "\U0001f7e1"
        lines.append(f"  Market Structure: {trend_icon} {trend}")

        if s['highs']:
            last_h = s['highs'][-1]
            lines.append(f"  Last Swing High: ${last_h['price']:.2f} ({last_h.get('label', '')}) @ {_fmt_date(last_h['date'])}")
        if s['lows']:
            last_l = s['lows'][-1]
            lines.append(f"  Last Swing Low: ${last_l['price']:.2f} ({last_l.get('label', '')}) @ {_fmt_date(last_l['date'])}")

        if s['bos']:
            for b in s['bos']:
                icon = "\U0001f7e2" if b['type'] == 'bullish' else "\U0001f534"
                lines.append(f"  BOS: {icon} {b['type'].title()} structure break @ ${b['level']:.2f}")
        else:
            lines.append(f"  BOS: None (no recent structure break)")

        if s['choch']:
            icon = "\U0001f7e2" if s['choch']['direction'] == 'bullish' else "\U0001f534"
            lines.append(f"  CHoCH: {icon} {s['choch']['direction'].title()} \u2014 trend change @ {_fmt_date(s['choch']['date'])}")
        else:
            lines.append(f"  CHoCH: None (no trend change detected)")

        lines.append(f"  Order Blocks:")
        if r['order_blocks']:
            for ob in r['order_blocks'][:3]:
                icon = "\U0001f7e2" if ob['type'] == 'bullish' else "\U0001f534"
                lines.append(f"    {icon} {ob['type'].title()}: ${ob['bottom']:.2f}\u2013${ob['top']:.2f}")
        else:
            lines.append(f"    None detected")

        lines.append(f"  Fair Value Gaps:")
        if r['fvgs']:
            for fvg in r['fvgs'][:3]:
                icon = "\U0001f7e2" if fvg['type'] == 'bullish' else "\U0001f534"
                lines.append(f"    {icon} {fvg['type'].title()}: ${fvg['bottom']:.2f}\u2013${fvg['top']:.2f}")
        else:
            lines.append(f"    None detected")

        # Advanced SMC
        lines.append(f"  Breaker Blocks:")
        if r.get('breaker_blocks'):
            for bb in r['breaker_blocks'][:3]:
                icon = "\U0001f7e2" if bb['type'] == 'bullish' else "\U0001f534"
                lines.append(f"    {icon} {bb['type'].title()}: ${bb['bottom']:.2f}–${bb['top']:.2f} ({bb['strength']})")
        else:
            lines.append(f"    None detected")

        lines.append(f"  Mitigation Blocks (filled FVGs):")
        if r.get('mitigation_blocks'):
            for mb in r['mitigation_blocks'][:3]:
                icon = "\U0001f7e2" if mb['type'] == 'bullish' else "\U0001f534"
                lines.append(f"    {icon} {mb['type'].title()}: ${mb['bottom']:.2f}–${mb['top']:.2f}")
        else:
            lines.append(f"    None detected")

        lines.append(f"  Reclaimed OBs:")
        if r.get('reclaimed_obs'):
            for rob in r['reclaimed_obs'][:3]:
                icon = "\U0001f7e2" if rob['type'] == 'bullish' else "\U0001f534"
                lines.append(f"    {icon} {rob['type'].title()}: ${rob['bottom']:.2f}–${rob['top']:.2f} (reclaimed ✅)")
        else:
            lines.append(f"    None detected")

        lines.append(f"  Liquidity Sweeps:")
        if r['sweeps']:
            for sw in r['sweeps']:
                if sw['type'] == 'bearish_sweep':
                    lines.append(f"    \U0001f534 Sell-side sweep above ${sw['level']:.2f}")
                else:
                    lines.append(f"    \U0001f7e2 Buy-side sweep below ${sw['level']:.2f}")
        else:
            lines.append(f"    None detected")

    lines.append("")
    lines.append("\u3010SMC SIGNAL\u3011")
    signal, signal_icon = _smc_signal(all_results)
    lines.append(f"  {signal_icon} {signal}")
    lines.append("")

    return "\n".join(lines)


def _smc_signal(all_results: Dict[str, Any]) -> Tuple[str, str]:
    score = 0.0
    details: List[str] = []

    for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
        if tf_name not in all_results or all_results[tf_name].get('error'):
            continue
        r = all_results[tf_name]
        w = TF_WEIGHTS_SMC.get(tf_name, 1)
        s = r['structure']
        trend = s['trend']

        tf_score = 0.0

        if trend == 'UP':
            tf_score += 2 * w
        elif trend == 'DOWN':
            tf_score -= 2 * w

        if s['choch']:
            tf_score += (1.5 * w) if s['choch']['direction'] == 'bullish' else (-1.5 * w)

        for b in s['bos']:
            tf_score += (0.5 * w) if b['type'] == 'bullish' else (-0.5 * w)

        bullish_obs = sum(1 for ob in r['order_blocks'] if ob['type'] == 'bullish')
        bearish_obs = sum(1 for ob in r['order_blocks'] if ob['type'] == 'bearish')
        tf_score += (0.5 * w) * (bullish_obs - bearish_obs)

        bullish_fvg = sum(1 for f in r['fvgs'] if f['type'] == 'bullish')
        bearish_fvg = sum(1 for f in r['fvgs'] if f['type'] == 'bearish')
        tf_score += (0.5 * w) * (bullish_fvg - bearish_fvg)

        # Advanced SMC concepts
        bullish_breakers = sum(1 for bb in r.get('breaker_blocks', []) if bb['type'] == 'bullish')
        bearish_breakers = sum(1 for bb in r.get('breaker_blocks', []) if bb['type'] == 'bearish')
        tf_score += (0.4 * w) * (bullish_breakers - bearish_breakers)

        bullish_reclaimed = sum(1 for rob in r.get('reclaimed_obs', []) if rob['type'] == 'bullish')
        bearish_reclaimed = sum(1 for rob in r.get('reclaimed_obs', []) if rob['type'] == 'bearish')
        tf_score += (0.6 * w) * (bullish_reclaimed - bearish_reclaimed)

        # Mitigation blocks: filled FVGs = decreased momentum
        mitigated = sum(1 for mb in r.get('mitigation_blocks', []))
        tf_score -= (0.1 * w) * mitigated

        bearish_sweeps = sum(1 for sw in r['sweeps'] if sw['type'] == 'bearish_sweep')
        bullish_sweeps = sum(1 for sw in r['sweeps'] if sw['type'] == 'bullish_sweep')
        tf_score += (0.3 * w) * (bullish_sweeps - bearish_sweeps)

        score += tf_score
        dir_str = "\U0001f7e2" if tf_score >= 0 else "\U0001f534"
        details.append(f"  {tf_name}: {trend} (score: {tf_score:+.1f}) {dir_str}")

    if score >= 8:
        return (
            f"STRONG BUY — {score:+.1f}\n"
            f"  SMC structure bullish across multiple timeframes. Look for buy on OB/FVG pullbacks.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f7e2",
        )
    elif score >= 3:
        return (
            f"BUY BIAS — {score:+.1f}\n"
            f"  SMC structure leaning bullish. Trade with the higher-TF structure.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f7e2",
        )
    elif score <= -8:
        return (
            f"STRONG SELL — {score:+.1f}\n"
            f"  SMC structure bearish across multiple timeframes. Look for sell on OB/FVG bounces.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f534",
        )
    elif score <= -3:
        return (
            f"SELL BIAS — {score:+.1f}\n"
            f"  SMC structure leaning bearish. Trade with the higher-TF structure.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f534",
        )
    elif score >= 1:
        return (
            f"CAUTIOUS BUY — {score:+.1f}\n"
            f"  Mixed but bullish SMC bias. Wait for clearer confirmation.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f7e1",
        )
    elif score <= -1:
        return (
            f"CAUTIOUS SELL — {score:+.1f}\n"
            f"  Mixed but bearish SMC bias. Wait for clearer confirmation.\n"
            f"Details:\n" + "\n".join(details),
            "\U0001f7e1",
        )
    else:
        return (
            f"NEUTRAL — {score:+.1f}\n"
            f"  SMC structure conflicting. No clear directional edge.\n"
            f"Details:\n" + "\n".join(details),
            "\u23f8\ufe0f",
        )


def smc_signal_compact(all_results: Dict[str, Any]) -> str:
    signal, icon = _smc_signal(all_results)
    return f"SMC Signal: {icon} {signal}"


def _fmt_date(d) -> str:
    if hasattr(d, 'strftime'):
        return d.strftime('%b %d %H:%M')
    return str(d)


def analyze_all_smc(tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
        if tf_name not in tf_data or tf_data[tf_name].empty:
            continue
        results[tf_name] = analyze_smc(tf_data[tf_name], tf_name)
    return results
