#!/usr/bin/env python3
"""
Trade CLI — data fetcher for Pi Agent.
4 phong cách giao dịch: Position · Swing · Day Trade · Scalping
+ Ichimoku Kinko Hyo + SMC + AI Analysis

Usage:
  python trade_cli.py data xau               # raw data
  python trade_cli.py analyze xau            # full AI analysis
  python trade_cli.py signal xau             # multi-TF signal
  python trade_cli.py smc xau                # SMC analysis
  python trade_cli.py chat xau "question"    # chat with DeepSeek
  python trade_cli.py position xau           # position trade analysis
  python trade_cli.py position_signal xau    # quick position signal
  python trade_cli.py swing xau              # swing trade analysis
  python trade_cli.py swing_signal xau       # quick swing signal
  python trade_cli.py daytrade xau           # day trade analysis
  python trade_cli.py daytrade_signal xau    # quick day trade signal
  python trade_cli.py scalp xau              # scalping analysis
  python trade_cli.py scalp_signal xau       # quick scalping signal
  python trade_cli.py ichimoku xau           # ichimoku analysis
  python trade_cli.py ichimoku_signal xau    # quick ichimoku signal
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import fetch_data
from position import run_position_analysis, run_position_signal
from swing import run_swing_analysis, run_swing_signal
from daytrade import run_daytrade_analysis, run_daytrade_signal
from scalp import run_scalp_analysis, run_scalp_signal
from ichimoku import run_ichimoku_analysis, run_ichimoku_signal
from core import (
    fetch_all_timeframes,
    determine_trend,
    build_confluence_summary,
    fmt_price,
    TF_ORDER,
    TF_WEIGHTS,
)
from instruments import INSTRUMENTS
from regime import detect_regime, regime_recommendation
from divergence import analyze_all_divergences
from risk_calculator import RiskCalculator, quick_risk_calc
from volume_profile import analyze_all_timeframes as analyze_vp_all, FullVPReport


def run_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    """Run full multi-timeframe analysis (same as analysis.py main)."""
    from analysis import format_report
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."
    return format_report(tf_data, cfg)


def run_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    """Generate unified multi-TF signal."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."

    trends = {}
    for tf in TF_ORDER:
        df = tf_data.get(tf)
        if df is not None and not df.empty:
            trend, score = determine_trend(df)
            trends[tf] = (trend, score)

    lines = [f"【SIGNAL】{cfg['display_name']}"]
    lines.append(f"{'TF':<8} {'Trend':<10} {'Score':<8}")
    lines.append("-" * 30)
    weighted = 0.0
    for tf in TF_ORDER:
        if tf in trends:
            t, s = trends[tf]
            w = TF_WEIGHTS.get(tf, 1)
            weighted += (1 if t == "UP" else -1 if t == "DOWN" else 0) * w
            lines.append(f"{tf:<8} {t:<10} {s:+d}")
    lines.append(f"\nWeighted: {weighted:+.1f}")

    if weighted >= 5:
        lines.append("→ 🟢 STRONG BUY")
    elif weighted >= 2:
        lines.append("→ 🟢 BUY BIAS")
    elif weighted <= -5:
        lines.append("→ 🔴 STRONG SELL")
    elif weighted <= -2:
        lines.append("→ 🔴 SELL BIAS")
    else:
        lines.append("→ 🟡 WAIT")

    lines.append(f"\nConfluence: {build_confluence_summary(tf_data)}")
    return "\n".join(lines)


def run_smc(instrument: str = "xau", no_cache: bool = False) -> str:
    """Run SMC analysis on all timeframes."""
    from smc import analyze_all_smc, format_smc_footer
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."
    smc_data = analyze_all_smc(tf_data)
    return format_smc_footer(smc_data)


def run_regime(instrument: str = "xau", no_cache: bool = False) -> str:
    """Run market regime detection."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."
    result = detect_regime(tf_data)
    report = result.detailed_report()
    # Add recommendations
    recs = regime_recommendation(result.regime)
    report += "\n" + "\n".join(recs)
    return report


def run_divergence(instrument: str = "xau", no_cache: bool = False) -> str:
    """Run RSI + MACD divergence analysis."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."
    report = analyze_all_divergences(tf_data)
    return report.summary()


def run_risk(instrument: str = "xau", account: float = 10000.0,
             risk_pct: float = 1.0, entry: float = 0.0,
             sl: float = 0.0, tp: float = 0.0) -> str:
    """Run risk calculator."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"

    # If no entry/SL given, use current price from data
    if entry == 0.0 or sl == 0.0:
        from core import fetch_all_timeframes
        from core import calculate_indicators
        tf_data = fetch_all_timeframes(cfg, use_cache=True)
        df = tf_data.get("1H") or tf_data.get("4H") or tf_data.get("Daily")
        if df is not None and not df.empty:
            last = df.iloc[-1]
            if entry == 0.0:
                entry = float(last['Close'])
            if sl == 0.0:
                atr = float(last.get('ATR', 10.0))
                sl = entry - atr * 2.0 if entry > 0 else 0.0
            if tp == 0.0:
                atr = float(last.get('ATR', 10.0))
                tp = entry + atr * 4.0 if entry > 0 else 0.0

    tp_opt = tp if tp > 0 else None
    return quick_risk_calc(
        account_size=account,
        risk_percent=risk_pct,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp_opt,
        instrument=instrument,
    )


def run_vp(instrument: str = "xau", no_cache: bool = False) -> str:
    """Run Volume Profile analysis."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."
    report = analyze_vp_all(tf_data)
    return report.summary()


def run_chat(instrument: str = "xau", question: str = "", no_cache: bool = False) -> str:
    """Chat with DeepSeek AI about current market data."""
    from analysis import format_report, call_deepseek
    from instruments import INSTRUMENTS

    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("DEEPSEEK_API_KEY=") and not DEEPSEEK_API_KEY:
                    DEEPSEEK_API_KEY = _line.split("=", 1)[1].strip().strip("\"'")

    if not DEEPSEEK_API_KEY:
        return "❌ DEEPSEEK_API_KEY not set. Add to .env file."

    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"

    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    if not tf_data:
        return "No data."

    report = format_report(tf_data, cfg)
    default_q = "Give me a detailed trading plan with entry, SL, TP, and R:R."
    q = question.strip() or default_q

    result = call_deepseek(report, q, DEEPSEEK_API_KEY, cfg)
    return result or "❌ DeepSeek API returned no response."


# ─── NEW MODES ─────────────────────────────────────────────────────────

def run_fundamental(instrument: str = "xau") -> str:
    """Run fundamental analysis: DXY, bond yields, economic calendar."""
    from fundamental import fundamental_report
    return fundamental_report(instrument)


def run_live(instrument: str = "") -> str:
    """Get live market prices."""
    from stream import get_live_report
    if instrument:
        return get_live_report(instruments=[instrument])
    return get_live_report()


def run_chart(
    instrument: str = "xau",
    timeframe: str = "Daily",
) -> str:
    """Generate a technical chart."""
    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"
    tf_data = fetch_all_timeframes(cfg, use_cache=True)
    if not tf_data:
        return "No data."

    from chart_generator import generate_chart, ChartConfig
    path = generate_chart(tf_data, instrument, timeframe)
    if path:
        return f"✅ Chart saved: {path}\n📈 Open the file to view {timeframe} chart for {cfg['display_name']}."
    return f"❌ Could not generate {timeframe} chart."


def run_learn(
    instrument: str = "xau",
    level: str = "intermediate",
    question: str = "",
) -> str:
    """Educational analysis with DeepSeek + Pi Agent follow-up."""
    from analysis import format_report, call_deepseek

    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("DEEPSEEK_API_KEY=") and not DEEPSEEK_API_KEY:
                    DEEPSEEK_API_KEY = _line.split("=", 1)[1].strip().strip("\"'")

    cfg = INSTRUMENTS.get(instrument)
    if not cfg:
        return f"Unknown: {instrument}"

    tf_data = fetch_all_timeframes(cfg, use_cache=True)
    if not tf_data:
        return "No data."

    report = format_report(tf_data, cfg)

    from education.prompt_templates import build_educational_prompt, build_user_prompt, MARKDOWN_EDUCATION_SECTION
    from education.conversation_flow import get_learning_summary

    # Generate lessons based on strategy
    strategy = "swing"  # Default strategy for learning
    from education.prompt_templates import generate_education_lessons
    education = generate_education_lessons(strategy, "WAIT")

    lines = [f"# 📚 {cfg['display_name']} — Educational Analysis ({level})", ""]

    if DEEPSEEK_API_KEY:
        lines.append("## 🤖 DeepSeek Analysis")
        lines.append("")
        user_prompt = build_user_prompt(report, cfg, style="educational")
        result = call_deepseek(report, question or "Give me a detailed educational analysis.", DEEPSEEK_API_KEY, cfg)
        if result:
            lines.append(result)
            lines.append("")
        else:
            lines.append("❌ DeepSeek API returned no response.")
            lines.append("")

    # Add educational content
    lines.append("## 📖 Educational Content")
    lines.append("")
    lines.append(education["lessons"])
    lines.append("")
    lines.append("### ⚠️ Common Mistakes")
    lines.append("")
    lines.append(education["mistakes"])
    lines.append("")
    lines.append("### 📚 Technical Knowledge")
    lines.append("")
    lines.append(education["technical_knowledge"])
    lines.append("")

    # Add learning summary
    lines.append(get_learning_summary(level))

    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trade_cli.py <mode> [instrument] [options]")
        print("Modes: data, analyze, signal, smc, chat,")
        print("       regime, divergence, risk, volume_profile,")
        print("       position, position_signal, swing, swing_signal,")
        print("       daytrade, daytrade_signal, scalp, scalp_signal,")
        print("       ichimoku, ichimoku_signal,")
        print("       fundamental, live, chart, learn")
        print("")
        print("risk options: account=10000 risk=1.0 entry=2650 sl=2640 tp=2670")
        print("chart options: instrument timeframe (e.g., xau Daily)")
        print("learn options: instrument level (e.g., xau intermediate)")
        sys.exit(1)

    mode = sys.argv[1]
    instrument = sys.argv[2] if len(sys.argv) > 2 else "xau"
    no_cache = "--no-cache" in sys.argv

    modes = {
        "data":              lambda: fetch_data(instrument, no_cache=no_cache),
        "analyze":           lambda: run_analysis(instrument, no_cache=no_cache),
        "signal":            lambda: run_signal(instrument, no_cache=no_cache),
        "smc":               lambda: run_smc(instrument, no_cache=no_cache),
        "regime":            lambda: run_regime(instrument, no_cache=no_cache),
        "volume_profile":    lambda: run_vp(instrument, no_cache=no_cache),
        "divergence":        lambda: run_divergence(instrument, no_cache=no_cache),
        "risk":              lambda: run_risk(instrument, no_cache=no_cache),
        "position":          lambda: run_position_analysis(instrument, no_cache=no_cache),
        "position_signal":   lambda: run_position_signal(instrument, no_cache=no_cache),
        "swing":             lambda: run_swing_analysis(instrument, no_cache=no_cache),
        "swing_signal":      lambda: run_swing_signal(instrument, no_cache=no_cache),
        "daytrade":          lambda: run_daytrade_analysis(instrument, no_cache=no_cache),
        "daytrade_signal":   lambda: run_daytrade_signal(instrument, no_cache=no_cache),
        "scalp":             lambda: run_scalp_analysis(instrument, no_cache=no_cache),
        "scalp_signal":      lambda: run_scalp_signal(instrument, no_cache=no_cache),
        "ichimoku":          lambda: run_ichimoku_analysis(instrument, no_cache=no_cache),
        "ichimoku_signal":   lambda: run_ichimoku_signal(instrument, no_cache=no_cache),
        "fundamental":       lambda: run_fundamental(instrument),
        "live":              lambda: run_live(instrument),
        "learn":             lambda: run_learn(instrument),
    }

    # Chart mode needs special parsing
    if mode == "chart":
        instr = sys.argv[2] if len(sys.argv) > 2 else "xau"
        tf = sys.argv[3] if len(sys.argv) > 3 else "Daily"
        print(run_chart(instr, tf))
        sys.exit(0)

    # Parse risk params
    if mode == "risk":
        # Parse key=value pairs from remaining args
        params = {"instrument": instrument, "account": 10000.0,
                   "risk_pct": 1.0, "entry": 0.0, "sl": 0.0, "tp": 0.0}
        for arg in sys.argv[2:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                if k in params:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        pass
        result = run_risk(**params)
        print(result)
        sys.exit(0)

    # Chat mode needs extra arg
    if mode == "chat":
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        instrument = "xau"
        no_cache = "--no-cache" in sys.argv
        question = question.replace("--no-cache", "").strip()
        result = run_chat(instrument, question, no_cache=no_cache)
        print(result)
        sys.exit(0)

    if mode in modes:
        result = modes[mode]()
    else:
        result = f"Unknown mode: {mode}. Use: {', '.join(modes.keys())}"

    print(result)
