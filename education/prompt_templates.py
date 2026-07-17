"""
prompt_templates.py — Structured Educational DeepSeek Prompt Templates

Cải thiện chất lượng đầu ra của AI:
  1. Buộc AI giải thích "TẠI SAO" cho mỗi quyết định
  2. Checklist học tập cho user
  3. Standard Markdown report format với sections:
     - "Bài học hôm nay"
     - "Sai lầm cần tránh"
     - "Kiến thức kỹ thuật"

Usage:
  from education.prompt_templates import build_educational_prompt
  prompt = build_educational_prompt(cfg, report_text, user_question)
"""

from typing import Dict, Any, Optional


def build_system_prompt(cfg: Dict[str, Any]) -> str:
    """Build compact system prompt — raw data, let AI do the thinking."""
    return (
        f"Bạn là chuyên gia phân tích kỹ thuật {cfg.get('prompt_instrument', '')} cho trader Việt Nam. "
        "Bạn nhận DỮ LIỆU THÔ (giá, indicators, levels) và tự phân tích toàn bộ.\n"
        "Trả lời BẰNG TIẾNG VIỆT, dùng số liệu cụ thể, không nói chung chung.\n"
        "Cấu trúc:\n"
        "1. Tóm tắt bias + 3 lý do chính (kèm số liệu)\n"
        "2. Entry zone, SL, TP, R:R cụ thể\n"
        "3. Rủi ro & cảnh báo\n"
        "4. Bài học (1-2 câu)\n"
        "Ngắn gọn, dưới 1000 từ."
    )


def build_user_prompt(
    report_text: str,
    cfg: Dict[str, Any],
    user_question: Optional[str] = None,
    style: str = "educational",
) -> str:
    if user_question:
        return f"Dữ liệu {cfg['prompt_instrument']}:\n\n{report_text}\n\nCâu hỏi: {user_question}"

    style_map = {
        "beginner": "Giải thích đơn giản như cho người mới.",
        "quick": "Chỉ cần: Bias, Entry/SL/TP, R:R, 1 bài học.",
    }
    extra = style_map.get(style, "Phân tích: bias, entry/SL/TP, rủi ro, bài học.")
    return f"Dữ liệu {cfg['prompt_instrument']}:\n\n{report_text}\n\n{extra}"


def build_deepseek_payload(
    system_prompt: str,
    user_prompt: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full payload for DeepSeek API.

    Args:
        system_prompt: System message
        user_prompt: User message
        cfg: Instrument config (for model selection)

    Returns:
        Dict ready to send to DeepSeek API
    """
    payload = {
        "model": cfg.get("deepseek_model", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False,
    }

    if cfg.get("deepseek_thinking", False):
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "high"

    return payload


# ─── STANDARD MARKDOWN REPORT TEMPLATES ───

MARKDOWN_REPORT_HEADER = """# 📊 {instrument} — {strategy} Analysis
**Generated:** {timestamp} UTC
**Style:** {style_description}
---

"""

MARKDOWN_BIAS_SECTION = """## 🎯 Signal: {bias_icon} {bias}

| Metric | Value |
|--------|-------|
| Weighted Score | {score:+.1f} |
| Risk/Trade | {risk_pct} |
| Min R:R | 1:{min_rr} |

"""

MARKDOWN_ENTRY_SECTION = """## 📋 Trade Plan

| Component | Level | R:R |
|-----------|-------|-----|
| **Entry Zone** | {entry_zone} | — |
| **Stop Loss** | {stop_loss} | — |
| **Take Profit 1** | {tp1} | 1:{rr1} |
| **Take Profit 2** | {tp2} | 1:{rr2} |
| **Take Profit 3** | {tp3} | 1:{rr3} |

### Key Levels
- Support: {support_levels}
- Resistance: {resistance_levels}

"""

MARKDOWN_EDUCATION_SECTION = """## 🧠 Bài Học Hôm Nay

{lessons}

## ⚠️ Sai Lầm Cần Tránh

{mistakes}

## 📚 Kiến Thức Kỹ Thuật

{technical_knowledge}

---

## ✅ Checklist Tự Kiểm Tra

- [ ] Đã xác nhận trend Daily?
- [ ] 4H structure có ủng hộ?
- [ ] RSI không quá overbought/oversold?
- [ ] MACD divergence?
- [ ] Volume xác nhận?
- [ ] SL đặt đúng kỹ thuật?
- [ ] R:R >= 1:2?
- [ ] Tránh tin tức quan trọng?
"""


def generate_education_lessons(strategy: str, bias: str) -> Dict[str, str]:
    """Generate educational content based on strategy and bias.

    Args:
        strategy: Trading style (position, swing, daytrade, scalp, ichimoku)
        bias: Signal bias (BUY, SELL, WAIT)

    Returns:
        Dict with 'lessons', 'mistakes', 'technical_knowledge' keys
    """
    lessons_map = {
        "position": {
            "default": (
                "1. **Xu hướng là bạn**: Position trading chỉ nên trade cùng chiều Daily trend. "
                "Không bao giờ chống lại SMA200 trên Daily.\n\n"
                "2. **Kiên nhẫn trả lời**: Vào lệnh scale-in, không FOMO. "
                "Nếu giá chạy xa mà không pullback, chờ cơ hội khác.\n\n"
                "3. **Đa dạng hóa**: Không đặt toàn bộ vốn vào 1 lệnh position. "
                "2-3 lệnh với các instrument khác nhau giúp giảm rủi ro."
            ),
        },
        "swing": {
            "default": (
                "1. **Fibonacci là công cụ, không phải thước thần**: "
                "Fib 0.618 là vùng retrace phổ biến nhất, nhưng không phải lúc nào cũng chạm.\n\n"
                "2. **Chờ xác nhận**: Không vào lệnh chỉ vì giá chạm vùng hỗ trợ. "
                "Chờ nến xác nhận (engulfing, pinbar, hammer).\n\n"
                "3. **Risk management**: Swing trade có thể chạm SL nhiều lần. "
                "Đảm bảo R:R tối thiểu 1:2 để bền vững."
            ),
        },
        "daytrade": {
            "default": (
                "1. **VWAP là thước đo định giá intraday**: Giá trên VWAP = bullish bias, "
                "dưới VWAP = bearish bias trong phiên.\n\n"
                "2. **Volume là chìa khóa**: Breakout không kèm volume spike thường là fakeout. "
                "Luôn kiểm tra volume trước khi vào lệnh.\n\n"
                "3. **Không giữ lệnh qua đêm**: Day trading = đóng vị thế trước EOD. "
                "Gap overnight có thể phá hủy mọi phân tích."
            ),
        },
        "scalp": {
            "default": (
                "1. **ATR càng nhỏ, spread càng quan trọng**: Với ATR nhỏ, "
                "spread có thể ăn mất 20-30% lợi nhuận. Chọn broker có spread thấp.\n\n"
                "2. **Không scalp ngược trend Daily**: Dù là scalping 5m, "
                "trade cùng chiều Daily vẫn cho tỷ lệ thắng cao hơn.\n\n"
                "3. **Dừng đúng lúc**: Nếu thua 2 lệnh liên tiếp, dừng lại. "
                "Scalping dễ bị revenge trading."
            ),
        },
        "ichimoku": {
            "default": (
                "1. **Kumo là vùng hỗ trợ/kháng cự động**: Giá trong mây = sideway. "
                "Chờ breakout hẳn khỏi mây rồi mới trade.\n\n"
                "2. **TK Cross cần Chikou xác nhận**: Tenkan cắt Kijun thôi chưa đủ. "
                "Chikou phải nằm trên/dưới giá 26 kỳ trước đó.\n\n"
                "3. **Kumo Twist = trend reversal tiềm năng**: Khi Senkou A cắt Senkou B, "
                "mây đổi màu — đây là tín hiệu mạnh nhưng hiếm."
            ),
        },
    }

    mistakes_map = {
        "position": {
            "default": (
                "❌ **Vào lệnh khi RSI > 70 hoặc < 30**: Position trade cần entry ở vùng giá trị, "
                "không phải vùng quá mua/quá bán.\n\n"
                "❌ **Đặt SL quá hẹp**: Position trade có biến động lớn. "
                "SL 3x ATR là tối thiểu.\n\n"
                "❌ **Kiểm tra lệnh mỗi ngày**: Position trade = nhìn xa. "
                "Check hàng ngày chỉ làm bạn panic."
            ),
        },
        "swing": {
            "default": (
                "❌ **Vào lệnh ngay khi thấy setup**: Luôn chờ pullback/retest. "
                "Chase giá là nguyên nhân số 1 thua lỗ swing.\n\n"
                "❌ **Dời SL về breakeven quá sớm**: Đợi ít nhất chạm TP1 trước, "
                "hoặc giá đi được 1 ATR.\n\n"
                "❌ **Trade nhiều cặp cùng lúc**: Tập trung 1-2 cặp, "
                "theo dõi sâu thay vì ôm đồm."
            ),
        },
        "daytrade": {
            "default": (
                "❌ **Giao dịch 30 phút trước tin tức**: Thị trường thường đi ngang "
                "hoặc biến động bất thường. Tránh bằng mọi giá.\n\n"
                "❌ **Dời SL rộng hơn khi đang thua**: Kỷ luật là tất cả. "
                "Nếu SL chạm, thoát ra và xem lại.\n\n"
                "❌ **Overtrading**: Quy tắc 3 lệnh/ngày. Hết 3 lệnh là dừng."
            ),
        },
        "scalp": {
            "default": (
                "❌ **Giữ lệnh quá lâu**: Scalping = 5-15 phút. Giữ lâu biến thành swing, "
                "phá hỏng kỳ vọng R:R.\n\n"
                "❌ **Không có SL**: Scalping không SL là tự sát. "
                "Spread + phí + biến động bất ngờ = cháy tài khoản.\n\n"
                "❌ **Scalp khi thị trường ít thanh khoản**: Tránh giờ nghỉ trưa, "
                "chờ London/NY Open."
            ),
        },
        "ichimoku": {
            "default": (
                "❌ **Trade trong mây (Kumo)**: Đây là vùng sideway. "
                "Chờ breakout hẳn về 1 phía.\n\n"
                "❌ **Chỉ dùng 1 TF Ichimoku**: Cần ít nhất 2 TF để xác nhận. "
                "Daily cho trend, 4H cho entry.\n\n"
                "❌ **Bỏ qua Chikou Span**: TK Cross mà Chikou không xác nhận = tín hiệu yếu."
            ),
        },
    }

    technical_map = {
        "position": (
            "**SMA200** — Đường trung bình 200 kỳ, được coi là 'đường sống' của thị trường. "
            "Giá trên SMA200 = bull market, dưới SMA200 = bear market.\n\n"
            "**Golden Cross / Death Cross** — SMA50 cắt lên (golden) hoặc cắt xuống (death) SMA200. "
            "Tín hiệu vĩ mô, không dùng cho entry ngắn hạn.\n\n"
            "**HH/HL Structure** — Higher High + Higher Low = uptrend. "
            "LH/LL = downtrend. Đây là kiến thức cơ bản nhất của price action."
        ),
        "swing": (
            "**Fibonacci Retracement** — Công cụ đo lường mức pullback trong xu hướng. "
            "0.618 là vùng retrace phổ biến nhất trong swing trading.\n\n"
            "**R:R (Risk:Reward)** — Tỷ lệ giữa rủi ro và lợi nhuận kỳ vọng. "
            "1:2 nghĩa là chấp nhận mất 1 để kiếm 2.\n\n"
            "**Engulfing Pattern** — Nến thân lớn bao trùm nến trước. "
            "Bullish engulfing ở support = tín hiệu mua mạnh."
        ),
        "daytrade": (
            "**VWAP** — Volume Weighted Average Price. "
            "Là giá trung bình có trọng số theo khối lượng trong phiên. "
            "Tổ chức lớn dùng VWAP để định giá.\n\n"
            "**Opening Range** — Khoảng giá 30-60 phút đầu phiên. "
            "Breakout khỏi opening range thường báo hiệu hướng đi cả ngày.\n\n"
            "**Volume Profile** — Phân bố khối lượng theo giá. "
            "POC (Point of Control) là vùng giá có nhiều giao dịch nhất."
        ),
        "scalp": (
            "**ATR (Average True Range)** — Đo lường biến động trung bình. "
            "ATR càng cao, biến động càng lớn, phù hợp scalping.\n\n"
            "**EMA5/EMA9** — Exponential Moving Average ngắn hạn. "
            "Crossover tạo tín hiệu mua/bán nhanh cho scalper.\n\n"
            "**10-bar Range** — Biên độ 10 nến gần nhất. "
            "Giá ở biên trên = momentum tăng, biên dưới = momentum giảm."
        ),
        "ichimoku": (
            "**Tenkan-sen (Conversion Line)** — Trung bình 9 kỳ (cao+thấp)/2. "
            "Phản ứng nhanh với giá, dùng như EMA9.\n\n"
            "**Kijun-sen (Base Line)** — Trung bình 26 kỳ. "
            "Đường hỗ trợ/kháng cự động, dùng làm trailing stop.\n\n"
            "**Kumo (Cloud)** — Senkou A và Senkou B tạo thành 'đám mây'. "
            "Mây dày = hỗ trợ/kháng cự mạnh. Mây mỏng = dễ break.\n\n"
            "**Chikou Span (Lagging Line)** — Giá đóng cửa dịch lùi 26 kỳ. "
            "Dùng để xác nhận xu hướng."
        ),
    }

    strategy = strategy.lower()
    lessons = lessons_map.get(strategy, {}).get("default", lessons_map.get("position", {}).get("default", ""))
    mistakes = mistakes_map.get(strategy, {}).get("default", mistakes_map.get("position", {}).get("default", ""))
    tech = technical_map.get(strategy, technical_map.get("position", ""))

    return {
        "lessons": lessons,
        "mistakes": mistakes,
        "technical_knowledge": tech,
    }
