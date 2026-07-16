"""
conversation_flow.py — Pi Agent Multi-Turn Reasoning & Học Tập

Design:
  Sau khi DeepSeek phân tích, Pi Agent follow-up để kiểm tra kiến thức user.
  Progressive learning levels:
    Level 1: Beginner — Hỏi về khái niệm cơ bản
    Level 2: Intermediate — Hỏi về reasoning, divergence, confluence
    Level 3: Advanced — Hỏi về risk management, multiple TF conflict

Usage:
  from education.conversation_flow import (
      get_follow_up_questions, get_knowledge_check,
      get_learning_summary, build_conversation_prompt,
  )

  # After DeepSeek analysis, suggest follow-up questions
  questions = get_follow_up_questions(level="intermediate", instrument="xau")

  # Quick knowledge check
  check = get_knowledge_check(level="beginner")
"""

import random
from typing import Dict, List, Optional, Any


# ─── LEARNING LEVELS ──────────────────────────────────────────────────────

LEVELS = {
    "beginner": {
        "name": "Người Mới",
        "description": "Học khái niệm cơ bản: xu hướng, hỗ trợ/kháng cự, RSI, MACD",
    },
    "intermediate": {
        "name": "Trung Cấp",
        "description": "Học về divergence, confluence, Fibonacci, SMC cơ bản",
    },
    "advanced": {
        "name": "Nâng Cao",
        "description": "Học về risk management, multiple TF conflict, market regime, Volume Profile",
    },
}


# ─── FOLLOW-UP QUESTIONS ──────────────────────────────────────────────────

FOLLOW_UP_QUESTIONS: Dict[str, List[str]] = {
    "beginner": [
        "📌 **Bạn có hiểu tại sao tôi chọn khung thời gian này để phân tích không?** "
        "Hãy giải thích Daily là gì và tại sao nó quan trọng.",

        "📌 **RSI hiện tại là bao nhiêu?** "
        "RSI > 70 nghĩa là gì? Còn RSI < 30 thì sao?",

        "📌 **Bạn có thấy vùng hỗ trợ và kháng cự trên chart không?** "
        "Vùng nào gần giá nhất hiện tại?",

        "📌 **EMA 9 và EMA 21 đang nằm thế nào?** "
        "Khi EMA9 ở trên EMA21 → xu hướng gì?",

        "📌 **MACD đang ở trên hay dưới đường Signal?** "
        "Điều này báo hiệu điều gì?",

        "📌 **Bạn có phân biệt được 'xu hướng tăng' và 'xu hướng giảm' không?** "
        "Hãy nhìn vào cấu trúc HH/HL hoặc LH/LL trên chart Daily.",
    ],
    "intermediate": [
        "📌 **Bạn có thấy divergence không?** "
        "RSI divergence là gì? Nó báo hiệu đảo chiều hay tiếp diễn?",

        "📌 **Có mâu thuẫn giữa các khung thời gian không?** "
        "Ví dụ: Daily UP nhưng 4H DOWN. Bạn sẽ ưu tiên TF nào và tại sao?",

        "📌 **Confluence score hiện tại là bao nhiêu?** "
        "Những yếu tố nào đóng góp vào confluence?",

        "📌 **Fibonacci retracement đang ở mức nào?** "
        "Mức 0.618 có ý nghĩa gì trong swing trading?",

        "📌 **SMC: bạn có thấy Order Block hoặc FVG nào không?** "
        "Order Block bullish nằm ở đâu? Nó có vai trò gì?",

        "📌 **Market regime hiện tại là gì?** "
        "Trending, Ranging hay Volatile? Bạn trade thế nào trong regime này?",
    ],
    "advanced": [
        "📌 **Bạn sẽ đặt Stop Loss ở đâu và tại sao?** "
        "Tính R:R hiện tại. R:R bao nhiêu là chấp nhận được?",

        "📌 **Position size của bạn là bao nhiêu với account $10,000?** "
        "Nếu risk 1%, SL 20 pips thì vào được bao nhiêu lot?",

        "📌 **Volume Profile nói gì về vùng giá hiện tại?** "
        "POC ở đâu? Value Area High/Low là bao nhiêu?",

        "📌 **Bạn có trade nếu Daily và 4H mâu thuẫn không?** "
        "Hãy giải thích chiến lược xử lý xung đột TF.",

        "📌 **Kịch bản nào sẽ làm setup này fail?** "
        "Điều kiện nào bạn sẽ hủy lệnh?",

        "📌 **Bạn có kiểm tra economic calendar trước khi vào lệnh không?** "
        "Có tin tức gì quan trọng trong 24h tới?",
    ],
}


def get_follow_up_questions(
    level: str = "intermediate",
    instrument: str = "xau",
    count: int = 3,
) -> List[str]:
    """Get follow-up questions for the user after analysis.

    Args:
        level: Learning level ('beginner', 'intermediate', 'advanced')
        instrument: Instrument ID (unused, for future customization)
        count: Number of questions to return

    Returns:
        List of question strings
    """
    questions = FOLLOW_UP_QUESTIONS.get(level, FOLLOW_UP_QUESTIONS["intermediate"])
    return random.sample(questions, min(count, len(questions)))


# ─── KNOWLEDGE CHECKS ─────────────────────────────────────────────────────

KNOWLEDGE_CHECKS: Dict[str, List[Dict[str, Any]]] = {
    "beginner": [
        {
            "question": "RSI > 70 báo hiệu điều gì?",
            "options": [
                "Thị trường đang quá mua (overbought), có thể giảm",
                "Thị trường đang quá bán (oversold), có thể tăng",
                "Xu hướng đang mạnh lên",
                "Không có ý nghĩa gì",
            ],
            "correct": 0,
            "explanation": "RSI > 70 = overbought. Giá đã tăng quá nhanh, có thể điều chỉnh giảm. "
                          "Không có nghĩa là phải bán ngay — cần kết hợp với các tín hiệu khác.",
        },
        {
            "question": "HH/HL là cấu trúc của xu hướng gì?",
            "options": [
                "Xu hướng giảm (Downtrend)",
                "Xu hướng tăng (Uptrend)",
                "Đi ngang (Sideway)",
                "Không xác định",
            ],
            "correct": 1,
            "explanation": "HH (Higher High) + HL (Higher Low) = Uptrend. "
                          "Giá đang tạo đỉnh sau cao hơn đỉnh trước, đáy sau cao hơn đáy trước.",
        },
        {
            "question": "MACD cắt lên trên đường Signal → tín hiệu gì?",
            "options": [
                "Bullish (momentum tăng)",
                "Bearish (momentum giảm)",
                "Trung tính",
                "Cần xác nhận thêm",
            ],
            "correct": 0,
            "explanation": "MACD cắt lên trên Signal = bullish crossover. "
                          "Momentum đang chuyển sang tăng. Tín hiệu mua tiềm năng.",
        },
        {
            "question": "Golden Cross là gì?",
            "options": [
                "SMA20 cắt lên SMA50",
                "SMA50 cắt lên SMA200",
                "EMA9 cắt lên EMA21",
                "Giá vượt đỉnh cũ",
            ],
            "correct": 1,
            "explanation": "Golden Cross = SMA50 cắt lên SMA200. "
                          "Tín hiệu dài hạn cho thấy xu hướng tăng vĩ mô.",
        },
    ],
    "intermediate": [
        {
            "question": "Regular Bullish Divergence là gì?",
            "options": [
                "Giá tạo đáy thấp hơn, RSI tạo đáy cao hơn",
                "Giá tạo đỉnh cao hơn, RSI tạo đỉnh thấp hơn",
                "Giá và RSI cùng tạo đáy cao hơn",
                "RSI > 70 và đang giảm",
            ],
            "correct": 0,
            "explanation": "Regular Bullish Divergence: giá tạo Lower Low nhưng RSI tạo Higher Low. "
                          "Báo hiệu đà giảm yếu dần, có thể đảo chiều tăng.",
        },
        {
            "question": "Khi nào nên ưu tiên tín hiệu Daily hơn 4H?",
            "options": [
                "Khi Daily và 4H mâu thuẫn, ưu tiên Daily vì khung lớn hơn",
                "Khi Daily và 4H mâu thuẫn, ưu tiên 4H vì nhạy hơn",
                "Luôn ưu tiên khung nhỏ vì vào lệnh chính xác hơn",
                "Không bao giờ ưu tiên, chờ cả 2 đồng thuận",
            ],
            "correct": 0,
            "explanation": "Daily là khung thời gian lớn hơn → xu hướng đáng tin hơn. "
                          "4H có thể nhiễu (noise). Luôn trade theo Daily bias, dùng 4H để tìm entry.",
        },
        {
            "question": "Confluence score cao nghĩa là gì?",
            "options": [
                "Nhiều tín hiệu đồng thuận → setup chất lượng cao",
                "Nhiều indicator đang báo mua → chắc chắn thắng",
                "Thị trường đang biến động mạnh",
                "Không có ý nghĩa đặc biệt",
            ],
            "correct": 0,
            "explanation": "Confluence = nhiều tín hiệu khác nhau cùng báo 1 hướng. "
                          "Càng nhiều confluence, xác suất thành công càng cao. "
                          "Nhưng không bao giờ 'chắc chắn thắng'.",
        },
    ],
    "advanced": [
        {
            "question": "Với account $10,000, risk 1%, SL 20 pips trên XAU, position size là bao nhiêu?",
            "options": [
                "0.5 lot",
                "0.05 lot",
                "1.0 lot",
                "0.1 lot",
            ],
            "correct": 1,
            "explanation": "Risk $ = $10,000 × 1% = $100. "
                          "Với XAU (100oz/lot), 1 pip = $10. SL=20 pips ($200 risk/lot full). "
                          "$100/$200 = 0.5 lot mini = 0.05 lot standard.",
        },
        {
            "question": "Khi nào nên hủy lệnh đang chờ (cancel pending order)?",
            "options": [
                "Khi có tin tức quan trọng bất ngờ",
                "Khi giá chưa chạm entry sau 24h",
                "Khi thấy 1 tín hiệu ngược trên TF nhỏ hơn",
                "Tất cả các lý do trên",
            ],
            "correct": 3,
            "explanation": "Tất cả đều là lý do chính đáng: "
                          "tin tức bất ngờ thay đổi cục diện, "
                          "giá không chạm entry có nghĩa kỳ vọng sai, "
                          "tín hiệu ngược trên TF nhỏ báo hiệu setup yếu.",
        },
        {
            "question": "Nếu 4H trending UP nhưng 15m có bearish divergence, bạn làm gì?",
            "options": [
                "Bỏ qua divergence, vào lệnh BUY theo 4H",
                "Chờ 15m divergence xong rồi mới BUY",
                "Vào lệnh SELL vì divergence",
                "Không trade vì mâu thuẫn",
            ],
            "correct": 1,
            "explanation": "4H UP là xu hướng chính → vẫn BUY, nhưng chờ pullback. "
                          "Bearish divergence 15m báo hiệu điều chỉnh ngắn hạn — "
                          "đây là cơ hội entry giá tốt hơn theo xu hướng chính.",
        },
    ],
}


def get_knowledge_check(level: str = "beginner") -> Dict[str, Any]:
    """Get a random knowledge check question.

    Args:
        level: Learning level

    Returns:
        Dict with question, options, correct answer index, explanation
    """
    checks = KNOWLEDGE_CHECKS.get(level, KNOWLEDGE_CHECKS["beginner"])
    return random.choice(checks)


def run_knowledge_check(level: str = "beginner", count: int = 3) -> str:
    """Generate a formatted knowledge check section.

    Args:
        level: Learning level
        count: Number of questions

    Returns:
        Formatted knowledge check string
    """
    checks = KNOWLEDGE_CHECKS.get(level, KNOWLEDGE_CHECKS["beginner"])
    selected = random.sample(checks, min(count, len(checks)))

    lines = [
        f"## 🧠 Kiểm Tra Kiến Thức — {LEVELS[level]['name']}",
        "",
    ]

    for i, check in enumerate(selected, 1):
        lines.append(f"### Câu {i}: {check['question']}")
        lines.append("")
        for j, opt in enumerate(check["options"]):
            marker = "✅" if j == check["correct"] else "⬜"
            lines.append(f"- {marker} {opt}")
        lines.append("")
        lines.append(f"**Giải thích**: {check['explanation']}")
        lines.append("")

    return "\n".join(lines)


# ─── CONVERSATION PROMPTS ─────────────────────────────────────────────────

def build_conversation_prompt(
    analysis_result: str,
    user_level: str = "intermediate",
    instrument: str = "xau",
) -> str:
    """Build a structured conversation prompt for Pi Agent.

    After DeepSeek analysis, Pi Agent uses this to:
      1. Summarize the analysis
      2. Ask follow-up questions
      3. Check user knowledge
      4. Provide learning materials

    Args:
        analysis_result: The analysis text from DeepSeek
        user_level: User's learning level
        instrument: Instrument ID

    Returns:
        Structured prompt for Pi Agent follow-up
    """
    follow_ups = get_follow_up_questions(user_level, instrument, count=3)
    knowledge = get_knowledge_check(user_level)

    prompt = f"""\
## 🎯 PHÂN TÍCH DEEPSEEK — TÓM TẮT

{analysis_result[:2000]}

---

## 📋 HƯỚNG DẪN PI AGENT FOLLOW-UP

Bạn là trợ giảng giao dịch. Sau khi DeepSeek phân tích, hãy:

### 1️⃣ Tóm tắt (1-2 câu)
Nêu bias chính và lý do quan trọng nhất.

### 2️⃣ Câu hỏi kiểm tra ({LEVELS[user_level]['name']})
Hỏi user {len(follow_ups)} câu để kiểm tra hiểu biết:
"""
    for i, q in enumerate(follow_ups, 1):
        prompt += f"\n{i}. {q}\n"

    prompt += f"""

### 3️⃣ Kiến thức bổ sung
Nếu user trả lời sai, giải thích đơn giản:
- **Câu hỏi**: {knowledge['question']}
- **Đáp án**: {knowledge['options'][knowledge['correct']]}
- **Giải thích**: {knowledge['explanation']}

### 4️⃣ Bài tập về nhà
Đề xuất 1 việc user nên tự làm trên chart để học thêm.

---

## ⚠️ QUY TẮC

• Không đưa ra lời khuyên giao dịch mới — chỉ dùng dữ liệu từ DeepSeek.
• Nếu user hỏi ngoài phạm vi, hãy nói "Tôi sẽ ghi nhận để phân tích sau."
• Khen ngợi khi user trả lời đúng, khích lệ khi sai.
• Luôn kết thúc bằng 1 câu hỏi mở để user tiếp tục học.
"""
    return prompt


def get_learning_summary(
    level: str = "beginner",
    instrument: str = "xau",
) -> str:
    """Generate a learning summary card for the user.

    Args:
        level: Learning level
        instrument: Instrument ID

    Returns:
        Formatted learning summary
    """
    level_info = LEVELS.get(level, LEVELS["beginner"])

    topics = {
        "beginner": [
            "Xu hướng (Trend) — HH/HL, LH/LL",
            "Hỗ trợ và Kháng cự (S/R)",
            "RSI — Quá mua/Quá bán",
            "MACD — Crossover, Centerline",
            "EMA/SMA — Golden/Death Cross",
            "Cấu trúc nến cơ bản (Engulfing, Pinbar)",
        ],
        "intermediate": [
            "Divergence — Regular & Hidden",
            "Confluence — Nhiều tín hiệu đồng thuận",
            "Fibonacci — Retracement & Extension",
            "SMC — Order Block, FVG, Liquidity",
            "Market Regime — Trending/Ranging/Volatile",
            "Volume Profile — POC, Value Area",
        ],
        "advanced": [
            "Risk Management — Position sizing, R:R",
            "Multi-TF Conflict Resolution",
            "Economic Calendar & Fundamental Analysis",
            "Advanced SMC — Breaker/Mitigation/Reclaimed",
            "Backtesting — Win rate, Profit factor",
            "Journal — Track & improve",
        ],
    }

    lines = [
        f"## 📚 Lộ Trình Học Tập — {level_info['name']}",
        "",
        level_info["description"],
        "",
        "### Các Chủ Đề Cần Nắm:",
        "",
    ]

    for topic in topics.get(level, topics["beginner"]):
        lines.append(f"- [ ] {topic}")

    lines.append("")
    lines.append("### Gợi Ý Học: ")
    lines.append("")

    tips = {
        "beginner": (
            "1. Mở chart Daily và tập nhận biết xu hướng bằng mắt.\n"
            "2. Thêm RSI(14) và MACD(12,26,9) — quan sát cách chúng phản ứng với giá.\n"
            "3. Vẽ hỗ trợ/kháng cự bằng tay — không dùng tự động."
        ),
        "intermediate": (
            "1. Tập phát hiện divergence — so sánh đáy/đỉnh giá với RSI.\n"
            "2. Dùng Volume Profile để xác định vùng giá trị.\n"
            "3. Kết hợp ít nhất 3 indicator trước khi đưa ra quyết định."
        ),
        "advanced": (
            "1. Backtest 100 lệnh trước khi trade thật.\n"
            "2. Ghi journal mỗi lệnh — học từ sai lầm.\n"
            "3. Xây dựng hệ thống — không trade cảm tính."
        ),
    }

    lines.append(tips.get(level, tips["beginner"]))
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "questions"
    level = sys.argv[2] if len(sys.argv) > 2 else "intermediate"

    if mode == "questions":
        questions = get_follow_up_questions(level, count=5)
        print(f"📌 Follow-up Questions ({LEVELS[level]['name']}):\n")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}\n")

    elif mode == "check":
        print(run_knowledge_check(level))

    elif mode == "learn":
        print(get_learning_summary(level))

    elif mode == "prompt":
        print(build_conversation_prompt("SAMPLE ANALYSIS TEXT...", level))
