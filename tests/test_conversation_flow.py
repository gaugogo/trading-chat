"""
Tests for education/conversation_flow.py — Multi-turn reasoning flow

Tests focus on:
  - get_follow_up_questions
  - get_knowledge_check
  - build_conversation_prompt
  - get_learning_summary
  - LEVELS configuration
  - KNOWLEDGE_CHECKS data integrity
  - FOLLOW_UP_QUESTIONS structure
"""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from education.conversation_flow import (
    get_follow_up_questions,
    get_knowledge_check,
    run_knowledge_check,
    build_conversation_prompt,
    get_learning_summary,
    LEVELS,
    FOLLOW_UP_QUESTIONS,
    KNOWLEDGE_CHECKS,
)


# ─── LEVELS TESTS ────────────────────────────────────────────────────────

class TestLevels:
    """Tests for LEVELS configuration."""

    def test_beginner_level_exists(self):
        assert "beginner" in LEVELS
        assert LEVELS["beginner"]["name"] == "Người Mới"

    def test_intermediate_level_exists(self):
        assert "intermediate" in LEVELS
        assert LEVELS["intermediate"]["name"] == "Trung Cấp"

    def test_advanced_level_exists(self):
        assert "advanced" in LEVELS
        assert LEVELS["advanced"]["name"] == "Nâng Cao"

    def test_all_levels_have_name_and_description(self):
        for level_id, level_data in LEVELS.items():
            assert "name" in level_data
            assert "description" in level_data
            assert isinstance(level_data["name"], str)
            assert isinstance(level_data["description"], str)
            assert len(level_data["name"]) > 0
            assert len(level_data["description"]) > 0


# ─── FOLLOW-UP QUESTIONS TESTS ──────────────────────────────────────────

class TestFollowUpQuestions:
    """Tests for get_follow_up_questions."""

    def test_beginner_questions(self):
        questions = get_follow_up_questions(level="beginner", count=2)
        assert len(questions) == 2
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 10

    def test_intermediate_questions(self):
        questions = get_follow_up_questions(level="intermediate", count=3)
        assert len(questions) == 3

    def test_advanced_questions(self):
        questions = get_follow_up_questions(level="advanced", count=3)
        assert len(questions) == 3

    def test_default_level(self):
        """Test default level is intermediate."""
        questions = get_follow_up_questions(count=1)
        assert len(questions) == 1

    def test_count_more_than_available(self):
        """Test requesting more questions than available."""
        questions = get_follow_up_questions(level="beginner", count=100)
        assert len(questions) <= len(FOLLOW_UP_QUESTIONS["beginner"])

    def test_zero_count(self):
        """Test requesting zero questions."""
        questions = get_follow_up_questions(count=0)
        assert len(questions) == 0

    def test_all_questions_are_unique(self):
        """Test returned questions are unique (random sampling)."""
        questions = get_follow_up_questions(level="intermediate", count=10)
        assert len(set(questions)) == len(questions)

    def test_beginner_questions_content(self):
        """Test beginner questions contain basic trading concepts."""
        questions = FOLLOW_UP_QUESTIONS["beginner"]
        all_text = " ".join(questions)
        assert "RSI" in all_text
        assert "MACD" in all_text
        assert "xu hướng" in all_text

    def test_intermediate_questions_content(self):
        """Test intermediate questions contain advanced concepts."""
        questions = FOLLOW_UP_QUESTIONS["intermediate"]
        all_text = " ".join(questions)
        assert "divergence" in all_text.lower()
        assert "confluence" in all_text.lower()
        assert "SMC" in all_text or "Order Block" in all_text

    def test_advanced_questions_content(self):
        """Test advanced questions contain expert concepts."""
        questions = FOLLOW_UP_QUESTIONS["advanced"]
        all_text = " ".join(questions)
        assert "R:R" in all_text or "risk" in all_text.lower()
        assert "Volume Profile" in all_text or "POC" in all_text


# ─── KNOWLEDGE CHECK TESTS ──────────────────────────────────────────────

class TestKnowledgeCheck:
    """Tests for get_knowledge_check."""

    def test_beginner_check(self):
        check = get_knowledge_check("beginner")
        assert "question" in check
        assert "options" in check
        assert "correct" in check
        assert "explanation" in check
        assert isinstance(check["correct"], int)
        assert 0 <= check["correct"] < len(check["options"])

    def test_intermediate_check(self):
        check = get_knowledge_check("intermediate")
        assert len(check["options"]) >= 2
        assert len(check["explanation"]) > 10

    def test_advanced_check(self):
        check = get_knowledge_check("advanced")
        assert len(check["options"]) >= 2

    def test_default_level(self):
        check = get_knowledge_check()
        assert check is not None

    def test_randomness(self):
        """Test that different calls may return different questions."""
        checks = set()
        for _ in range(20):
            check = get_knowledge_check("beginner")
            checks.add(check["question"])
        # With 4 beginner questions, should get at least 2 different ones
        assert len(checks) >= 2

    def test_correct_answer_matches_option(self):
        """Test that correct index points to a valid option."""
        for level, checks in KNOWLEDGE_CHECKS.items():
            for check in checks:
                assert 0 <= check["correct"] < len(check["options"])

    def test_all_options_have_content(self):
        """Test all options are non-empty strings."""
        for level, checks in KNOWLEDGE_CHECKS.items():
            for check in checks:
                for opt in check["options"]:
                    assert isinstance(opt, str)
                    assert len(opt) > 0

    def test_all_explanations_have_content(self):
        """Test all explanations are non-empty strings."""
        for level, checks in KNOWLEDGE_CHECKS.items():
            for check in checks:
                assert isinstance(check["explanation"], str)
                assert len(check["explanation"]) > 10


# ─── RUN KNOWLEDGE CHECK TESTS ──────────────────────────────────────────

class TestRunKnowledgeCheck:
    """Tests for run_knowledge_check."""

    def test_formatting(self):
        result = run_knowledge_check("beginner", count=2)
        assert isinstance(result, str)
        assert "Kiểm Tra Kiến Thức" in result
        assert "Người Mới" in result
        assert "Câu 1" in result
        assert "Giải thích" in result

    def test_count_control(self):
        result = run_knowledge_check("beginner", count=3)
        assert result.count("Câu ") == 3

    def test_default_count(self):
        result = run_knowledge_check()
        assert result is not None

    def test_all_levels(self):
        for level in ["beginner", "intermediate", "advanced"]:
            result = run_knowledge_check(level)
            level_name = LEVELS[level]["name"]
            assert level_name in result


# ─── BUILD CONVERSATION PROMPT TESTS ────────────────────────────────────

class TestBuildConversationPrompt:
    """Tests for build_conversation_prompt."""

    def test_basic_structure(self):
        prompt = build_conversation_prompt(
            analysis_result="This is a sample analysis of XAUUSD showing bullish bias...",
            user_level="intermediate",
            instrument="xau",
        )
        assert isinstance(prompt, str)
        assert "PHÂN TÍCH DEEPSEEK" in prompt
        assert "PI AGENT FOLLOW-UP" in prompt
        assert "Câu hỏi kiểm tra" in prompt
        assert "Trung Cấp" in prompt

    def test_includes_analysis(self):
        analysis = "XAUUSD is trending UP on Daily"
        prompt = build_conversation_prompt(analysis, "beginner", "xau")
        assert analysis in prompt

    def test_includes_follow_up_questions(self):
        prompt = build_conversation_prompt("Analysis text...", "beginner", "xau")
        assert "📌" in prompt  # Follow-up questions have 📌 marker

    def test_includes_knowledge_check(self):
        prompt = build_conversation_prompt("Analysis text...", "beginner", "xau")
        assert "Câu hỏi" in prompt
        assert "Giải thích" in prompt

    def test_includes_rules(self):
        prompt = build_conversation_prompt("Analysis text...", "beginner", "xau")
        assert "QUY TẮC" in prompt
        assert "không đưa ra lời khuyên" in prompt.lower()

    def test_different_levels(self):
        for level in ["beginner", "intermediate", "advanced"]:
            prompt = build_conversation_prompt("Analysis text...", level, "xau")
            assert LEVELS[level]["name"] in prompt


# ─── LEARNING SUMMARY TESTS ─────────────────────────────────────────────

class TestLearningSummary:
    """Tests for get_learning_summary."""

    def test_beginner_topics(self):
        summary = get_learning_summary("beginner")
        assert isinstance(summary, str)
        assert "Lộ Trình Học Tập" in summary
        assert "Người Mới" in summary
        assert "Xu hướng" in summary
        assert "RSI" in summary
        assert "MACD" in summary

    def test_intermediate_topics(self):
        summary = get_learning_summary("intermediate")
        assert "Trung Cấp" in summary
        assert "Divergence" in summary
        assert "Confluence" in summary
        assert "Fibonacci" in summary

    def test_advanced_topics(self):
        summary = get_learning_summary("advanced")
        assert "Nâng Cao" in summary
        assert "Risk Management" in summary
        assert "Backtesting" in summary or "Backtest" in summary
        assert "Journal" in summary

    def test_all_levels_have_checkboxes(self):
        for level in ["beginner", "intermediate", "advanced"]:
            summary = get_learning_summary(level)
            assert "[ ]" in summary  # All have unchecked learning topics

    def test_all_levels_have_tips(self):
        for level in ["beginner", "intermediate", "advanced"]:
            summary = get_learning_summary(level)
            assert "Gợi Ý Học" in summary

    def test_default_level(self):
        summary = get_learning_summary()
        assert summary is not None


# ─── DATA INTEGRITY TESTS ───────────────────────────────────────────────

class TestDataIntegrity:
    """Tests for data integrity of all conversation data."""

    def test_follow_up_questions_structure(self):
        """Test FOLLOW_UP_QUESTIONS has all levels."""
        for level in ["beginner", "intermediate", "advanced"]:
            assert level in FOLLOW_UP_QUESTIONS
            assert len(FOLLOW_UP_QUESTIONS[level]) > 0

    def test_knowledge_checks_structure(self):
        """Test KNOWLEDGE_CHECKS has all levels."""
        for level in ["beginner", "intermediate", "advanced"]:
            assert level in KNOWLEDGE_CHECKS
            assert len(KNOWLEDGE_CHECKS[level]) > 0

    def test_all_levels_match(self):
        """Test that level keys match between all data sources."""
        level_keys = set(LEVELS.keys())
        assert level_keys == set(FOLLOW_UP_QUESTIONS.keys())
        assert level_keys == set(KNOWLEDGE_CHECKS.keys())

    def test_no_duplicate_questions(self):
        """Test no duplicate questions within each level."""
        for level, questions in FOLLOW_UP_QUESTIONS.items():
            assert len(questions) == len(set(questions))

    def test_no_duplicate_check_questions(self):
        """Test no duplicate check questions within each level."""
        for level, checks in KNOWLEDGE_CHECKS.items():
            questions = [c["question"] for c in checks]
            assert len(questions) == len(set(questions))
