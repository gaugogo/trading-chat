"""
Tests for journal.py — Signal Journal (SQLite).

Run with: pytest tests/test_journal.py -v
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from journal import SignalJournal, SignalRecord


# ─── FIXTURES ───

@pytest.fixture
def journal():
    """Create a journal with temp DB file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield SignalJournal(db_path=db_path)
    os.unlink(db_path)


@pytest.fixture
def journal_with_signals(journal):
    """Journal with some test signals."""
    ids = []
    signals_data = [
        ("xau", "swing", "BUY", 4.5, 0.8, 2650.0, 2630.0, 2700.0, 2.5, "Daily uptrend"),
        ("xau", "swing", "BUY", 3.0, 0.6, 2660.0, 2640.0, 2720.0, 3.0, "Pullback to support"),
        ("btc", "daytrade", "SELL", -3.5, 0.7, 45000.0, 45500.0, 44000.0, 2.0, "Resistance test"),
        ("xau", "position", "STRONG_BUY", 6.0, 0.9, 2600.0, 2500.0, 2800.0, 5.0, "Golden cross"),
        ("gbp", "scalp", "SELL", -2.0, 0.55, 1.2650, 1.2680, 1.2600, 1.6, "5m breakdown"),
    ]

    for instr, strat, bias, score, conf, entry, sl, tp, rr, reason in signals_data:
        sid = journal.record_signal(
            instrument=instr, strategy=strat, bias=bias,
            score=score, confidence=conf,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            rr_ratio=rr, reason=reason,
        )
        ids.append(sid)

    # Record outcomes for some
    journal.record_outcome(ids[0], 2710.0, "WIN", 2.26, "Good swing trade")
    journal.record_outcome(ids[1], 2640.0, "LOSS", -0.75, "Stop hit on news")
    journal.record_outcome(ids[2], 44200.0, "WIN", 1.78, "Good day trade")
    journal.record_outcome(ids[3], 2750.0, "WIN", 5.77, "Perfect position")

    return journal, ids


# ─── TESTS: Record ───

class TestRecord:
    def test_record_signal(self, journal):
        sid = journal.record_signal(
            instrument="xau", strategy="swing", bias="BUY",
            score=4.5, confidence=0.8,
            entry_price=2650.0, stop_loss=2630.0, take_profit=2700.0,
            rr_ratio=2.5, reason="Daily trend + pullback",
        )
        assert sid > 0

    def test_record_signal_minimal(self, journal):
        sid = journal.record_signal(
            instrument="btc", strategy="daytrade", bias="SELL",
        )
        assert sid > 0

    def test_record_outcome_not_found(self, journal):
        result = journal.record_outcome(9999, 100.0, "WIN", 1.0)
        assert result is False


# ─── TESTS: Query ───

class TestQuery:
    def test_get_recent_empty(self, journal):
        signals = journal.get_recent_signals(limit=5)
        assert len(signals) == 0

    def test_get_recent_signals(self, journal_with_signals):
        journal, ids = journal_with_signals
        signals = journal.get_recent_signals(limit=10)
        assert len(signals) == 5

    def test_get_recent_filter_instrument(self, journal_with_signals):
        journal, ids = journal_with_signals
        signals = journal.get_recent_signals(instrument="btc")
        assert len(signals) == 1
        assert signals[0].instrument == "btc"

    def test_get_recent_filter_strategy(self, journal_with_signals):
        journal, ids = journal_with_signals
        signals = journal.get_recent_signals(strategy="swing")
        assert len(signals) == 2

    def test_get_open_signals(self, journal_with_signals):
        journal, ids = journal_with_signals
        open_signals = journal.get_open_signals()
        assert len(open_signals) == 1  # gbp/scalp has no outcome
        assert open_signals[0].strategy == "scalp"

    def test_get_signal_by_id(self, journal_with_signals):
        journal, ids = journal_with_signals
        s = journal.get_signal_by_id(ids[0])
        assert s is not None
        assert s.instrument == "xau"
        assert s.strategy == "swing"
        assert s.bias == "BUY"

    def test_get_signal_by_id_not_found(self, journal):
        s = journal.get_signal_by_id(9999)
        assert s is None


# ─── TESTS: Accuracy Report ───

class TestAccuracy:
    def test_accuracy_report_empty(self, journal):
        report = journal.accuracy_report()
        assert "No closed signals" in report

    def test_accuracy_report_with_data(self, journal_with_signals):
        journal, ids = journal_with_signals
        report = journal.accuracy_report()
        assert "3W / 1L" in report
        assert "75.0%" in report  # 3 wins / 4 closed
        assert "Profit Factor" in report

    def test_accuracy_report_filter_strategy(self, journal_with_signals):
        journal, ids = journal_with_signals
        report = journal.accuracy_report(strategy="swing")
        assert "swing" in report
        assert "2" in report  # 2 swing signals

    def test_accuracy_report_filter_days(self, journal_with_signals):
        journal, ids = journal_with_signals
        report = journal.accuracy_report(days=1)
        assert "No closed signals" in report or "signals" in report


# ─── TESTS: Export ───

class TestExport:
    def test_export_json(self, journal_with_signals, tmp_path):
        journal, ids = journal_with_signals
        path = str(tmp_path / "test_export.json")
        result_path = journal.export_json(path=path)
        assert os.path.exists(result_path)

        with open(result_path) as f:
            data = json.load(f)
        assert len(data) == 5  # 5 signals

    def test_export_empty(self, journal, tmp_path):
        path = str(tmp_path / "empty_export.json")
        result_path = journal.export_json(path=path)
        assert os.path.exists(result_path)
        with open(result_path) as f:
            data = json.load(f)
        assert len(data) == 0


# ─── TESTS: Clear ───

class TestClear:
    def test_clear_without_confirm(self, journal_with_signals):
        journal, ids = journal_with_signals
        result = journal.clear_all(confirm=False)
        assert result is False
        assert len(journal.get_recent_signals()) > 0

    def test_clear_with_confirm(self, journal_with_signals):
        journal, ids = journal_with_signals
        result = journal.clear_all(confirm=True)
        assert result is True
        assert len(journal.get_recent_signals()) == 0


# ─── TESTS: SignalRecord ───

class TestSignalRecord:
    def test_to_dict(self):
        record = SignalRecord(
            id=1, instrument="xau", strategy="swing", bias="BUY",
            entry_price=2650.0, stop_loss=2630.0,
        )
        d = record.to_dict()
        assert d["id"] == 1
        assert d["instrument"] == "xau"
        assert "created_at" in d

    def test_to_dict_omits_none(self):
        record = SignalRecord(
            id=1, instrument="xau", strategy="swing", bias="BUY",
        )
        d = record.to_dict()
        assert "entry_price" not in d  # None, so omitted
