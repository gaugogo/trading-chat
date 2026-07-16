"""
journal.py — Signal Journal (SQLite)

Lưu mọi signal đã generate + outcome để:
  - Tính accuracy per style/TF
  - Phân tích hiệu suất theo thời gian
  - Học từ lịch sử giao dịch

Usage:
  from journal import SignalJournal
  journal = SignalJournal()

  # Ghi signal mới
  journal.record_signal(
      instrument="xau",
      strategy="swing",
      bias="BUY",
      entry_price=2650.0,
      stop_loss=2630.0,
      take_profit=2700.0,
      reason="Daily uptrend + 4H pullback to SMA20",
  )

  # Ghi outcome sau khi đóng lệnh
  journal.record_outcome(signal_id=1, exit_price=2680.0, result="WIN", pnl_pct=1.2)

  # Báo cáo accuracy
  report = journal.accuracy_report()
  print(report)

  # CLI
  python journal.py                          # xem recent signals
  python journal.py --stats                  # accuracy stats
  python journal.py --export                 # export to JSON
"""

import os
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ─── DB PATH ────────────────────────────────────────────────────────────

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "signal_journal.db"


# ─── DATA CLASS ─────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    """Single signal record."""
    id: Optional[int] = None
    instrument: str = ""
    strategy: str = ""          # 'position', 'swing', 'daytrade', 'scalp', 'ichimoku'
    bias: str = ""              # 'STRONG_BUY', 'BUY', 'BUY_BIAS', 'NEUTRAL', 'SELL_BIAS', 'SELL', 'STRONG_SELL'
    score: float = 0.0          # Normalized score (-10 to +10)
    confidence: float = 0.0     # 0.0 to 1.0

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rr_ratio: Optional[float] = None

    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    result: Optional[str] = None   # 'WIN', 'LOSS', 'BREAKEVEN', None (open)
    pnl_pct: Optional[float] = None

    reason: str = ""             # Lý do signal
    notes: str = ""              # Ghi chú sau khi đóng lệnh
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ─── JOURNAL ────────────────────────────────────────────────────────────

class SignalJournal:
    """SQLite-backed signal journal.

    Records every signal + outcome for performance tracking.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument      TEXT NOT NULL,
                    strategy        TEXT NOT NULL,
                    bias            TEXT NOT NULL,
                    score           REAL DEFAULT 0.0,
                    confidence      REAL DEFAULT 0.0,
                    entry_price     REAL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    rr_ratio        REAL,
                    exit_price      REAL,
                    exit_date       TEXT,
                    result          TEXT,
                    pnl_pct         REAL,
                    reason          TEXT DEFAULT '',
                    notes           TEXT DEFAULT '',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_signals_instrument ON signals(instrument);
                CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
                CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
            """)

    # ─── RECORD SIGNAL ──────────────────────────────────────────────────

    def record_signal(
        self,
        instrument: str,
        strategy: str,
        bias: str,
        score: float = 0.0,
        confidence: float = 0.0,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        rr_ratio: Optional[float] = None,
        reason: str = "",
    ) -> int:
        """Record a new signal and return its ID.

        Args:
            instrument: 'xau', 'btc', 'gbp'
            strategy: 'position', 'swing', 'daytrade', 'scalp', 'ichimoku'
            bias: Signal bias
            score: Normalized score (-10 to +10)
            confidence: 0.0 to 1.0
            entry_price: Suggested entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            rr_ratio: Risk:Reward ratio
            reason: Signal reasoning text

        Returns:
            Signal ID (primary key)
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO signals
                   (instrument, strategy, bias, score, confidence,
                    entry_price, stop_loss, take_profit, rr_ratio, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (instrument, strategy, bias, score, confidence,
                 entry_price, stop_loss, take_profit, rr_ratio, reason),
            )
            return cursor.lastrowid

    # ─── RECORD OUTCOME ────────────────────────────────────────────────

    def record_outcome(
        self,
        signal_id: int,
        exit_price: float,
        result: str,
        pnl_pct: float,
        notes: str = "",
    ) -> bool:
        """Record outcome for a closed signal.

        Args:
            signal_id: Signal ID from record_signal()
            exit_price: Price at exit
            result: 'WIN', 'LOSS', or 'BREAKEVEN'
            pnl_pct: P&L as percentage
            notes: Optional notes about the trade

        Returns:
            True if updated, False if signal not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE signals SET
                    exit_price = ?, exit_date = datetime('now'),
                    result = ?, pnl_pct = ?, notes = ?,
                    updated_at = datetime('now')
                   WHERE id = ?""",
                (exit_price, result, pnl_pct, notes, signal_id),
            )
            return cursor.rowcount > 0

    # ─── QUERIES ───────────────────────────────────────────────────────

    def get_recent_signals(
        self,
        limit: int = 20,
        instrument: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[SignalRecord]:
        """Get recent signals with optional filters."""
        with self._connect() as conn:
            query = "SELECT * FROM signals WHERE 1=1"
            params: List = []

            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_open_signals(self) -> List[SignalRecord]:
        """Get signals that haven't been closed yet."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE result IS NULL ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_signal_by_id(self, signal_id: int) -> Optional[SignalRecord]:
        """Get a single signal by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SignalRecord:
        return SignalRecord(
            id=row["id"],
            instrument=row["instrument"],
            strategy=row["strategy"],
            bias=row["bias"],
            score=row["score"],
            confidence=row["confidence"],
            entry_price=row["entry_price"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            rr_ratio=row["rr_ratio"],
            exit_price=row["exit_price"],
            exit_date=row["exit_date"],
            result=row["result"],
            pnl_pct=row["pnl_pct"],
            reason=row["reason"] or "",
            notes=row["notes"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ─── STATISTICS ────────────────────────────────────────────────────

    def accuracy_report(
        self,
        instrument: Optional[str] = None,
        strategy: Optional[str] = None,
        days: Optional[int] = None,
    ) -> str:
        """Generate accuracy report as formatted string.

        Args:
            instrument: Filter by instrument
            strategy: Filter by strategy
            days: Only include signals from last N days

        Returns:
            Formatted report string
        """
        with self._connect() as conn:
            query = "SELECT * FROM signals WHERE result IS NOT NULL"
            params: List = []

            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            if days:
                query += " AND created_at >= datetime('now', ?)"
                params.append(f"-{days} days")

            rows = conn.execute(query, params).fetchall()
            signals = [self._row_to_record(r) for r in rows]

        if not signals:
            return "📭 No closed signals found."

        total = len(signals)
        wins = sum(1 for s in signals if s.result == "WIN")
        losses = sum(1 for s in signals if s.result == "LOSS")
        be = sum(1 for s in signals if s.result == "BREAKEVEN")

        win_rate = (wins / total * 100) if total > 0 else 0.0
        total_pnl = sum(s.pnl_pct or 0 for s in signals)
        avg_win = sum(s.pnl_pct or 0 for s in signals if s.result == "WIN") / max(wins, 1)
        avg_loss = sum(abs(s.pnl_pct or 0) for s in signals if s.result == "LOSS") / max(losses, 1)

        # Profit factor
        gross_profit = sum(s.pnl_pct or 0 for s in signals if s.result == "WIN")
        gross_loss = abs(sum(s.pnl_pct or 0 for s in signals if s.result == "LOSS"))
        profit_factor = gross_profit / max(gross_loss, 0.001)

        # Per strategy breakdown
        strategies = set(s.strategy for s in signals)
        strategy_lines = []
        for strat in sorted(strategies):
            s_signals = [s for s in signals if s.strategy == strat]
            s_total = len(s_signals)
            s_wins = sum(1 for s in s_signals if s.result == "WIN")
            s_wr = s_wins / s_total * 100 if s_total > 0 else 0
            s_pnl = sum(s.pnl_pct or 0 for s in s_signals)
            strategy_lines.append(f"  {strat:<10} {s_total:>3} signals  {s_wr:>5.1f}% WR  {s_pnl:>+7.2f}% PnL")

        lines = [
            "=" * 60,
            "  📊 SIGNAL JOURNAL — ACCURACY REPORT",
            "=" * 60,
            "",
            f"  Period: {signals[-1].created_at[:10] if signals else 'N/A'} → "
            f"{signals[0].created_at[:10] if signals else 'N/A'}",
            f"  Total Signals: {total}",
            f"  Win Rate:      {win_rate:.1f}%",
            f"  Profit Factor: {profit_factor:.2f}",
            f"  Net P&L:       {total_pnl:+.2f}%",
            f"  Avg Win:       {avg_win:+.2f}%",
            f"  Avg Loss:      {avg_loss:.2f}%",
            "",
            f"  Breakdown: {wins}W / {losses}L / {be}BE",
            "",
            "  Per Strategy:",
            *strategy_lines,
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def export_json(self, path: Optional[str] = None) -> str:
        """Export all signals to JSON file.

        Args:
            path: Output path (default: 'signal_journal_export.json')

        Returns:
            Path to exported file
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY created_at DESC"
            ).fetchall()
            signals = [self._row_to_record(r) for r in rows]

        data = [s.to_dict() for s in signals]

        path = path or f"signal_journal_export_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    def clear_all(self, confirm: bool = False) -> bool:
        """Clear all signal records.

        Args:
            confirm: Must be True to actually delete

        Returns:
            True if deleted
        """
        if not confirm:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM signals")
            conn.commit()
        # VACUUM must be outside transaction
        with self._connect() as conn:
            conn.execute("VACUUM")
        return True


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Signal Journal — trade performance tracker")
    parser.add_argument("--stats", "-s", action="store_true", help="Show accuracy report")
    parser.add_argument("--export", "-e", action="store_true", help="Export to JSON")
    parser.add_argument("--instrument", "-i", help="Filter by instrument")
    parser.add_argument("--strategy", help="Filter by strategy")
    parser.add_argument("--days", type=int, default=30, help="Days back for stats (default: 30)")
    parser.add_argument("--recent", "-r", type=int, default=10, help="Show recent N signals (default: 10)")
    parser.add_argument("--open", "-o", action="store_true", help="Show open signals")
    parser.add_argument("--clear", action="store_true", help="Clear all data (requires --confirm)")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive action")

    args = parser.parse_args()
    journal = SignalJournal()

    if args.clear:
        if journal.clear_all(confirm=args.confirm):
            print("✅ All signal records cleared.")
        else:
            print("❌ Clear cancelled. Use --confirm to force.")
        return

    if args.stats:
        print(journal.accuracy_report(
            instrument=args.instrument,
            strategy=args.strategy,
            days=args.days,
        ))
        return

    if args.export:
        path = journal.export_json()
        print(f"📁 Exported to: {path}")
        return

    if args.open:
        signals = journal.get_open_signals()
        if not signals:
            print("📭 No open signals.")
            return
        print(f"📋 {len(signals)} open signal(s):")
        print_signal_table(signals)
        return

    # Default: show recent signals
    signals = journal.get_recent_signals(
        limit=args.recent,
        instrument=args.instrument,
        strategy=args.strategy,
    )

    if not signals:
        print("📭 No signals recorded yet.")
        print("Use: python journal.py --record ... to add signals from your strategy code.")
        return

    print(f"📋 Recent {len(signals)} signal(s):")
    print_signal_table(signals)


def print_signal_table(signals: List[SignalRecord]):
    """Print signals as a formatted table."""
    header = f"{'ID':<4} {'Instr':<6} {'Strategy':<10} {'Bias':<14} {'Score':<7} {'Price':<10} {'Result':<10} {'PnL':<8} {'Date':<20}"
    print(header)
    print("-" * len(header))
    for s in signals:
        price = f"${s.entry_price:.2f}" if s.entry_price else "N/A"
        pnl = f"{s.pnl_pct:+.2f}%" if s.pnl_pct is not None else ""
        result = s.result or "OPEN"
        print(f"{s.id:<4} {s.instrument:<6} {s.strategy:<10} {s.bias:<14} {s.score:+.1f}    {price:<10} {result:<10} {pnl:<8} {s.created_at[:19]}")


if __name__ == "__main__":
    main()
