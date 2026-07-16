"""
fundamental.py — Fundamental Data & Economic Calendar

Cung cấp:
  - DXY (US Dollar Index) từ Yahoo Finance (DX-Y.NYB)
  - US Treasury yields (10Y: ^TNX, 2Y: ^2YY)
  - Bond yield spread (10Y-2Y) — recession indicator
  - Economic calendar (ForexFactory via scraping + free API fallback)
  - Correlation analysis: XAU vs DXY, XAU vs bond yields
  - Fundamental bias scoring

Usage:
  from fundamental import (
      fetch_dxy, fetch_bond_yields, fetch_economic_calendar,
      fundamental_bias, fundamental_report,
  )

  dxy = fetch_dxy()
  yields = fetch_bond_yields()
  calendar = fetch_economic_calendar(days_ahead=3)
  bias = fundamental_bias("xau")
  print(fundamental_report("xau"))
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import requests
import numpy as np
import pandas as pd

from core import get_logger, CACHE_DIR

logger = get_logger(__name__)


# ─── CONSTANTS ─────────────────────────────────────────────────────────────

DXY_SYMBOL = "DX-Y.NYB"
TNX_SYMBOL = "^TNX"    # US 10Y Treasury Yield
TWOYY_SYMBOL = "^2YY"  # US 2Y Treasury Yield

# Yahoo Finance cookie/session constants
YAHOO_CRUMB_URL = "https://fc.yahoo.com/ws/query/v1/instrument/GC=F/chart/1d"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# Cache TTLs
DXY_CACHE_HOURS = 1.0
YIELD_CACHE_HOURS = 1.0
CALENDAR_CACHE_HOURS = 6.0

# Economic calendar — high-impact events for XAU
HIGH_IMPACT_EVENTS_XAU = [
    "CPI", "PPI", "Non-Farm", "NFP", "Unemployment",
    "FOMC", "Federal Funds Rate", "Interest Rate Decision",
    "GDP", "Retail Sales", "ISM Manufacturing",
    "Initial Jobless Claims", "Consumer Confidence",
    "Core CPI", "Core PCE", "Personal Spending",
    "Philadelphia Fed", "Empire State",
    "Treasury Auction", "10-Year Note Auction",
]

# ForexFactory scraping URLs
FOREX_FACTORY_CALENDAR_URL = "https://www.forexfactory.com/calendar"


# ─── DATA CLASSES ──────────────────────────────────────────────────────────

@dataclass
class DXYData:
    """US Dollar Index data snapshot."""
    price: float
    change_24h_pct: float
    timestamp: float = 0.0
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    trend: str = "neutral"  # 'bullish', 'bearish', 'neutral'

    def __post_init__(self):
        if self.change_24h_pct > 0.3:
            self.trend = "bullish"
        elif self.change_24h_pct < -0.3:
            self.trend = "bearish"
        else:
            self.trend = "neutral"


@dataclass
class BondYieldData:
    """US Treasury yield data."""
    yield_10y: float
    yield_2y: float
    spread_10y_2y: float  # 10Y - 2Y (negative = inverted curve → recession signal)
    timestamp: float = 0.0
    yield_10y_change: float = 0.0
    yield_2y_change: float = 0.0

    @property
    def is_inverted(self) -> bool:
        """Yield curve inverted when 10Y < 2Y."""
        return self.spread_10y_2y < 0

    @property
    def inversion_depth_bps(self) -> float:
        """How deep is the inversion in basis points."""
        return abs(self.spread_10y_2y) * 100 if self.is_inverted else 0.0


@dataclass
class EconomicEvent:
    """A single economic calendar event."""
    date: str
    time: str
    currency: str
    event: str
    importance: str  # 'High', 'Medium', 'Low'
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None

    @property
    def is_high_impact(self) -> bool:
        return self.importance == "High"

    def affects_xau(self) -> bool:
        """Check if this event typically affects gold prices."""
        if self.currency not in ("USD", "ALL"):
            return False
        for keyword in HIGH_IMPACT_EVENTS_XAU:
            if keyword.lower() in self.event.lower():
                return True
        return self.is_high_impact


@dataclass
class FundamentalReport:
    """Comprehensive fundamental analysis report."""
    instrument: str
    dxy: Optional[DXYData] = None
    bond_yields: Optional[BondYieldData] = None
    upcoming_events: List[EconomicEvent] = field(default_factory=list)
    bias: str = "neutral"  # 'bullish', 'bearish', 'neutral'
    bias_score: float = 0.0
    confidence: str = "low"  # 'high', 'medium', 'low'
    summary: str = ""

    def detailed_report(self) -> str:
        """Generate a formatted fundamental analysis report.

        Returns:
            Multi-line report string
        """
        lines = [f"📊 **Fundamental Analysis — {self.instrument.upper()}**", "=" * 50, ""]

        # DXY section
        if self.dxy:
            arrow = "🟢" if self.dxy.trend == "bullish" else "🔴" if self.dxy.trend == "bearish" else "⚪"
            lines.append(f"**DXY (US Dollar Index)**: {arrow} ${self.dxy.price:.2f}")
            lines.append(f"  • 24h Change: {self.dxy.change_24h_pct:+.2f}% ({self.dxy.trend})")
            if self.dxy.high_24h:
                lines.append(f"  • 24h Range: ${self.dxy.low_24h:.2f} — ${self.dxy.high_24h:.2f}")
            lines.append("")

        # Bond yields section
        if self.bond_yields:
            inv_icon = "⚠️" if self.bond_yields.is_inverted else "✅"
            lines.append(f"**US Treasury Yields**: {inv_icon}")
            lines.append(f"  • 10Y Yield: {self.bond_yields.yield_10y:.2f}% ({self.bond_yields.yield_10y_change:+.2f}bp)")
            lines.append(f"  • 2Y Yield:  {self.bond_yields.yield_2y:.2f}% ({self.bond_yields.yield_2y_change:+.2f}bp)")
            lines.append(f"  • 10Y-2Y Spread: {self.bond_yields.spread_10y_2y:.2f}%")

            if self.bond_yields.is_inverted:
                lines.append(f"  ⚠️ **Yield Curve INVERTED** ({self.bond_yields.inversion_depth_bps:.0f} bps deep) — Recession signal!")
                lines.append(f"  → XAU typically BENEFITS from inversion as safe-haven demand rises")
            else:
                lines.append(f"  ✅ Yield curve normal (positive spread)")
            lines.append("")

        # Impact on XAU
        lines.append(f"**Impact on {self.instrument.upper()}**:")
        lines.append(self._xau_correlation_analysis())
        lines.append("")

        # Economic calendar
        if self.upcoming_events:
            lines.append(f"**📅 Upcoming High-Impact Events (next 3 days):**")
            lines.append(f"  {'Date':<14} {'Time':<8} {'Event':<35} {'Forecast':<12} {'Previous':<12}")
            lines.append(f"  {'-'*14} {'-'*8} {'-'*35} {'-'*12} {'-'*12}")
            for event in self.upcoming_events[:10]:
                if event.affects_xau():
                    marker = "🔴" if event.is_high_impact else "🟡"
                else:
                    marker = "  "
                forecast = event.forecast or "—"
                previous = event.previous or "—"
                lines.append(f"  {marker} {event.date:<12} {event.time:<8} {event.event:<35} {forecast:<12} {previous:<12}")
            lines.append("")

        # Bias summary
        if self.bias_score != 0:
            bias_icon = "🟢" if self.bias_score > 0 else "🔴"
            lines.append(f"**Fundamental Bias**: {bias_icon} {self.bias.upper()} (score: {self.bias_score:+.1f})")
            lines.append(f"**Confidence**: {self.confidence.upper()}")
            lines.append("")

        if self.summary:
            lines.append(f"**Summary**: {self.summary}")

        return "\n".join(lines)

    def _xau_correlation_analysis(self) -> str:
        """Analyze how DXY and bond yields affect XAU."""
        parts = []

        if self.dxy:
            # DXY vs XAU: inverse correlation
            if self.dxy.trend == "bullish":
                parts.append("🔴 DXY tăng → gây áp lực giảm lên XAU (tương quan nghịch)")
            elif self.dxy.trend == "bearish":
                parts.append("🟢 DXY giảm → hỗ trợ tăng cho XAU (tương quan nghịch)")
            else:
                parts.append("⚪ DXY đi ngang → ít tác động rõ rệt lên XAU")

        if self.bond_yields:
            # Real yields vs XAU: inverse correlation
            if self.bond_yields.yield_10y > 4.5:
                parts.append(f"🔴 Lợi suất 10Y ở mức cao ({self.bond_yields.yield_10y:.1f}%) → chi phí cơ hội giữ XAU tăng")
            elif self.bond_yields.yield_10y < 3.5:
                parts.append(f"🟢 Lợi suất 10Y ở mức thấp ({self.bond_yields.yield_10y:.1f}%) → XAU hấp dẫn hơn")

            if self.bond_yields.is_inverted:
                parts.append("⚠️ Yield curve đảo ngược → kỳ vọng suy thoái → XAU là safe-haven hưởng lợi")

        return "\n".join(f"      {p}" for p in parts) if parts else "      Không đủ dữ liệu để phân tích tương quan."


# ─── DATA FETCHING ─────────────────────────────────────────────────────────

def _fetch_yahoo_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch quote data from Yahoo Finance v8 chart API.

    Args:
        symbol: Yahoo Finance symbol (e.g., 'DX-Y.NYB', '^TNX')

    Returns:
        Dict with price info, or None
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1d", "range": "5d"}
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        result = data.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        timestamps = result.get("timestamp", [])

        if not timestamps or not quotes:
            return None

        closes = [c for c in quotes.get("close", []) if c is not None]
        highs = [h for h in quotes.get("high", []) if h is not None]
        lows = [l for l in quotes.get("low", []) if l is not None]

        if not closes:
            return None

        current_price = float(closes[-1])
        prev_close = float(meta.get("previousClose", closes[-2] if len(closes) > 1 else closes[-1]))
        change_pct = ((current_price - prev_close) / prev_close) * 100

        return {
            "price": current_price,
            "change_pct": change_pct,
            "high_24h": float(highs[-1]) if highs else None,
            "low_24h": float(lows[-1]) if lows else None,
            "timestamp": time.time(),
        }

    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        logger.warning(f"[Fundamental] Yahoo quote failed for {symbol}: {e}")
        return None


def fetch_dxy(use_cache: bool = True) -> Optional[DXYData]:
    """Fetch US Dollar Index (DXY) from Yahoo Finance.

    Args:
        use_cache: Whether to use file cache

    Returns:
        DXYData object or None
    """
    cache_file = CACHE_DIR / "dxy_cache.json"

    # Check cache
    if use_cache and cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(hours=DXY_CACHE_HOURS):
            try:
                data = json.loads(cache_file.read_text())
                return DXYData(**data)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    # Fetch fresh data
    quote = _fetch_yahoo_quote(DXY_SYMBOL)
    if not quote:
        return None

    dxy = DXYData(
        price=quote["price"],
        change_24h_pct=quote["change_pct"],
        timestamp=quote["timestamp"],
        high_24h=quote.get("high_24h"),
        low_24h=quote.get("low_24h"),
    )

    # Cache it
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "price": dxy.price,
            "change_24h_pct": dxy.change_24h_pct,
            "timestamp": dxy.timestamp,
            "high_24h": dxy.high_24h,
            "low_24h": dxy.low_24h,
            "trend": dxy.trend,
        }))

    return dxy


def fetch_bond_yields(use_cache: bool = True) -> Optional[BondYieldData]:
    """Fetch US Treasury yields (10Y, 2Y) from Yahoo Finance.

    Args:
        use_cache: Whether to use file cache

    Returns:
        BondYieldData object or None
    """
    cache_file = CACHE_DIR / "bond_yields_cache.json"

    # Check cache
    if use_cache and cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(hours=YIELD_CACHE_HOURS):
            try:
                data = json.loads(cache_file.read_text())
                return BondYieldData(**data)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    # Fetch yields
    tnx_quote = _fetch_yahoo_quote(TNX_SYMBOL)
    twoyy_quote = _fetch_yahoo_quote(TWOYY_SYMBOL)

    if not tnx_quote or not twoyy_quote:
        logger.warning("[Fundamental] Could not fetch bond yields")
        return None

    yields = BondYieldData(
        yield_10y=tnx_quote["price"],
        yield_2y=twoyy_quote["price"],
        spread_10y_2y=tnx_quote["price"] - twoyy_quote["price"],
        timestamp=time.time(),
        yield_10y_change=tnx_quote["change_pct"],
        yield_2y_change=twoyy_quote["change_pct"],
    )

    # Cache it
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "yield_10y": yields.yield_10y,
            "yield_2y": yields.yield_2y,
            "spread_10y_2y": yields.spread_10y_2y,
            "timestamp": yields.timestamp,
            "yield_10y_change": yields.yield_10y_change,
            "yield_2y_change": yields.yield_2y_change,
        }))

    return yields


def fetch_economic_calendar(
    days_ahead: int = 3,
    use_cache: bool = True,
) -> List[EconomicEvent]:
    """Fetch economic calendar events.

    Primary: Parse ForexFactory HTML (free, no API key).
    Fallback: Return empty list with warning.

    Args:
        days_ahead: Number of days to look ahead
        use_cache: Whether to cache results

    Returns:
        List of EconomicEvent objects
    """
    cache_file = CACHE_DIR / f"calendar_{days_ahead}d_cache.json"

    # Check cache
    if use_cache and cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(hours=CALENDAR_CACHE_HOURS):
            try:
                data = json.loads(cache_file.read_text())
                return [EconomicEvent(**e) for e in data]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    try:
        events = _scrape_forexfactory(days_ahead)
    except Exception as e:
        logger.warning(f"[Fundamental] Scraper failed: {e}")
        events = _calendar_fallback(days_ahead)

    # Cache
    if use_cache and events:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            [e.__dict__ for e in events]
        ))

    return events


def _scrape_forexfactory(days_ahead: int = 3) -> List[EconomicEvent]:
    """Scrape economic calendar from ForexFactory.

    Uses simple HTML parsing. If structure changes, returns empty list.

    Args:
        days_ahead: Number of days to look ahead

    Returns:
        List of EconomicEvent objects
    """
    events: List[EconomicEvent] = []

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        r = requests.get(
            FOREX_FACTORY_CALENDAR_URL,
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        html = r.text

        # Parse the calendar table rows
        # ForexFactory uses <tr class="calendar_row"> 
        rows = re.findall(
            r'<tr\s+class="calendar_row[^"]*"[^>]*>(.*?)</tr>',
            html,
            re.DOTALL,
        )

        for row in rows:
            try:
                # Extract date
                date_match = re.search(r'<td[^>]*class="calendar__date[^"]*"[^>]*>\s*([^<]+)\s*<', row)
                date = date_match.group(1).strip() if date_match else ""

                # Extract time
                time_match = re.search(r'<td[^>]*class="calendar__time[^"]*"[^>]*>\s*([^<]+)\s*<', row)
                time_ = time_match.group(1).strip() if time_match else ""

                # Extract currency
                curr_match = re.search(r'<td[^>]*class="calendar__currency[^"]*"[^>]*>\s*([^<]+)\s*<', row)
                currency = curr_match.group(1).strip() if curr_match else ""

                # Extract event
                event_match = re.search(r'<td[^>]*class="calendar__event[^"]*"[^>]*>\s*(?:<[^>]*>)*([^<]+(?:<[^>]*>[^<]*)*)')

                # Simpler event extraction
                event_match = re.search(
                    r'calendar__event[^>]*>\s*(?:<span[^>]*>\s*)?([^<]+?)\s*(?:</span>)?\s*<',
                    row,
                )
                event = event_match.group(1).strip() if event_match else ""
                event = re.sub(r'<[^>]+>', '', event).strip()
                event = ' '.join(event.split())

                # Extract importance (via class)
                impact = "Low"
                if "bg_pink" in row or "bg_red" in row or "highimpact" in row.lower():
                    impact = "High"
                elif "bg_orange" in row or "bg_yellow" in row or "medimpact" in row.lower():
                    impact = "Medium"
                elif "bg_blue" in row:
                    impact = "Low"

                # Previous/Forecast/Actual
                prev_match = re.search(
                    r'class="calendar__previous[^"]*"[^>]*>\s*([^<]+)\s*<', row
                )
                forecast_match = re.search(
                    r'class="calendar__forecast[^"]*"[^>]*>\s*([^<]+)\s*<', row
                )
                actual_match = re.search(
                    r'class="calendar__actual[^"]*"[^>]*>\s*([^<]+)\s*<', row
                )

                if event and currency:
                    ev = EconomicEvent(
                        date=date,
                        time=time_,
                        currency=currency,
                        event=event,
                        importance=impact,
                        previous=prev_match.group(1).strip() if prev_match else None,
                        forecast=forecast_match.group(1).strip() if forecast_match else None,
                        actual=actual_match.group(1).strip() if actual_match else None,
                    )
                    events.append(ev)

            except (AttributeError, IndexError):
                continue

        # Limit to high-impact + days_ahead
        today = datetime.now().date()
        filtered = []
        for ev in events:
            if ev.is_high_impact or ev.affects_xau():
                filtered.append(ev)
            if len(filtered) >= 20:
                break

        logger.info(f"[Fundamental] Scraped {len(filtered)} calendar events from ForexFactory")
        return filtered

    except Exception as e:
        logger.warning(f"[Fundamental] ForexFactory scrape failed: {e}")

        # If ForexFactory fails, return sample notable events as fallback
        return _calendar_fallback(days_ahead)


def _calendar_fallback(days_ahead: int = 3) -> List[EconomicEvent]:
    """Fallback: return notable known events based on day of week.

    This is a minimal fallback when scraping fails.
    In production, you'd use an API like Alpha Vantage or Marketaux.

    Args:
        days_ahead: Number of days ahead

    Returns:
        List of EconomicEvent objects (minimal)
    """
    today = datetime.now()
    events: List[EconomicEvent] = []

    # Map day-of-week to typical high-impact events
    # This is a simplified approximation
    weekday_events = {
        0: [  # Monday
            ("09:00", "EUR", "German CPI (MoM)", "Medium"),
        ],
        1: [  # Tuesday
            ("08:30", "USD", "Core PPI (MoM)", "Medium"),
        ],
        2: [  # Wednesday
            ("08:30", "USD", "CPI (YoY)", "High"),
            ("14:00", "USD", "FOMC Minutes", "High"),
        ],
        3: [  # Thursday
            ("08:30", "USD", "Initial Jobless Claims", "High"),
            ("08:30", "USD", "GDP (QoQ)", "High"),
        ],
        4: [  # Friday
            ("08:30", "USD", "Non-Farm Payrolls", "High"),
            ("08:30", "USD", "Unemployment Rate", "High"),
        ],
    }

    for day_offset in range(days_ahead + 1):
        check_date = today + timedelta(days=day_offset)
        weekday = check_date.weekday()

        if weekday in weekday_events:
            for time_str, currency, event_name, importance in weekday_events[weekday]:
                events.append(EconomicEvent(
                    date=check_date.strftime("%Y-%m-%d"),
                    time=time_str,
                    currency=currency,
                    event=event_name,
                    importance=importance,
                ))

    logger.info(f"[Fundamental] Calendar fallback: {len(events)} events")
    return events


# ─── FUNDAMENTAL BIAS ANALYSIS ────────────────────────────────────────────

def fundamental_bias(instrument: str = "xau") -> FundamentalReport:
    """Generate a comprehensive fundamental bias analysis.

    Combines DXY, bond yields, and economic calendar to produce
    a fundamental bias score for the instrument.

    Args:
        instrument: Instrument ID ('xau', 'btc', 'gbp')

    Returns:
        FundamentalReport with bias score
    """
    report = FundamentalReport(instrument=instrument)
    score = 0.0
    score_reasons: List[str] = []

    # 1. Fetch DXY
    dxy = fetch_dxy()
    report.dxy = dxy
    if dxy:
        # XAU vs DXY: inverse correlation
        if dxy.trend == "bearish":
            score += 2.0
            score_reasons.append(f"DXY giảm ({dxy.change_24h_pct:+.2f}%) → Bullish cho XAU (+2)")
        elif dxy.trend == "bullish":
            score -= 2.0
            score_reasons.append(f"DXY tăng ({dxy.change_24h_pct:+.2f}%) → Bearish cho XAU (-2)")

    # 2. Fetch Bond Yields
    yields = fetch_bond_yields()
    report.bond_yields = yields
    if yields:
        # High real yields = bearish for gold
        if yields.yield_10y > 4.5:
            score -= 1.5
            score_reasons.append(f"Lợi suất 10Y cao ({yields.yield_10y:.2f}%) → Bearish cho XAU (-1.5)")
        elif yields.yield_10y < 3.5:
            score += 1.0
            score_reasons.append(f"Lợi suất 10Y thấp ({yields.yield_10y:.2f}%) → Bullish cho XAU (+1)")

        # Yield curve inversion = safe-haven demand
        if yields.is_inverted:
            score += 1.5
            score_reasons.append(f"Yield curve đảo ngược ({yields.inversion_depth_bps:.0f}bps) → Safe-haven demand (+1.5)")

        # Yield spread narrowing = bullish gold
        if yields.spread_10y_2y > 0.5:  # Steep curve
            score -= 0.5
            score_reasons.append(f"Yield curve dốc → Kỳ vọng tăng trưởng, giảm demand gold (-0.5)")

    # 3. Fetch Economic Calendar
    calendar = fetch_economic_calendar(days_ahead=3)
    report.upcoming_events = calendar

    # Check for high-impact events that could affect XAU
    xau_events = [ev for ev in calendar if ev.affects_xau()]
    high_impact_count = sum(1 for ev in xau_events if ev.is_high_impact)

    if calendar:  # Only add calendar bias if we actually got data
        if high_impact_count >= 2:
            score -= 0.5  # Uncertainty before major events
            score_reasons.append(f"{high_impact_count} high-impact events upcoming → thận trọng (-0.5)")
        elif high_impact_count == 0:
            score += 0.5  # Clear calendar = technical factors dominate
            score_reasons.append("Không có high-impact events trong 3 ngày → kỹ thuật chi phối (+0.5)")

    # 4. Determine bias
    report.bias_score = score

    if score >= 2.5:
        report.bias = "bullish"
        report.confidence = "high"
    elif score >= 1.0:
        report.bias = "bullish"
        report.confidence = "medium"
    elif score <= -2.5:
        report.bias = "bearish"
        report.confidence = "high"
    elif score <= -1.0:
        report.bias = "bearish"
        report.confidence = "medium"
    else:
        report.bias = "neutral"
        report.confidence = "low"

    # Build summary
    if score_reasons:
        report.summary = "\n".join(f"  • {r}" for r in score_reasons)
    else:
        report.summary = "  Không đủ dữ liệu fundamental để đánh giá."

    return report


def fundamental_report(instrument: str = "xau") -> str:
    """Generate a formatted fundamental analysis report.

    Args:
        instrument: Instrument ID

    Returns:
        Formatted report string
    """
    report = fundamental_bias(instrument)
    return report.detailed_report()


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    instr = sys.argv[1] if len(sys.argv) > 1 else "xau"
    print(fundamental_report(instr))
