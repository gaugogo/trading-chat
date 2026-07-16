# 📊 Trading Tools — Kế Hoạch Cải Tiến

> **Mục tiêu**: Nâng cao độ chính xác phân tích, chất lượng dữ liệu, và đảm bảo đầu ra giúp user học được từ DeepSeek + Pi Agent.
>
> **Tiến độ**: ✅ Tuần 1 + Tuần 2 đã hoàn tất (82 tests, 15 tools)

---

## 🎯 NHÓM 1: NÂNG CAO CHẤT LƯỢNG DỮ LIỆU

| # | Vấn đề hiện tại | Giải pháp | Ưu tiên | Trạng thái |
|---|---|---|---|---|
| 1.1 | **1 nguồn dữ liệu** (Yahoo Finance), dễ lỗi/giới hạn rate limit | Thêm **fallback provider** (Alpha Vantage). Implement retry + circuit breaker. | 🔴 CAO | ✅ `data_provider.py` |
| 1.2 | **Cache TTL tĩnh**: Daily 2h, 4H 1h, 1H 30ph, 15m 15ph, 5m 5ph | **Adaptive cache**: tự động giảm TTL khi thị trường mở cửa, tăng khi đóng. Session-based cache tránh stale data. | 🔴 CAO | ✅ `data_provider.py` |
| 1.3 | **Spot price scraping** từ Investing.com — fragile, dễ vỡ khi site đổi layout | Thêm **multi-source spot**: Yahoo Finance quote fallback, validate cross-source. | 🟡 MED | ✅ `data_provider.py` (Yahoo fallback) + `core.py` (cross-validation) |
| 1.4 | **Không có real-time tick data** cho scalping | WebSocket stream từ Binance (BTC/ETH) + Yahoo polling (XAU/GBP). | 🟢 LOW | ✅ `stream.py` |
| 1.5 | **Không có dữ liệu fundamental** | Economic calendar (ForexFactory API/scraping), bond yields, DXY index, fundamental bias scoring. | 🟡 MED | ✅ `fundamental.py` |

---

## 🎯 NHÓM 2: CẢI THIỆN ĐỘ CHÍNH XÁC PHÂN TÍCH

| # | Vấn đề hiện tại | Giải pháp | Ưu tiên | Trạng thái |
|---|---|---|---|---|
| 2.1 | **Scoring system rời rạc**: position/swing/daytrade/scalp mỗi cái có cách tính score khác nhau, không nhất quán | Tạo **unified scoring engine**: 1 class tính điểm từ indicators, hỗ trợ weight tùy chỉnh qua config. | 🔴 CAO | ✅ `scoring_engine.py` |
| 2.2 | **Không có backtesting** để validate độ chính xác của signal | **Backtest module**: chạy signal trên historical data, tính win rate, profit factor, Sharpe ratio. Output báo cáo cho user học từ quá khứ. | 🔴 CAO | ✅ `backtest.py` |
| 2.3 | **Không tracked performance** của signal đã generate | **Signal journal (SQLite)**: lưu mọi signal + outcome. Cho phép xem accuracy per style/TF. | 🟡 MED | ✅ `journal.py` |
| 2.4 | **Divergence detection** bị thiếu hoàn toàn | Thêm **RSI divergence** (bullish/bearish) + **MACD divergence** tự động. | 🟡 MED | ✅ `divergence.py` |
| 2.5 | **Volume Profile** quá đơn giản (chỉ 10 bins) | Cải thiện Volume Profile với **dynamic bins** + delta volume (buy vs sell pressure). | 🟡 MED | ✅ `volume_profile.py` |
| 2.6 | **SMC** thiếu nhiều khái niệm nâng cao | Thêm **Breaker Block**, **Mitigation Block**, **Reclaimed OB** vào SMC module. | 🟢 LOW | ✅ `smc.py` (breaker + mitigation + reclaimed) |
| 2.7 | **Không có Market Regime detection** | Thêm **regime filter** (trending/ranging/volatile) dựa trên ADX + ATR + BB Width. | 🟡 MED | ✅ `regime.py` |

---

## 🎯 NHÓM 3: ĐẢM BẢO ĐẦU RA CHẤT LƯỢNG — HỌC TẬP

| # | Vấn đề hiện tại | Giải pháp | Ưu tiên | Trạng thái |
|---|---|---|---|---|
| 3.1 | **DeepSeek prompt thô sơ**: chỉ "trả lời tiếng Việt", không giải thích reasoning | **Structured educational prompt**: yêu cầu AI giải thích "TẠI SAO" cho mỗi quyết định, kèm checklist học tập. | 🔴 CAO | ✅ `education/prompt_templates.py` |
| 3.2 | **Output format** không nhất quán giữa các module | **Markdown Report Template** chuẩn: mỗi report có section "Bài học hôm nay", "Sai lầm cần tránh", "Kiến thức kỹ thuật". | 🔴 CAO | ✅ `education/prompt_templates.py` |
| 3.3 | **Không có glossary/kiến thức** cho người mới | Thêm `knowledge.md` chứa giải thích từng indicator, pattern. Link vào report output. | 🟡 MED | ✅ `education/knowledge.md` |
| 3.4 | **Pi Agent không tận dụng multi-turn reasoning** | Thiết kế **conversation flow**: sau DeepSeek phân tích, Pi Agent follow-up kiểm tra kiến thức user. | 🟡 MED | ✅ `education/conversation_flow.py` |
| 3.5 | **Không có chart visualization** để học | **Chart generation** (matplotlib): vẽ chart + entry/SL/TP annotations, SMC zones, export PNG. | 🟢 LOW | ✅ `chart_generator.py` |
| 3.6 | **Risk management education** bị thiếu | **Risk Calculator tool**: user nhập account size → tính position size, max contracts, daily loss limit. | 🟡 MED | ✅ `risk_calculator.py` |

---

## 🎯 NHÓM 4: KIẾN TRÚC & TỔ CHỨC CODE

| # | Vấn đề hiện tại | Giải pháp | Ưu tiên | Trạng thái |
|---|---|---|---|---|
| 4.1 | **2 phiên bản extension** (global 2 tools vs local 15 tools) gây rối | **Hợp nhất**: giữ 1 bản 15 tools, dùng config (`TRADE_DISABLE`/`TRADE_ENABLE_ONLY`) để bật/tắt. | 🔴 CAO | ✅ `.pi/extensions/trading-tools.ts` |
| 4.2 | **Code duplicated**: `determine_trend`, `fmt_price`, `calculate_indicators` trùng khắp module | Tạo `core.py` chứa shared functions. Refactor toàn bộ module import từ core. | 🔴 CAO | ✅ `core.py` + refactor all |
| 4.3 | **Không có logging/error handling** | Thêm Python `logging`: log errors, API failures, cache hits/misses. Gửi alert khi data source down. | 🟡 MED | ✅ `core.py` (logging setup + calls) + `data_provider.py` + các module khác |
| 4.4 | **Config hardcoded** trong từng module (ATR multipliers, TF weights, cache TTLs) | Tạo `config.yaml` chứa tất cả parameters. Load 1 lần, override qua env vars. | 🟡 MED | ✅ `config.yaml` |
| 4.5 | **Không có automated tests** | `pytest` tests cho: indicator calculation, signal logic, data fetching (mock). | 🟡 MED | ✅ 191 tests (8 test files) |
| 4.6 | **Yahoo Finance API** là hard dependency | `data_provider.py` abstract base class. Yahoo là 1 implementation, dễ swap sang provider khác. | 🟡 MED | ✅ `data_provider.py` |
| 4.7 | **Hardcoded DeepSeek model** trong `instruments.py` | Cho phép override model qua `DEEPSEEK_MODEL` env var hoặc CLI flag `--model`. | 🟢 LOW | ✅ `instruments.py` (env override DEEPSEEK_MODEL, DEEPSEEK_THINKING) |

---

## 📋 LỘ TRÌNH TRIỂN KHAI

### ✅ Tuần 1: Data Quality + Code Refactor (Nền tảng) — HOÀN TẤT

```
├── 1.1  DataProvider abstract + multi-source fallback      ✅ data_provider.py
├── 1.2  Adaptive cache system                              ✅ data_provider.py
├── 4.1  Hợp nhất extension (15 tools + config)             ✅ .pi/extensions/trading-tools.ts
├── 4.2  Tạo core.py, refactor shared functions             ✅ core.py + all modules
├── 4.4  config.yaml                                        ✅ config.yaml
├── 4.5  Thêm pytest test suite (82 tests)                  ✅ tests/
└── 4.6  Abstract data provider pattern                     ✅ data_provider.py
```

### ✅ Tuần 2: Accuracy + Learning Output — HOÀN TẤT

```
├── 2.1  Unified scoring engine                             ✅ scoring_engine.py
├── 2.2  Backtest module                                    ✅ backtest.py
├── 2.3  Signal journal (SQLite)                            ✅ journal.py
├── 3.1  Structured educational DeepSeek prompt             ✅ education/prompt_templates.py
└── 3.2  Standardized Markdown report template              ✅ education/prompt_templates.py
```

### ✅ Tuần 3: Advanced Features — HOÀN TẤT
```
├── 1.5  Economic calendar + DXY/bond yields correlation    ✅ fundamental.py
├── 2.4  RSI + MACD divergence detection                    ✅ divergence.py
├── 2.5  Improved Volume Profile (dynamic bins, delta)      ✅ volume_profile.py
├── 2.7  Market regime filter (ADX-based)                   ✅ regime.py
├── 3.4  Pi Agent multi-turn reasoning flow                 ✅ education/conversation_flow.py
└── 3.6  Risk Calculator tool                               ✅ risk_calculator.py
```

### ✅ Tuần 4: Polish + Hardening — HOÀN TẤT
```
├── 1.3  Multi-source spot price validation                 ✅ data_provider.py + core.py
├── 1.4  Real-time WebSocket data (low TF)                  ✅ stream.py
├── 2.6  Advanced SMC (Breaker/Mitigation/Reclaimed blocks) ✅ smc.py
├── 3.5  Chart generation with matplotlib                   ✅ chart_generator.py
├── 4.3  Structured logging system                          ✅ core.py + các module
├── 4.7  Override DeepSeek model via env/CLI                ✅ instruments.py
└── 3.3  knowledge.md glossary                              ✅ education/knowledge.md
```

---

## 🚀 ƯU TIÊN BẮT ĐẦU (Top 3 Impact)

| Thứ tự | Mục | Lý do | Trạng thái |
|---|---|---|---|
| **1** | 4.1 + 4.2 — Hợp nhất extension + Tạo `core.py` | Nền tảng cho mọi thay đổi sau. | ✅ |
| **2** | 3.1 + 3.2 — Structured prompt + Report template | Cải thiện ngay output cho user. | ✅ |
| **3** | 2.2 — Backtest module | Đo lường được accuracy. | ✅ |

---

## 📁 CẤU TRÚC THƯ MỤC HIỆN TẠI

```
trading-chat/
├── core.py                  # Shared: indicators, trend, formatting, data fetching
├── data_provider.py         # Abstract base + Yahoo/AlphaVantage + circuit breaker + adaptive cache
├── scoring_engine.py        # Unified weighted scoring system (Bias enum, confidence)
├── backtest.py              # Historical backtesting runner (3 signal generators)
├── journal.py               # SQLite signal journal (accuracy tracking)
├── analysis.py              # Multi-TF analysis + DeepSeek integration (refactored, imports core)
├── mcp_server.py            # Data output formatter (refactored, imports core)
├── trade_cli.py             # CLI entry point (19+ modes)
├── config.yaml              # All tunable parameters
├── regime.py                # Market regime detection (ADX + BB Width + ATR)
├── divergence.py            # RSI + MACD divergence (regular & hidden)
├── risk_calculator.py       # Position sizing, R:R, margin, daily loss limit
├── volume_profile.py        # Dynamic bins, delta volume, imbalance
├── stream.py                # Real-time WebSocket (Binance) + polling (Yahoo)
├── fundamental.py           # DXY, bond yields, economic calendar, fundamental bias
├── chart_generator.py       # matplotlib OHLCV chart with indicators + annotations
│
├── position.py              # Position trading (imports core)
├── swing.py                 # Swing trading (imports core)
├── daytrade.py              # Day trading (imports core)
├── scalp.py                 # Scalping (imports core)
├── ichimoku.py              # Ichimoku (imports core)
├── smc.py                   # Smart Money Concepts (standalone)
├── instruments.py           # Instrument definitions (XAU, BTC, GBP)
│
├── education/
│   ├── prompt_templates.py  # Educational DeepSeek prompt templates
│   ├── knowledge.md         # Glossary & kiến thức giao dịch
│   └── conversation_flow.py # Multi-turn reasoning flow (3 levels)
│
├── tests/
│   ├── test_core.py              # 17 tests
│   ├── test_data_provider.py     # 28 tests
│   ├── test_scoring_engine.py    # 20 tests
│   ├── test_journal.py           # 17 tests
│   ├── test_regime.py            # 18 tests: market regime
│   ├── test_divergence.py        # 19 tests: RSI/MACD divergence
│   ├── test_risk_calculator.py   # 18 tests: risk management
│   ├── test_volume_profile.py    # 20 tests: volume profile
│   ├── test_stream.py            # 31 tests: live streaming
│   ├── test_fundamental.py       # 56 tests: fundamental analysis
│   ├── test_conversation_flow.py # 43 tests: multi-turn learning
│   └── test_chart_generator.py   # 27 tests: chart visualization
│
├── cache/                   # OHLCV cache (session-aware: open/closed)
├── data/                    # SQLite database (signal_journal.db)
├── charts/                  # Generated chart PNG files
├── .pi/extensions/
│   └── trading-tools.ts     # Pi Agent extension — 23 tools + config enable/disable
│
└── requirements.txt         # requests, pandas, numpy, ta, pyyaml, pytest, matplotlib, websocket-client
```

## 📊 HIỆN TRẠNG

| Metric | Giá trị |
|--------|---------|
| **Tổng files** | 32+ Python modules + knowledge.md + conversation flow |
| **Tests** | 321 tests, tất cả đều pass |
| **Extension tools** | 23 (configurable via TRADE_DISABLE / TRADE_ENABLE_ONLY) |
| **Data providers** | 2 (Yahoo Finance + Alpha Vantage) |
| **Cache** | Adaptive theo phiên giao dịch (forex/crypto/stock) |
| **Backtest strategies** | 3 (swing, position, daytrade) |
| **Journal DB** | SQLite với accuracy tracking |
| **Regime detection** | ADX + BB Width + ATR ratio → Trending/Ranging/Volatile |
| **Divergence detection** | RSI + MACD regular/hidden bullish/bearish |
| **Risk calculator** | Position size, R:R, margin, daily loss limit |
| **Real-time stream** | Binance WebSocket (BTC/ETH) + Yahoo polling (XAU/GBP) |
| **Fundamental analysis** | DXY, US 10Y/2Y yields, yield spread, economic calendar |
| **Chart generation** | matplotlib: OHLCV + indicators + annotations + SMC zones |
| **Conversation flow** | 3-level progressive learning, knowledge checks, follow-up Q&A |
| **Config** | `config.yaml` + env var override |
