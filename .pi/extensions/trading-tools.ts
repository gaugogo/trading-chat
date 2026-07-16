/**
 * Pi Trading Tools Extension — 15 Công Cụ Giao Dịch + AI + SMC
 *
 * ── Tính năng ──
 *   • Streaming progress: hiển thị "Đang tải dữ liệu..." khi chạy
 *   • Result caching: cache 30s cho data/signal để tránh gọi Python liên tục
 *   • Metadata trong details: instrument, strategy, thời gian chạy
 *   • promptSnippet/Guidelines: giúp LLM chọn đúng tool
 *   • Signal hủy: tôn trọng abort signal
 *
 * ── Cấu hình ──
 *   TRADE_DISABLE=data,analyze      tắt tool (cách nhau bằng dấu phẩy)
 *   TRADE_ENABLE_ONLY=swing,scalp   chỉ bật các tool này
 *   TRADE_CACHE_TTL=30              cache TTL (giây, mặc định 30)
 *
 * Hỗ trợ: XAUUSD (Gold), BTC/USD (Bitcoin), GBP/USD (Cable)
 * Phong cách: Position · Swing · Day Trade · Scalp · Ichimoku · SMC
 */

import type { ExtensionAPI, ToolExecuteParams } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execSync } from "node:child_process";

const CLI = `${process.cwd()}/trade_cli.py`;
const CACHE_TTL_MS = (parseInt(process.env.TRADE_CACHE_TTL || "30", 10) || 30) * 1000;

// ─── SIMPLE IN-MEMORY CACHE ──────────────────────────────────────────────

const resultCache = new Map<string, { data: string; expiry: number }>();

function cacheGet(key: string): string | undefined {
  const entry = resultCache.get(key);
  if (entry && Date.now() < entry.expiry) return entry.data;
  resultCache.delete(key);
  return undefined;
}

function cacheSet(key: string, data: string) {
  resultCache.set(key, { data, expiry: Date.now() + CACHE_TTL_MS });
  // Evict old entries if cache grows too large
  if (resultCache.size > 100) {
    const oldest = resultCache.entries().next().value;
    if (oldest) resultCache.delete(oldest[0]);
  }
}

// ─── RUN PYTHON WITH STREAMING ────────────────────────────────────────────

async function runWithProgress(
  args: string,
  signal?: AbortSignal,
  onUpdate?: ToolExecuteParams["onUpdate"],
  timeout = 120_000,
): Promise<string> {
  const cacheKey = args;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  onUpdate?.([{ type: "text", text: "⏳ Đang tải dữ liệu thị trường..." }]);

  const cmd = `python3 "${CLI}" ${args}`;
  
  return new Promise<string>((resolve, reject) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    // Forward outer signal to inner controller
    if (signal) {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    try {
      const stdout = execSync(cmd, {
        cwd: process.cwd(),
        timeout,
        encoding: "utf-8",
        maxBuffer: 20 * 1024 * 1024,  // 20MB
        stdio: ["pipe", "pipe", "pipe"],
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const result = stdout.slice(-20_000);
      cacheSet(cacheKey, result);
      resolve(result);
    } catch (err: any) {
      clearTimeout(timeoutId);
      const errorMsg = (err.stdout || "") + (err.stderr ? "\n" + err.stderr : "") || err.message;
      reject(new Error(errorMsg.slice(-5_000)));
    }
  });
}

// ─── CONFIG: Enable/Disable tools ──────────────────────────────────────

const DISABLED_TOOLS = new Set(
  (process.env.TRADE_DISABLE || "").split(",").map(s => s.trim()).filter(Boolean),
);
const ENABLE_ONLY = new Set(
  (process.env.TRADE_ENABLE_ONLY || "").split(",").map(s => s.trim()).filter(Boolean),
);

function isEnabled(name: string): boolean {
  if (ENABLE_ONLY.size > 0) return ENABLE_ONLY.has(name);
  return !DISABLED_TOOLS.has(name);
}

// ─── SHARED PARAM SCHEMA ────────────────────────────────────────────────

const InstrumentParam = Type.Object({
  instrument: Type.Optional(Type.String({
    description: "Công cụ: 'xau' (XAUUSD), 'btc' (BTC/USD), 'gbp' (GBP/USD). Mặc định: xau",
  })),
});

const InstrumentWithQuestion = Type.Object({
  instrument: Type.Optional(Type.String({ description: "Công cụ: 'xau', 'btc', 'gbp'" })),
  question: Type.String({ description: "Câu hỏi cho AI về thị trường hiện tại" }),
});

const RiskParams = Type.Object({
  instrument: Type.Optional(Type.String({ description: "Công cụ: 'xau', 'btc', 'gbp'" })),
  account: Type.Optional(Type.Number({ description: "Số dư tài khoản (USD), mặc định 10000" })),
  risk_pct: Type.Optional(Type.Number({ description: "% rủi ro mỗi lệnh, mặc định 1.0" })),
  entry: Type.Optional(Type.Number({ description: "Giá entry (mặc định: giá hiện tại)" })),
  sl: Type.Optional(Type.Number({ description: "Stop Loss (mặc định: 2x ATR)" })),
  tp: Type.Optional(Type.Number({ description: "Take Profit (mặc định: 4x ATR)" })),
});

// ─── REGISTER TOOL HELPER (optimized) ────────────────────────────────────

function registerTool(
  pi: ExtensionAPI,
  name: string,
  label: string,
  description: string,
  promptSnippet: string | undefined,
  promptGuidelines: string[] | undefined,
  parameters: any,
  buildArgs: (params: any) => string,
  timeout = 120_000,
) {
  if (!isEnabled(name)) return;

  pi.registerTool({
    name,
    label,
    description,
    promptSnippet,
    promptGuidelines,
    parameters,
    async execute(_toolCallId, params, signal, onUpdate, _ctx) {
      const t0 = Date.now();
      const args = buildArgs(params);

      try {
        const text = await runWithProgress(args, signal, onUpdate, timeout);
        return {
          content: [{ type: "text", text }],
          details: {
            instrument: params.instrument || "xau",
            tool: name,
            runtime_ms: Date.now() - t0,
            cached: resultCache.has(args),
          },
        };
      } catch (err: any) {
        return {
          content: [{ type: "text", text: `❌ Lỗi: ${err.message}` }],
          details: { error: true, instrument: params.instrument || "xau", tool: name },
          isError: true,
        };
      }
    },
  });
}

// ─── EXPORT ──────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {

  // ═══════════════════════════════════════════════════════════════════
  //  DATA & GENERAL TOOLS
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_data", "Dữ Liệu Thô",
    "Lấy dữ liệu OHLCV + indicators đa khung thời gian (Daily, 4H, 1H, 15m, 5m). " +
      "Trả về: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, hỗ trợ/kháng cự. KHÔNG có AI. " +
      "Dùng tool này TRƯỚC khi phân tích để có dữ liệu thực tế.",
    "Fetch multi-TF market data with indicators: RSI, MACD, SMA, BB, ATR, key levels",
    ["Dùng trade_data trước để lấy dữ liệu thị trường thực tế cho XAU/BTC/GBP.",
     "Sau đó dùng Pi Agent + DeepSeek để phân tích dữ liệu này."],
    InstrumentParam,
    (p) => `data ${p.instrument || "xau"}`,
    60_000,
  );

  registerTool(pi,
    "trade_signal", "Tín Hiệu Giao Dịch",
    "Tín hiệu tổng hợp đa khung thời gian từ 5 TF (Daily→5m). " +
      "Điểm số có trọng số → STRONG BUY / BUY BIAS / WAIT / SELL BIAS / STRONG SELL.",
    "Get weighted multi-TF signal: BUY/SELL with score",
    undefined,
    InstrumentParam,
    (p) => `signal ${p.instrument || "xau"}`,
    60_000,
  );

  registerTool(pi,
    "trade_analyze", "Phân Tích Giao Dịch",
    "Phân tích kỹ thuật đa khung thời gian chi tiết. " +
      "Báo cáo gồm: xu hướng từng TF, RSI, MACD, BB, EMA, S/R, confluence analysis. " +
      "Dữ liệu thô để Pi Agent tự phân tích — không gọi DeepSeek.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `analyze ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_smc", "Phân Tích SMC",
    "Phân tích Smart Money Concepts: cấu trúc thị trường (BOS/CHoCH), " +
      "order blocks, fair value gaps (FVG), liquidity sweeps. Trên D, 4H, 1H, 15m, 5m.",
    "Run SMC analysis: market structure, order blocks, FVG, liquidity sweeps",
    undefined,
    InstrumentParam,
    (p) => `smc ${p.instrument || "xau"}`,
    120_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  MARKET REGIME + DIVERGENCE + RISK
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_regime", "Phân Tích Market Regime",
    "Phân tích trạng thái thị trường: Trending (UP/DOWN), Ranging, Volatile, Choppy. " +
      "Dùng ADX (trend strength), BB Width (volatility), ATR ratio. " +
      "Kết quả kèm khuyến nghị giao dịch phù hợp với regime.",
    "Detect market regime: trending/ranging/volatile with recommendations",
    ["Dùng trade_regime trước khi trade để biết thị trường đang ở trạng thái nào.",
     "Trending → trend-follow, Ranging → mean-reversion, Volatile → giảm position size."],
    InstrumentParam,
    (p) => `regime ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_divergence", "Phân Tích Divergence",
    "Phát hiện RSI và MACD divergence (Regular & Hidden) trên mọi khung thời gian. " +
      "Regular Bullish/Bearish → đảo chiều. Hidden Bullish/Bearish → tiếp diễn xu hướng. " +
      "Báo cáo kèm độ mạnh tín hiệu.",
    "Detect RSI + MACD divergences: regular/hidden bullish/bearish",
    undefined,
    InstrumentParam,
    (p) => `divergence ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_vp", "Volume Profile",
    "Phân tích Volume Profile với dynamic bins, delta volume (buy/sell pressure). " +
      "POC, Value Area (VAH/VAL), volume imbalance. Trên D/4H/1H/15m/5m.",
    "Volume Profile: POC, VAH/VAL, delta volume, imbalance across TFs",
    undefined,
    InstrumentParam,
    (p) => `volume_profile ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_risk", "Máy Tính Rủi Ro",
    "Tính toán position size, R:R, margin, daily loss limit. " +
      "Nhập: account size, risk%, entry, SL, TP. " +
      "Trả về: lot size, dollar risk, R:R, margin, cảnh báo.",
    "Risk calculator: position sizing, R:R, margin, daily loss limit",
    undefined,
    RiskParams,
    (p) => {
      const instr = p.instrument || "xau";
      const account = p.account || 10000;
      const risk = p.risk_pct || 1.0;
      const entry = p.entry || 0;
      const sl = p.sl || 0;
      const tp = p.tp || 0;
      return `risk ${instr} account=${account} risk_pct=${risk} entry=${entry} sl=${sl} tp=${tp}`;
    },
    60_000,
  );

  registerTool(pi,
    "trade_chat", "Hỏi AI Giao Dịch",
    "Gửi câu hỏi cho DeepSeek AI với context dữ liệu thị trường hiện tại. " +
      "Yêu cầu DEEPSEEK_API_KEY trong .env",
    undefined,
    undefined,
    InstrumentWithQuestion,
    (p) => {
      const instr = p.instrument || "xau";
      const q = (p.question || "Give me a trading plan.").replace(/"/g, '\\"');
      return `chat ${instr} "${q}"`;
    },
    120_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  POSITION TRADING — Hold weeks to months
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_position", "Phân Tích Position",
    "Position Trading: Daily/4H/1H. SMA50/200 Golden/Death Cross, macro trend, " +
      "HH/HL structure, monthly/quarterly/yearly S/R, pivot points, ATR-based SL/TP.",
    "Long-term position analysis: SMA200, macro trend, monthly S/R",
    undefined,
    InstrumentParam,
    (p) => `position ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_position_signal", "Tín Hiệu Position",
    "Tín hiệu position nhanh: SMA200, macro trend strength. " +
      "STRONG BUY / BUY BIAS / WAIT / SELL BIAS / STRONG SELL.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `position_signal ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  SWING TRADING — Hold 1-5 days
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_swing", "Phân Tích Swing",
    "Swing Trading: Daily/4H/1H. Fibonacci retracement/extension, key S/R, " +
      "SMA alignment, entry zone, SL/TP với ATR, Risk:Reward.",
    "Swing analysis: Fibonacci, key S/R, entry zone with R:R",
    undefined,
    InstrumentParam,
    (p) => `swing ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_swing_signal", "Tín Hiệu Swing",
    "Tín hiệu swing nhanh từ Daily + 4H. " +
      "STRONG BUY / BUY BIAS / WAIT / SELL BIAS / STRONG SELL.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `swing_signal ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  DAY TRADING — Intraday, close EOD
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_daytrade", "Phân Tích Day Trade",
    "Day Trading: 4H/1H/15m/5m. VWAP, Volume Profile POC/VAH/VAL, " +
      "Opening Range, EMA ribbon, engulfing patterns, entry/exit zones.",
    "Intraday analysis: VWAP, Volume Profile, Opening Range",
    undefined,
    InstrumentParam,
    (p) => `daytrade ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_daytrade_signal", "Tín Hiệu Day Trade",
    "Tín hiệu day trade nhanh từ 4H/1H/15m. BUY / SELL / WAIT.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `daytrade_signal ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  SCALPING — 5-15 minutes
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_scalp", "Phân Tích Scalping",
    "Scalping: 1H/15m/5m. EMA5/9 cross, RSI7, momentum, " +
      "entry/exit zones, SL/TP, R:R.",
    "Scalp analysis: fast EMA cross, RSI7, tight SL/TP",
    undefined,
    InstrumentParam,
    (p) => `scalp ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_scalp_signal", "Tín Hiệu Scalping",
    "Tín hiệu scalping nhanh từ 1H/15m/5m. BUY / SELL / WAIT.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `scalp_signal ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  ICHIMOKU — Cloud analysis
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_ichimoku", "Phân Tích Ichimoku",
    "Ichimoku Kinko Hyo: Tenkan/Kijun, Kumo (cloud), TK Cross, Chikou Span. " +
      "D/4H/1H/15m. Cloud breakout, Kumo twist, entry zones.",
    "Ichimoku cloud analysis: Kumo, TK Cross, Chikou",
    undefined,
    InstrumentParam,
    (p) => `ichimoku ${p.instrument || "xau"}`,
    120_000,
  );

  registerTool(pi,
    "trade_ichimoku_signal", "Tín Hiệu Ichimoku",
    "Tín hiệu ichimoku nhanh: cloud position, TK cross. " +
      "STRONG BUY / BUY BIAS / NEUTRAL / SELL BIAS / STRONG SELL.",
    undefined,
    undefined,
    InstrumentParam,
    (p) => `ichimoku_signal ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  NEW: FUNDAMENTAL ANALYSIS — DXY, bond yields, economic calendar
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_fundamental", "Phân Tích Cơ Bản",
    "Phân tích fundamental: DXY (US Dollar Index), US Treasury yields (10Y/2Y), " +
      "yield spread (inversion signal), economic calendar events. " +
      "Đánh giá tác động lên XAU (tương quan nghịch DXY, safe-haven khi yield đảo ngược). " +
      "Kết quả: Fundamental Bias score + confidence level.",
    "Fundamental analysis: DXY, bond yields, economic calendar, XAU correlation",
    ["Dùng trade_fundamental trước trade để kiểm tra bias từ vĩ mô.",
     "DXY tăng → bearish XAU, DXY giảm → bullish XAU (tương quan nghịch).",
     "Yield curve đảo ngược (10Y < 2Y) → recession signal → XAU hưởng lợi safe-haven."],
    InstrumentParam,
    (p) => `fundamental ${p.instrument || "xau"}`,
    60_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  NEW: LIVE PRICE STREAM — real-time prices
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_live", "Giá Trực Tiếp",
    "Giá thị trường trực tiếp: BTC/ETH (Binance WebSocket real-time), " +
      "XAU/GBP (Yahoo Finance polling 30s). " +
      "Hiển thị: giá hiện tại, thay đổi 24h (% + màu), nguồn dữ liệu, độ trễ.",
    "Live market prices: crypto (WebSocket) + forex/commodity (polling)",
    ["Dùng trade_live để kiểm tra giá real-time trước khi phân tích.",
     "BTC/ETH: real-time WebSocket từ Binance, độ trễ <1s.",
     "XAU/GBP: polling 30s từ Yahoo Finance."],
    Type.Object({
      instrument: Type.Optional(Type.String({
        description: "Công cụ: 'btc', 'eth', 'xau', 'gbp'. Mặc định: all",
      })),
    }),
    (p) => `live ${p.instrument || ""}`,
    30_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  NEW: CHART GENERATION — visual chart with indicators
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_chart", "Biểu Đồ Kỹ Thuật",
    "Tạo biểu đồ kỹ thuật trực quan: nến OHLCV, EMA/SMA/BB indicators, " +
      "RSI + MACD subplots, Volume bars. " +
      "Hỗ trợ annotation entry/SL/TP, S/R levels, SMC zones. " +
      "Kết quả: link đến file PNG trong thư mục charts/.",
    "Generate technical chart: candlesticks, indicators, annotations",
    ["Dùng trade_chart để có hình ảnh trực quan cho phân tích.",
     "Chart được lưu trong thư mục charts/ dưới dạng PNG.",
     "Kết hợp với trade_signal để thấy entry/SL/TP trên chart."],
    Type.Object({
      instrument: Type.Optional(Type.String({ description: "Công cụ: 'xau', 'btc', 'gbp'" })),
      timeframe: Type.Optional(Type.String({ description: "Khung thời gian: 'Daily', '4H', '1H', '15m', '5m'. Mặc định: Daily" })),
    }),
    (p) => `chart ${p.instrument || "xau"} ${p.timeframe || "Daily"}`,
    120_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  NEW: DEEPSEEK EDUCATIONAL ANALYSIS — structured learning
  // ═══════════════════════════════════════════════════════════════════

  registerTool(pi,
    "trade_learn", "Học Giao Dịch",
    "Phân tích giáo dục có cấu trúc: DeepSeek phân tích + Pi Agent follow-up. " +
      "Gồm: bài học hôm nay, sai lầm cần tránh, kiến thức kỹ thuật, checklist tự kiểm tra. " +
      "Chọn level: 'beginner' (người mới), 'intermediate' (trung cấp), 'advanced' (nâng cao). " +
      "Yêu cầu DEEPSEEK_API_KEY trong .env",
    "Educational analysis: DeepSeek + Pi Agent multi-turn learning",
    ["Dùng trade_learn để vừa phân tích vừa học.",
     "Chọn level phù hợp: beginner → intermediate → advanced.",
     "Sau khi DeepSeek trả lời, Pi Agent sẽ hỏi follow-up để kiểm tra kiến thức."],
    Type.Object({
      instrument: Type.Optional(Type.String({ description: "Công cụ: 'xau', 'btc', 'gbp'" })),
      level: Type.Optional(Type.String({ description: "Trình độ: 'beginner', 'intermediate', 'advanced'. Mặc định: intermediate" })),
      question: Type.Optional(Type.String({ description: "Câu hỏi tùy chỉnh (nếu có)" })),
    }),
    (p) => {
      const instr = p.instrument || "xau";
      const level = p.level || "intermediate";
      const q = p.question ? ` "${p.question.replace(/"/g, '\\"')}"` : "";
      return `learn ${instr} ${level}${q}`;
    },
    120_000,
  );

  // ═══════════════════════════════════════════════════════════════════
  //  STARTUP
  // ═══════════════════════════════════════════════════════════════════

  pi.on("session_start", async (_event, ctx) => {
    const disabled = process.env.TRADE_DISABLE || "";
    const only = process.env.TRADE_ENABLE_ONLY || "";
    let msg = "📊 Trading tools: · data · signal · fundamental · live · chart · learn · position · swing · daytrade · scalp · smc · regime · divergence · risk · vp";
    if (disabled) msg += `  [disabled: ${disabled}]`;
    if (only) msg += `  [only: ${only}]`;
    ctx.ui.notify(msg, "info");
  });
}
