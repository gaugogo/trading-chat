# 📚 Kiến Thức Giao Dịch — Glossary

> Tài liệu giải thích các indicator, pattern, và khái niệm được sử dụng trong Trading Tools.
> Mục đích: giúp người mới học và hiểu reasoning đằng sau mỗi tín hiệu.

---

## 📈 Chỉ Báo Kỹ Thuật (Technical Indicators)

### 1. SMA (Simple Moving Average)
- **Định nghĩa**: Trung bình cộng giá đóng cửa trong N phiên.
- **Thông số**: SMA 20 (ngắn hạn), SMA 50 (trung hạn), SMA 200 (dài hạn).
- **Ý nghĩa**:
  - Giá trên SMA → xu hướng tăng.
  - Giá dưới SMA → xu hướng giảm.
  - SMA20 cắt lên SMA50 → "Golden Cross" (tín hiệu mua).
  - SMA50 cắt xuống SMA20 → "Death Cross" (tín hiệu bán).
- **Hạn chế**: Chậm, phản ứng trễ với giá mới.

### 2. EMA (Exponential Moving Average)
- **Định nghĩa**: Trung bình động có trọng số, ưu tiên giá gần nhất.
- **Thông số**: EMA 9 (rất ngắn), EMA 21 (ngắn hạn).
- **So với SMA**: EMA phản ứng nhanh hơn với biến động giá, phù hợp cho trading ngắn hạn.
- **Cách dùng**: EMA9 > EMA21 → bullish; EMA9 < EMA21 → bearish.

### 3. RSI (Relative Strength Index)
- **Định nghĩa**: Đo tốc độ và biên độ thay đổi giá (0-100).
- **Thông số**: RSI 14 (mặc định), RSI 7 (scalping nhanh).
- **Ngưỡng**:
  - RSI > 70 → Quá mua (overbought), có thể đảo chiều giảm.
  - RSI < 30 → Quá bán (oversold), có thể đảo chiều tăng.
  - RSI 50 → Ngưỡng phân định xu hướng.
- **Divergence** (xem mục riêng bên dưới).

### 4. MACD (Moving Average Convergence Divergence)
- **Cấu tạo**: 3 thành phần — MACD line, Signal line, Histogram.
- **Thông số**: MACD(12, 26, 9) — fast 12, slow 26, signal 9.
- **Cách đọc**:
  - MACD > Signal → bullish momentum.
  - MACD < Signal → bearish momentum.
  - Histogram dương và tăng → momentum mạnh lên.
  - Histogram âm và giảm → momentum yếu đi.
- **Centerline cross**: MACD > 0 → bullish; < 0 → bearish.

### 5. Bollinger Bands (BB)
- **Cấu tạo**: Middle band (SMA 20) + Upper/Lower bands (2 độ lệch chuẩn).
- **Ý nghĩa**:
  - Giá chạm Upper band → quá mua, có thể đảo chiều.
  - Giá chạm Lower band → quá bán, có thể đảo chiều.
  - Bands mở rộng → biến động tăng.
  - Bands thu hẹp → biến động giảm (nén, sắp breakout).
- **BB Width** = (Upper - Lower) / Middle — đo biến động tương đối.

### 6. ATR (Average True Range)
- **Định nghĩa**: Đo biến động trung bình trong N phiên.
- **Công thức**: ATR = EMA của True Range (High - Low, |High - Close_prev|, |Low - Close_prev|).
- **Cách dùng**:
  - Đặt Stop Loss: 1-3x ATR tùy strategy.
  - Đặt Take Profit: 2-5x ATR.
  - ATR tăng → biến động tăng.
  - ATR giảm → biến động giảm.

### 7. ADX (Average Directional Index)
- **Định nghĩa**: Đo độ mạnh của xu hướng (không phải hướng).
- **Thông số**: ADX 14.
- **Ngưỡng**:
  - ADX ≥ 30 → xu hướng mạnh.
  - ADX 20-30 → xu hướng vừa.
  - ADX < 20 → thị trường đi ngang (ranging).
- **+DI/-DI**: +DI > -DI → bullish; -DI > +DI → bearish.

### 8. Volume Profile
- **Định nghĩa**: Phân bố khối lượng giao dịch theo mức giá.
- **Thành phần**:
  - **POC** (Point of Control): Mức giá có khối lượng giao dịch lớn nhất.
  - **VA** (Value Area): Vùng giá chứa 70% tổng khối lượng (VAH = đỉnh, VAL = đáy).
  - **Delta Volume**: Buy volume - Sell volume (dương = bullish, âm = bearish).
- **Ý nghĩa**:
  - POC là vùng giá "cân bằng" — nơi người mua và người bán đồng thuận.
  - Giá trên VA → bullish bias.
  - Giá dưới VA → bearish bias.
  - Volume gap → giá có thể di chuyển nhanh qua vùng này.

---

## 🔄 Divergence

### Regular Divergence (Đảo chiều)
| Loại | Giá | Indicator | Ý nghĩa |
|------|-----|-----------|---------|
| **Bullish** | Đáy thấp hơn | Đáy cao hơn | Đà giảm yếu → sắp tăng |
| **Bearish** | Đỉnh cao hơn | Đỉnh thấp hơn | Đà tăng yếu → sắp giảm |

### Hidden Divergence (Tiếp diễn)
| Loại | Giá | Indicator | Ý nghĩa |
|------|-----|-----------|---------|
| **Bullish** | Đáy cao hơn | Đáy thấp hơn | Xu hướng tăng còn mạnh |
| **Bearish** | Đỉnh thấp hơn | Đỉnh cao hơn | Xu hướng giảm còn mạnh |

- **RSI Divergence**: Phổ biến nhất, dùng RSI 14.
- **MACD Divergence**: Dùng MACD line.
- **Độ mạnh**: Regular > Hidden; nhiều TF cùng divergence → tín hiệu mạnh.

---

## 🌡️ Market Regime

| Regime | ADX | BB Width | ATR | Chiến lược phù hợp |
|--------|-----|----------|-----|-------------------|
| **Strong Uptrend** | ≥ 30 | Normal | Ổn định | Trend-follow LONG |
| **Uptrend** | 20-30 | Normal | Ổn định | Trend-follow LONG |
| **Ranging** | < 20 | Thấp | Thấp | Mean-reversion |
| **Downtrend** | 20-30 | Normal | Ổn định | Trend-follow SHORT |
| **Strong Downtrend** | ≥ 30 | Normal | Ổn định | Trend-follow SHORT |
| **Volatile** | < 20 | Cao | Cao | Chờ/giảm size |
| **Choppy** | < 15 | Cao | Bất thường | KHÔNG trade |

---

## 🧠 SMC (Smart Money Concepts)

### Cấu trúc thị trường
- **HH** (Higher High): Đỉnh cao hơn đỉnh trước → uptrend.
- **HL** (Higher Low): Đáy cao hơn đáy trước → uptrend.
- **LH** (Lower High): Đỉnh thấp hơn đỉnh trước → downtrend.
- **LL** (Lower Low): Đáy thấp hơn đáy trước → downtrend.
- **BOS** (Break of Structure): Giá phá vỡ đỉnh/đáy gần nhất → xu hướng tiếp diễn.
- **CHoCH** (Change of Character): Thay đổi cấu trúc — HH→LH hoặc HL→LL → đảo chiều.

### Order Block (OB)
- **Định nghĩa**: Nến/zone giá cuối cùng trước khi giá impulsively di chuyển.
- **Bullish OB**: Nến giảm cuối cùng trước một đợt tăng mạnh.
- **Bearish OB**: Nến tăng cuối cùng trước một đợt giảm mạnh.
- **Cách dùng**: Entry tại OB, SL dưới/trên OB.

### Fair Value Gap (FVG)
- **Định nghĩa**: Khoảng trống giá giữa 3 nến liên tiếp (do mất cân bằng lệnh).
- **Bullish FVG**: Low(candle_i) > High(candle_i-2) — gap lên.
- **Bearish FVG**: High(candle_i) < Low(candle_i-2) — gap xuống.
- **Cách dùng**: FVG thường được "fill" sau đó — chờ giá quay lại FVG để entry.

### Breaker Block
- **Định nghĩa**: OB cũ bị phá vỡ, sau đó đổi vai trò thành kháng cự/hỗ trợ mới.
- **Cơ chế**: OB thất bại → smart money đã đổi hướng → breaker block hình thành.
- **Cách dùng**: Entry tại breaker block theo hướng mới.

### Mitigation Block
- **Định nghĩa**: FVG đã được lấp đầy (giá quay lại và fill gap).
- **Ý nghĩa**: Khi FVG được fill, động lượng giảm — có thể đảo chiều.

### Reclaimed OB
- **Định nghĩa**: OB bị xâm phạm nhưng giá đóng cửa trở lại bên ngoài.
- **Ý nghĩa**: OB vẫn còn hiệu lực — tín hiệu mạnh mẽ rằng OB sẽ giữ.

---

## 📐 Risk Management

### Position Sizing
```
Position Size = (Account × Risk%) / (SL Pips × Pip Value)
```
- **1% Rule**: Không rủi ro quá 1% tài khoản mỗi lệnh.
- **2% Rule**: Tổng rủi ro tất cả lệnh đang mở ≤ 2%.

### R:R (Risk:Reward)
- **Tối thiểu**: 1:2 (risk 1 để kiếm 2).
- **Công thức**: R:R = (Entry - TP) / (Entry - SL) (cho LONG).
- **Win rate cần**: WR = 1 / (1 + R:R).

### Daily Loss Limit
- **Khuyến nghị**: Giới hạn thua lỗ 3% tài khoản/ngày.
- **Khi chạm limit**: Dừng giao dịch, xem lại chiến lược.

### Position Types
- **Position**: Giữ tuần-tháng. SL rộng (3x ATR Daily), TP xa (8-12x ATR).
- **Swing**: Giữ 1-5 ngày. SL 2x ATR 4H, TP 5x ATR.
- **Day Trade**: Giữ trong ngày. SL 1.5x ATR 15m, TP 3.5x ATR.
- **Scalp**: Giữ 5-15 phút. SL 1x ATR 5m, TP 2.5x ATR.

---

## 📊 Chiến Thuật Giao Dịch

### Trend Following
- **Khi nào**: Thị trường trending (ADX > 25).
- **Cách làm**: Mua pullback trong uptrend, bán pullback trong downtrend.
- **Entry**: Chờ giá chạm EMA21/SMA20 trong xu hướng.
- **Stop Loss**: Dưới swing low gần nhất (LONG) / trên swing high (SHORT).

### Mean Reversion
- **Khi nào**: Thị trường ranging (ADX < 20, BB thu hẹp).
- **Cách làm**: Mua gần BB Lower, bán gần BB Upper.
- **Stop Loss**: Ngoài BB band.
- **Rủi ro**: Range có thể breakout bất kỳ lúc nào.

### Breakout Trading
- **Khi nào**: Khung giá nén (BB thu hẹp, volume thấp).
- **Cách làm**: Đợi breakout khỏi range với volume.
- **Entry**: Khi giá đóng cửa ngoài range + xác nhận volume.
- **Stop Loss**: Trong range.

---

## 🔗 Liên Kết

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) — Kế hoạch phát triển
- `core.py` — Shared utilities
- `regime.py` — Market regime detection
- `divergence.py` — RSI/MACD divergence
- `risk_calculator.py` — Risk management
- `volume_profile.py` — Volume Profile analysis
- `smc.py` — Smart Money Concepts
- `scoring_engine.py` — Unified scoring system
