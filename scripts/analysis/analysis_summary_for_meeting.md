# 社会的孤立・孤独研究 解析進捗まとめ

## 1. 研究目的

本研究では、スマートフォンセンシングデータを用いて、

- 社会的孤立（LSNS）
- 孤独感（UCLA）

を推定・理解することを目的とする。

特に、

- 「孤立しているが孤独ではない」
- 「孤立していないが孤独」

といった discordance に着目し、
行動・移動・接触・スマホ利用との関係性を分析する。

---

# 2. 使用データ

## 2.1 Questionnaire

- UCLA Loneliness Scale
- LSNS-6
- PANAS
- 好奇心尺度
  - 拡散的好奇心
  - 特殊的好奇心
- デモグラ
  - 年齢
  - 性別
  - 婚姻

---

## 2.2 Smartphone sensing

### GPS
- 滞在場所
- 移動距離
- 行動範囲
- location diversity

### Bluetooth
- 周辺接触
- 接触多様性
- repeated_device_ratio

### Activity
- stationary
- walking
- automotive
- active movement

### Screen
- screen ON/OFF
- night screen ratio

### WiFi
- home WiFi ratio
- WiFi entropy

### Network
- WiFi / mobile 切替

### Weather
- 天候
- 気温
- 気圧

### Battery
- 充電頻度
- 夜間充電

### EMA
momentary affect を取得。

---

# 3. 実装済み解析

---

# 3.1 EMA解析

## 実装

- EMA master 作成
- 注意確認問題フィルタ
- EMA session validity 判定

---

## 結果

### negative affect

- stationary_ratio ↑
  → negative affect ↑

- active_movement_ratio ↑
  → negative affect ↓

- night_screen_ratio ↑
  → negative affect ↑

### positive affect

- unique social devices ↑
  → positive affect ↑

- outdoor mobility ↑
  → positive affect ↑

---

# 3.2 Mixed Effects Model

## 実装

within-person change を考慮。

```text
answer_numeric ~ feature + (1 | participant_id)