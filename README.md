# isolation-loneliness-discordance

社会的孤立 (LSNS-6) と孤独感 (UCLA-3) の **不協和 (discordance)** を、スマートフォン由来のマルチモーダルセンシングデータから検証する研究プロジェクト。

## 研究の問い

「**孤立**(社会ネットワークが乏しい)と **孤独**(主観的に孤独を感じる)は概念的に独立しており、両者が一致しない人々(不協和タイプ)が存在する」という枠組みのもとで、

- **discordance タイプ**(`isolated_lonely` / `isolated_not_lonely` / `not_isolated_lonely` / `not_isolated_not_lonely`)を客観行動データで分離できるか?
- pre/post 期間での主観評価の変化と行動指標の変化は対応するか?

を主問いとする。

## 実験設計

| 項目 | 値 |
|---|---|
| 参加者 | 14 名 (高齢者中心) |
| 期間 | 14 日間 × pre / post の 2 phase |
| 主観評価 | LSNS-6 (社会ネットワーク) / UCLA-3 (孤独感) / GAD-7 / K10 / PSS / TIPI / 多次元好奇心尺度 |
| 客観評価 | AWARE Framework (iOS) で連続収集 |

## 取得モダリティの状態

| モダリティ | DB テーブル | 状態 | 主要特徴量 |
|---|---|---|---|
| GPS | `locations` | 完成 | 自宅滞在割合、行動半径、訪問場所数、移動距離、最大/平均速度 |
| Activity | `plugin_ios_activity_recognition` | 完成 | 静止/歩行/走行/車移動/自転車の各割合、能動的移動・外出移動割合 |
| Bluetooth | `bluetooth` | 完成 | 検出件数、社会接触候補 vs personal device 分離、強 RSSI 比、夜間検出比、反復接触比 |
| Screen | `screen` | 完成 | ON/OFF カウント、画面セッション数/日、夜間画面 ON 割合 |
| Home context (派生) | (上記の統合) | 完成 | 自宅文脈での補助指標 |
| EMA | `esm` / `plugin_ios_esm` | 仕上げ中 | 感情パターン、回答時のセンサ文脈 |
| Wi-Fi | `sensor_wifi` | 未着手 | (自宅/外出補強用) |
| Network | `network` | 未着手 | (自宅/外出補強用) |
| Weather | `plugin_openweather` | 未着手 | (天気と外出量の交絡統制用) |
| Battery / Barometer / Gravity / Notification | 各種 | 未着手 | — |

## ディレクトリ構成

クリーンアーキテクチャをベースにした 3 層構造を採用している。

```
src/
├── domain/                       純粋ロジック (I/O 禁止)
│   ├── scoring/                  LSNS / UCLA / discordance / GAD-7 のカットオフ判定
│   └── features/                 geo (haversine), home 推定, 位置特徴量計算
├── application/                  ユースケース (オーケストレーション)
│   ├── pipelines/                BuildXxxMaster 9 個 (questionnaire / metadata / sensing / analysis)
│   └── analysis/                 FitBinaryGEE, FitMultinomialLogit + FitResult dataclass
└── infrastructure/               外部依存
    ├── database/                 LocationRepository / DeviceRepository (parameterized SQL)
    └── storage/                  CSV Repository (paths.py, questionnaire / metadata / sensing / analysis)

scripts/                          薄いエントリポイント (Build*.run() を呼ぶだけ)
├── preprocessing/                生データクリーニング
├── metadata/                     参加者・phase 期間メタデータ生成
├── features/{location,bluetooth,activity,home,screen,ema,merge}/  特徴量生成
├── analysis/                     31 本の分析スクリプト
├── visualization/
└── debug/

tests/                            domain / infrastructure / application を網羅 (80 passed)
data/                             生データと中間出力 (.gitignore 対象)
results/                          プロットと最終成果物
configs/                          パラメータ設定
notebooks/                        探索用
```

## セットアップ

```bash
# 仮想環境
python -m venv env
source env/bin/activate

# 依存関係
pip install -r requirements.txt

# 開発用インストール
pip install -e .

# DB 接続情報 (MySQL / AWARE)
cp .env.example .env  # DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME を設定
```

## データ生成パイプライン

スクリプトはプロジェクトルートから実行する。生データ → 分析直前マスタまでの典型的な実行順:

```bash
# 1. メタデータ
python scripts/metadata/create_participant_mapping.py
python scripts/metadata/create_participant_phase_periods.py
python scripts/metadata/create_sensing_periods.py

# 2. 質問紙
python scripts/preprocessing/preprocess_questionnaire.py
python scripts/preprocessing/create_psychology_master.py
python scripts/metadata/create_label_master.py

# 3. GPS (生 → クリーン → 標準化)
python scripts/preprocessing/remove_gps_jumps.py
python scripts/features/location/create_phase_location_features_from_clean.py
python scripts/preprocessing/filter_gps_feature_outliers.py
python scripts/preprocessing/standardize_gps_features.py

# 4. その他センサ
python scripts/preprocessing/clean_bluetooth_logs.py
python scripts/features/bluetooth/create_phase_bluetooth_social_features.py
python scripts/features/activity/create_phase_activity_features.py
python scripts/features/screen/create_phase_screen_features.py
python scripts/features/home/create_home_context_features.py

# 5. EMA
python scripts/features/ema/create_ema_master.py
python scripts/features/ema/filter_valid_ema_sessions.py
python scripts/features/ema/create_ema_context_dataset.py

# 6. 統合マスタ
python scripts/features/merge/create_multimodal_feature_master.py
python scripts/features/merge/create_analysis_ready_master.py
```

## 分析の実行

```bash
# 標準的な単変量推論 (GEE binomial + 多項ロジ)
python scripts/analysis/run_standard_inference.py

# 混合効果モデル
python scripts/analysis/run_mixed_effects_model.py

# メイン分析パッケージ (cross-sectional + 縦断 + 群比較を一括)
python scripts/analysis/run_main_analysis.py
```

各分析スクリプトは [data/analysis/](data/analysis/) または [results/plots/](results/plots/) に出力する。

## テスト

```bash
pytest tests/ -v
```

domain (scoring / features) と infrastructure (storage Repository) と application (pipelines / analysis) を 80 ケースでカバー。

## 主要な所見 (2026-05-10 時点、N=14)

- **行動範囲↑ × 社会ネット豊富さ↑**: radius_of_gyration_km × LSNS で Spearman ρ=0.64, **p=0.034** (n=11)
- **車移動↑ × 社会ネット貧弱化**: automotive_ratio × LSNS で ρ=−0.55, **p=0.043** (n=14)
- **GPS と Bluetooth の整合性**: radius_of_gyration_km × 社会接触候補デバイス数で ρ=0.79, **p=0.012** (n=9)
- **discordance タイプの物理プロファイル**: 「孤立・非孤独」(引きこもり型: home_stay=0.96, radius=1km) と「非孤立・孤独」(群衆の中の孤独型: home_stay=0.71, radius=8.7km) が**正反対**の行動パターンとして分離
- **不協和は動的**: 14 日で 5/13 名が discordance タイプ遷移

## 制約・限界

- N=14 で群分布が極端に偏り(10 : 2 : 1 : 1)、特に「孤立・孤独」群は pre データで n=0
- Bluetooth は 1 日中央値 0.14 件と疎で、AWARE 稼働率の交絡を排除できない
- 縦断 Δ 相関は実質 n=11 で検出力不足
- これらは大規模サンプル(N≥50、各群 n≥10)での再検証が必要

## 開発上のメモ

- **DataFrame は infrastructure–application 境界で使う**(エンティティ変換は不要)
- **Repository はコンストラクタでパスを差し替え可能**にしてテストで `tmp_path` 注入できるように
- **欠測値は明示的に `None` を伝播**(暗黙の 0 埋めは禁止)
- DB クエリは **パラメタライズドクエリ**(SQL injection 対策)
- 設定値(カットオフ、データパス、`HOME_RADIUS_KM` 等)は **定数として明示**

## ライセンス

(未定)
