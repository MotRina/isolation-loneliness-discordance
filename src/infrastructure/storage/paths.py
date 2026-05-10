"""データセット・成果物のファイルパス定義。

スクリプトはプロジェクトルートから実行される前提で、相対パスを保持する。
パスを変更する場合はここ一箇所を編集すれば全 Repository に伝搬する。
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Questionnaire
QUESTIONNAIRE_RAW = DATA_DIR / "questionnaire" / "raw" / "questionnaire.csv"
QUESTIONNAIRE_MASTER = DATA_DIR / "questionnaire" / "processed" / "questionnaire_master.csv"
PSYCHOLOGY_MASTER = DATA_DIR / "questionnaire" / "processed" / "psychology_master.csv"
LABEL_MASTER = DATA_DIR / "questionnaire" / "processed" / "label_master.csv"

# Metadata
PARTICIPANT_MAPPING = DATA_DIR / "metadata" / "participant_mapping.csv"
PARTICIPANT_PHASE_PERIODS = DATA_DIR / "metadata" / "participant_phase_periods.csv"
PARTICIPANT_SENSING_PERIODS = DATA_DIR / "metadata" / "participant_sensing_periods.csv"

# Sensing features
LOCATION_FEATURES = DATA_DIR / "sensing" / "processed" / "location_features.csv"
PHASE_LOCATION_FEATURES = DATA_DIR / "sensing" / "processed" / "phase_location_features.csv"

# Analysis
ANALYSIS_MASTER = DATA_DIR / "analysis" / "analysis_master.csv"

# Plots
PLOTS_DIR = RESULTS_DIR / "plots"
DISCORDANCE_LOCATION_PLOT = PLOTS_DIR / "discordance_location_jp.png"
