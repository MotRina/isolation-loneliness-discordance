# scripts/features/ema/create_ema_context_dataset.py

from pathlib import Path

import pandas as pd


EMA_PATH = "data/questionnaire/processed/ema_master.csv"
SENSOR_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/ema_context_dataset.csv"


def main():
    ema_path = Path(EMA_PATH)

    if not ema_path.exists():
        print(f"EMA file does not exist: {EMA_PATH}")
        print("まず ema_master.csv を作成してください。")
        return

    ema_df = pd.read_csv(EMA_PATH)
    sensor_df = pd.read_csv(SENSOR_PATH)

    print("\n=== EMA ===")
    print(ema_df.head())

    print("\n=== Sensor master ===")
    print(sensor_df.head())

    # 将来：
    # EMA回答時刻の直前30分/60分/3時間のsensor特徴量を結合する
    # ここでは参加者単位・phase単位の結合土台だけ作る

    merged_df = ema_df.merge(
        sensor_df,
        on=["participant_id", "phase"],
        how="left",
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== EMA context dataset ===")
    print(merged_df.head())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()