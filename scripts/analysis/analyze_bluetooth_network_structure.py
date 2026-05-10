# scripts/analysis/analyze_bluetooth_network_structure.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/bluetooth_network_structure_summary.csv"
OUTPUT_DIR = Path("results/plots/bluetooth_network_structure")


def classify_contact_structure(row):
    repeated = row["repeated_device_ratio"]
    unique_per_day = row["unique_possible_social_devices_per_day"]

    if pd.isna(repeated) or pd.isna(unique_per_day):
        return "unknown"

    if repeated >= 0.6 and unique_per_day <= 1.0:
        return "固定的・少数接触"

    if repeated < 0.6 and unique_per_day > 1.0:
        return "多様な接触"

    if repeated >= 0.6 and unique_per_day > 1.0:
        return "固定接触＋多接触"

    return "少数・一時的接触"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["phase"] == "pre")
        & (df["is_valid_bluetooth"] == True)
    ].copy()

    df["contact_structure_type"] = df.apply(
        classify_contact_structure,
        axis=1,
    )

    summary_df = (
        df.groupby("contact_structure_type")
        .agg(
            n=("participant_id", "count"),
            mean_ucla=("ucla_total", "mean"),
            median_ucla=("ucla_total", "median"),
            mean_lsns=("lsns_total", "mean"),
            median_lsns=("lsns_total", "median"),
            mean_repeated=("repeated_device_ratio", "mean"),
            mean_unique_bt=("unique_possible_social_devices_per_day", "mean"),
        )
        .reset_index()
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Bluetooth contact structure summary ===")
    print(summary_df)

    print("\n=== participant-level ===")
    print(
        df[
            [
                "participant_id",
                "discordance_type",
                "ucla_total",
                "lsns_total",
                "unique_possible_social_devices_per_day",
                "repeated_device_ratio",
                "night_bluetooth_ratio",
                "contact_structure_type",
            ]
        ]
    )

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="unique_possible_social_devices_per_day",
        y="repeated_device_ratio",
        hue="discordance_type",
        style="contact_structure_type",
        s=100,
    )

    for _, row in df.iterrows():
        plt.text(
            row["unique_possible_social_devices_per_day"],
            row["repeated_device_ratio"],
            row["participant_id"],
            fontsize=8,
        )

    plt.axhline(0.6, linestyle="--")
    plt.axvline(1.0, linestyle="--")

    plt.title("Bluetooth接触構造：多様性 × 反復性")
    plt.xlabel("社会接触候補デバイス数/日")
    plt.ylabel("反復検出デバイス割合")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "bluetooth_contact_structure_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()