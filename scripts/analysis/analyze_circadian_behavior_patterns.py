# scripts/analysis/analyze_circadian_behavior_patterns.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


LOCATION_PATH = "data/sensing/processed/clean_phase_location_logs.csv"

# 存在すれば読む。なければスキップ。
SCREEN_CANDIDATES = [
    "data/sensing/processed/clean_screen_logs.csv",
    "data/sensing/processed/screen_logs.csv",
]

BLUETOOTH_CANDIDATES = [
    "data/sensing/processed/clean_bluetooth_logs.csv",
    "data/sensing/processed/bluetooth_logs.csv",
]

ACTIVITY_CANDIDATES = [
    "data/sensing/processed/clean_activity_logs.csv",
    "data/sensing/processed/activity_logs.csv",
]

MASTER_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_DIR = Path("data/analysis/circadian")
OUTPUT_PATH = OUTPUT_DIR / "circadian_behavior_summary.csv"
CORR_PATH = OUTPUT_DIR / "circadian_loneliness_correlation.csv"


def read_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            print(f"Load: {path}")
            return pd.read_csv(path)

    print(f"Skip missing candidates: {paths}")
    return pd.DataFrame()


def safe_read(path):
    path = Path(path)

    if not path.exists():
        print(f"Skip missing: {path}")
        return pd.DataFrame()

    print(f"Load: {path}")
    return pd.read_csv(path)


def prepare_time(df):
    if df.empty:
        return df

    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["timestamp"],
                unit="ms",
                errors="coerce",
            )
        else:
            return pd.DataFrame()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "participant_id",
            "datetime",
        ]
    ).copy()

    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    df["is_evening"] = df["hour"].isin([18, 19, 20, 21])
    df["is_daytime"] = df["hour"].between(9, 17)

    return df


def summarize_location(df):
    if df.empty:
        return pd.DataFrame()

    if "distance_from_previous_km" not in df.columns:
        df["distance_from_previous_km"] = 0.0

    df["night_distance_component"] = df["distance_from_previous_km"].where(
        df["is_night"],
        0.0,
    )

    df["evening_distance_component"] = df["distance_from_previous_km"].where(
        df["is_evening"],
        0.0,
    )

    summary = (
        df.groupby("participant_id")
        .agg(
            circadian_location_log_count=("datetime", "count"),
            circadian_night_location_ratio=("is_night", "mean"),
            circadian_evening_location_ratio=("is_evening", "mean"),
            circadian_daytime_location_ratio=("is_daytime", "mean"),
            circadian_weekend_location_ratio=("is_weekend", "mean"),
            circadian_night_distance_km=("night_distance_component", "sum"),
            circadian_evening_distance_km=("evening_distance_component", "sum"),
            circadian_total_distance_km=("distance_from_previous_km", "sum"),
        )
        .reset_index()
    )

    summary["circadian_night_distance_ratio"] = (
        summary["circadian_night_distance_km"]
        / summary["circadian_total_distance_km"].replace(0, pd.NA)
    )

    summary["circadian_evening_distance_ratio"] = (
        summary["circadian_evening_distance_km"]
        / summary["circadian_total_distance_km"].replace(0, pd.NA)
    )

    return summary


def summarize_screen(df):
    if df.empty:
        return pd.DataFrame()

    if "screen_status" in df.columns:
        screen_on = df[df["screen_status"] == 2].copy()
    else:
        screen_on = df.copy()

    if screen_on.empty:
        return pd.DataFrame()

    summary = (
        screen_on.groupby("participant_id")
        .agg(
            circadian_screen_on_count=("datetime", "count"),
            circadian_night_screen_on_count=("is_night", "sum"),
            circadian_evening_screen_on_count=("is_evening", "sum"),
            circadian_weekend_screen_on_ratio=("is_weekend", "mean"),
        )
        .reset_index()
    )

    summary["circadian_night_screen_ratio"] = (
        summary["circadian_night_screen_on_count"]
        / summary["circadian_screen_on_count"].replace(0, pd.NA)
    )

    summary["circadian_evening_screen_ratio"] = (
        summary["circadian_evening_screen_on_count"]
        / summary["circadian_screen_on_count"].replace(0, pd.NA)
    )

    return summary


def summarize_bluetooth(df):
    if df.empty:
        return pd.DataFrame()

    device_col = "bt_address" if "bt_address" in df.columns else None

    agg_dict = {
        "circadian_bluetooth_log_count": ("datetime", "count"),
        "circadian_night_bluetooth_count": ("is_night", "sum"),
        "circadian_evening_bluetooth_count": ("is_evening", "sum"),
    }

    if device_col:
        agg_dict["circadian_unique_bluetooth_devices"] = (device_col, "nunique")

    summary = (
        df.groupby("participant_id")
        .agg(**agg_dict)
        .reset_index()
    )

    summary["circadian_night_bluetooth_ratio"] = (
        summary["circadian_night_bluetooth_count"]
        / summary["circadian_bluetooth_log_count"].replace(0, pd.NA)
    )

    summary["circadian_evening_bluetooth_ratio"] = (
        summary["circadian_evening_bluetooth_count"]
        / summary["circadian_bluetooth_log_count"].replace(0, pd.NA)
    )

    return summary


def summarize_activity(df):
    if df.empty:
        return pd.DataFrame()

    if "activity_name" not in df.columns:
        if "activity_type" in df.columns:
            df["activity_name"] = df["activity_type"].astype(str)
        else:
            df["activity_name"] = "unknown"

    active_keywords = [
        "walking",
        "running",
        "cycling",
        "automotive",
        "徒歩",
        "歩行",
        "自転車",
        "車",
    ]

    df["is_active"] = (
        df["activity_name"]
        .astype(str)
        .str.lower()
        .apply(lambda x: any(k.lower() in x for k in active_keywords))
    )

    return (
        df.groupby("participant_id")
        .agg(
            circadian_activity_log_count=("datetime", "count"),
            circadian_night_activity_ratio=("is_night", "mean"),
            circadian_evening_activity_ratio=("is_evening", "mean"),
            circadian_daytime_activity_ratio=("is_daytime", "mean"),
            circadian_active_ratio=("is_active", "mean"),
        )
        .reset_index()
    )


def merge_parts(parts):
    summary = None

    for part in parts:
        if part is None or part.empty:
            continue

        if summary is None:
            summary = part
        else:
            summary = summary.merge(
                part,
                on="participant_id",
                how="outer",
            )

    if summary is None:
        return pd.DataFrame()

    return summary


def correlate_with_outcomes(summary_df):
    master = safe_read(MASTER_PATH)

    if summary_df.empty or master.empty:
        return pd.DataFrame()

    master = master[master["phase"] == "pre"].copy()

    outcome_cols = [
        col for col in [
            "participant_id",
            "ucla_total",
            "lsns_total",
            "ucla_lonely",
            "lsns_isolated",
            "discordance_type",
        ]
        if col in master.columns
    ]

    master = master[outcome_cols].copy()

    merged = summary_df.merge(
        master,
        on="participant_id",
        how="left",
    )

    outcomes = [
        col for col in [
            "ucla_total",
            "lsns_total",
        ]
        if col in merged.columns
    ]

    # summary_df 側の列だけを相関対象にすることで、merge後の列名衝突を回避
    features = [
        col for col in summary_df.columns
        if col != "participant_id"
    ]

    rows = []

    for outcome in outcomes:
        for feature in features:
            if feature not in merged.columns:
                continue

            use = merged.dropna(
                subset=[
                    outcome,
                    feature,
                ]
            )

            if len(use) < 5:
                continue

            if use[feature].nunique() <= 1:
                continue

            r, p = spearmanr(
                use[feature],
                use[outcome],
            )

            rows.append({
                "outcome": outcome,
                "feature": feature,
                "n": len(use),
                "spearman_r": r,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    location_df = prepare_time(
        safe_read(LOCATION_PATH)
    )

    screen_df = prepare_time(
        read_first_existing(SCREEN_CANDIDATES)
    )

    bluetooth_df = prepare_time(
        read_first_existing(BLUETOOTH_CANDIDATES)
    )

    activity_df = prepare_time(
        read_first_existing(ACTIVITY_CANDIDATES)
    )

    parts = [
        summarize_location(location_df),
        summarize_screen(screen_df),
        summarize_bluetooth(bluetooth_df),
        summarize_activity(activity_df),
    ]

    summary_df = merge_parts(parts)

    summary_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    corr_df = correlate_with_outcomes(summary_df)

    corr_df.to_csv(
        CORR_PATH,
        index=False,
    )

    print("\n=== Circadian behavior summary ===")
    print(summary_df.head())

    print("\n=== Circadian correlation ===")
    if corr_df.empty:
        print(corr_df)
    else:
        print(
            corr_df
            .sort_values("p_value")
            .head(30)
        )

    print(f"\nSaved summary to: {OUTPUT_PATH}")
    print(f"Saved correlation to: {CORR_PATH}")


if __name__ == "__main__":
    main()