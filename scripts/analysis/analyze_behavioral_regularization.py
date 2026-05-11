# scripts/analysis/analyze_behavioral_regularization.py

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


SCREEN_PATH = "data/sensing/processed/clean_screen_logs.csv"
LOCATION_PATH = "data/sensing/processed/clean_phase_location_logs.csv"
MASTER_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_DIR = Path("data/analysis/behavioral_regularization")
OUTPUT_PATH = OUTPUT_DIR / "behavioral_regularization_summary.csv"
CORR_PATH = OUTPUT_DIR / "behavioral_regularization_correlation.csv"


def safe_read(path):
    path = Path(path)
    if not path.exists():
        print(f"Skip missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_datetime(df):
    if df.empty:
        return df
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["participant_id", "datetime"]).copy()
    df["date"] = df["datetime"].dt.date
    df["hour_float"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    return df


def summarize_screen_rhythm(df):
    if df.empty:
        return pd.DataFrame()

    if "screen_status" in df.columns:
        df = df[df["screen_status"] == 2].copy()

    daily = (
        df.groupby(["participant_id", "date"])
        .agg(
            first_screen_hour=("hour_float", "min"),
            last_screen_hour=("hour_float", "max"),
            screen_count=("datetime", "count"),
        )
        .reset_index()
    )

    return (
        daily.groupby("participant_id")
        .agg(
            active_days_screen=("date", "nunique"),
            mean_first_screen_hour=("first_screen_hour", "mean"),
            sd_first_screen_hour=("first_screen_hour", "std"),
            mean_last_screen_hour=("last_screen_hour", "mean"),
            sd_last_screen_hour=("last_screen_hour", "std"),
            mean_screen_count=("screen_count", "mean"),
            sd_screen_count=("screen_count", "std"),
        )
        .reset_index()
    )


def summarize_mobility_rhythm(df):
    if df.empty:
        return pd.DataFrame()

    if "distance_from_previous_km" not in df.columns:
        df["distance_from_previous_km"] = 0

    moving = df[df["distance_from_previous_km"] > 0.05].copy()

    if moving.empty:
        return pd.DataFrame()

    daily = (
        moving.groupby(["participant_id", "date"])
        .agg(
            first_movement_hour=("hour_float", "min"),
            last_movement_hour=("hour_float", "max"),
            movement_log_count=("datetime", "count"),
        )
        .reset_index()
    )

    return (
        daily.groupby("participant_id")
        .agg(
            active_days_movement=("date", "nunique"),
            mean_first_movement_hour=("first_movement_hour", "mean"),
            sd_first_movement_hour=("first_movement_hour", "std"),
            mean_last_movement_hour=("last_movement_hour", "mean"),
            sd_last_movement_hour=("last_movement_hour", "std"),
            mean_movement_log_count=("movement_log_count", "mean"),
        )
        .reset_index()
    )


def correlate(summary):
    master = safe_read(MASTER_PATH)
    if master.empty:
        return pd.DataFrame()

    master = master[master["phase"] == "pre"].copy()
    merged = summary.merge(master, on="participant_id", how="left")

    features = [c for c in summary.columns if c != "participant_id"]
    outcomes = ["ucla_total", "lsns_total"]

    rows = []
    for outcome in outcomes:
        if outcome not in merged.columns:
            continue
        for feature in features:
            use = merged.dropna(subset=[outcome, feature])
            if len(use) < 5 or use[feature].nunique() <= 1:
                continue
            r, p = spearmanr(use[feature], use[outcome])
            rows.append({"outcome": outcome, "feature": feature, "n": len(use), "spearman_r": r, "p_value": p})
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    screen = prepare_datetime(safe_read(SCREEN_PATH))
    loc = prepare_datetime(safe_read(LOCATION_PATH))

    s1 = summarize_screen_rhythm(screen)
    s2 = summarize_mobility_rhythm(loc)

    if s1.empty:
        summary = s2
    elif s2.empty:
        summary = s1
    else:
        summary = s1.merge(s2, on="participant_id", how="outer")

    summary.to_csv(OUTPUT_PATH, index=False)
    corr = correlate(summary)
    corr.to_csv(CORR_PATH, index=False)

    print("\n=== Behavioral regularization ===")
    print(summary.head())
    print("\n=== Correlation ===")
    print(corr.sort_values("p_value").head(30) if not corr.empty else corr)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {CORR_PATH}")


if __name__ == "__main__":
    main()