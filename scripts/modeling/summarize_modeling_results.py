# scripts/modeling/summarize_modeling_results.py

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("data/modeling/summary")
OUTPUT_PATH = OUTPUT_DIR / "modeling_results_summary.csv"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "modeling_results_summary.md"


INPUT_FILES = {
    "loneliness_prediction": "data/modeling/loneliness_prediction/loneliness_prediction_metrics.csv",
    "modality_ablation": "data/modeling/modality_ablation/modality_ablation_metrics.csv",
    "longitudinal_delta_regression": "data/modeling/longitudinal_prediction/longitudinal_delta_regression_metrics.csv",
    "longitudinal_change_classification": "data/modeling/longitudinal_prediction/longitudinal_loneliness_change_classification_metrics.csv",
    "within_person_ema": "data/modeling/within_person_prediction/within_person_ema_prediction_metrics.csv",
    "feature_selection_lasso": "data/modeling/feature_selection/lasso_selected_features.csv",
    "feature_selection_rfe": "data/modeling/feature_selection/rfe_selected_features.csv",
    "feature_selection_rf": "data/modeling/feature_selection/random_forest_feature_importance.csv",
}


def load_csv(path):
    path = Path(path)

    if not path.exists():
        print(f"Skip: {path} does not exist")
        return None

    return pd.read_csv(path)


def summarize_loneliness_prediction(df):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for target in df["target"].dropna().unique():
        target_df = df[df["target"] == target].copy()

        best_mae = target_df.sort_values("mae").iloc[0]
        best_rmse = target_df.sort_values("rmse").iloc[0]
        best_r2 = target_df.sort_values("r2", ascending=False).iloc[0]

        rows.extend([
            {
                "section": "loneliness_prediction",
                "target": target,
                "summary_type": "best_mae",
                "model": best_mae["model"],
                "metric": "mae",
                "value": best_mae["mae"],
                "note": "Lower MAE is better.",
            },
            {
                "section": "loneliness_prediction",
                "target": target,
                "summary_type": "best_rmse",
                "model": best_rmse["model"],
                "metric": "rmse",
                "value": best_rmse["rmse"],
                "note": "Lower RMSE is better.",
            },
            {
                "section": "loneliness_prediction",
                "target": target,
                "summary_type": "best_r2",
                "model": best_r2["model"],
                "metric": "r2",
                "value": best_r2["r2"],
                "note": "Higher R2 is better. Negative R2 suggests weak predictive performance.",
            },
        ])

    return pd.DataFrame(rows)


def summarize_modality_ablation(df):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for target in df["target"].dropna().unique():
        target_df = df[df["target"] == target].copy()

        best_row = target_df.sort_values("mae").iloc[0]

        rows.append({
            "section": "modality_ablation",
            "target": target,
            "summary_type": "best_modality_by_mae",
            "model": best_row["model"],
            "metric": "mae",
            "value": best_row["mae"],
            "note": f"Best modality: {best_row['modality']}, feature_count={best_row['feature_count']}",
        })

        for modality in target_df["modality"].dropna().unique():
            modality_df = target_df[target_df["modality"] == modality]
            best_modality_row = modality_df.sort_values("mae").iloc[0]

            rows.append({
                "section": "modality_ablation",
                "target": target,
                "summary_type": "modality_detail",
                "model": best_modality_row["model"],
                "metric": "mae",
                "value": best_modality_row["mae"],
                "note": f"modality={modality}, r2={best_modality_row['r2']:.3f}",
            })

    return pd.DataFrame(rows)


def summarize_longitudinal(df_reg, df_cls):
    rows = []

    if df_reg is not None and not df_reg.empty:
        for target in df_reg["target"].dropna().unique():
            target_df = df_reg[df_reg["target"] == target].copy()
            best_row = target_df.sort_values("mae").iloc[0]

            rows.append({
                "section": "longitudinal_delta_regression",
                "target": target,
                "summary_type": "best_delta_prediction_by_mae",
                "model": best_row["model"],
                "metric": "mae",
                "value": best_row["mae"],
                "note": f"rmse={best_row['rmse']:.3f}, r2={best_row['r2']:.3f}",
            })

    if df_cls is not None and not df_cls.empty:
        if "balanced_accuracy" in df_cls.columns:
            best_row = df_cls.sort_values(
                "balanced_accuracy",
                ascending=False,
            ).iloc[0]

            rows.append({
                "section": "longitudinal_change_classification",
                "target": "loneliness_worsened",
                "summary_type": "best_balanced_accuracy",
                "model": best_row["model"],
                "metric": "balanced_accuracy",
                "value": best_row["balanced_accuracy"],
                "note": f"f1={best_row.get('f1', None)}, status={best_row.get('status', None)}",
            })

    return pd.DataFrame(rows)


def summarize_within_person_ema(df):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    best_row = df.sort_values("mae").iloc[0]

    rows.append({
        "section": "within_person_ema",
        "target": best_row["target"],
        "summary_type": "best_within_person_ema_by_mae",
        "model": best_row["model"],
        "metric": "mae",
        "value": best_row["mae"],
        "note": f"rmse={best_row['rmse']:.3f}, r2={best_row['r2']:.3f}, feature_type={best_row['feature_type']}",
    })

    return pd.DataFrame(rows)


def summarize_feature_selection(lasso_df, rfe_df, rf_df):
    rows = []

    if lasso_df is not None and not lasso_df.empty:
        selected_df = lasso_df[lasso_df["selected"] == True].copy()

        for target in selected_df["target"].dropna().unique():
            target_df = selected_df[selected_df["target"] == target].copy()
            top_df = target_df.sort_values(
                "abs_coefficient",
                ascending=False,
            ).head(10)

            for _, row in top_df.iterrows():
                rows.append({
                    "section": "feature_selection_lasso",
                    "target": target,
                    "summary_type": "selected_feature",
                    "model": "LASSO",
                    "metric": "abs_coefficient",
                    "value": row["abs_coefficient"],
                    "note": f"{row['feature']} coef={row['coefficient']:.3f}",
                })

    if rfe_df is not None and not rfe_df.empty:
        selected_df = rfe_df[rfe_df["selected"] == True].copy()

        for target in selected_df["target"].dropna().unique():
            for _, row in selected_df[selected_df["target"] == target].iterrows():
                rows.append({
                    "section": "feature_selection_rfe",
                    "target": target,
                    "summary_type": "selected_feature",
                    "model": "RFE",
                    "metric": "ranking",
                    "value": row["ranking"],
                    "note": row["feature"],
                })

    if rf_df is not None and not rf_df.empty:
        for target in rf_df["target"].dropna().unique():
            top_df = (
                rf_df[rf_df["target"] == target]
                .sort_values("importance", ascending=False)
                .head(10)
            )

            for _, row in top_df.iterrows():
                rows.append({
                    "section": "feature_selection_rf",
                    "target": target,
                    "summary_type": "top_feature_importance",
                    "model": "RandomForest",
                    "metric": "importance",
                    "value": row["importance"],
                    "note": row["feature"],
                })

    return pd.DataFrame(rows)


def create_markdown(summary_df):
    lines = []

    lines.append("# Modeling Results Summary")
    lines.append("")
    lines.append("## 1. Overall interpretation")
    lines.append("")
    lines.append(
        "- Prediction performance should be interpreted cautiously because the participant-level sample size is small."
    )
    lines.append(
        "- Results are most useful as exploratory evidence for identifying meaningful sensing features."
    )
    lines.append(
        "- EMA / within-person analyses are especially important because they use many momentary observations."
    )
    lines.append("")

    for section in summary_df["section"].dropna().unique():
        section_df = summary_df[summary_df["section"] == section].copy()

        lines.append(f"## {section}")
        lines.append("")

        for _, row in section_df.iterrows():
            lines.append(
                f"- target={row['target']}, "
                f"type={row['summary_type']}, "
                f"model={row['model']}, "
                f"{row['metric']}={row['value']:.4f}, "
                f"{row['note']}"
            )

        lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loneliness_df = load_csv(INPUT_FILES["loneliness_prediction"])
    modality_df = load_csv(INPUT_FILES["modality_ablation"])
    longitudinal_reg_df = load_csv(INPUT_FILES["longitudinal_delta_regression"])
    longitudinal_cls_df = load_csv(INPUT_FILES["longitudinal_change_classification"])
    within_person_df = load_csv(INPUT_FILES["within_person_ema"])
    lasso_df = load_csv(INPUT_FILES["feature_selection_lasso"])
    rfe_df = load_csv(INPUT_FILES["feature_selection_rfe"])
    rf_df = load_csv(INPUT_FILES["feature_selection_rf"])

    summary_parts = [
        summarize_loneliness_prediction(loneliness_df),
        summarize_modality_ablation(modality_df),
        summarize_longitudinal(longitudinal_reg_df, longitudinal_cls_df),
        summarize_within_person_ema(within_person_df),
        summarize_feature_selection(lasso_df, rfe_df, rf_df),
    ]

    summary_df = pd.concat(
        [part for part in summary_parts if part is not None and not part.empty],
        ignore_index=True,
    )

    summary_df.to_csv(OUTPUT_PATH, index=False)

    markdown_text = create_markdown(summary_df)

    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print("\n=== Modeling results summary ===")
    print(summary_df.head(50))

    print(f"\nSaved CSV to: {OUTPUT_PATH}")
    print(f"Saved Markdown to: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()