import matplotlib
matplotlib.use("Agg")

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

from src.infrastructure.storage import AnalysisMasterRepository, paths


def main():
    master_repo = AnalysisMasterRepository()
    df = master_repo.load()

    # preのみ使用
    df = df[df["phase"] == "pre"]

    label_map = {
        "isolated_lonely": "孤立・孤独",
        "isolated_not_lonely": "孤立・非孤独",
        "not_isolated_lonely": "非孤立・孤独",
        "not_isolated_not_lonely": "非孤立・非孤独",
    }

    df["discordance_type_jp"] = df["discordance_type"].map(label_map)

    group_df = (
        df.groupby("discordance_type_jp")["unique_location_bins_per_day"]
        .mean()
        .reset_index()
    )

    print(group_df)

    plt.figure(figsize=(10, 5))
    plt.bar(
        group_df["discordance_type_jp"],
        group_df["unique_location_bins_per_day"],
    )
    plt.ylabel("1日あたりの訪問場所種類数")
    plt.xlabel("孤立・孤独タイプ")
    plt.title("孤立・孤独タイプ別の移動多様性")
    plt.xticks(rotation=10)
    plt.tight_layout()

    output_path = paths.DISCORDANCE_LOCATION_PLOT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
