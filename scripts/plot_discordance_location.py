import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


CSV_PATH = "data/analysis/analysis_master.csv"
OUTPUT_PATH = "results/plots/discordance_location_jp.png"

# =========================
# CSV読み込み
# =========================
df = pd.read_csv(CSV_PATH)

# preのみ使用
df = df[df["phase"] == "pre"]

# =========================
# 日本語ラベルへ変換
# =========================
label_map = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}

df["discordance_type_jp"] = (
    df["discordance_type"]
    .map(label_map)
)

# =========================
# group平均
# =========================
group_df = (
    df.groupby("discordance_type_jp")[
        "unique_location_bins_per_day"
    ]
    .mean()
    .reset_index()
)

print(group_df)

# =========================
# plot
# =========================
plt.figure(figsize=(10, 5))

plt.bar(
    group_df["discordance_type_jp"],
    group_df["unique_location_bins_per_day"]
)

plt.ylabel("1日あたりの訪問場所種類数")
plt.xlabel("孤立・孤独タイプ")

plt.title(
    "孤立・孤独タイプ別の移動多様性"
)

plt.xticks(rotation=10)

plt.tight_layout()

# =========================
# 保存
# =========================
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved plot to: {OUTPUT_PATH}")