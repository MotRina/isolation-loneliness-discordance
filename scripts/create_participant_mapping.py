import json
import pandas as pd

from src.infrastructure.database.connection import create_db_engine

engine = create_db_engine()

query = """
SELECT
    device_id,
    data
FROM aware_device
"""

df = pd.read_sql(query, engine)

# JSON展開
df["parsed"] = df["data"].apply(json.loads)

# participant_id抽出
df["participant_id"] = df["parsed"].apply(
    lambda x: x.get("label")
)

# 必要列だけ
mapping_df = df[["participant_id", "device_id"]]

# 重複除去
mapping_df = mapping_df.drop_duplicates()

print(mapping_df)

# 保存
mapping_df.to_csv(
    "data/metadata/participant_mapping.csv",
    index=False
)