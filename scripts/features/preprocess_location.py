import json
import pandas as pd


def parse_location_data(df: pd.DataFrame):

    parsed = df["data"].apply(json.loads)

    location_df = pd.DataFrame({
        "timestamp": parsed.apply(
            lambda x: x.get("timestamp")
        ),
        "latitude": parsed.apply(
            lambda x: x.get("latitude")
        ),
        "longitude": parsed.apply(
            lambda x: x.get("longitude")
        ),
        "accuracy": parsed.apply(
            lambda x: x.get("accuracy")
        ),
    })

    return location_df