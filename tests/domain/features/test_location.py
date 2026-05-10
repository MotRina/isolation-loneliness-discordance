import json

import pandas as pd

from src.domain.features.location import (
    create_location_features,
    empty_location_features,
    parse_location_dataframe,
    parse_location_json,
)


def test_parse_location_json_valid():
    raw = json.dumps({"double_latitude": 35.0, "double_longitude": 139.0})
    parsed = parse_location_json(raw)
    assert parsed["double_latitude"] == 35.0


def test_parse_location_json_invalid_returns_empty_dict():
    assert parse_location_json("not-json") == {}
    assert parse_location_json("") == {}


def test_parse_location_dataframe_filters_low_accuracy():
    raw_df = pd.DataFrame({
        "timestamp": [1000, 2000, 3000],
        "data": [
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
            json.dumps({"double_latitude": 35.1, "double_longitude": 139.1, "accuracy": 100}),
            json.dumps({"double_latitude": 35.2, "double_longitude": 139.2, "accuracy": 30}),
        ],
    })
    parsed = parse_location_dataframe(raw_df, accuracy_threshold=50.0)
    assert len(parsed) == 2
    assert set(parsed["latitude"]) == {35.0, 35.2}


def test_parse_location_dataframe_no_filter_when_threshold_none():
    raw_df = pd.DataFrame({
        "timestamp": [1000, 2000],
        "data": [
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
            json.dumps({"double_latitude": 35.1, "double_longitude": 139.1, "accuracy": 9999}),
        ],
    })
    parsed = parse_location_dataframe(raw_df, accuracy_threshold=None)
    assert len(parsed) == 2


def test_parse_location_dataframe_drops_missing_coordinates():
    raw_df = pd.DataFrame({
        "timestamp": [1000, 2000],
        "data": [
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
            json.dumps({"accuracy": 10}),
        ],
    })
    parsed = parse_location_dataframe(raw_df)
    assert len(parsed) == 1


def test_create_location_features_on_empty_returns_empty_template():
    empty_df = pd.DataFrame(
        columns=["timestamp", "latitude", "longitude", "accuracy", "datetime"]
    )
    assert create_location_features(empty_df) == empty_location_features()


def test_create_location_features_keys():
    expected_keys = set(empty_location_features().keys())
    raw_df = pd.DataFrame({
        "timestamp": [1000, 90_000_000],
        "data": [
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
            json.dumps({"double_latitude": 35.1, "double_longitude": 139.1, "accuracy": 10}),
        ],
    })
    parsed = parse_location_dataframe(raw_df)
    features = create_location_features(parsed)
    assert set(features.keys()) == expected_keys


def test_home_stay_ratio_when_all_points_at_home():
    raw_df = pd.DataFrame({
        "timestamp": [1000, 90_000_000],
        "data": [
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
            json.dumps({"double_latitude": 35.0, "double_longitude": 139.0, "accuracy": 10}),
        ],
    })
    parsed = parse_location_dataframe(raw_df)
    features = create_location_features(parsed)
    assert features["home_stay_ratio"] == 1.0
    assert features["away_from_home_ratio"] == 0.0
    assert features["total_distance_km"] == 0.0
