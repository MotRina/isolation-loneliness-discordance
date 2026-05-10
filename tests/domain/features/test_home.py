import pandas as pd

from src.domain.features.home import estimate_home_location


def _make_df(timestamps, latitudes, longitudes):
    return pd.DataFrame({
        "datetime": pd.to_datetime(timestamps),
        "latitude": latitudes,
        "longitude": longitudes,
    })


def test_uses_nighttime_median():
    df = _make_df(
        timestamps=[
            "2026-01-01 23:00",  # 夜
            "2026-01-02 02:00",  # 夜
            "2026-01-02 12:00",  # 昼 (無視される)
        ],
        latitudes=[35.0, 35.2, 40.0],
        longitudes=[139.0, 139.2, 145.0],
    )
    lat, lon = estimate_home_location(df)
    assert lat == 35.1
    assert lon == 139.1


def test_falls_back_to_all_when_no_nighttime():
    df = _make_df(
        timestamps=["2026-01-01 12:00", "2026-01-01 15:00"],
        latitudes=[35.0, 35.2],
        longitudes=[139.0, 139.2],
    )
    lat, lon = estimate_home_location(df)
    assert lat == 35.1
    assert lon == 139.1


def test_includes_22_oclock_and_6_oclock_inclusive():
    df = _make_df(
        timestamps=["2026-01-01 22:00", "2026-01-02 06:00", "2026-01-01 12:00"],
        latitudes=[35.0, 35.2, 99.0],
        longitudes=[139.0, 139.2, 99.0],
    )
    lat, lon = estimate_home_location(df)
    assert lat == 35.1
    assert lon == 139.1
