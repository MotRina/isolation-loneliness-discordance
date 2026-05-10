import math

from src.domain.features.geo import EARTH_RADIUS_KM, haversine_km


def test_same_point_returns_zero():
    assert haversine_km(35.0, 139.0, 35.0, 139.0) == 0.0


def test_known_short_distance():
    # 1 度緯度差 = 約 111 km
    distance = haversine_km(35.0, 139.0, 36.0, 139.0)
    assert math.isclose(distance, 111.19, abs_tol=0.5)


def test_tokyo_to_osaka_approximate():
    # 東京 (35.68, 139.69) → 大阪 (34.69, 135.50)
    distance = haversine_km(35.68, 139.69, 34.69, 135.50)
    assert 390 < distance < 410


def test_antipodes_is_half_earth_circumference():
    distance = haversine_km(0.0, 0.0, 0.0, 180.0)
    assert math.isclose(distance, math.pi * EARTH_RADIUS_KM, abs_tol=0.01)


def test_symmetric():
    a_to_b = haversine_km(35.0, 139.0, 34.0, 138.0)
    b_to_a = haversine_km(34.0, 138.0, 35.0, 139.0)
    assert math.isclose(a_to_b, b_to_a)
