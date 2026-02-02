"""Unit tests for outlier detection functions."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.feature_engineering.outliers import (
    apply_consensus_outlier_filter,
    detect_iqr_outliers,
    detect_mahalanobis_outliers,
    detect_zscore_outliers,
)


def _base_gdf(n: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tree_id": [f"T{i:03d}" for i in range(n)],
            "city": ["berlin"] * n,
            "genus_latin": ["TILIA"] * n,
            "geometry": [Point(float(i), float(i)) for i in range(n)],
        },
        crs=PROJECT_CRS,
    )


def test_detect_zscore_outliers_flags_extreme_values():
    gdf = _base_gdf(6)
    gdf["F1"] = [0, 0, 0, 0, 0, 100]
    gdf["F2"] = [0, 0, 0, 0, 0, 100]
    gdf["F3"] = [0, 0, 0, 0, 0, 100]

    flags = detect_zscore_outliers(gdf, ["F1", "F2", "F3"], z_threshold=2.0, min_feature_count=2)
    assert flags.sum() == 1
    assert bool(flags.iloc[-1]) is True


def test_detect_mahalanobis_outliers_flags_multivariate():
    gdf = _base_gdf(5)
    gdf["genus_latin"] = ["TILIA"] * 5
    gdf["F1"] = [0.0, 0.1, 0.0, 0.1, 10.0]
    gdf["F2"] = [0.0, 0.0, 0.1, 0.1, 10.0]

    flags = detect_mahalanobis_outliers(gdf, ["F1", "F2"], alpha=0.3)
    assert flags.sum() == 1
    assert bool(flags.iloc[-1]) is True


def test_detect_iqr_outliers_flags_height_extremes():
    gdf = _base_gdf(4)
    gdf["genus_latin"] = ["TILIA"] * 4
    gdf["city"] = ["berlin"] * 4
    gdf["CHM_1m"] = [10.0, 11.0, 12.0, 100.0]

    flags = detect_iqr_outliers(gdf, height_column="CHM_1m", multiplier=1.5)
    assert flags.sum() == 1
    assert bool(flags.iloc[-1]) is True


def test_apply_consensus_outlier_filter_adds_metadata():
    gdf = _base_gdf(4)
    z_flags = pd.Series([True, True, False, False], index=gdf.index)
    m_flags = pd.Series([True, True, True, False], index=gdf.index)
    i_flags = pd.Series([True, False, False, False], index=gdf.index)

    result, stats = apply_consensus_outlier_filter(gdf, z_flags, m_flags, i_flags)

    assert len(result) == len(gdf)
    assert result["outlier_severity"].tolist() == ["high", "medium", "low", "none"]
    assert stats["trees_removed"] == 0
    assert stats["high_count"] == 1
    assert stats["medium_count"] == 1
    assert stats["low_count"] == 1
    assert stats["none_count"] == 1
