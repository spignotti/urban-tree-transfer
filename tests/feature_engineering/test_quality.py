"""Unit tests for Phase 2 data quality functions."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import get_all_s2_features, load_feature_config
from urban_tree_transfer.feature_engineering.quality import (
    add_is_conifer_column,
    analyze_nan_distribution,
    apply_temporal_selection,
    compute_chm_engineered_features,
    filter_by_plant_year,
    filter_deciduous_genera,
    filter_nan_trees,
    filter_ndvi_plausibility,
    interpolate_features_within_tree,
)


def create_test_trees(n: int = 10) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tree_id": [f"T{i:03d}" for i in range(n)],
            "city": ["berlin"] * n,
            "genus_latin": ["TILIA"] * n,
            "species_latin": [pd.NA] * n,
            "genus_german": [pd.NA] * n,
            "species_german": [pd.NA] * n,
            "plant_year": pd.Series([pd.NA] * n, dtype="Int64"),
            "height_m": np.linspace(10.0, 12.0, n),
            "tree_type": [pd.NA] * n,
            "position_corrected": [False] * n,
            "correction_distance": np.zeros(n, dtype=float),
            "CHM_1m": np.linspace(5.0, 20.0, n),
            "geometry": [Point(i, i) for i in range(n)],
        },
        crs=PROJECT_CRS,
    )


def test_filter_deciduous_genera():
    gdf = create_test_trees(4)
    gdf.loc[2, "genus_latin"] = "PINUS"
    gdf.loc[3, "genus_latin"] = "PICEA"

    filtered = filter_deciduous_genera(gdf, deciduous_genera=["TILIA", "ACER"])
    assert len(filtered) == 2
    assert set(filtered["genus_latin"]) == {"TILIA"}


def test_add_is_conifer_column_basic_classification():
    gdf = create_test_trees(4)
    gdf["genus_latin"] = ["TILIA", "PINUS", "PICEA", "ACER"]

    result = add_is_conifer_column(gdf, ["PINUS", "PICEA"])

    assert result["is_conifer"].tolist() == [False, True, True, False]


def test_add_is_conifer_column_empty_list_marks_all_false():
    gdf = create_test_trees(3)
    gdf["genus_latin"] = ["TILIA", "PINUS", "PICEA"]

    result = add_is_conifer_column(gdf, [])

    assert result["is_conifer"].tolist() == [False, False, False]


def test_add_is_conifer_column_is_case_insensitive():
    gdf = create_test_trees(3)
    gdf["genus_latin"] = ["pinus", "Picea", "Tilia"]

    result = add_is_conifer_column(gdf, ["PINUS", "PICEA"])

    assert result["is_conifer"].tolist() == [True, True, False]


def test_filter_by_plant_year():
    gdf = create_test_trees(4)
    gdf["plant_year"] = pd.Series([2010, 2018, 2020, pd.NA], dtype="Int64")

    filtered = filter_by_plant_year(gdf, max_year=2018)
    assert len(filtered) == 3
    assert filtered["plant_year"].isna().sum() == 1


def test_apply_temporal_selection():
    gdf = create_test_trees(1)
    feature_config = load_feature_config()
    s2_features = get_all_s2_features(feature_config)

    feature_data: dict[str, float] = {}
    for month in range(1, 13):
        for feature in s2_features:
            feature_data[f"{feature}_{month:02d}"] = 1.0

    feature_df = pd.DataFrame(feature_data, index=gdf.index)
    gdf = gpd.GeoDataFrame(pd.concat([gdf, feature_df], axis=1), crs=PROJECT_CRS)

    selected_months = [3, 4, 5, 6, 7, 8]
    result = apply_temporal_selection(gdf, selected_months, feature_config)

    assert isinstance(result, gpd.GeoDataFrame)
    assert str(result.crs) == PROJECT_CRS
    assert result.geometry.name == "geometry"
    assert "CHM_1m" in result.columns
    assert f"{s2_features[0]}_01" not in result.columns
    assert f"{s2_features[0]}_03" in result.columns
    assert f"{s2_features[0]}_08" in result.columns


def test_interpolate_within_tree_interior_nan():
    gdf = create_test_trees(2)
    gdf["NDVI_04"] = [0.2, 0.8]
    gdf["NDVI_05"] = [np.nan, 0.9]
    gdf["NDVI_06"] = [0.4, 1.0]

    feature_cols = ["NDVI_04", "NDVI_05", "NDVI_06"]
    selected_months = [4, 5, 6]

    interpolated = interpolate_features_within_tree(
        gdf, feature_cols, selected_months, max_edge_nan_months=1
    )
    assert np.isclose(interpolated.loc[0, "NDVI_05"], 0.3)

    gdf_alt = gdf.copy()
    gdf_alt.loc[1, "NDVI_04"] = 0.1
    gdf_alt.loc[1, "NDVI_05"] = 0.2
    gdf_alt.loc[1, "NDVI_06"] = 0.3

    interpolated_alt = interpolate_features_within_tree(
        gdf_alt, feature_cols, selected_months, max_edge_nan_months=1
    )
    assert np.isclose(interpolated_alt.loc[0, "NDVI_05"], 0.3)


def test_interpolate_within_tree_edge_nan():
    gdf = create_test_trees(2)
    gdf["NDVI_04"] = [np.nan, np.nan]
    gdf["NDVI_05"] = [0.3, np.nan]
    gdf["NDVI_06"] = [0.5, 0.6]

    feature_cols = ["NDVI_04", "NDVI_05", "NDVI_06"]
    selected_months = [4, 5, 6]

    interpolated = interpolate_features_within_tree(
        gdf, feature_cols, selected_months, max_edge_nan_months=1
    )
    assert np.isclose(interpolated.loc[0, "NDVI_04"], 0.3)
    assert pd.isna(interpolated.loc[1, "NDVI_04"])
    assert pd.isna(interpolated.loc[1, "NDVI_05"])


def test_interpolate_end_edge_nan():
    """Test backward-fill for end-edge NaN."""
    gdf = create_test_trees(1)
    gdf["NDVI_04"] = [0.3]
    gdf["NDVI_05"] = [0.5]
    gdf["NDVI_06"] = [np.nan]

    feature_cols = ["NDVI_04", "NDVI_05", "NDVI_06"]
    selected_months = [4, 5, 6]

    interpolated = interpolate_features_within_tree(
        gdf, feature_cols, selected_months, max_edge_nan_months=1
    )
    assert np.isclose(interpolated.loc[0, "NDVI_06"], 0.5)


def test_compute_chm_zscore():
    gdf = create_test_trees(3)
    gdf["city"] = ["berlin", "berlin", "leipzig"]
    gdf["CHM_1m"] = [10.0, 14.0, 20.0]

    result = compute_chm_engineered_features(gdf)

    berlin_mean = 12.0
    berlin_std = np.std([10.0, 14.0], ddof=1)
    expected_z = (10.0 - berlin_mean) / berlin_std

    assert np.isclose(result.loc[0, "CHM_1m_zscore"], expected_z)
    assert result.loc[2, "CHM_1m_percentile"] == 100.0


def test_compute_chm_zscore_zero_std():
    """Test CHM zscore when all values are identical (std=0)."""
    gdf = create_test_trees(3)
    gdf["city"] = ["berlin"] * 3
    gdf["CHM_1m"] = [10.0, 10.0, 10.0]

    result = compute_chm_engineered_features(gdf)
    assert np.all(result["CHM_1m_zscore"] == 0.0)


def test_compute_chm_percentile_all_nan():
    """Test CHM percentile when all CHM values are NaN."""
    gdf = create_test_trees(3)
    gdf["city"] = ["berlin"] * 3
    gdf["CHM_1m"] = [np.nan, np.nan, np.nan]

    result = compute_chm_engineered_features(gdf)
    assert result["CHM_1m_percentile"].isna().all()


def test_compute_chm_engineered_features_genus_specific():
    """Test genus-specific CHM normalization."""
    data = {
        "tree_id": range(1, 21),
        "city": ["berlin"] * 20,
        "genus_latin": ["QUERCUS"] * 10 + ["MALUS"] * 10,
        "CHM_1m": [
            *[15, 18, 20, 22, 25, 28, 30, 32, 33, 35],
            *[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        "geometry": [Point(i, i) for i in range(1, 21)],
    }
    gdf = gpd.GeoDataFrame(data, crs=PROJECT_CRS)

    result = compute_chm_engineered_features(gdf)

    quercus = result[result["genus_latin"] == "QUERCUS"]["CHM_1m_percentile"]
    malus = result[result["genus_latin"] == "MALUS"]["CHM_1m_percentile"]

    assert 45 <= quercus.mean() <= 55
    assert 45 <= malus.mean() <= 55

    quercus_z = result[result["genus_latin"] == "QUERCUS"]["CHM_1m_zscore"]
    malus_z = result[result["genus_latin"] == "MALUS"]["CHM_1m_zscore"]

    assert abs(quercus_z.mean()) < 0.01
    assert abs(malus_z.mean()) < 0.01
    assert 0.95 < quercus_z.std() < 1.05
    assert 0.95 < malus_z.std() < 1.05


def test_compute_chm_engineered_features_rare_genus_fallback():
    """Test city-level fallback for rare genera (<10 samples)."""
    data = {
        "tree_id": range(1, 16),
        "city": ["berlin"] * 15,
        "genus_latin": ["QUERCUS"] * 10 + ["RARE_GENUS"] * 5,
        "CHM_1m": list(range(10, 25)),
        "geometry": [Point(i, i) for i in range(1, 16)],
    }
    gdf = gpd.GeoDataFrame(data, crs=PROJECT_CRS)

    result = compute_chm_engineered_features(gdf)

    rare_features = result[result["genus_latin"] == "RARE_GENUS"]
    assert rare_features["CHM_1m_zscore"].notna().all()
    assert rare_features["CHM_1m_percentile"].notna().all()


def test_filter_ndvi_plausibility():
    gdf = create_test_trees(3)
    gdf["NDVI_04"] = [0.2, 0.4, 0.6]
    gdf["NDVI_05"] = [0.1, 0.5, 0.2]

    filtered = filter_ndvi_plausibility(gdf, ["NDVI_04", "NDVI_05"], min_threshold=0.3)
    assert len(filtered) == 2
    assert set(filtered["tree_id"]) == {"T001", "T002"}


def test_interpolate_mixed_interior_edge_nan():
    """Test combination of interior and edge NaN."""
    gdf = create_test_trees(1)
    gdf["NDVI_03"] = [np.nan]
    gdf["NDVI_04"] = [0.4]
    gdf["NDVI_05"] = [np.nan]
    gdf["NDVI_06"] = [0.6]

    feature_cols = ["NDVI_03", "NDVI_04", "NDVI_05", "NDVI_06"]
    selected_months = [3, 4, 5, 6]

    interpolated = interpolate_features_within_tree(
        gdf, feature_cols, selected_months, max_edge_nan_months=1
    )
    assert np.isclose(interpolated.loc[0, "NDVI_03"], 0.4)
    assert np.isclose(interpolated.loc[0, "NDVI_05"], 0.5)


def test_analyze_nan_distribution():
    gdf = create_test_trees(3)
    gdf["B2_01"] = [1.0, np.nan, 2.0]
    gdf["B2_02"] = [np.nan, np.nan, 3.0]

    stats = analyze_nan_distribution(gdf, ["B2_01", "B2_02"])
    assert list(stats.columns) == ["feature", "nan_count", "nan_percentage"]
    assert stats.loc[0, "nan_count"] == 2


def test_filter_nan_trees():
    gdf = create_test_trees(2)
    gdf["NDVI_04"] = [np.nan, 0.2]
    gdf["NDVI_05"] = [np.nan, 0.3]
    gdf["NDVI_06"] = [0.4, 0.4]

    filtered = filter_nan_trees(gdf, ["NDVI_04", "NDVI_05", "NDVI_06"], max_nan_months=1)
    assert len(filtered) == 1
    assert filtered["tree_id"].iloc[0] == "T001"


def test_crs_preservation():
    """Ensure all functions preserve CRS."""
    gdf = create_test_trees(5)
    gdf["NDVI_04"] = [0.5] * 5
    gdf["NDVI_05"] = [0.6] * 5
    original_crs = gdf.crs

    filtered = filter_deciduous_genera(gdf, ["TILIA"])
    assert filtered.crs == original_crs

    filtered = filter_by_plant_year(gdf, max_year=2020)
    assert filtered.crs == original_crs

    filtered = filter_ndvi_plausibility(gdf, ["NDVI_04", "NDVI_05"])
    assert filtered.crs == original_crs

    result = compute_chm_engineered_features(gdf)
    assert result.crs == original_crs

    result = interpolate_features_within_tree(gdf, ["NDVI_04", "NDVI_05"], [4, 5])
    assert result.crs == original_crs


def test_interpolate_within_tree_empty_gdf():
    gdf = create_test_trees(0)
    gdf["NDVI_04"] = pd.Series(dtype=float)
    gdf["NDVI_05"] = pd.Series(dtype=float)
    gdf["NDVI_06"] = pd.Series(dtype=float)

    result = interpolate_features_within_tree(gdf, ["NDVI_04", "NDVI_05", "NDVI_06"], [4, 5, 6])

    assert result.empty


def test_filter_nan_trees_ignores_non_temporal_columns():
    gdf = create_test_trees(2)
    gdf["CHM_1m_zscore"] = [0.1, 0.2]
    gdf["NDVI_04"] = [np.nan, 0.2]
    gdf["NDVI_05"] = [0.3, 0.4]

    # With new parameters: max_edge_nan_months=1, min_valid_months=2
    # Tree 1 has 1 edge NaN and only 1 valid month (NDVI_05), so gets removed with default min_valid_months=2
    # Set min_valid_months=1 to keep both trees
    filtered = filter_nan_trees(
        gdf,
        ["CHM_1m_zscore", "NDVI_04", "NDVI_05"],
        max_nan_months=1,
        max_edge_nan_months=1,
        min_valid_months=1,
    )

    assert len(filtered) == 2
