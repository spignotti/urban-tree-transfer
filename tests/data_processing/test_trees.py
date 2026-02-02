"""Unit tests for tree harmonization."""

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point

from urban_tree_transfer.data_processing.trees import harmonize_trees
from urban_tree_transfer.utils.strings import normalize_city_name


def test_harmonize_trees_schema(berlin_config):
    """Harmonized trees should have consistent schema."""
    raw = GeoDataFrame(
        {
            "gisid": [1],
            "gattung": ["Acer"],
            "art_bot": ["Acer platanoides"],
            "gattung_deutsch": ["Ahorn"],
            "art_deutsch": ["Spitzahorn"],
            "pflanzjahr": [1990],
            "baumhoehe": [10.5],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:25833",
    )

    harmonized = harmonize_trees(raw, berlin_config)
    expected_columns = [
        "tree_id",
        "city",
        "genus_latin",
        "species_latin",
        "genus_german",
        "species_german",
        "plant_year",
        "height_m",
        "tree_type",
        "geometry",
    ]
    assert list(harmonized.columns) == expected_columns

    assert harmonized["tree_id"].apply(lambda value: isinstance(value, str)).all()
    assert harmonized["tree_id"].dtype == object
    assert harmonized["plant_year"].dtype == pd.Int64Dtype()
    assert harmonized["height_m"].dtype == pd.Float64Dtype()
    assert harmonized["city"].iloc[0] == normalize_city_name(berlin_config["name"])
