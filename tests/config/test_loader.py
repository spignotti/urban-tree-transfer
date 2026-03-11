"""Tests for config loader."""

from urban_tree_transfer.config.loader import (
    get_config_dir,
    get_coniferous_genera,
    load_city_config,
    load_feature_config,
)


def test_config_dir_exists():
    """Config directory should exist within the package."""
    config_dir = get_config_dir()
    assert config_dir.exists()
    assert (config_dir / "cities").exists()


def test_load_berlin_config():
    """Berlin config should load with required keys."""
    config = load_city_config("berlin")
    assert config["name"] == "Berlin"
    assert "boundaries" in config
    assert "trees" in config
    assert "elevation" in config


def test_load_leipzig_config():
    """Leipzig config should load with required keys."""
    config = load_city_config("leipzig")
    assert config["name"] == "Leipzig"
    assert "boundaries" in config
    assert "trees" in config
    assert "elevation" in config


def test_get_coniferous_genera_matches_feature_config():
    """Coniferous genera helper should return configured genera."""
    feature_config = load_feature_config()

    genera = get_coniferous_genera(feature_config)

    assert genera
    assert genera == feature_config["genus_classification"]["coniferous"]
