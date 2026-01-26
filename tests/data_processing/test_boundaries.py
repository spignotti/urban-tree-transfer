"""Unit tests for boundary helpers."""

from shapely.geometry import MultiPolygon, Polygon

from urban_tree_transfer.data_processing.boundaries import _build_ogc_filter, _largest_polygon


def test_build_ogc_filter():
    """OGC filter XML should be well-formed."""
    xml = _build_ogc_filter("name", "Leipzig", "ave")
    assert (
        'xmlns:ave="http://repository.gdi-de.org/schemas/adv/produkt/alkis-vereinfacht/2.0"' in xml
    )
    assert "<PropertyName>ave:name</PropertyName>" in xml
    assert "<Literal>Leipzig</Literal>" in xml


def test_largest_polygon_multipolygon():
    """Should return largest polygon from MultiPolygon."""
    small = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    large = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    multi = MultiPolygon([small, large])
    result = _largest_polygon(multi)
    assert result.equals(large)


def test_largest_polygon_single():
    """Should return same polygon if not MultiPolygon."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = _largest_polygon(poly)
    assert result.equals(poly)
