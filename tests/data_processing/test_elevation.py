"""Unit tests for elevation helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.data_processing.elevation import (
    DEFAULT_NODATA,
    XYZ_RESOLUTION,
    _xyz_to_geotiff,
)


def test_xyz_to_geotiff_basic():
    """Convert simple XYZ data to GeoTIFF."""
    with TemporaryDirectory() as tmpdir:
        xyz_path = Path(tmpdir) / "test.txt"
        tif_path = Path(tmpdir) / "test.tif"

        # Create simple 3x3 XYZ data (1m resolution)
        # Origin at (100, 200), values increasing with position
        xyz_content = """\
100.5 202.5 10.0
101.5 202.5 11.0
102.5 202.5 12.0
100.5 201.5 13.0
101.5 201.5 14.0
102.5 201.5 15.0
100.5 200.5 16.0
101.5 200.5 17.0
102.5 200.5 18.0
"""
        xyz_path.write_text(xyz_content)

        result_path = _xyz_to_geotiff(xyz_path, tif_path)

        assert result_path == tif_path
        assert tif_path.exists()

        with rasterio.open(tif_path) as src:
            assert str(src.crs) == PROJECT_CRS
            assert src.nodata == DEFAULT_NODATA
            data = src.read(1)
            # Check grid dimensions (3x3 at 1m resolution)
            assert data.shape == (3, 3)
            # Check some values (top-left is highest Y, so row 0 = Y 202.5)
            assert data[0, 0] == 10.0  # X=100.5, Y=202.5
            assert data[0, 2] == 12.0  # X=102.5, Y=202.5
            assert data[2, 0] == 16.0  # X=100.5, Y=200.5
            assert data[2, 2] == 18.0  # X=102.5, Y=200.5


def test_xyz_to_geotiff_preserves_resolution():
    """Verify 1m resolution is maintained in transform."""
    with TemporaryDirectory() as tmpdir:
        xyz_path = Path(tmpdir) / "test.txt"

        # 2x2 grid at 1m spacing
        xyz_content = """\
0.5 1.5 1.0
1.5 1.5 2.0
0.5 0.5 3.0
1.5 0.5 4.0
"""
        xyz_path.write_text(xyz_content)

        tif_path = _xyz_to_geotiff(xyz_path)

        with rasterio.open(tif_path) as src:
            # Transform should have 1m pixel size
            assert abs(src.transform.a - XYZ_RESOLUTION) < 0.001
            assert abs(src.transform.e + XYZ_RESOLUTION) < 0.001  # Negative for Y


def test_xyz_to_geotiff_handles_sparse_data():
    """Handle XYZ data with gaps (sparse point cloud)."""
    with TemporaryDirectory() as tmpdir:
        xyz_path = Path(tmpdir) / "sparse.txt"

        # Only corners of a 3x3 grid
        xyz_content = """\
0.5 2.5 1.0
2.5 2.5 2.0
0.5 0.5 3.0
2.5 0.5 4.0
"""
        xyz_path.write_text(xyz_content)

        tif_path = _xyz_to_geotiff(xyz_path)

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            # Grid should be 3x3
            assert data.shape == (3, 3)
            # Corners should have values
            assert data[0, 0] == 1.0
            assert data[0, 2] == 2.0
            assert data[2, 0] == 3.0
            assert data[2, 2] == 4.0
            # Center should be nodata
            assert data[1, 1] == DEFAULT_NODATA


def test_xyz_to_geotiff_berlin_coordinates():
    """Test with realistic Berlin EPSG:25833 coordinates."""
    with TemporaryDirectory() as tmpdir:
        xyz_path = Path(tmpdir) / "berlin.txt"

        # Realistic Berlin coordinates (similar to actual data)
        xyz_content = """\
369999.50 5808323.50 29.68
369997.50 5808324.50 29.65
369998.50 5808324.50 29.57
369999.50 5808324.50 29.51
"""
        xyz_path.write_text(xyz_content)

        tif_path = _xyz_to_geotiff(xyz_path)

        with rasterio.open(tif_path) as src:
            assert str(src.crs) == PROJECT_CRS
            # Check bounds are in expected range
            assert 369990 < src.bounds.left < 370000
            assert 5808320 < src.bounds.bottom < 5808330
            # Should have valid elevation data
            data = src.read(1)
            valid_data = data[data != DEFAULT_NODATA]
            assert len(valid_data) == 4
            assert np.all((valid_data > 29.0) & (valid_data < 30.0))
