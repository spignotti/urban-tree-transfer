"""Project-wide constants."""

# Coordinate Systems
PROJECT_CRS = "EPSG:25833"  # UTM Zone 33N (Berlin & Leipzig native)

# Processing Parameters
RANDOM_SEED = 42
CHM_REFERENCE_YEAR = 2021
MIN_SAMPLES_PER_GENUS = 500

# Sentinel-2 Bands
SPECTRAL_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
VEGETATION_INDICES = [
    "NDVI",
    "EVI",
    "GNDVI",
    "NDre1",
    "NDVIre",
    "CIre",
    "IRECI",
    "RTVIcore",
    "NDWI",
    "MSI",
    "NDII",
    "kNDVI",
    "VARI",
]

# CHM Parameters
CHM_MIN_VALID = -2.0  # Values below set to 0
CHM_MAX_VALID = 50.0  # Values above set to NoData

# GDAL Options
GDAL_COMPRESS_OPTIONS = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
