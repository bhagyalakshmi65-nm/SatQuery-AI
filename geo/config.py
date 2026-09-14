"""
Configuration settings and logging setup for the Geo/GIS Engine.
"""

import logging

# Logger setup
logger = logging.getLogger("GeoEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default Coordinate Reference Systems (CRS)
DEFAULT_CRS = "EPSG:4326"           # WGS 84 Geographic Coordinates (Lat/Lon)
DEFAULT_PROJECTED_CRS = "EPSG:3857" # Web Mercator (used for default metric projections)

# Supported Geometry Types (Shapely standard types)
SUPPORTED_GEOMETRY_TYPES = [
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection"
]

# Supported Measurement Units
DEFAULT_DISTANCE_UNIT = "meters"
DEFAULT_AREA_UNIT = "square_meters"

DISTANCE_UNIT_FACTORS = {
    "meters": 1.0,
    "m": 1.0,
    "kilometers": 0.001,
    "km": 0.001,
    "feet": 3.28084,
    "ft": 3.28084,
    "miles": 0.000621371,
    "mi": 0.000621371
}

AREA_UNIT_FACTORS = {
    "square_meters": 1.0,
    "sq_m": 1.0,
    "m2": 1.0,
    "square_kilometers": 1e-6,
    "sq_km": 1e-6,
    "km2": 1e-6,
    "hectares": 1e-4,
    "ha": 1e-4,
    "acres": 0.000247105,
    "square_feet": 10.7639,
    "sq_ft": 10.7639
}

# Validation & Healing Settings
STRICT_VALIDATION = False
AUTO_FIX_INVALID_GEOMETRY = True

# Supported Raster Image Extensions
SUPPORTED_RASTER_EXTENSIONS = [".tif", ".tiff", ".geotiff", ".jp2", ".png", ".jpg"]
