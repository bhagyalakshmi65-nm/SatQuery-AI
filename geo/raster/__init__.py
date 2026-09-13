"""
Raster package initialization.
"""

from geo.raster.metadata import get_raster_metadata
from geo.raster.bounds import get_raster_bounds
from geo.raster.coordinate_system import (
    pixel_to_coord,
    coord_to_pixel,
    transform_crs
)
from geo.raster.pixel_conversion import (
    pixel_to_geo,
    geo_to_pixel
)
from geo.raster.raster_analysis import (
    get_raster_crs,
    get_raster_resolution
)

__all__ = [
    "get_raster_metadata",
    "get_raster_bounds",
    "pixel_to_coord",
    "coord_to_pixel",
    "transform_crs",
    "pixel_to_geo",
    "geo_to_pixel",
    "get_raster_crs",
    "get_raster_resolution"
]
