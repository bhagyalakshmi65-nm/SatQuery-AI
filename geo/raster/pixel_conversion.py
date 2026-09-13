"""
Pixel and geographic coordinate conversion module using raster geotransforms.
"""

from typing import Any, Dict, Tuple, Union
import rasterio.transform

from geo.config import logger
from geo.utils.result_utils import format_response, format_error


def pixel_to_geo(
    x: float,
    y: float,
    transform: Union[rasterio.transform.Affine, Tuple, list]
) -> Dict[str, Any]:
    """
    Convert image pixel coordinates (x column, y row) into geographic / spatial coordinates (lon/x, lat/y).
    
    Args:
        x: Pixel column index.
        y: Pixel row index.
        transform: Raster Affine transformation matrix or 6-element tuple.
        
    Returns:
        Structured response dictionary containing spatial coordinates (longitude, latitude).
    """
    try:
        if isinstance(transform, (tuple, list)):
            aff = rasterio.transform.Affine(*transform[:6])
        else:
            aff = transform

        try:
            geo_x, geo_y = aff @ (x, y)
        except TypeError:
            geo_x, geo_y = aff * (x, y)

        return format_response(
            operation="pixel_to_geo",
            success=True,
            data={
                "pixel_x": float(x),
                "pixel_y": float(y),
                "longitude": float(geo_x),
                "latitude": float(geo_y),
                "spatial_x": float(geo_x),
                "spatial_y": float(geo_y),
                "coordinates": (float(geo_x), float(geo_y))
            }
        )
    except Exception as e:
        return format_error("pixel_to_geo", str(e))


def geo_to_pixel(
    longitude: float,
    latitude: float,
    transform: Union[rasterio.transform.Affine, Tuple, list]
) -> Dict[str, Any]:
    """
    Convert geographic / spatial coordinates (longitude/x, latitude/y) into image pixel coordinates (col, row).
    
    Args:
        longitude: Spatial X coordinate / longitude.
        latitude: Spatial Y coordinate / latitude.
        transform: Raster Affine transformation matrix or 6-element tuple.
        
    Returns:
        Structured response dictionary containing pixel column and row indices.
    """
    try:
        if isinstance(transform, (tuple, list)):
            aff = rasterio.transform.Affine(*transform[:6])
        else:
            aff = transform

        inv_aff = ~aff
        try:
            pixel_x, pixel_y = inv_aff @ (longitude, latitude)
        except TypeError:
            pixel_x, pixel_y = inv_aff * (longitude, latitude)

        return format_response(
            operation="geo_to_pixel",
            success=True,
            data={
                "longitude": float(longitude),
                "latitude": float(latitude),
                "spatial_x": float(longitude),
                "spatial_y": float(latitude),
                "pixel_x": round(float(pixel_x), 2),
                "pixel_y": round(float(pixel_y), 2),
                "pixel_col": int(round(pixel_x)),
                "pixel_row": int(round(pixel_y))
            }
        )
    except Exception as e:
        return format_error("geo_to_pixel", str(e))
