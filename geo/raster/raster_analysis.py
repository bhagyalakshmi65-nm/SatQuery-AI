"""
Specific raster dataset analysis and inspection utilities.
"""

import os
from typing import Any, Dict
import rasterio

from geo.config import logger
from geo.utils.result_utils import format_response, format_error


def get_raster_crs(image_path: str) -> Dict[str, Any]:
    """
    Extract the Coordinate Reference System (CRS) of a raster dataset.
    
    Args:
        image_path: File path to the raster image.
        
    Returns:
        Structured response dictionary containing raster CRS string and EPSG code.
    """
    if not os.path.exists(image_path):
        return format_error("get_raster_crs", f"Raster file not found at '{image_path}'")

    try:
        with rasterio.open(image_path) as dataset:
            crs_str = dataset.crs.to_string() if dataset.crs else "Unknown"
            epsg = dataset.crs.to_epsg() if dataset.crs else None

            return format_response(
                operation="get_raster_crs",
                success=True,
                data={
                    "crs": crs_str,
                    "epsg": epsg,
                    "is_geographic": dataset.crs.is_geographic if dataset.crs else None
                }
            )
    except Exception as e:
        return format_error("get_raster_crs", str(e))


def get_raster_resolution(image_path: str) -> Dict[str, Any]:
    """
    Extract pixel resolution (width and height) of a raster dataset in spatial CRS units.
    
    Args:
        image_path: File path to the raster image.
        
    Returns:
        Structured response dictionary containing pixel resolution tuple.
    """
    if not os.path.exists(image_path):
        return format_error("get_raster_resolution", f"Raster file not found at '{image_path}'")

    try:
        with rasterio.open(image_path) as dataset:
            res_x, res_y = abs(float(dataset.res[0])), abs(float(dataset.res[1]))

            return format_response(
                operation="get_raster_resolution",
                success=True,
                data={
                    "resolution": (res_x, res_y),
                    "resolution_x": res_x,
                    "resolution_y": res_y,
                    "unit": dataset.crs.linear_units if dataset.crs and not dataset.crs.is_geographic else "degrees"
                }
            )
    except Exception as e:
        return format_error("get_raster_resolution", str(e))
