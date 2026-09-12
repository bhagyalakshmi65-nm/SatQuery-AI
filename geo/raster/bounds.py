"""
Raster spatial bounding box module.
"""

import os
from typing import Any, Dict, Union
import rasterio
from pyproj import Transformer
from shapely.geometry import box, Polygon, mapping
import shapely.ops

from geo.config import DEFAULT_CRS
from geo.utils import format_response, format_error, validate_crs


def get_raster_bounds(
    image_path: str,
    target_crs: Union[str, int, Any] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Extract spatial bounding box of a raster image in native and reprojected CRS.
    
    Args:
        image_path: File path to the raster image.
        target_crs: Target CRS to reproject bounds into (default: EPSG:4326).
        
    Returns:
        Structured response dictionary containing bounding box polygons and coordinate extents.
    """
    if not os.path.exists(image_path):
        return format_response(
            operation="get_raster_bounds",
            success=False,
            error=f"Raster file not found at path '{image_path}'"
        )

    try:
        with rasterio.open(image_path) as src:
            native_bounds = [
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top)
            ]
            native_poly = box(*native_bounds)
            native_crs_str = src.crs.to_string() if src.crs else None

            target_crs_obj = validate_crs(target_crs)
            
            if src.crs and src.crs != target_crs_obj:
                transformer = Transformer.from_crs(src.crs, target_crs_obj, always_xy=True)
                reprojected_poly = shapely.ops.transform(transformer.transform, native_poly)
                reprojected_bounds = list(reprojected_poly.bounds)
            else:
                reprojected_poly = native_poly
                reprojected_bounds = native_bounds

            return format_response(
                operation="get_raster_bounds",
                success=True,
                data={
                    "native_bounds": native_bounds,
                    "native_crs": native_crs_str,
                    "target_crs": target_crs_obj.to_string(),
                    "target_bounds": reprojected_bounds,
                    "geometry": reprojected_poly,
                    "geojson": mapping(reprojected_poly)
                }
            )
    except Exception as e:
        return format_response(
            operation="get_raster_bounds",
            success=False,
            error=f"Failed to extract raster bounds: {str(e)}"
        )
