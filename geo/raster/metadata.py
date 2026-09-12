"""
Raster metadata extraction module.
"""

import os
from typing import Any, Dict, Union
import rasterio

from geo.utils.validation import format_response


def get_raster_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract geospatial and structural metadata from a GeoTIFF or geospatial raster image.
    
    Extracted fields:
        width: Image width in pixels
        height: Image height in pixels
        crs: Coordinate Reference System string
        bounds: [left, bottom, right, top] spatial extent
        transform: 6-element Affine transform matrix tuple
        resolution: (pixel_width, pixel_height) in spatial CRS units
        bands: Number of raster bands
        driver: Raster format driver (e.g. GTiff)
        nodata: Nodata pixel value if specified
        
    Args:
        image_path: Absolute or relative file path to the raster image.
        
    Returns:
        Structured response dictionary containing raster metadata.
    """
    if not os.path.exists(image_path):
        return format_response(
            operation="get_raster_metadata",
            success=False,
            error=f"Raster file not found at path '{image_path}'"
        )

    try:
        with rasterio.open(image_path) as dataset:
            crs_str = dataset.crs.to_string() if dataset.crs else "Unprojected / Unknown"
            bounds = [
                float(dataset.bounds.left),
                float(dataset.bounds.bottom),
                float(dataset.bounds.right),
                float(dataset.bounds.top)
            ]
            aff = dataset.transform
            transform_tuple = (aff.a, aff.b, aff.c, aff.d, aff.e, aff.f)
            res = (abs(float(dataset.res[0])), abs(float(dataset.res[1])))

            metadata = {
                "width": int(dataset.width),
                "height": int(dataset.height),
                "crs": crs_str,
                "bounds": bounds,
                "transform": transform_tuple,
                "resolution": res,
                "bands": int(dataset.count),
                "driver": str(dataset.driver),
                "nodata": dataset.nodata
            }

            return format_response(
                operation="get_raster_metadata",
                success=True,
                data=metadata
            )
    except Exception as e:
        return format_response(
            operation="get_raster_metadata",
            success=False,
            error=f"Failed to read raster metadata: {str(e)}"
        )
