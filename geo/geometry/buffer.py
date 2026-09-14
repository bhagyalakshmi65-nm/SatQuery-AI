"""
Buffer creation module for geospatial geometries.
"""

import math
from typing import Any, Dict, Union
import pyproj
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
import shapely.ops
from shapely.geometry import mapping

from geo.config import DEFAULT_CRS
from geo.utils import ensure_shapely_geometry, validate_crs, format_response, format_error, estimate_utm_crs


def create_buffer(
    geometry: Union[BaseGeometry, Dict[str, Any], str],
    distance_m: float,
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Create a spatial buffer around a geometry by a specified distance in meters.
    
    If the geometry is in geographic coordinates (lat/lon), it is reprojected to a metric 
    projected CRS (UTM), buffered by distance_m, and reprojected back to the original CRS.
    
    Args:
        geometry: Input geometry (Shapely object, GeoJSON dict, or WKT string).
        distance_m: Buffer radius in meters.
        crs: Coordinate Reference System of input geometry (default: EPSG:4326).
        
    Returns:
        Structured response dictionary containing the buffered Shapely geometry and GeoJSON mapping.
    """
    try:
        if distance_m is None or not isinstance(distance_m, (int, float)) or isinstance(distance_m, bool):
            return format_error("buffer", "Invalid distance_m: distance must be a valid number")
        if math.isnan(distance_m) or math.isinf(distance_m):
            return format_error("buffer", "Invalid distance_m: distance cannot be NaN or infinity")
        if distance_m < 0:
            return format_error("buffer", "Invalid distance_m: distance cannot be negative")

        geom = ensure_shapely_geometry(geometry)
        if geom.is_empty:
            return format_error("buffer", "Geometry is empty")

        crs_obj = validate_crs(crs)

        if crs_obj.is_geographic:
            centroid = geom.centroid
            utm_crs = estimate_utm_crs(centroid.x, centroid.y)
            
            # Forward transformer: Geographic -> UTM
            to_utm = Transformer.from_crs(crs_obj, utm_crs, always_xy=True)
            proj_geom = shapely.ops.transform(to_utm.transform, geom)
            
            # Buffer in meters
            buffered_proj = proj_geom.buffer(distance_m)
            
            # Reverse transformer: UTM -> Geographic
            to_geo = Transformer.from_crs(utm_crs, crs_obj, always_xy=True)
            buffered_geom = shapely.ops.transform(to_geo.transform, buffered_proj)
        else:
            buffered_geom = geom.buffer(distance_m)

        return format_response(
            operation="buffer",
            success=True,
            data={
                "distance_m": distance_m,
                "geometry": buffered_geom,
                "geojson": mapping(buffered_geom),
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_response(
            operation="buffer",
            success=False,
            error=str(e)
        )
