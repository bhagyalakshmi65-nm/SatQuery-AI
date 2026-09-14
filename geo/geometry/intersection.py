"""
Intersection calculation module for geospatial geometries.
"""

from typing import Any, Dict, Union
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping
from pyproj import Transformer
import shapely.ops

from geo.config import DEFAULT_CRS
from geo.utils import ensure_shapely_geometry, validate_crs, format_response, format_error, estimate_utm_crs


def find_intersection(
    geometry1: Union[BaseGeometry, Dict[str, Any], str],
    geometry2: Union[BaseGeometry, Dict[str, Any], str],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Find the spatial intersection between two geometries and extract useful metadata.
    
    Args:
        geometry1: First input geometry.
        geometry2: Second input geometry.
        crs: Coordinate Reference System of geometries (default: EPSG:4326).
        
    Returns:
        Structured response dictionary containing intersection status, geometry, and area.
    """
    try:
        g1 = ensure_shapely_geometry(geometry1)
        g2 = ensure_shapely_geometry(geometry2)

        if g1.is_empty or g2.is_empty:
            return format_response(
                operation="intersection",
                success=True,
                data={
                    "intersects": False,
                    "geometry": None,
                    "geojson": None,
                    "intersection_area_m2": 0.0
                }
            )

        crs_obj = validate_crs(crs)

        intersects = g1.intersects(g2)
        if not intersects:
            return format_response(
                operation="intersection",
                success=True,
                data={
                    "intersects": False,
                    "geometry": None,
                    "geojson": None,
                    "intersection_area_m2": 0.0
                }
            )

        intersection_geom = g1.intersection(g2)
        
        # Calculate intersection area in square meters if intersecting geometry is 2D (Polygon/MultiPolygon)
        intersection_area_m2 = 0.0
        if intersection_geom.geom_type in ["Polygon", "MultiPolygon", "GeometryCollection"]:
            if crs_obj.is_geographic:
                centroid = intersection_geom.centroid
                utm_crs = estimate_utm_crs(centroid.x, centroid.y)
                transformer = Transformer.from_crs(crs_obj, utm_crs, always_xy=True)
                proj_geom = shapely.ops.transform(transformer.transform, intersection_geom)
                intersection_area_m2 = proj_geom.area
            else:
                intersection_area_m2 = intersection_geom.area

        return format_response(
            operation="intersection",
            success=True,
            data={
                "intersects": True,
                "geometry": intersection_geom,
                "geojson": mapping(intersection_geom),
                "intersection_area_m2": round(float(intersection_area_m2), 4)
            }
        )
    except Exception as e:
        return format_response(
            operation="intersection",
            success=False,
            error=str(e)
        )
