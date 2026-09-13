"""
Geometry union, difference, and centroid operations.
"""

from typing import Any, Dict, List, Union
import pyproj
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geo.config import DEFAULT_CRS, logger
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs
from geo.utils.result_utils import format_response, format_error


def union_geometries(
    geometries: List[Union[BaseGeometry, Dict[str, Any], Any]],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Combine multiple geometries into a single unified geometry (unary union).
    
    Args:
        geometries: List of Shapely geometries or GeoJSON dicts.
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary containing unified geometry and GeoJSON mapping.
    """
    if not geometries:
        return format_error("union_geometries", "Geometry list is empty")

    try:
        logger.info(f"Performing unary union on {len(geometries)} geometries")
        crs_obj = validate_crs(crs)
        sh_geoms = [
            ensure_shapely_geometry(g["geometry"] if isinstance(g, dict) and "geometry" in g else g)
            for g in geometries
        ]

        unified_geom = unary_union(sh_geoms)

        return format_response(
            operation="union_geometries",
            success=True,
            data={
                "geometry": unified_geom,
                "geojson": mapping(unified_geom),
                "geom_type": unified_geom.geom_type,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("union_geometries", str(e))


def difference(
    geometry1: Union[BaseGeometry, Dict[str, Any], Any],
    geometry2: Union[BaseGeometry, Dict[str, Any], Any],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Compute the spatial difference (geometry1 minus geometry2).
    
    Args:
        geometry1: Source geometry.
        geometry2: Geometry to subtract / exclude.
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary with difference geometry.
    """
    try:
        logger.info("Computing geometry spatial difference")
        crs_obj = validate_crs(crs)
        g1 = ensure_shapely_geometry(geometry1["geometry"] if isinstance(geometry1, dict) and "geometry" in geometry1 else geometry1)
        g2 = ensure_shapely_geometry(geometry2["geometry"] if isinstance(geometry2, dict) and "geometry" in geometry2 else geometry2)

        diff_geom = g1.difference(g2)

        return format_response(
            operation="difference",
            success=True,
            data={
                "geometry": diff_geom,
                "geojson": mapping(diff_geom),
                "geom_type": diff_geom.geom_type,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("difference", str(e))


def calculate_centroid(
    geometry: Union[BaseGeometry, Dict[str, Any], Any],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Calculate the geometric centroid of a geometry.
    
    Args:
        geometry: Input geometry.
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary containing centroid coordinates and GeoJSON Point.
    """
    try:
        crs_obj = validate_crs(crs)
        g = ensure_shapely_geometry(geometry["geometry"] if isinstance(geometry, dict) and "geometry" in geometry else geometry)
        if g.is_empty:
            return format_error("calculate_centroid", "Cannot calculate centroid for empty geometry")
        centroid = g.centroid

        return format_response(
            operation="calculate_centroid",
            success=True,
            data={
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
                "coordinates": (float(centroid.x), float(centroid.y)),
                "geometry": centroid,
                "geojson": mapping(centroid),
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("calculate_centroid", str(e))
