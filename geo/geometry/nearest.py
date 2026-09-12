"""
Nearest feature discovery module.
"""

from typing import Any, Dict, List, Optional, Union
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from geo.config import DEFAULT_CRS, logger
from geo.geometry.distance import calculate_distance
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs
from geo.utils.result_utils import format_response, format_error


def find_nearest(
    source: Union[BaseGeometry, Dict[str, Any], Any],
    objects: List[Union[BaseGeometry, Dict[str, Any], Any]],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Find the closest object to a source geometry from a list of objects.
    
    Args:
        source: Source geometry (e.g. airport or query point).
        objects: List of target candidate objects or geometries (e.g. buildings).
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary with nearest object, distance_m, and index.
    """
    if not objects:
        return format_error("nearest", "Target objects list is empty")

    try:
        logger.info(f"Finding nearest feature among {len(objects)} candidate objects")
        crs_obj = validate_crs(crs)
        src_geom = ensure_shapely_geometry(source["geometry"] if isinstance(source, dict) and "geometry" in source else source)

        min_dist_m = float("inf")
        nearest_obj = None
        nearest_idx = -1

        for idx, obj in enumerate(objects):
            target_geom = ensure_shapely_geometry(obj["geometry"] if isinstance(obj, dict) and "geometry" in obj else obj)
            dist_res = calculate_distance(src_geom, target_geom, crs=crs_obj, unit="meters")
            
            if dist_res["success"]:
                d_m = dist_res["distance_m"]
                if d_m < min_dist_m:
                    min_dist_m = d_m
                    nearest_obj = obj
                    nearest_idx = idx

        if nearest_idx == -1:
            return format_error("nearest", "Could not compute distance to candidate objects")

        logger.info(f"Nearest object found at index {nearest_idx} with distance {min_dist_m:.2f} meters")

        return format_response(
            operation="nearest",
            success=True,
            data={
                "distance_m": round(min_dist_m, 2),
                "object": nearest_obj,
                "index": nearest_idx,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("nearest", str(e))
