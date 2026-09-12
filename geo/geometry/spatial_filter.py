"""
Spatial object filtering module.
"""

from typing import Any, Dict, List, Union
import pyproj
from shapely.geometry.base import BaseGeometry

from geo.config import DEFAULT_CRS, logger
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs
from geo.utils.result_utils import format_response, format_error


def filter_objects_within(
    objects: List[Union[BaseGeometry, Dict[str, Any], Any]],
    target_geometry: Union[BaseGeometry, Dict[str, Any], str],
    predicate: str = "within",
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Filter a collection of objects (geometries or dicts containing geometries) based on 
    a spatial predicate relationship with a target geometry.
    
    Supported predicates: within, intersects, contains, overlaps, touches.
    
    Args:
        objects: List of Shapely geometries, GeoJSON dicts, or dictionaries containing geometry keys.
        target_geometry: Target spatial geometry (e.g., river buffer).
        predicate: Predicate to apply ("within", "intersects", "contains").
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary with count of matching objects, filtered objects, and indices.
    """
    try:
        logger.info(f"Filtering {len(objects)} objects using spatial predicate '{predicate}'")
        target = ensure_shapely_geometry(target_geometry)
        crs_obj = validate_crs(crs)

        matching_objects = []
        matching_indices = []

        pred_lower = predicate.lower()

        for idx, obj in enumerate(objects):
            if isinstance(obj, dict) and "geometry" in obj:
                geom = ensure_shapely_geometry(obj["geometry"])
            else:
                geom = ensure_shapely_geometry(obj)

            if pred_lower == "within":
                match = geom.within(target)
            elif pred_lower == "intersects":
                match = geom.intersects(target)
            elif pred_lower == "contains":
                match = geom.contains(target)
            elif pred_lower == "overlaps":
                match = geom.overlaps(target)
            elif pred_lower == "touches":
                match = geom.touches(target)
            else:
                match = geom.within(target)

            if match:
                matching_objects.append(obj)
                matching_indices.append(idx)

        logger.info(f"Spatial filtering completed: {len(matching_objects)} of {len(objects)} objects matched")

        return format_response(
            operation="spatial_filter",
            success=True,
            data={
                "count": len(matching_objects),
                "predicate": pred_lower,
                "objects": matching_objects,
                "matching_indices": matching_indices,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("spatial_filter", str(e))
