"""
Spatial relationship and filtering module for geospatial geometries.
"""

from typing import Any, Dict, List, Union
import pyproj
from shapely.geometry.base import BaseGeometry

from geo.config import DEFAULT_CRS
from geo.utils import ensure_shapely_geometry, validate_crs, format_response, format_error


def check_spatial_relation(
    geometry1: Union[BaseGeometry, Dict[str, Any], str],
    geometry2: Union[BaseGeometry, Dict[str, Any], str],
    predicate: str = "intersects",
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Check spatial relationship predicate between two geometries.
    
    Supported predicates: intersects, contains, within, touches, overlaps, disjoint, crosses, equals.
    
    Args:
        geometry1: First input geometry.
        geometry2: Second input geometry.
        predicate: Predicate name string.
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary with predicate result boolean.
    """
    try:
        g1 = ensure_shapely_geometry(geometry1)
        g2 = ensure_shapely_geometry(geometry2)
        crs_obj = validate_crs(crs)

        pred_lower = predicate.lower()
        if pred_lower == "intersects":
            result = g1.intersects(g2)
        elif pred_lower == "contains":
            result = g1.contains(g2)
        elif pred_lower == "within":
            result = g1.within(g2)
        elif pred_lower == "touches":
            result = g1.touches(g2)
        elif pred_lower == "overlaps":
            result = g1.overlaps(g2)
        elif pred_lower == "disjoint":
            result = g1.disjoint(g2)
        elif pred_lower == "crosses":
            result = g1.crosses(g2)
        elif pred_lower == "equals":
            result = g1.equals(g2)
        else:
            valid_preds = "intersects, contains, within, touches, overlaps, disjoint, crosses, equals"
            raise ValueError(f"Unsupported spatial predicate '{predicate}'. Supported: {valid_preds}")

        return format_response(
            operation=f"spatial_relation_{pred_lower}",
            success=True,
            data={
                "predicate": pred_lower,
                "result": bool(result),
                pred_lower: bool(result)
            }
        )
    except Exception as e:
        return format_response(
            operation=f"spatial_relation_{predicate}",
            success=False,
            error=str(e)
        )


def is_intersects(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="intersects", crs=crs)


def is_contains(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="contains", crs=crs)


def is_within(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="within", crs=crs)


def is_touches(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="touches", crs=crs)


def is_overlaps(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="overlaps", crs=crs)


def is_disjoint(g1: Any, g2: Any, crs: Any = DEFAULT_CRS) -> Dict[str, Any]:
    return check_spatial_relation(g1, g2, predicate="disjoint", crs=crs)


def filter_objects_within(
    objects: List[Union[BaseGeometry, Dict[str, Any], Any]],
    target_geometry: Union[BaseGeometry, Dict[str, Any], str],
    predicate: str = "within",
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Filter a collection of objects (geometries or dicts containing geometries) based on 
    a spatial predicate relationship with a target geometry.
    
    Example use-case:
        Filter building bounding boxes/polygons that fall within a 100m river buffer.
        
    Args:
        objects: List of Shapely geometries, GeoJSON dicts, or dictionaries containing geometry keys.
        target_geometry: Target spatial geometry (e.g., river buffer).
        predicate: Predicate to apply ("within", "intersects", "contains").
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary with count of matching objects, filtered objects, and indices.
    """
    try:
        target = ensure_shapely_geometry(target_geometry)
        crs_obj = validate_crs(crs)

        matching_objects = []
        matching_indices = []

        pred_lower = predicate.lower()

        for idx, obj in enumerate(objects):
            # Extract geometry if object is a dict with 'geometry' key, else treat object as geometry
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

        return format_response(
            operation="filter_objects_within",
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
        return format_response(
            operation="filter_objects_within",
            success=False,
            error=str(e)
        )
