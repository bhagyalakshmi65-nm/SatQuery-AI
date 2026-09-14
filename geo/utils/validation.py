"""
Geometry validation and conversion utilities.
"""

from typing import Any, Dict, Optional, Tuple, Union
from shapely.geometry import shape, Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
try:
    from shapely import explain_valid
except ImportError:
    try:
        from shapely.validation import explain_valid
    except ImportError:
        def explain_valid(geom): return "Invalid geometry"
import shapely.wkt

from geo.config import (
    DEFAULT_CRS,
    AUTO_FIX_INVALID_GEOMETRY,
    STRICT_VALIDATION,
    SUPPORTED_GEOMETRY_TYPES,
    logger
)
from geo.utils.result_utils import format_response, format_error


def validate_geometry(
    geometry: Union[BaseGeometry, Dict[str, Any], str, Tuple, list, None]
) -> Dict[str, Any]:
    """
    Validate a geometry for existence, non-emptiness, correctness, and supported geometry type.
    
    Args:
        geometry: Geometry input (Shapely object, GeoJSON dict, WKT string, or tuple/list).
        
    Returns:
        Structured response dictionary with validation status.
    """
    if geometry is None:
        return format_error("validate_geometry", "Geometry input is None or missing")

    try:
        sh_geom = ensure_shapely_geometry(geometry, auto_fix=False)
    except Exception as e:
        return format_error("validate_geometry", f"Failed to parse geometry: {str(e)}")

    if sh_geom.is_empty:
        return format_error("validate_geometry", "Geometry is empty")

    if sh_geom.geom_type not in SUPPORTED_GEOMETRY_TYPES:
        return format_error(
            "validate_geometry",
            f"Unsupported geometry type '{sh_geom.geom_type}'. Supported: {SUPPORTED_GEOMETRY_TYPES}"
        )

    if not sh_geom.is_valid:
        reason = explain_valid(sh_geom)
        return format_error("validate_geometry", f"Invalid geometry: {reason}")

    return format_response(
        operation="validate_geometry",
        success=True,
        data={
            "valid": True,
            "geometry": sh_geom,
            "geom_type": sh_geom.geom_type,
            "is_empty": False
        }
    )


def ensure_shapely_geometry(
    geom_input: Union[BaseGeometry, Dict[str, Any], str, Tuple, list],
    auto_fix: bool = True
) -> BaseGeometry:
    """
    Convert various geometry input types into a valid Shapely BaseGeometry object.
    
    Args:
        geom_input: Shapely Geometry, GeoJSON dict, Feature dict, WKT string, or coordinate list/tuple.
        auto_fix: If True, uses shapely.validation.make_valid to heal invalid geometries.
        
    Returns:
        Shapely BaseGeometry object.
        
    Raises:
        ValueError: If conversion fails or geometry is invalid and auto-fix is False.
    """
    if geom_input is None:
        raise ValueError("Geometry input cannot be None")
        
    geom = None
    
    if isinstance(geom_input, BaseGeometry):
        geom = geom_input
    elif isinstance(geom_input, dict):
        if geom_input.get("type") == "Feature" and "geometry" in geom_input:
            geom = shape(geom_input["geometry"])
        elif "type" in geom_input and "coordinates" in geom_input:
            geom = shape(geom_input)
        elif geom_input.get("type") == "FeatureCollection" and "features" in geom_input:
            geoms = [shape(f["geometry"]) for f in geom_input["features"] if "geometry" in f]
            geom = GeometryCollection(geoms)
        else:
            raise ValueError(f"Unrecognized GeoJSON format: {geom_input}")
    elif isinstance(geom_input, str):
        try:
            geom = shapely.wkt.loads(geom_input)
        except Exception:
            raise ValueError(f"Could not parse string as WKT geometry: '{geom_input}'")
    elif isinstance(geom_input, (tuple, list)):
        if len(geom_input) == 2:
            geom = Point(geom_input[0], geom_input[1])
        elif len(geom_input) >= 3 and isinstance(geom_input[0], (tuple, list)):
            geom = Polygon(geom_input)
        else:
            raise ValueError(f"Cannot infer geometry from list/tuple: {geom_input}")
    else:
        raise ValueError(f"Unsupported geometry input type: {type(geom_input)}")
        
    if not geom.is_valid:
        if auto_fix or AUTO_FIX_INVALID_GEOMETRY:
            logger.info("Healing invalid geometry using shapely.validation.make_valid")
            geom = make_valid(geom)
        elif STRICT_VALIDATION:
            raise ValueError(f"Invalid geometry detected: {geom_input}")
            
    return geom
