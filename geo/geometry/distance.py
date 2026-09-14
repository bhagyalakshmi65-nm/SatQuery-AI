"""
Distance calculation module for geographic and projected geometries.
"""

from typing import Any, Dict, Optional, Union, Tuple
import pyproj
from pyproj import Geod, Transformer
import shapely.ops
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from geo.config import DEFAULT_CRS, DISTANCE_UNIT_FACTORS, logger
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs, estimate_utm_crs
from geo.utils.result_utils import format_response, format_error


def calculate_distance(
    point1: Union[BaseGeometry, Dict[str, Any], Tuple[float, float], list],
    point2: Union[BaseGeometry, Dict[str, Any], Tuple[float, float], list],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS,
    unit: str = "meters"
) -> Dict[str, Any]:
    """
    Calculate geographic or projected distance between two geometries/coordinates.
    
    Supports: Point-to-Point, Point-to-Line, Point-to-Polygon, Polygon-to-Polygon.
    
    Args:
        point1: First geometry (Shapely Point/Line/Polygon, GeoJSON, or (lon, lat) tuple).
        point2: Second geometry.
        crs: Coordinate Reference System of input geometries (default: EPSG:4326).
        unit: Target unit for distance result ("meters", "kilometers", "feet", "miles").
        
    Returns:
        Structured response dictionary containing calculated distance_m and unit.
    """
    try:
        logger.info("Calculating metric distance between geometries")
        g1 = ensure_shapely_geometry(point1)
        g2 = ensure_shapely_geometry(point2)

        if g1.is_empty or g2.is_empty:
            return format_error("distance", "Cannot calculate distance for empty geometry")

        crs_obj = validate_crs(crs)

        if crs_obj.is_geographic:
            lon1, lat1 = g1.centroid.x, g1.centroid.y
            lon2, lat2 = g2.centroid.x, g2.centroid.y
            
            if g1.geom_type == "Point" and g2.geom_type == "Point":
                geod = Geod(ellps="WGS84")
                _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)
            else:
                # Reproject to estimated local UTM zone to compute exact minimum metric distance
                utm_crs = estimate_utm_crs((lon1 + lon2) / 2.0, (lat1 + lat2) / 2.0)
                transformer = Transformer.from_crs(crs_obj, utm_crs, always_xy=True)
                
                g1_proj = shapely.ops.transform(transformer.transform, g1)
                g2_proj = shapely.ops.transform(transformer.transform, g2)
                distance_m = g1_proj.distance(g2_proj)
        else:
            distance_m = g1.distance(g2)

        unit_lower = unit.lower()
        if unit_lower not in DISTANCE_UNIT_FACTORS:
            valid_units = ", ".join(DISTANCE_UNIT_FACTORS.keys())
            raise ValueError(f"Unsupported unit '{unit}'. Valid units are: {valid_units}")
            
        factor = DISTANCE_UNIT_FACTORS[unit_lower]
        converted_distance = float(distance_m * factor)

        return format_response(
            operation="distance",
            success=True,
            data={
                "distance": round(converted_distance, 4),
                "distance_m": round(float(distance_m), 4),
                "unit": unit,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("distance", str(e))
