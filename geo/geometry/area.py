"""
Area calculation and statistics module for polygon geometries.
"""

from typing import Any, Dict, List, Union
import pyproj
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
import shapely.ops

from geo.config import DEFAULT_CRS, AREA_UNIT_FACTORS, logger
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs, estimate_utm_crs
from geo.utils.result_utils import format_response, format_error


def calculate_area(
    polygon: Union[BaseGeometry, Dict[str, Any], str],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS,
    unit: str = "square_meters"
) -> Dict[str, Any]:
    """
    Calculate the real-world surface area of a polygon geometry.
    
    Automatically transforms geographic geometries (lat/lon) into an appropriate projected equal-area CRS
    before calculating area.
    
    Args:
        polygon: Input polygon geometry.
        crs: Coordinate Reference System of input polygon (default: EPSG:4326).
        unit: Target area unit ("square_meters", "square_kilometers", "hectares", "acres", "square_feet").
        
    Returns:
        Structured response dictionary containing calculated area_m2 and unit.
    """
    try:
        logger.info("Calculating polygon surface area")
        geom = ensure_shapely_geometry(polygon)
        if geom.is_empty:
            return format_error("area", "Cannot calculate area for empty polygon")

        crs_obj = validate_crs(crs)

        if geom.geom_type not in ["Polygon", "MultiPolygon", "GeometryCollection"]:
            raise ValueError(f"Area calculation requires Polygon or MultiPolygon geometry, got '{geom.geom_type}'")

        if crs_obj.is_geographic:
            centroid = geom.centroid
            utm_crs = estimate_utm_crs(centroid.x, centroid.y)
            transformer = Transformer.from_crs(crs_obj, utm_crs, always_xy=True)
            proj_geom = shapely.ops.transform(transformer.transform, geom)
            area_m2 = proj_geom.area
        else:
            area_m2 = geom.area

        unit_lower = unit.lower()
        if unit_lower not in AREA_UNIT_FACTORS:
            valid_units = ", ".join(AREA_UNIT_FACTORS.keys())
            raise ValueError(f"Unsupported unit '{unit}'. Valid units are: {valid_units}")
            
        factor = AREA_UNIT_FACTORS[unit_lower]
        converted_area = float(area_m2 * factor)

        return format_response(
            operation="area",
            success=True,
            data={
                "area": round(converted_area, 4),
                "area_m2": round(float(area_m2), 4),
                "unit": unit,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("area", str(e))


def calculate_area_statistics(
    geometries: List[Union[BaseGeometry, Dict[str, Any], str]],
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS,
    unit: str = "square_meters"
) -> Dict[str, Any]:
    """
    Calculate area statistics across multiple polygon geometries (total, average, min, max area).
    
    Args:
        geometries: List of polygon geometries or dicts.
        crs: Coordinate Reference System.
        unit: Measurement unit.
        
    Returns:
        Structured response dictionary containing total_area, average_area, min_area, max_area.
    """
    if not geometries:
        return format_error("area_statistics", "Geometry list is empty")

    try:
        areas_m2 = []
        for g in geometries:
            res = calculate_area(g, crs=crs, unit="square_meters")
            if res["success"]:
                areas_m2.append(res["area_m2"])

        if not areas_m2:
            return format_error("area_statistics", "No valid polygons found to calculate area statistics")

        total_m2 = float(sum(areas_m2))
        avg_m2 = float(total_m2 / len(areas_m2))
        min_m2 = float(min(areas_m2))
        max_m2 = float(max(areas_m2))

        unit_lower = unit.lower()
        factor = AREA_UNIT_FACTORS.get(unit_lower, 1.0)

        return format_response(
            operation="area_statistics",
            success=True,
            data={
                "count": len(areas_m2),
                "total_area": round(total_m2 * factor, 4),
                "total_area_m2": round(total_m2, 4),
                "average_area": round(avg_m2 * factor, 4),
                "average_area_m2": round(avg_m2, 4),
                "min_area": round(min_m2 * factor, 4),
                "min_area_m2": round(min_m2, 4),
                "max_area": round(max_m2 * factor, 4),
                "max_area_m2": round(max_m2, 4),
                "unit": unit
            }
        )
    except Exception as e:
        return format_error("area_statistics", str(e))
