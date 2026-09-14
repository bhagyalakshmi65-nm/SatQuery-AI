"""
Coordinate Reference System (CRS) management and projection utilities.
"""

from typing import Any, Dict, List, Tuple, Union
import pyproj
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
import shapely.ops

from geo.config import DEFAULT_CRS, DEFAULT_PROJECTED_CRS, logger
from geo.utils.result_utils import format_response, format_error


def validate_crs(crs_input: Union[str, int, CRS, Dict[str, Any], None]) -> CRS:
    """
    Validate and parse a Coordinate Reference System input into a pyproj.CRS instance.
    
    Args:
        crs_input: EPSG string ("EPSG:4326"), integer (4326), PyProj CRS object, or dict.
        
    Returns:
        pyproj.CRS instance.
        
    Raises:
        ValueError: If CRS input cannot be parsed or is invalid.
    """
    if crs_input is None:
        crs_input = DEFAULT_CRS
        
    try:
        if isinstance(crs_input, CRS):
            return crs_input
        elif isinstance(crs_input, int):
            return CRS.from_epsg(crs_input)
        elif isinstance(crs_input, str):
            return CRS.from_user_input(crs_input)
        elif isinstance(crs_input, dict):
            return CRS.from_dict(crs_input)
        else:
            return CRS.from_user_input(crs_input)
    except Exception as e:
        raise ValueError(f"Invalid Coordinate Reference System (CRS) '{crs_input}': {str(e)}")


def estimate_utm_crs(lon: float, lat: float) -> CRS:
    """
    Estimate the optimal WGS84 UTM projected CRS for a given longitude and latitude.
    
    Args:
        lon: Longitude (-180 to 180)
        lat: Latitude (-90 to 90)
        
    Returns:
        pyproj.CRS corresponding to the appropriate UTM zone.
    """
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    return CRS.from_epsg(epsg_code)


def get_projected_crs(geometry: BaseGeometry, crs: Union[str, int, CRS, None] = None) -> CRS:
    """
    Determine an appropriate metric projected CRS for a geometry.
    If the current CRS is geographic (lat/lon), returns the estimated local UTM CRS.
    If already projected, returns the validated CRS.
    """
    crs_obj = validate_crs(crs)
    if crs_obj.is_geographic:
        centroid = geometry.centroid
        return estimate_utm_crs(centroid.x, centroid.y)
    return crs_obj


def transform_geometry(
    geometry: BaseGeometry,
    source_crs: Union[str, int, CRS],
    target_crs: Union[str, int, CRS]
) -> BaseGeometry:
    """
    Reproject a Shapely geometry from source_crs to target_crs.
    """
    src_obj = validate_crs(source_crs)
    dst_obj = validate_crs(target_crs)

    if src_obj == dst_obj:
        return geometry

    transformer = Transformer.from_crs(src_obj, dst_obj, always_xy=True)
    return shapely.ops.transform(transformer.transform, geometry)


def transform_coordinates(
    coords: Union[Tuple[float, float], List[Tuple[float, float]]],
    source_crs: Union[str, int, CRS],
    target_crs: Union[str, int, CRS]
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Reproject coordinate tuple(s) from source_crs to target_crs.
    """
    src_obj = validate_crs(source_crs)
    dst_obj = validate_crs(target_crs)

    if src_obj == dst_obj:
        return coords

    transformer = Transformer.from_crs(src_obj, dst_obj, always_xy=True)
    if isinstance(coords, tuple) and len(coords) == 2 and isinstance(coords[0], (int, float)):
        x, y = transformer.transform(coords[0], coords[1])
        return (float(x), float(y))
    elif isinstance(coords, list):
        res = []
        for x, y in coords:
            tx, ty = transformer.transform(x, y)
            res.append((float(tx), float(ty)))
        return res
    else:
        raise ValueError(f"Unsupported coordinate format: {coords}")
