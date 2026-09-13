"""
Geometry conversion utilities for YOLO bounding boxes and spatial geometries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pyproj
from shapely.geometry import box, Polygon, mapping
from shapely.geometry.base import BaseGeometry
import rasterio.transform

from geo.config import DEFAULT_CRS, logger
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.crs_utils import validate_crs
from geo.utils.result_utils import format_response, format_error


def bbox_to_geometry(
    bbox: Union[List[float], Tuple[float, float, float, float]],
    image_transform: Optional[Union[rasterio.transform.Affine, Tuple, List]] = None,
    crs: Union[str, int, pyproj.CRS] = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Convert a 2D bounding box [x1, y1, x2, y2] into a spatial Polygon geometry.
    
    If image_transform (GeoTIFF Affine transform) is provided, converts pixel-based bounding box
    coordinates into geographic / projected spatial coordinates.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2] (minx, miny, maxx, maxy or pixel coordinates).
        image_transform: Rasterio Affine transform matrix or 6-element tuple (a, b, c, d, e, f).
        crs: Coordinate Reference System.
        
    Returns:
        Structured response dictionary containing Shapely Polygon and GeoJSON mapping.
    """
    try:
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must contain exactly 4 coordinates [x1, y1, x2, y2], got {len(bbox)}")

        x1, y1, x2, y2 = bbox

        if image_transform is not None:
            if isinstance(image_transform, (list, tuple)):
                if len(image_transform) == 6:
                    aff = rasterio.transform.Affine(*image_transform)
                elif len(image_transform) == 9:
                    aff = rasterio.transform.Affine(
                        image_transform[0], image_transform[1], image_transform[2],
                        image_transform[3], image_transform[4], image_transform[5]
                    )
                else:
                    raise ValueError(f"Invalid affine transform tuple length: {len(image_transform)}")
            elif isinstance(image_transform, rasterio.transform.Affine):
                aff = image_transform
            else:
                raise ValueError(f"Unsupported transform type: {type(image_transform)}")

            try:
                spatial_x1, spatial_y1 = aff @ (x1, y1)
                spatial_x2, spatial_y2 = aff @ (x2, y2)
                spatial_x3, spatial_y3 = aff @ (x2, y1)
                spatial_x4, spatial_y4 = aff @ (x1, y2)
            except TypeError:
                spatial_x1, spatial_y1 = aff * (x1, y1)
                spatial_x2, spatial_y2 = aff * (x2, y2)
                spatial_x3, spatial_y3 = aff * (x2, y1)
                spatial_x4, spatial_y4 = aff * (x1, y2)

            poly_geom = Polygon([
                (spatial_x1, spatial_y1),
                (spatial_x3, spatial_y3),
                (spatial_x2, spatial_y2),
                (spatial_x4, spatial_y4),
                (spatial_x1, spatial_y1)
            ])
            is_georeferenced = True
        else:
            minx = min(x1, x2)
            maxx = max(x1, x2)
            miny = min(y1, y2)
            maxy = max(y1, y2)
            poly_geom = box(minx, miny, maxx, maxy)
            is_georeferenced = False

        crs_obj = validate_crs(crs)

        return format_response(
            operation="bbox_to_geometry",
            success=True,
            data={
                "geometry": poly_geom,
                "geojson": mapping(poly_geom),
                "bbox": [poly_geom.bounds[0], poly_geom.bounds[1], poly_geom.bounds[2], poly_geom.bounds[3]],
                "is_georeferenced": is_georeferenced,
                "crs": crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_error("bbox_to_geometry", str(e))


def geometry_to_bbox(
    geometry: Union[BaseGeometry, Dict[str, Any], Any]
) -> Dict[str, Any]:
    """
    Extract the bounding box [minx, miny, maxx, maxy] from a geometry.
    
    Args:
        geometry: Input geometry.
        
    Returns:
        Structured response dictionary with bounding box list [minx, miny, maxx, maxy].
    """
    try:
        geom = ensure_shapely_geometry(geometry["geometry"] if isinstance(geometry, dict) and "geometry" in geometry else geometry)
        bounds = list(geom.bounds)  # [minx, miny, maxx, maxy]
        
        return format_response(
            operation="geometry_to_bbox",
            success=True,
            data={
                "bbox": [float(b) for b in bounds],
                "minx": float(bounds[0]),
                "miny": float(bounds[1]),
                "maxx": float(bounds[2]),
                "maxy": float(bounds[3])
            }
        )
    except Exception as e:
        return format_error("geometry_to_bbox", str(e))
