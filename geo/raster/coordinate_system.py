"""
Coordinate system transformation utilities for rasters and geometries.
"""

from typing import Any, Dict, Tuple, Union
import rasterio.transform
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
import shapely.ops

from geo.utils import ensure_shapely_geometry, validate_crs, format_response, format_error


def pixel_to_coord(
    pixel_x: float,
    pixel_y: float,
    transform: Union[rasterio.transform.Affine, Tuple, list]
) -> Dict[str, Any]:
    """
    Convert image pixel coordinates (column, row) to spatial coordinates (x, y).
    
    Args:
        pixel_x: Pixel column index (x).
        pixel_y: Pixel row index (y).
        transform: Raster Affine transformation matrix or 6-element tuple.
        
    Returns:
        Structured response dictionary with (x, y) spatial coordinates.
    """
    try:
        if isinstance(transform, (tuple, list)):
            aff = rasterio.transform.Affine(*transform[:6])
        else:
            aff = transform

        try:
            x, y = aff @ (pixel_x, pixel_y)
        except TypeError:
            x, y = aff * (pixel_x, pixel_y)
        return format_response(
            operation="pixel_to_coord",
            success=True,
            data={
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "spatial_x": float(x),
                "spatial_y": float(y),
                "coordinates": (float(x), float(y))
            }
        )
    except Exception as e:
        return format_response(
            operation="pixel_to_coord",
            success=False,
            error=str(e)
        )


def coord_to_pixel(
    spatial_x: float,
    spatial_y: float,
    transform: Union[rasterio.transform.Affine, Tuple, list]
) -> Dict[str, Any]:
    """
    Convert spatial coordinates (x, y) to image pixel coordinates (col, row).
    
    Args:
        spatial_x: Spatial X coordinate / longitude.
        spatial_y: Spatial Y coordinate / latitude.
        transform: Raster Affine transformation matrix or 6-element tuple.
        
    Returns:
        Structured response dictionary with (pixel_x, pixel_y) column and row indices.
    """
    try:
        if isinstance(transform, (tuple, list)):
            aff = rasterio.transform.Affine(*transform[:6])
        else:
            aff = transform

        inv_aff = ~aff
        try:
            pixel_x, pixel_y = inv_aff @ (spatial_x, spatial_y)
        except TypeError:
            pixel_x, pixel_y = inv_aff * (spatial_x, spatial_y)

        return format_response(
            operation="coord_to_pixel",
            success=True,
            data={
                "spatial_x": spatial_x,
                "spatial_y": spatial_y,
                "pixel_x": round(float(pixel_x), 2),
                "pixel_y": round(float(pixel_y), 2),
                "pixel_col": int(round(pixel_x)),
                "pixel_row": int(round(pixel_y))
            }
        )
    except Exception as e:
        return format_response(
            operation="coord_to_pixel",
            success=False,
            error=str(e)
        )


def transform_crs(
    geometry: Union[BaseGeometry, Dict[str, Any], str],
    source_crs: Union[str, int, Any],
    target_crs: Union[str, int, Any]
) -> Dict[str, Any]:
    """
    Reproject a geometry from a source Coordinate Reference System to a target CRS.
    
    Args:
        geometry: Input geometry (Shapely object, GeoJSON dict, or WKT).
        source_crs: Source CRS (e.g., EPSG:4326 or EPSG:3857).
        target_crs: Target CRS.
        
    Returns:
        Structured response dictionary with reprojected geometry.
    """
    try:
        geom = ensure_shapely_geometry(geometry)
        src_crs_obj = validate_crs(source_crs)
        dst_crs_obj = validate_crs(target_crs)

        if src_crs_obj == dst_crs_obj:
            return format_response(
                operation="transform_crs",
                success=True,
                data={
                    "geometry": geom,
                    "source_crs": src_crs_obj.to_string(),
                    "target_crs": dst_crs_obj.to_string()
                }
            )

        transformer = Transformer.from_crs(src_crs_obj, dst_crs_obj, always_xy=True)
        reprojected_geom = shapely.ops.transform(transformer.transform, geom)

        return format_response(
            operation="transform_crs",
            success=True,
            data={
                "geometry": reprojected_geom,
                "source_crs": src_crs_obj.to_string(),
                "target_crs": dst_crs_obj.to_string()
            }
        )
    except Exception as e:
        return format_response(
            operation="transform_crs",
            success=False,
            error=str(e)
        )
