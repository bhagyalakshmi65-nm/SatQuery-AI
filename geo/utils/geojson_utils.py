"""
GeoJSON and GeoDataFrame conversion utilities.
"""

import json
from typing import Any, Dict, List, Optional, Union
from shapely.geometry import mapping, shape, GeometryCollection
from shapely.geometry.base import BaseGeometry
import geopandas as gpd

from geo.config import DEFAULT_CRS
from geo.utils.validation import ensure_shapely_geometry
from geo.utils.result_utils import format_response, format_error


def geometry_to_geojson(
    geometry: Union[BaseGeometry, gpd.GeoDataFrame, gpd.GeoSeries, List[BaseGeometry]],
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert Shapely geometry, list of geometries, or GeoDataFrame to GeoJSON dictionary format.
    
    Args:
        geometry: Shapely BaseGeometry, GeoDataFrame, GeoSeries, or list of geometries.
        properties: Optional metadata dictionary to attach to the GeoJSON Feature.
        
    Returns:
        Structured result containing GeoJSON dictionary representation.
    """
    try:
        if isinstance(geometry, gpd.GeoDataFrame):
            geojson_dict = json.loads(geometry.to_json())
        elif isinstance(geometry, gpd.GeoSeries):
            geojson_dict = json.loads(geometry.to_json())
        elif isinstance(geometry, list):
            features = []
            for item in geometry:
                sh_geom = ensure_shapely_geometry(item)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(sh_geom),
                    "properties": properties or {}
                })
            geojson_dict = {
                "type": "FeatureCollection",
                "features": features
            }
        else:
            sh_geom = ensure_shapely_geometry(geometry)
            geom_mapping = mapping(sh_geom)
            if properties is not None:
                geojson_dict = {
                    "type": "Feature",
                    "geometry": geom_mapping,
                    "properties": properties
                }
            else:
                geojson_dict = geom_mapping
                
        return format_response(
            operation="geometry_to_geojson",
            success=True,
            data={"geojson": geojson_dict}
        )
    except Exception as e:
        return format_error("geometry_to_geojson", str(e))


def geojson_to_geometry(
    geojson_data: Union[Dict[str, Any], str]
) -> Dict[str, Any]:
    """
    Convert GeoJSON dict or GeoJSON string into Shapely geometry object(s).
    
    Args:
        geojson_data: GeoJSON Python dict or JSON string.
        
    Returns:
        Structured result containing Shapely geometry object.
    """
    try:
        if isinstance(geojson_data, str):
            geojson_dict = json.loads(geojson_data)
        else:
            geojson_dict = geojson_data
            
        sh_geom = ensure_shapely_geometry(geojson_dict)
        return format_response(
            operation="geojson_to_geometry",
            success=True,
            data={
                "geometry": sh_geom,
                "geometry_type": sh_geom.geom_type
            }
        )
    except Exception as e:
        return format_error("geojson_to_geometry", str(e))


def geodataframe_to_geojson(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Convert a GeoPandas GeoDataFrame into a GeoJSON dictionary.
    """
    try:
        geojson_dict = json.loads(gdf.to_json())
        return format_response(
            operation="geodataframe_to_geojson",
            success=True,
            data={"geojson": geojson_dict, "feature_count": len(gdf)}
        )
    except Exception as e:
        return format_error("geodataframe_to_geojson", str(e))


def geojson_to_geodataframe(
    geojson_data: Union[Dict[str, Any], str],
    crs: str = DEFAULT_CRS
) -> Dict[str, Any]:
    """
    Convert a GeoJSON dictionary or string into a GeoPandas GeoDataFrame.
    """
    try:
        if isinstance(geojson_data, str):
            geojson_dict = json.loads(geojson_data)
        else:
            geojson_dict = geojson_data

        if geojson_dict.get("type") == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geojson_dict["features"], crs=crs)
        elif geojson_dict.get("type") == "Feature":
            gdf = gpd.GeoDataFrame.from_features([geojson_dict], crs=crs)
        else:
            sh_geom = shape(geojson_dict)
            gdf = gpd.GeoDataFrame(geometry=[sh_geom], crs=crs)

        return format_response(
            operation="geojson_to_geodataframe",
            success=True,
            data={"geodataframe": gdf, "feature_count": len(gdf)}
        )
    except Exception as e:
        return format_error("geojson_to_geodataframe", str(e))
