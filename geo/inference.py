"""
GeoEngine Main Inference Interface for SATQuery AI.

Exposes a clean, unified interface for the AI Agent to execute spatial calculations,
filtering, raster analysis, geometry conversions, and spatial queries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from shapely.geometry import Polygon, LineString

from geo.config import DEFAULT_CRS, logger
from geo.geometry import (
    calculate_distance,
    calculate_area,
    calculate_area_statistics,
    create_buffer,
    find_intersection,
    check_spatial_relation,
    is_intersects,
    is_contains,
    is_within,
    is_touches,
    is_overlaps,
    is_disjoint,
    filter_objects_within,
    find_nearest,
    union_geometries,
    difference,
    calculate_centroid,
    bbox_to_geometry,
    geometry_to_bbox
)
from geo.raster import (
    get_raster_metadata,
    get_raster_bounds,
    get_raster_crs,
    get_raster_resolution,
    pixel_to_coord,
    coord_to_pixel,
    pixel_to_geo,
    geo_to_pixel,
    transform_crs
)
from geo.utils import (
    geometry_to_geojson,
    geojson_to_geometry,
    geodataframe_to_geojson,
    geojson_to_geodataframe,
    validate_geometry,
    validate_crs,
    transform_geometry,
    transform_coordinates,
    get_projected_crs,
    ensure_shapely_geometry,
    format_response,
    format_error
)


class GeoEngine:
    """
    Main Geo/GIS Engine class exposing standalone geospatial analysis tools for AI Agents.
    """

    def __init__(self, default_crs: str = DEFAULT_CRS):
        """
        Initialize the GeoEngine with a default CRS.
        """
        self.default_crs = default_crs

    # 1. Distance & Area
    def calculate_distance(self, point1: Any, point2: Any, crs: Optional[str] = None, unit: str = "meters") -> Dict[str, Any]:
        """Calculate distance between two points or geometries in specified units (default: meters)."""
        target_crs = crs or self.default_crs
        return calculate_distance(point1, point2, crs=target_crs, unit=unit)

    def calculate_area(self, polygon: Any, crs: Optional[str] = None, unit: str = "square_meters") -> Dict[str, Any]:
        """Calculate surface area of a polygon geometry in real-world metric units."""
        target_crs = crs or self.default_crs
        return calculate_area(polygon, crs=target_crs, unit=unit)

    def calculate_area_statistics(self, geometries: List[Any], crs: Optional[str] = None, unit: str = "square_meters") -> Dict[str, Any]:
        """Calculate total, average, min, and max surface area across multiple polygons."""
        target_crs = crs or self.default_crs
        return calculate_area_statistics(geometries, crs=target_crs, unit=unit)

    # 2. Buffer & Intersection
    def create_buffer(self, geometry: Any, distance_m: float, crs: Optional[str] = None) -> Dict[str, Any]:
        """Create a metric buffer zone around a geometry by distance in meters."""
        target_crs = crs or self.default_crs
        return create_buffer(geometry, distance_m, crs=target_crs)

    def find_intersection(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        """Find spatial intersection geometry and metadata between two geometries."""
        target_crs = crs or self.default_crs
        return find_intersection(geometry1, geometry2, crs=target_crs)

    # 3. Spatial Relationships & Filtering
    def check_spatial_relation(self, geometry1: Any, geometry2: Any, predicate: str = "intersects", crs: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a spatial relationship predicate (intersects, contains, within, etc.)."""
        target_crs = crs or self.default_crs
        return check_spatial_relation(geometry1, geometry2, predicate=predicate, crs=target_crs)

    def is_within(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="within", crs=crs)

    def is_contains(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="contains", crs=crs)

    def is_intersects(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="intersects", crs=crs)

    def is_touches(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="touches", crs=crs)

    def is_overlaps(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="overlaps", crs=crs)

    def is_disjoint(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        return self.check_spatial_relation(geometry1, geometry2, predicate="disjoint", crs=crs)

    def filter_objects_within(self, objects: List[Any], target_geometry: Any, predicate: str = "within", crs: Optional[str] = None) -> Dict[str, Any]:
        """Filter a list of spatial objects/detections based on target geometry."""
        target_crs = crs or self.default_crs
        return filter_objects_within(objects, target_geometry, predicate=predicate, crs=target_crs)

    def find_nearest(self, source: Any, objects: List[Any], crs: Optional[str] = None) -> Dict[str, Any]:
        """Find closest object from candidate objects to source geometry with metric distance."""
        target_crs = crs or self.default_crs
        return find_nearest(source, objects, crs=target_crs)

    # 4. Set Operations & Centroid
    def union_geometries(self, geometries: List[Any], crs: Optional[str] = None) -> Dict[str, Any]:
        """Combine multiple geometries into a single unified geometry."""
        target_crs = crs or self.default_crs
        return union_geometries(geometries, crs=target_crs)

    def difference(self, geometry1: Any, geometry2: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        """Compute spatial difference (geometry1 minus geometry2)."""
        target_crs = crs or self.default_crs
        return difference(geometry1, geometry2, crs=target_crs)

    def calculate_centroid(self, geometry: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        """Calculate geometric centroid of a geometry."""
        target_crs = crs or self.default_crs
        return calculate_centroid(geometry, crs=target_crs)

    # 5. Bounding Box Conversions
    def bbox_to_geometry(self, bbox: List[float], image_transform: Optional[Any] = None, crs: Optional[str] = None) -> Dict[str, Any]:
        """Convert bounding box [x1, y1, x2, y2] into spatial Polygon geometry."""
        target_crs = crs or self.default_crs
        return bbox_to_geometry(bbox, image_transform=image_transform, crs=target_crs)

    def geometry_to_bbox(self, geometry: Any) -> Dict[str, Any]:
        """Extract bounding box [minx, miny, maxx, maxy] from geometry."""
        return geometry_to_bbox(geometry)

    # 6. CRS Management
    def validate_crs(self, crs_input: Any) -> Dict[str, Any]:
        """Validate CRS string/code."""
        try:
            c = validate_crs(crs_input)
            return format_response("validate_crs", True, data={"crs": c.to_string(), "is_geographic": c.is_geographic})
        except Exception as e:
            return format_error("validate_crs", str(e))

    def transform_geometry(self, geometry: Any, source_crs: str, target_crs: str) -> Dict[str, Any]:
        """Reproject geometry from source CRS to target CRS."""
        return transform_crs(geometry, source_crs, target_crs)

    def transform_coordinates(self, coords: Any, source_crs: str, target_crs: str) -> Dict[str, Any]:
        """Reproject coordinate tuple(s) from source CRS to target CRS."""
        try:
            res = transform_coordinates(coords, source_crs, target_crs)
            return format_response("transform_coordinates", True, data={"coordinates": res, "source_crs": source_crs, "target_crs": target_crs})
        except Exception as e:
            return format_error("transform_coordinates", str(e))

    # 7. GeoJSON & DataFrames
    def geometry_to_geojson(self, geometry: Any, properties: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert Shapely geometry/GeoDataFrame to GeoJSON dict."""
        return geometry_to_geojson(geometry, properties=properties)

    def geojson_to_geometry(self, geojson_data: Any) -> Dict[str, Any]:
        """Convert GeoJSON dict/string to Shapely geometry."""
        return geojson_to_geometry(geojson_data)

    def geodataframe_to_geojson(self, gdf: Any) -> Dict[str, Any]:
        """Convert GeoPandas GeoDataFrame to GeoJSON dict."""
        return geodataframe_to_geojson(gdf)

    def geojson_to_geodataframe(self, geojson_data: Any, crs: Optional[str] = None) -> Dict[str, Any]:
        """Convert GeoJSON dict to GeoPandas GeoDataFrame."""
        target_crs = crs or self.default_crs
        return geojson_to_geodataframe(geojson_data, crs=target_crs)

    # 8. Raster Analysis & Pixel Conversions
    def get_raster_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract raster dataset metadata (dimensions, CRS, bounds, transform, bands, resolution)."""
        return get_raster_metadata(image_path)

    def get_raster_bounds(self, image_path: str, target_crs: Optional[str] = None) -> Dict[str, Any]:
        """Extract spatial bounds of a raster image."""
        crs_val = target_crs or self.default_crs
        return get_raster_bounds(image_path, target_crs=crs_val)

    def get_raster_crs(self, image_path: str) -> Dict[str, Any]:
        """Extract raster CRS."""
        return get_raster_crs(image_path)

    def get_raster_resolution(self, image_path: str) -> Dict[str, Any]:
        """Extract raster pixel resolution."""
        return get_raster_resolution(image_path)

    def pixel_to_geo(self, x: float, y: float, transform: Any) -> Dict[str, Any]:
        """Map pixel coordinates (column x, row y) to geographic spatial coordinates."""
        return pixel_to_geo(x, y, transform)

    def geo_to_pixel(self, longitude: float, latitude: float, transform: Any) -> Dict[str, Any]:
        """Map geographic coordinates (lon, lat) to image pixel coordinates."""
        return geo_to_pixel(longitude, latitude, transform)

    # 9. Geometry Validation
    def validate_geometry(self, geometry: Any) -> Dict[str, Any]:
        """Validate geometry for non-emptiness, correctness, and supported geometry type."""
        return validate_geometry(geometry)

    # 10. Spatial Query Interface for AI Agent
    def execute_spatial_query(
        self,
        operation: str,
        geometry1: Any,
        geometry2: Optional[Any] = None,
        distance_m: Optional[float] = None,
        objects: Optional[List[Any]] = None,
        crs: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified spatial query entrypoint designed for easy invocation by the AI Agent.
        
        Supported operations: 'within', 'near', 'distance', 'area', 'buffer', 'intersects', 
        'contains', 'filter', 'nearest', 'union', 'difference', 'centroid', 'metadata'.
        """
        op = operation.lower()
        target_crs = crs or self.default_crs

        if op == "buffer":
            return self.create_buffer(geometry1, distance_m=distance_m or 0.0, crs=target_crs)
        elif op in ["within", "intersects", "contains", "touches", "overlaps", "disjoint"]:
            return self.check_spatial_relation(geometry1, geometry2, predicate=op, crs=target_crs)
        elif op == "distance":
            return self.calculate_distance(geometry1, geometry2, crs=target_crs, unit=kwargs.get("unit", "meters"))
        elif op == "area":
            return self.calculate_area(geometry1, crs=target_crs, unit=kwargs.get("unit", "square_meters"))
        elif op == "filter":
            return self.filter_objects_within(objects or [], geometry1, predicate=kwargs.get("predicate", "within"), crs=target_crs)
        elif op == "nearest":
            return self.find_nearest(geometry1, objects or [], crs=target_crs)
        elif op == "union":
            return self.union_geometries(objects or [geometry1, geometry2], crs=target_crs)
        elif op == "difference":
            return self.difference(geometry1, geometry2, crs=target_crs)
        elif op == "centroid":
            return self.calculate_centroid(geometry1, crs=target_crs)
        elif op == "metadata":
            return self.get_raster_metadata(geometry1)
        else:
            return format_error("execute_spatial_query", f"Unsupported query operation '{operation}'")

    # 11. Mock Satellite Analysis Demonstration
    def run_mock_analysis(self) -> Dict[str, Any]:
        """
        Execute a complete end-to-end mock satellite analysis query:
        "How many buildings are within 100 meters of the river?"
        """
        from geo.demo import run_demo
        return run_demo()
