"""
Geometry package initialization.
"""

from geo.geometry.distance import calculate_distance
from geo.geometry.area import calculate_area, calculate_area_statistics
from geo.geometry.buffer import create_buffer
from geo.geometry.intersection import find_intersection
from geo.geometry.spatial_relation import (
    check_spatial_relation,
    is_intersects,
    is_contains,
    is_within,
    is_touches,
    is_overlaps,
    is_disjoint
)
from geo.geometry.spatial_filter import filter_objects_within
from geo.geometry.nearest import find_nearest
from geo.geometry.union import union_geometries, difference, calculate_centroid
from geo.geometry.conversion import bbox_to_geometry, geometry_to_bbox

__all__ = [
    "calculate_distance",
    "calculate_area",
    "calculate_area_statistics",
    "create_buffer",
    "find_intersection",
    "check_spatial_relation",
    "is_intersects",
    "is_contains",
    "is_within",
    "is_touches",
    "is_overlaps",
    "is_disjoint",
    "filter_objects_within",
    "find_nearest",
    "union_geometries",
    "difference",
    "calculate_centroid",
    "bbox_to_geometry",
    "geometry_to_bbox"
]
