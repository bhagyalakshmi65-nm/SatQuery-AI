"""
Standalone Mock Demonstration for GeoEngine.

Scenario:
"How many buildings are within 100 meters of the river?"

This script creates synthetic geometries (river LineString and 10 mock building Polygons),
buffers the river by 100 meters, finds intersecting buildings, calculates their total surface area,
and returns a structured dictionary response.
"""

import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapely.geometry import Polygon, LineString
from geo.inference import GeoEngine


def run_demo():
    geo = GeoEngine()

    # 1. Create a synthetic river LineString (Longitude, Latitude coordinates)
    river_coords = [(77.5900, 12.9700), (77.5920, 12.9720), (77.5950, 12.9750), (77.5980, 12.9780)]
    river_line = LineString(river_coords)

    # 2. Create 10 synthetic building polygons (some within 100m of river, some further away)
    building_data = [
        {"id": "b1", "coords": [(77.5905, 12.9702), (77.5907, 12.9702), (77.5907, 12.9704), (77.5905, 12.9704)]},
        {"id": "b2", "coords": [(77.5918, 12.9719), (77.5920, 12.9719), (77.5920, 12.9721), (77.5918, 12.9721)]},
        {"id": "b3", "coords": [(77.5948, 12.9749), (77.5952, 12.9749), (77.5952, 12.9751), (77.5948, 12.9751)]},
        {"id": "b4", "coords": [(77.5975, 12.9778), (77.5978, 12.9778), (77.5978, 12.9780), (77.5975, 12.9780)]},
        {"id": "b5", "coords": [(77.5910, 12.9712), (77.5912, 12.9712), (77.5912, 12.9714), (77.5910, 12.9714)]},
        {"id": "b6", "coords": [(77.5935, 12.9736), (77.5938, 12.9736), (77.5938, 12.9738), (77.5935, 12.9738)]},
        # Far buildings (> 100m away)
        {"id": "b7", "coords": [(77.6050, 12.9850), (77.6052, 12.9850), (77.6052, 12.9852), (77.6050, 12.9852)]},
        {"id": "b8", "coords": [(77.6100, 12.9900), (77.6102, 12.9900), (77.6102, 12.9902), (77.6100, 12.9902)]},
        {"id": "b9", "coords": [(77.6150, 12.9950), (77.6152, 12.9950), (77.6152, 12.9952), (77.6150, 12.9952)]},
        {"id": "b10", "coords": [(77.6200, 13.0000), (77.6202, 13.0000), (77.6202, 13.0002), (77.6200, 13.0002)]}
    ]

    buildings = [
        {"id": b["id"], "geometry": Polygon(b["coords"])}
        for b in building_data
    ]

    # 3. Create a 100-meter buffer around the river
    buffer_res = geo.create_buffer(river_line, distance_m=100)
    if not buffer_res.get("success"):
        return {"error": f"Buffer creation failed: {buffer_res.get('error')}"}

    river_buffer = buffer_res["geometry"]

    # 4. Find buildings intersecting / within the 100m buffer
    filter_res = geo.filter_objects_within(buildings, river_buffer, predicate="intersects")
    if not filter_res.get("success"):
        return {"error": f"Spatial filtering failed: {filter_res.get('error')}"}

    matching_buildings = filter_res["objects"]

    # 5. Calculate total area of matching buildings in square meters
    total_area_m2 = 0.0
    for b in matching_buildings:
        area_res = geo.calculate_area(b["geometry"], unit="square_meters")
        if area_res.get("success"):
            total_area_m2 += area_res["area_m2"]

    # 6. Format structured result
    result = {
        "query": "buildings within 100m of river",
        "operation": "mock_satellite_analysis",
        "success": True,
        "building_count": len(matching_buildings),
        "total_building_area_m2": round(total_area_m2, 2)
    }

    return result


if __name__ == "__main__":
    demo_output = run_demo()
    print(json.dumps(demo_output, indent=4))
