# Geo Engine — SATQuery AI

The **Geo Engine** is a high-performance, beginner-friendly geospatial processing engine built for **SATQuery AI**. It empowers the system to understand geographic satellite data, perform real-world distance and area measurements, apply metric buffers, execute spatial relationships and filtering, convert bounding boxes, and analyze GeoTIFF satellite imagery.

---

## 1. What is the GeoEngine?

The **GeoEngine** is a Python library designed to handle vector geometry operations and raster metadata extractions in real-world geographic units (meters, square meters, kilometers, etc.). 

It translates raw pixel-level computer vision outputs (such as bounding boxes from YOLO or segmentation masks from SegFormer) into georeferenced spatial objects on Earth, enabling complex spatial reasoning.

---

## 2. Why SATQuery Needs a GeoEngine

Satellite imagery consists of pixels, but user queries are asked in terms of real-world physical geography:
> *"How many buildings are within 100 meters of the river?"*
> *"What is the total surface area of flooded agricultural land in hectares?"*

An AI Agent cannot answer these questions using pixel coordinates alone because pixel distances depend on satellite resolution and latitude. 

The **GeoEngine** bridges this gap by:
1. Converting image pixels into geographic coordinates (Latitude/Longitude or projected UTM meters).
2. Calculating accurate surface areas and metric distances on the Earth's surface.
3. Filtering detected objects against geographic buffers and spatial boundaries.
4. Returning standardized, structured JSON-serializable responses directly to the AI Agent.

---

## 3. Main Capabilities

The GeoEngine provides 15+ core geospatial capabilities:

1. **Distance Calculation**: Accurate point-to-point, point-to-line, and polygon-to-polygon distances in metric units (meters, kilometers, feet, miles) using geodetic or projected UTM transforms.
2. **Area Calculation & Statistics**: Surface area measurements in square meters, square kilometers, hectares, acres, or square feet for single polygons or batch feature collections.
3. **Buffer Creation**: Metric buffer zones around points, lines, or polygons by exact distance in meters (`create_buffer(geometry, distance_m)`).
4. **Intersection Discovery**: Geometric intersection extraction between spatial objects with overlapping area calculation.
5. **Spatial Relationships**: Evaluates spatial topological predicates (`is_within`, `is_contains`, `is_intersects`, `is_touches`, `is_overlaps`, `is_disjoint`).
6. **Spatial Filtering**: Batch filtering of detected objects against a target geographic region (`filter_objects_within`).
7. **CRS Handling**: Coordinate Reference System validation, reprojection, and automatic local UTM projection selection (`validate_crs`, `transform_geometry`).
8. **GeoJSON Conversion**: Seamless conversions between Shapely geometries, GeoPandas GeoDataFrames, and standard GeoJSON dictionaries.
9. **Raster Metadata Extraction**: Extracts GeoTIFF dimensions, CRS, bounding extent, pixel resolution, band count, and Affine transform matrices (`get_raster_metadata`).
10. **Bounding-Box Conversion**: Converts 2D pixel bounding boxes `[x1, y1, x2, y2]` into georeferenced spatial polygons (`bbox_to_geometry`) and extracts bounding boxes from geometries (`geometry_to_bbox`).
11. **Nearest Feature Search**: Finds the closest object to a reference geometry from candidate objects (`find_nearest`).
12. **Union Operations**: Combines multiple spatial geometries into a single unified geometry (`union_geometries`).
13. **Spatial Difference**: Subtracts one spatial geometry from another (`difference`).
14. **Geometry Validation**: Comprehensive validation checking missing, empty, or invalid status with healing (`validate_geometry`).
15. **Pixel-to-Geographic Coordinate Conversion**: Maps pixel indices `(col, row)` to spatial coordinates `(lon, lat)` and vice versa (`pixel_to_geo`, `geo_to_pixel`).

---

## 4. Installation Instructions

Ensure you have Python 3.10+ installed. Install the required dependencies from the project root:

```bash
pip install -r geo/requirements.txt
```

### Dependencies Included:
- `geopandas` (>= 1.0.0)
- `shapely` (>= 2.0.0)
- `rasterio` (>= 1.3.0)
- `pyproj` (>= 3.4.0)
- `numpy` (>= 1.24.0)
- `pandas` (>= 2.0.0)

---

## 5. How the AI Agent Uses GeoEngine as a Tool

The central AI Agent integrates with `GeoEngine` via `geo/inference.py`.

### A. Direct Method Calls
The Agent can import `GeoEngine` and call public functions directly:

```python
from geo import GeoEngine

geo = GeoEngine()

# Create a 100-meter buffer around a river LineString
river_buffer = geo.create_buffer(river_geometry, distance_m=100)

# Filter buildings within the buffer
nearby_buildings = geo.filter_objects_within(building_polygons, river_buffer["geometry"], predicate="intersects")
```

### B. Unified Spatial Query Interface (`execute_spatial_query`)
The Agent can call a single, deterministic function passing the operation name as a string:

```python
response = geo.execute_spatial_query(
    operation="buffer",
    geometry1=river_geometry,
    distance_m=100
)
```

---

## 6. Standardized Response Format

All public GeoEngine functions return clean, structured Python dictionaries.

### Successful Response Example:
```json
{
    "operation": "buffer",
    "success": true,
    "distance_m": 100.0,
    "geometry": "<Shapely Polygon>",
    "geojson": {
        "type": "Polygon",
        "coordinates": [...]
    },
    "crs": "EPSG:4326"
}
```

### Error Response Example:
```json
{
    "operation": "buffer",
    "success": false,
    "error": "Invalid distance_m: distance cannot be negative"
}
```

---

## 7. Example Query & Result Scenario

### Scenario:
*"How many buildings are within 100 meters of the river?"*

### Execution Code:
```python
from geo.demo import run_demo

result = run_demo()
print(result)
```

### Structured Output:
```json
{
    "query": "buildings within 100m of river",
    "building_count": 6,
    "total_building_area_m2": 3845.4
}
```

---

## 8. How to Run the Demonstration & Tests

### Run the Standalone Mock Demo:
```bash
python geo/demo.py
```

### Run the Complete Unit Test Suite:
```bash
python -m unittest -v geo/test_geo.py
```
