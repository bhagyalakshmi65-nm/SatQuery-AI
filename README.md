# SATQuery AI — Geospatial & Satellite Image Analysis System

SATQuery AI is an AI-powered satellite image analysis system where a central AI Agent understands natural-language questions and orchestrates specialized analysis modules (YOLO11 object detection, SegFormer semantic segmentation, Land-Cover classification, and the Geo/GIS Engine).

---

## Geo/GIS Engine

The **Geo/GIS Engine** is the core geospatial reasoning module located in [`geo/`](file:///c:/Users/Keerthana/OneDrive/Desktop/SatQuery-AI/geo/).

### Key Capabilities

- **Distance & Area Calculations** (Geodetic and equal-area projected coordinate transformations)
- **Metric Buffer Generation** (e.g. 100m river buffers)
- **Spatial Intersections & Relationship Predicates** (`intersects`, `contains`, `within`, `touches`, `overlaps`, `disjoint`)
- **Spatial Object Filtering** (`filter_objects_within`)
- **YOLO Bounding Box to Spatial Geometry Conversion** (Pixel bbox to Lat/Lon polygon via Affine transform)
- **GeoTIFF Raster Metadata Extraction** (CRS, bounds, resolution, transform matrix, bands)
- **GeoJSON Serialization & Deserialization**

### Quick Start

```python
from geo import GeoEngine

geo = GeoEngine()

# Example: Run mock spatial analysis pipeline
result = geo.run_mock_analysis()
print(result)
```

### Documentation & Testing

- Detailed API guide: [`geo/README.md`](file:///c:/Users/Keerthana/OneDrive/Desktop/SatQuery-AI/geo/README.md)
- Unit tests & mock analysis: [`geo/test_geo.py`](file:///c:/Users/Keerthana/OneDrive/Desktop/SatQuery-AI/geo/test_geo.py)

```bash
python -m unittest geo/test_geo.py
```