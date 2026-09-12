"""
Comprehensive Unit Test Suite & Mock Spatial Analysis Demonstration for Geo Engine.
"""

import os
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon, LineString

from geo.inference import GeoEngine


class TestGeoEngine(unittest.TestCase):
    """
    Test cases for all 16 core requirements of the Geo/GIS Engine.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = GeoEngine()

        # Create a synthetic GeoTIFF file for raster metadata, bounds, and pixel conversion tests
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_geotiff_path = os.path.join(cls.temp_dir.name, "synthetic_sat_image.tif")

        width, height = 256, 256
        cls.transform = from_origin(77.5900, 12.9800, 0.0001, 0.0001)  # top-left lon/lat, pixel size
        data = np.zeros((3, height, width), dtype=np.uint8)

        with rasterio.open(
            cls.temp_geotiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=cls.transform,
        ) as dst:
            dst.write(data)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_distance(self):
        """Test 1: Distance calculation between points and geometries in meters."""
        p1 = (77.5900, 12.9700)
        p2 = (77.5910, 12.9710)
        res = self.engine.calculate_distance(p1, p2, crs="EPSG:4326", unit="meters")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["operation"], "distance")
        self.assertGreater(res["distance_m"], 100.0)
        self.assertEqual(res["unit"], "meters")

    def test_02_polygon_area(self):
        """Test 2: Polygon area calculation in square meters and area statistics."""
        poly_coords = [(77.5900, 12.9700), (77.5910, 12.9700), (77.5910, 12.9710), (77.5900, 12.9710)]
        res = self.engine.calculate_area(poly_coords, crs="EPSG:4326", unit="square_meters")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["operation"], "area")
        self.assertGreater(res["area_m2"], 10000.0)
        self.assertEqual(res["unit"], "square_meters")

        # Test area statistics
        stats_res = self.engine.calculate_area_statistics([poly_coords, poly_coords], crs="EPSG:4326")
        self.assertTrue(stats_res["success"])
        self.assertEqual(stats_res["count"], 2)
        self.assertGreater(stats_res["total_area_m2"], 20000.0)

    def test_03_buffer_creation(self):
        """Test 3: Metric buffer creation around a geometry."""
        point = (77.5900, 12.9700)
        res = self.engine.create_buffer(point, distance_m=100, crs="EPSG:4326")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["distance_m"], 100)
        self.assertIsNotNone(res["geometry"])
        self.assertEqual(res["geometry"].geom_type, "Polygon")

    def test_04_intersection(self):
        """Test 4: Intersection discovery between geometries."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        
        res = self.engine.find_intersection(poly1, poly2, crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertTrue(res["intersects"])
        self.assertIsNotNone(res["geometry"])
        self.assertAlmostEqual(res["intersection_area_m2"], 1.0, places=2)

    def test_05_contains(self):
        """Test 5: Spatial predicate 'contains'."""
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        inner_point = Point(5, 5)

        res = self.engine.is_contains(container, inner_point, crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertTrue(res["result"])

    def test_06_within(self):
        """Test 6: Spatial predicate 'within'."""
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        inner_point = Point(5, 5)

        res = self.engine.is_within(inner_point, container, crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertTrue(res["result"])

    def test_07_spatial_filtering(self):
        """Test 7: Spatial object filtering against target geometry."""
        target_zone = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        test_points = [Point(1, 1), Point(5, 5), Point(20, 20), Point(50, 50)]

        res = self.engine.filter_objects_within(test_points, target_zone, predicate="within", crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertEqual(res["count"], 2)

    def test_08_crs_conversion(self):
        """Test 8: CRS transformation and coordinate remapping."""
        point_wgs84 = Point(77.5900, 12.9700)
        res = self.engine.transform_geometry(point_wgs84, source_crs="EPSG:4326", target_crs="EPSG:3857")
        
        self.assertTrue(res["success"])
        self.assertNotEqual(res["geometry"].x, 77.5900)

    def test_09_geojson_conversion(self):
        """Test 9: GeoJSON conversion utilities."""
        original_poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        
        res_geojson = self.engine.geometry_to_geojson(original_poly)
        self.assertTrue(res_geojson["success"])
        geojson_dict = res_geojson["geojson"]

        res_sh = self.engine.geojson_to_geometry(geojson_dict)
        self.assertTrue(res_sh["success"])
        self.assertTrue(original_poly.equals(res_sh["geometry"]))

    def test_10_raster_metadata_extraction(self):
        """Test 10: GeoTIFF raster metadata, bounds, CRS, and resolution extraction."""
        res = self.engine.get_raster_metadata(self.temp_geotiff_path)
        self.assertTrue(res["success"])
        self.assertEqual(res["width"], 256)
        self.assertEqual(res["height"], 256)

        crs_res = self.engine.get_raster_crs(self.temp_geotiff_path)
        self.assertTrue(crs_res["success"])
        self.assertIn("EPSG:4326", crs_res["crs"])

    def test_11_yolo_bbox_conversion(self):
        """Test 11: YOLO bounding box conversion & geometry_to_bbox."""
        bbox = [10, 10, 50, 50]
        res_poly = self.engine.bbox_to_geometry(bbox, image_transform=self.transform, crs="EPSG:4326")
        self.assertTrue(res_poly["success"])
        self.assertTrue(res_poly["is_georeferenced"])

        # Test geometry_to_bbox
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        res_bbox = self.engine.geometry_to_bbox(poly)
        self.assertTrue(res_bbox["success"])
        self.assertEqual(res_bbox["bbox"], [0.0, 0.0, 10.0, 10.0])

    def test_12_nearest_feature(self):
        """Test 12: Nearest feature search."""
        source = Point(0, 0)
        candidates = [Point(10, 10), Point(2, 2), Point(50, 50)]
        
        res = self.engine.find_nearest(source, candidates, crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertEqual(res["index"], 1)
        self.assertAlmostEqual(res["distance_m"], np.sqrt(8), places=1)

    def test_13_union(self):
        """Test 13: Unary union of geometries."""
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        
        res = self.engine.union_geometries([p1, p2], crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertEqual(res["geometry"].area, 6.0)

    def test_14_difference(self):
        """Test 14: Spatial difference between geometries."""
        p1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        p2 = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
        
        res = self.engine.difference(p1, p2, crs="EPSG:3857")
        self.assertTrue(res["success"])
        self.assertAlmostEqual(res["geometry"].area, 8.0)

    def test_15_geometry_validation(self):
        """Test 15: Geometry validation checking missing, empty, or invalid geometries."""
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        res_valid = self.engine.validate_geometry(valid_poly)
        self.assertTrue(res_valid["success"])
        self.assertTrue(res_valid["valid"])

        res_none = self.engine.validate_geometry(None)
        self.assertFalse(res_none["success"])

    def test_16_pixel_coordinate_conversion(self):
        """Test 16: Pixel-to-geographic and geographic-to-pixel coordinate conversion."""
        px, py = 10, 20
        res_geo = self.engine.pixel_to_geo(px, py, transform=self.transform)
        self.assertTrue(res_geo["success"])
        lon, lat = res_geo["longitude"], res_geo["latitude"]

        res_px = self.engine.geo_to_pixel(lon, lat, transform=self.transform)
        self.assertTrue(res_px["success"])
        self.assertEqual(res_px["pixel_col"], 10)
        self.assertEqual(res_px["pixel_row"], 20)

    def test_17_mock_satellite_analysis(self):
        """Test 17: Mock Satellite Analysis Demonstration pipeline."""
        res = self.engine.run_mock_analysis()
        self.assertTrue(res["success"])
        self.assertEqual(res["building_count"], 6)
        self.assertGreater(res["total_building_area_m2"], 0.0)

    def test_18_invalid_and_empty_geometry_handling(self):
        """Test 18: Handling of empty, missing, and invalid geometries returning structured errors."""
        empty_poly = Polygon()
        res_empty = self.engine.create_buffer(empty_poly, distance_m=50)
        self.assertFalse(res_empty["success"])
        self.assertIn("error", res_empty)

        res_area_empty = self.engine.calculate_area(empty_poly)
        self.assertFalse(res_area_empty["success"])
        self.assertIn("error", res_area_empty)

        res_none = self.engine.calculate_distance(None, Point(0, 0))
        self.assertFalse(res_none["success"])
        self.assertIn("error", res_none)

    def test_19_invalid_distance_handling(self):
        """Test 19: Handling of negative, NaN, or non-numeric buffer distances."""
        point = Point(77.59, 12.97)
        res_neg = self.engine.create_buffer(point, distance_m=-100)
        self.assertFalse(res_neg["success"])
        self.assertIn("negative", res_neg["error"])

        res_str = self.engine.create_buffer(point, distance_m="invalid")
        self.assertFalse(res_str["success"])
        self.assertIn("error", res_str)

    def test_20_invalid_geojson_handling(self):
        """Test 20: Handling of malformed GeoJSON strings and dicts."""
        res_malformed_json = self.engine.geojson_to_geometry("{invalid_json: true")
        self.assertFalse(res_malformed_json["success"])
        self.assertIn("error", res_malformed_json)

        res_invalid_dict = self.engine.geojson_to_geometry({"type": "UnknownType"})
        self.assertFalse(res_invalid_dict["success"])
        self.assertIn("error", res_invalid_dict)

    def test_21_missing_raster_handling(self):
        """Test 21: Handling of missing raster file paths."""
        fake_path = "non_existent_raster_12345.tif"
        res_meta = self.engine.get_raster_metadata(fake_path)
        self.assertFalse(res_meta["success"])
        self.assertIn("not found", res_meta["error"].lower())

        res_bounds = self.engine.get_raster_bounds(fake_path)
        self.assertFalse(res_bounds["success"])
        self.assertIn("not found", res_bounds["error"].lower())

    def test_22_standalone_mock_demo(self):
        """Test 22: Verify standalone geo/demo.py execution and output structure."""
        from geo.demo import run_demo
        res = run_demo()
        self.assertTrue(res["success"])
        self.assertEqual(res["query"], "buildings within 100m of river")
        self.assertEqual(res["building_count"], 6)
        self.assertAlmostEqual(res["total_building_area_m2"], 3845.4, places=1)


if __name__ == "__main__":
    unittest.main()
