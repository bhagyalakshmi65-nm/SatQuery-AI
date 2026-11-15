"""Tests for dynamic specialist execution, DAG resolution, GIS compounding, and envelope standardization."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from backend.agent.executor import AnalysisExecutor
from backend.agent.planner import AnalysisPlanner
from backend.agent.query_parser import SemanticQueryParser
from backend.agent.inspector import ImageInspector
from backend.agent.registry import get_tool
from backend.agent.schemas import AnalysisPlan


class TestAnalysisExecutor(unittest.TestCase):
    """Verify specialist model execution, compound pipeline DAGs, and standardized envelopes."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dir_path = Path(cls.temp_dir.name)

        # Create test image with distinct features
        cls.test_image_path = cls.dir_path / "satellite_test.png"
        img_arr = np.zeros((512, 512, 3), dtype=np.uint8)
        img_arr[:256, :] = [34, 139, 34]    # Forest green (vegetation)
        img_arr[256:320, :] = [30, 144, 255] # Deep sky blue (river)
        img_arr[320:, :] = [160, 160, 160]  # Concrete / urban built-up
        Image.fromarray(img_arr).save(cls.test_image_path)

        # Panel 1 and Panel 2 images for comparative queries
        cls.panel_a_path = cls.dir_path / "panel_a.png"
        cls.panel_b_path = cls.dir_path / "panel_b.png"
        Image.fromarray(np.full((256, 256, 3), [40, 150, 40], dtype=np.uint8)).save(cls.panel_a_path)
        Image.fromarray(np.full((256, 256, 3), [180, 100, 60], dtype=np.uint8)).save(cls.panel_b_path)

        cls.inspector = ImageInspector()
        cls.parser = SemanticQueryParser()
        cls.planner = AnalysisPlanner()
        cls.executor = AnalysisExecutor()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_standardized_envelope_format(self):
        """All specialist tools must return standard envelope {success, model, results, visualization, error}."""
        # Test SegFormer
        seg_tool = get_tool("segformer")
        self.assertIsNotNone(seg_tool)
        seg_res = seg_tool.execute(self.test_image_path)
        for key in ["success", "model", "results", "visualization", "error"]:
            self.assertIn(key, seg_res, f"Key '{key}' missing from SegFormer envelope")
        self.assertTrue(seg_res["success"])
        self.assertEqual(seg_res["model"], "SegFormer")

        # Test GIS distance_filter
        gis_tool = get_tool("gis")
        self.assertIsNotNone(gis_tool)
        gis_res = gis_tool.execute(
            operation="distance_filter",
            objects=[{"name": "building", "box": [100, 100, 150, 150]}],
            reference_polygon=[[100, 120], [200, 120]],
            threshold_meters=100.0,
        )
        for key in ["success", "model", "results", "visualization", "error"]:
            self.assertIn(key, gis_res, f"Key '{key}' missing from GIS envelope")
        self.assertTrue(gis_res["success"])
        self.assertEqual(gis_res["model"], "GIS")

    def test_compound_spatial_distance_query(self):
        """Compound query 'buildings within 100m of river' executes YOLO + SegFormer + GIS distance filter."""
        query = "How many buildings are located within 100 meters of the river?"
        inspection = self.inspector.inspect([self.test_image_path])
        qu = self.parser.understand(query, inspection)
        plan = self.planner.plan(qu, inspection)

        self.assertTrue(qu.requires_gis)
        self.assertTrue(plan.requires_gis)
        self.assertIn("GIS", plan.selected_tools)

        results, exec_steps = self.executor.execute(
            plan=plan,
            query_understanding=qu,
            images=[self.test_image_path],
        )

        # Check that YOLO, SegFormer, and GIS were executed
        self.assertIn("yolo", results)
        self.assertIn("segformer", results)
        self.assertIn("gis", results)

        gis_res = results["gis"]
        self.assertTrue(gis_res["success"])
        self.assertEqual(gis_res["results"]["operation"], "distance_filter")
        self.assertEqual(gis_res["results"]["threshold_meters"], 100.0)
        self.assertIn("matches_count", gis_res["results"])

        # Check execution trace steps
        trace_text = " ".join(exec_steps)
        self.assertIn("YOLO11", trace_text)
        self.assertIn("SegFormer", trace_text)
        self.assertIn("GIS", trace_text)

    def test_compound_containment_query(self):
        """Compound query 'buildings inside urban area' executes SegFormer + YOLO + GIS containment."""
        query = "How many buildings are located inside the urban area?"
        inspection = self.inspector.inspect([self.test_image_path])
        qu = self.parser.understand(query, inspection)
        plan = self.planner.plan(qu, inspection)

        self.assertTrue(plan.requires_gis)
        self.assertIn("GIS", plan.selected_tools)

        results, exec_steps = self.executor.execute(
            plan=plan,
            query_understanding=qu,
            images=[self.test_image_path],
        )

        self.assertIn("gis", results)
        gis_res = results["gis"]
        self.assertTrue(gis_res["success"])
        self.assertEqual(gis_res["results"]["operation"], "containment_filter")
        self.assertEqual(gis_res["results"]["container"], "urban")

    def test_comparative_feature_delta_execution(self):
        """Comparative query 'Did vegetation decrease?' on 2 images executes SegFormer on both and GIS area delta."""
        query = "Did vegetation decrease between these two images?"
        inspection = self.inspector.inspect([self.panel_a_path, self.panel_b_path])
        qu = self.parser.understand(query, inspection)
        plan = self.planner.plan(qu, inspection)

        results, exec_steps = self.executor.execute(
            plan=plan,
            query_understanding=qu,
            images=[self.panel_a_path, self.panel_b_path],
        )

        self.assertIn("segformer", results)
        self.assertIn("gis", results)
        gis_res = results["gis"]
        self.assertTrue(gis_res["success"])
        self.assertEqual(gis_res["results"]["operation"], "area_delta")
        self.assertIn("percentage_change", gis_res["results"])
        self.assertIn("trend", gis_res["results"])

    def test_resilience_to_model_failure(self):
        """Failure in a specialist model must NOT crash the pipeline, but return failure envelope."""
        # Non-existent image path
        fake_path = self.dir_path / "non_existent_satellite.png"
        query = "Find all buildings."
        qu = self.parser.understand(query)
        plan = self.planner.plan(qu)

        results, exec_steps = self.executor.execute(
            plan=plan,
            query_understanding=qu,
            images=[fake_path],
        )

        self.assertIn("yolo", results)
        # Should record error without crashing
        self.assertFalse(results["yolo"]["success"])
        self.assertIsNotNone(results["yolo"]["error"])


if __name__ == "__main__":
    unittest.main()
