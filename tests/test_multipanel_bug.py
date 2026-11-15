"""Tests for multi-panel / collage inspection, uninvented labeling, and bug resolution."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from backend.agent.agent import SatQueryAgent
from backend.agent.inspector import ImageInspector
from backend.agent.schemas import SceneLayout, TaskType
from backend.agent.router import TaskRouter


class TestMultiPanelBugResolution(unittest.TestCase):
    """Verify bug resolution and multi-panel / collage logic."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dir_path = Path(cls.temp_dir.name)

        # 1. Single scene image (512x512)
        cls.single_scene_path = cls.dir_path / "single_scene.png"
        img_single = Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))
        img_single.save(cls.single_scene_path)

        # 2. Side-by-side 2-panel collage (1024x512) with divider line
        cls.collage_path = cls.dir_path / "side_by_side_collage.png"
        collage_arr = np.zeros((512, 1024, 3), dtype=np.uint8)
        collage_arr[:, :510] = [80, 140, 90]   # Panel 1 (left)
        collage_arr[:, 510:514] = [255, 255, 255] # Divider line
        collage_arr[:, 514:] = [140, 110, 70]  # Panel 2 (right)
        Image.fromarray(collage_arr).save(cls.collage_path)

        # 3. Two separate scene images
        cls.scene_t1_path = cls.dir_path / "scene_t1.png"
        cls.scene_t2_path = cls.dir_path / "scene_t2.png"
        Image.fromarray(np.full((512, 512, 3), 100, dtype=np.uint8)).save(cls.scene_t1_path)
        Image.fromarray(np.full((512, 512, 3), 160, dtype=np.uint8)).save(cls.scene_t2_path)

        cls.agent = SatQueryAgent()
        cls.inspector = ImageInspector()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_bug_regression_single_image_tell_about_it(self):
        """CRITICAL: Exact user bug reproduction and fix.
        Single image + 'Analyze this image tell about it?' must NEVER fail with
        'Input Error: Bi-temporal change detection requires exactly two images.'
        """
        response = self.agent.analyze(
            query="Analyze this image tell about it?",
            images=[self.single_scene_path],
        )
        self.assertIsNone(response.error)
        self.assertNotIn("Input Error: Bi-temporal change detection requires exactly two images", response.answer)
        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_DESCRIPTION.value)
        self.assertGreater(len(response.answer), 20)
        self.assertNotIn("###", response.answer)

    def test_collage_without_comparison_query_does_not_force_change_detection(self):
        """A collage with 2 panels must NOT trigger change detection when query is descriptive."""
        inspection = self.inspector.inspect([self.collage_path])
        self.assertEqual(inspection.scene_count, 2)
        self.assertEqual(inspection.layout, SceneLayout.SIDE_BY_SIDE)
        self.assertTrue(inspection.can_separate_panels)

        response = self.agent.analyze(
            query="Describe this satellite image in detail.",
            images=[self.collage_path],
        )
        self.assertIsNone(response.error)
        self.assertNotEqual(response.task, TaskType.BI_TEMPORAL_CHANGE.value)
        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_DESCRIPTION.value)
        self.assertNotIn("Bi-temporal change detection requires exactly two images", response.answer)

    def test_uninvented_panel_labels_and_extraction(self):
        """Collage without verified dates must be labeled Panel A, Panel B, never invented dates."""
        inspection = self.inspector.inspect([self.collage_path])
        self.assertGreaterEqual(len(inspection.panels), 2)
        self.assertEqual(inspection.panels[0].display_name, "Panel A")
        self.assertEqual(inspection.panels[1].display_name, "Panel B")

        # Verify extraction creates valid cropped images
        extracted = self.inspector.extract_panels(self.collage_path, inspection)
        self.assertEqual(len(extracted), 2)
        for p in extracted:
            self.assertTrue(p.exists())
            with Image.open(p) as img:
                self.assertGreater(img.width, 400)
                self.assertGreater(img.height, 400)

    def test_collage_with_comparison_query_extracts_panels_and_compares(self):
        """A collage with comparison query automatically extracts panels and performs comparison."""
        response = self.agent.analyze(
            query="What changed between these panels?",
            images=[self.collage_path],
        )
        # Should NOT fail input validation!
        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.BI_TEMPORAL_CHANGE.value)
        # Verify extraction step logged in execution trace
        trace_str = " ".join(response.execution_trace.steps)
        self.assertIn("Multi-Panel Extraction", trace_str)
        self.assertGreater(len(response.answer), 20)
        self.assertNotIn("###", response.answer)

    def test_two_separate_images_with_comparison_query(self):
        """Two uploaded files + comparison query routes to bi_temporal_change."""
        response = self.agent.analyze(
            query="Compare these two images and identify what changed.",
            images=[self.scene_t1_path, self.scene_t2_path],
        )
        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.BI_TEMPORAL_CHANGE.value)
        self.assertTrue(
            "surface change" in response.answer.lower()
            or "comparison" in response.answer.lower()
            or "change" in response.answer.lower()
        )

    def test_building_query_routes_to_yolo11(self):
        """Counting / object detection query routes to YOLO11."""
        response = self.agent.analyze(
            query="How many buildings are visible in this scene?",
            images=[self.single_scene_path],
        )
        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_OBJECT_DETECTION.value)
        self.assertIn("YOLO11", response.analysis_plan.selected_models)
        self.assertIn("YOLO11", " ".join(response.execution_trace.steps))


if __name__ == "__main__":
    unittest.main()
