"""Unit tests for Image Understanding & Scene Inspection module."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from backend.agent.agent import SatQueryAgent
from backend.agent.inspector import ImageInspector
from backend.agent.schemas import SceneLayout, TaskType


class TestImageInspector(unittest.TestCase):
    """Test suite covering scene layout detection, panel separation, remote sensing, and neutrality."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.td = Path(cls.temp_dir.name)
        cls.inspector = ImageInspector()

        # 1. Create a single scene synthetic satellite image (512x512)
        cls.single_img_path = cls.td / "sentinel2_scene_01.png"
        np.random.seed(42)
        single_arr = (np.random.rand(512, 512, 3) * 200).astype(np.uint8)
        # Add high-texture aerial features
        single_arr[100:150, 100:150] = [30, 180, 50]  # green field
        single_arr[200:300, 200:300] = [120, 120, 120]  # urban block
        Image.fromarray(single_arr).save(cls.single_img_path)

        # 2. Create a side-by-side comparison image (1000x500)
        # Left: 500x500 pre-event, Right: 500x500 post-event, vertical divider seam
        cls.sbs_img_path = cls.td / "flood_before_and_after.png"
        left_panel = np.full((500, 500, 3), 60, dtype=np.uint8)
        right_panel = np.full((500, 500, 3), 190, dtype=np.uint8)
        sbs_arr = np.hstack([left_panel, right_panel])
        # Add a 2-pixel black divider seam at x=500
        sbs_arr[:, 499:501] = [0, 0, 0]
        Image.fromarray(sbs_arr).save(cls.sbs_img_path)

        # 3. Create a 4-panel 2x2 grid collage (400x400)
        cls.grid_img_path = cls.td / "tile_monitoring_collage.png"
        p1 = np.full((200, 200, 3), 40, dtype=np.uint8)
        p2 = np.full((200, 200, 3), 120, dtype=np.uint8)
        p3 = np.full((200, 200, 3), 180, dtype=np.uint8)
        p4 = np.full((200, 200, 3), 240, dtype=np.uint8)
        top = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        grid_arr = np.vstack([top, bottom])
        # Add 2-pixel seams
        grid_arr[:, 199:201] = [0, 0, 0]
        grid_arr[199:201, :] = [0, 0, 0]
        Image.fromarray(grid_arr).save(cls.grid_img_path)

        # 4. Two separate images for multi-file testing
        cls.file_a_path = cls.td / "S2A_20210501_t1.png"
        cls.file_b_path = cls.td / "S2A_20210915_t2.png"
        Image.fromarray(np.full((256, 256, 3), 80, dtype=np.uint8)).save(cls.file_a_path)
        Image.fromarray(np.full((256, 256, 3), 160, dtype=np.uint8)).save(cls.file_b_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Test 1: Single Satellite Image
    # -------------------------------------------------------------------------
    def test_single_satellite_image(self):
        """1 file with 1 scene -> file_count = 1, scene_count = 1, layout = single."""
        res = self.inspector.inspect([self.single_img_path])

        self.assertEqual(res.file_count, 1)
        self.assertEqual(res.scene_count, 1)
        self.assertEqual(res.layout, SceneLayout.SINGLE)
        self.assertFalse(res.can_separate_panels)
        self.assertTrue(res.is_remote_sensing)
        self.assertGreaterEqual(res.remote_sensing_confidence, 0.7)
        self.assertEqual(len(res.panels), 1)
        self.assertEqual(res.panels[0].width, 512)
        self.assertEqual(res.panels[0].height, 512)

    # -------------------------------------------------------------------------
    # Test 2: Side-by-Side Satellite Image (1 file -> 2 scenes)
    # -------------------------------------------------------------------------
    def test_side_by_side_satellite_image(self):
        """1 file with side-by-side layout -> file_count = 1, scene_count = 2, layout = side_by_side."""
        res = self.inspector.inspect([self.sbs_img_path])

        self.assertEqual(res.file_count, 1)
        self.assertEqual(res.scene_count, 2)
        self.assertEqual(res.layout, SceneLayout.SIDE_BY_SIDE)
        self.assertTrue(res.can_separate_panels)
        self.assertEqual(len(res.panels), 2)

        # Left panel
        p0 = res.panels[0]
        self.assertAlmostEqual(p0.width, 500, delta=10)
        self.assertEqual(p0.height, 500)
        self.assertEqual(p0.bbox[0], 0)  # ymin
        self.assertEqual(p0.bbox[1], 0)  # xmin

        # Right panel
        p1 = res.panels[1]
        self.assertAlmostEqual(p1.width, 500, delta=10)
        self.assertEqual(p1.height, 500)
        self.assertEqual(p1.bbox[3], 1000)  # xmax

    # -------------------------------------------------------------------------
    # Test 3: Grid Collage Image (1 file -> 4 scenes)
    # -------------------------------------------------------------------------
    def test_grid_collage_image(self):
        """1 file with 2x2 grid collage -> file_count = 1, scene_count = 4, layout = grid_collage."""
        res = self.inspector.inspect([self.grid_img_path])

        self.assertEqual(res.file_count, 1)
        self.assertEqual(res.scene_count, 4)
        self.assertEqual(res.layout, SceneLayout.GRID_COLLAGE)
        self.assertTrue(res.can_separate_panels)
        self.assertEqual(len(res.panels), 4)

        for p in res.panels:
            self.assertAlmostEqual(p.width, 200, delta=10)
            self.assertAlmostEqual(p.height, 200, delta=10)

    # -------------------------------------------------------------------------
    # Test 4: Two Separate Images (2 files -> 2 scenes)
    # -------------------------------------------------------------------------
    def test_multi_file_scenes(self):
        """2 separate files -> file_count = 2, scene_count = 2, layout = multi_file."""
        res = self.inspector.inspect([self.file_a_path, self.file_b_path])

        self.assertEqual(res.file_count, 2)
        self.assertEqual(res.scene_count, 2)
        self.assertEqual(res.layout, SceneLayout.MULTI_FILE)
        self.assertTrue(res.can_separate_panels)
        self.assertEqual(len(res.panels), 2)
        self.assertEqual(res.panels[0].file_index, 0)
        self.assertEqual(res.panels[1].file_index, 1)

    # -------------------------------------------------------------------------
    # Test 5: Critical Rule: FILE COUNT != SCENE COUNT
    # -------------------------------------------------------------------------
    def test_file_count_not_equal_to_scene_count(self):
        """Verify that file count is strictly decoupled from scene count."""
        # Case A: 1 file, 2 scenes
        res_sbs = self.inspector.inspect([self.sbs_img_path])
        self.assertNotEqual(res_sbs.file_count, res_sbs.scene_count)
        self.assertEqual(res_sbs.file_count, 1)
        self.assertEqual(res_sbs.scene_count, 2)

        # Case B: 1 file, 4 scenes
        res_grid = self.inspector.inspect([self.grid_img_path])
        self.assertNotEqual(res_grid.file_count, res_grid.scene_count)
        self.assertEqual(res_grid.file_count, 1)
        self.assertEqual(res_grid.scene_count, 4)

        # Case C: 2 files, 3 scenes (1 single + 1 side-by-side)
        res_mixed = self.inspector.inspect([self.single_img_path, self.sbs_img_path])
        self.assertNotEqual(res_mixed.file_count, res_mixed.scene_count)
        self.assertEqual(res_mixed.file_count, 2)
        self.assertEqual(res_mixed.scene_count, 3)

    # -------------------------------------------------------------------------
    # Test 6: Neutrality (No Routing Decisions Made by Inspector)
    # -------------------------------------------------------------------------
    def test_neutrality_no_routing_decision(self):
        """Inspector must purely characterize the visual content without forcing task selection."""
        res_sbs = self.inspector.inspect([self.sbs_img_path])

        # Verify that ImageSceneInspection has NO task routing fields
        self.assertFalse(hasattr(res_sbs, "task"))
        self.assertFalse(hasattr(res_sbs, "selected_tool"))
        self.assertFalse(hasattr(res_sbs, "route"))

        # Even with 2 detected panels, it strictly outputs layout and panel info
        self.assertEqual(res_sbs.scene_count, 2)
        self.assertEqual(res_sbs.layout, SceneLayout.SIDE_BY_SIDE)

    # -------------------------------------------------------------------------
    # Test 7: Temporal Cues Extraction
    # -------------------------------------------------------------------------
    def test_temporal_cues_extraction(self):
        """Verify date and temporal keyword detection in filenames and metadata."""
        # 1. Multi-file with dates in filename (S2A_20210501_t1 and S2A_20210915_t2)
        res = self.inspector.inspect([self.file_a_path, self.file_b_path])
        self.assertTrue(res.has_temporal_cues)
        self.assertIn("20210501", res.detected_dates)
        self.assertIn("20210915", res.detected_dates)

        # 2. Side-by-side with "before_and_after" in filename
        res_sbs = self.inspector.inspect([self.sbs_img_path])
        self.assertTrue(res_sbs.has_temporal_cues)

        # 3. Single image with no temporal cues
        res_single = self.inspector.inspect([self.single_img_path])
        self.assertFalse(res_single.has_temporal_cues)

    # -------------------------------------------------------------------------
    # Test 8: Panel Cropping
    # -------------------------------------------------------------------------
    def test_panel_cropping(self):
        """Verify crop_panels extracts valid sub-images matching panel dimensions."""
        res = self.inspector.inspect([self.sbs_img_path])
        self.assertEqual(len(res.panels), 2)

        with tempfile.TemporaryDirectory() as crop_td:
            cropped = self.inspector.crop_panels(self.sbs_img_path, res.panels, output_dir=crop_td)
            self.assertEqual(len(cropped), 2)

            for crop_p in cropped:
                self.assertTrue(crop_p.exists())
                with Image.open(crop_p) as im:
                    self.assertAlmostEqual(im.size[0], 500, delta=10)
                    self.assertEqual(im.size[1], 500)

    # -------------------------------------------------------------------------
    # Test 9: SatQueryAgent Integration
    # -------------------------------------------------------------------------
    def test_agent_integration_includes_scene_inspection(self):
        """Verify SatQueryAgent.analyze() executes scene inspection and populates response."""
        agent = SatQueryAgent()
        response = agent.analyze(
            query="Describe this satellite scene.",
            images=[str(self.single_img_path)],
        )

        self.assertIsNotNone(response.scene_inspection)
        self.assertEqual(response.scene_inspection.scene_count, 1)
        self.assertEqual(response.scene_inspection.layout, SceneLayout.SINGLE)
        self.assertTrue(response.scene_inspection.is_remote_sensing)

        # Execution trace should record the image understanding step
        trace_str = " ".join(response.execution_trace.steps)
        self.assertIn("Image Understanding:", trace_str)

    # -------------------------------------------------------------------------
    # Test 10: Empty Images Handled Gracefully
    # -------------------------------------------------------------------------
    def test_empty_images_handled_gracefully(self):
        """Calling inspect with empty list returns clean default without exceptions."""
        res = self.inspector.inspect([])
        self.assertEqual(res.file_count, 0)
        self.assertEqual(res.scene_count, 0)
        self.assertEqual(res.layout, SceneLayout.UNKNOWN)
        self.assertFalse(res.is_remote_sensing)
        self.assertEqual(len(res.panels), 0)


if __name__ == "__main__":
    unittest.main()

