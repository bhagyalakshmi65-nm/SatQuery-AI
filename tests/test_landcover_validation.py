"""
Unit and Integration Tests for SATQuery AI Land-Cover Input Validation and Visualization.

Tests:
1. Rejection of standard 3-channel visual imagery (PNG, JPG) for 12-band multispectral classification.
2. Acceptance and execution of genuine 12-band Sentinel-2 surface reflectance inputs.
3. Automatic routing of RGB optical imagery to SegFormer for surface segmentation and area measurement.
4. Truthful natural language generation for State 1, State 2, and State 3.
5. Verification of UI display labels ("Source Imagery (RGB Optical)" vs "Sentinel-2 Multispectral Input (12-Band)").
"""

import os
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import tifffile

from backend.agent.agent import SatQueryAgent
from backend.agent.modality import ModalityInspector
from backend.agent.schemas import TaskType
from ml.landcover.inference import validate_landcover_input, classify_landcover


class TestLandCoverValidation(unittest.TestCase):
    """Test suite verifying land-cover input validation and routing."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("backend/temp_uploads/test_lc")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a valid 3-channel optical RGB image (PNG)
        cls.rgb_png_path = str(cls.test_dir / "test_rgb_optical.png")
        rgb_arr = np.random.randint(20, 220, (256, 256, 3), dtype=np.uint8)
        # Create some green vegetation bias
        rgb_arr[:128, :, 1] = np.clip(rgb_arr[:128, :, 1] + 50, 0, 255)
        Image.fromarray(rgb_arr).save(cls.rgb_png_path)

        # 2. Create a genuine 12-band Sentinel-2 GeoTIFF
        cls.s2_12band_path = str(cls.test_dir / "test_sentinel2_12band.tif")
        cube_12b = np.random.randint(100, 3000, (12, 120, 120), dtype=np.uint16)
        tifffile.imwrite(cls.s2_12band_path, cube_12b)

        # 3. Create a 3-channel GeoTIFF (RGB visual raster)
        cls.rgb_tiff_path = str(cls.test_dir / "test_rgb_raster.tif")
        rgb_cube = np.random.randint(0, 255, (3, 120, 120), dtype=np.uint8)
        tifffile.imwrite(cls.rgb_tiff_path, rgb_cube)

        cls.agent = SatQueryAgent()

    def test_rgb_png_input_validation_fails_for_12band(self):
        """Standard RGB visual images must be rejected by validate_landcover_input."""
        val = validate_landcover_input(self.rgb_png_path)
        self.assertFalse(val["valid"])
        self.assertEqual(val["modality"], "rgb")
        self.assertEqual(val["bands"], 3)
        self.assertIn("12-band Sentinel-2 surface reflectance", val["message"])

    def test_rgb_tiff_input_validation_fails_for_12band(self):
        """3-channel visual TIFFs must be rejected by validate_landcover_input."""
        val = validate_landcover_input(self.rgb_tiff_path)
        self.assertFalse(val["valid"])
        self.assertEqual(val["modality"], "rgb")
        self.assertEqual(val["bands"], 3)

    def test_12band_geotiff_input_validation_succeeds(self):
        """Genuine 12-band Sentinel-2 GeoTIFFs must be accepted."""
        val = validate_landcover_input(self.s2_12band_path)
        self.assertTrue(val["valid"])
        self.assertEqual(val["modality"], "sentinel2_multispectral")
        self.assertEqual(val["bands"], 12)

    def test_classify_landcover_rejects_rgb_truthfully(self):
        """Direct classify_landcover call on RGB must return success=False and clear notice."""
        res = classify_landcover(self.rgb_png_path)
        self.assertFalse(res["success"])
        self.assertEqual(res["results"]["classes"], [])
        self.assertIsNone(res["visualization"])
        self.assertIn("Multispectral land-cover classification requires 12-band Sentinel-2", res["error"])

    def test_classify_landcover_executes_on_12band(self):
        """Direct classify_landcover call on 12-band input must run and produce classes + visualization."""
        res = classify_landcover(self.s2_12band_path, threshold=0.35, visualize=True)
        self.assertTrue(res["success"])
        self.assertGreater(len(res["results"]["classes"]), 0)
        self.assertIsNotNone(res["visualization"])
        self.assertTrue(os.path.exists(res["visualization"]))

    def test_agent_routes_rgb_image_to_segformer(self):
        """User asking for land cover on an RGB image gets routed to SegFormer with real metrics."""
        resp = self.agent.analyze(
            query="Analyze the land cover distribution in this image",
            images=[self.rgb_png_path],
        )
        self.assertEqual(resp.task, "single_image_landcover")
        self.assertIn("SegFormer", resp.execution_trace.tools_used)
        self.assertNotIn("BigEarthNet-ResNet50-SeCo-12B", resp.execution_trace.tools_used)

        # Real percentages should be populated
        self.assertIn("vegetation_coverage_percent", resp.percentages)
        self.assertGreater(len(resp.visualizations), 0)

        # Answer should describe empirical surface coverage rather than refusal
        self.assertNotIn("Multispectral land-cover classification requires 12-band", resp.answer)
        self.assertIn("Regional surface analysis", resp.answer)

    def test_agent_routes_12band_input_to_bigearthnet(self):
        """User asking for land cover on 12-band input gets routed to BigEarthNet classifier."""
        resp = self.agent.analyze(
            query="Perform multispectral land cover classification on this Sentinel-2 scene",
            images=[self.s2_12band_path],
        )
        self.assertEqual(resp.task, "single_image_landcover")
        self.assertIn("BigEarthNet-ResNet50-SeCo-12B", resp.execution_trace.tools_used)
        self.assertGreater(len(resp.visualizations), 0)
        self.assertIn("Multispectral land-cover classification", resp.answer)

    def test_modality_inspector_labeling(self):
        """ModalityInspector accurately labels RGB vs 12-band inputs."""
        info_rgb = ModalityInspector.inspect(self.rgb_png_path)
        self.assertEqual(info_rgb.probable_modality, "optical")
        self.assertFalse(info_rgb.is_multispectral_native)
        self.assertIn("Optical", info_rgb.display_label)

        info_12b = ModalityInspector.inspect(self.s2_12band_path)
        self.assertEqual(info_12b.probable_modality, "optical")
        self.assertTrue(info_12b.is_multispectral_native)
        self.assertEqual(info_12b.channels, 12)
        self.assertEqual(info_12b.display_label, "Sentinel-2 Multispectral Input (12-Band)")


if __name__ == "__main__":
    unittest.main()
