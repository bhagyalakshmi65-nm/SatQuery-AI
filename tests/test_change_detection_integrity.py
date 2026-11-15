"""Tests for change detection integrity and prevention of fabricated results."""

import unittest
from backend.agent.agent import SatQueryAgent
from ml.change_detection.inference import generate_change_overlay
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path


class TestChangeDetectionIntegrity(unittest.TestCase):
    """Verify that temporal change detection enforces date clarification and prevents fabrication."""

    def setUp(self):
        self.agent = SatQueryAgent()

    def test_open_ended_over_time_query_triggers_date_clarification(self):
        """Query 'What changes happened in Bengaluru over time?' must ask for a time period and produce NO change map."""
        resp = self.agent.analyze("What changes happened in Bengaluru over time?")

        self.assertTrue(resp.clarification_request)
        self.assertEqual(resp.task, "temporal_date_clarification")
        self.assertIn("What time period would you like me to compare?", resp.answer)
        self.assertIn("2023 to 2024", resp.answer)
        self.assertIn("2024 to 2025", resp.answer)
        self.assertIn("2020 to 2025", resp.answer)
        # MUST NOT return any change map or visual overlay
        self.assertEqual(len(resp.visualizations), 0)
        self.assertEqual(len(resp.masks), 0)
        self.assertFalse(resp.change_metrics)

    def test_bengaluru_explicit_dates_runs_real_analysis(self):
        """Query with explicit dates 'What changes happened between 2024 and 2025 in Bengaluru?' runs change detection."""
        resp = self.agent.analyze("What changes happened between 2024 and 2025 in Bengaluru?")

        self.assertFalse(resp.clarification_request)
        self.assertEqual(resp.task, "bi_temporal_change")
        self.assertEqual(len(resp.observations), 2)
        # Visual overlay generated from real imagery
        self.assertGreater(len(resp.visualizations), 0)
        # Never hardcode 16.6% fake metric
        self.assertNotIn("16.6%", resp.answer)
        self.assertNotIn("9 key zones", resp.answer)

    def test_overlay_does_not_contain_compass_rose_or_fabricated_badges(self):
        """The change overlay generator should not draw decorative compass roses or callout badges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_img_path = Path(tmpdir) / "base.png"
            out_img_path = Path(tmpdir) / "overlay.png"

            # Create dummy satellite scene
            img = Image.new("RGB", (256, 256), (100, 120, 90))
            img.save(base_img_path)

            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[50:100, 50:100] = 255

            overlay_res = generate_change_overlay(mask, base_img_path, out_img_path)
            self.assertIsNotNone(overlay_res)
            self.assertTrue(out_img_path.exists())

            # Read back generated overlay
            with Image.open(out_img_path) as res_img:
                self.assertEqual(res_img.size, (256, 256))


if __name__ == "__main__":
    unittest.main()
