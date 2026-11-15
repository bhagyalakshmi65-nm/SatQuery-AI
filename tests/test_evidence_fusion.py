"""Tests for Phase 7 Evidence Fusion, Anti-Hallucination Grounding, and Fused Responses."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from backend.agent.agent import SatQueryAgent
from backend.agent.evidence import EvidenceAggregator
from backend.agent.guard import GroundingGuard
from backend.agent.schemas import TaskType


class TestEvidenceFusion(unittest.TestCase):
    """Verify evidence fusion, factual attribution, why_this_answer, and unsupported query handling."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dir_path = Path(cls.temp_dir.name)

        # Create structured test satellite image
        cls.test_image_path = cls.dir_path / "satellite_synthetic.png"
        img_arr = np.zeros((512, 512, 3), dtype=np.uint8)
        img_arr[:256, :] = [34, 139, 34]     # Vegetation
        img_arr[256:320, :] = [30, 144, 255] # River / water
        img_arr[320:, :] = [160, 160, 160]   # Urban built-up
        Image.fromarray(img_arr).save(cls.test_image_path)

        cls.agent = SatQueryAgent()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_compound_distance_evidence_fusion(self):
        """Test compound distance query: YOLO + SegFormer + GIS produces exact count and concise attribution."""
        query = "How many buildings are within 100m of the river?"
        response = self.agent.analyze(
            query=query,
            images=[self.test_image_path],
        )

        self.assertIsNone(response.error)
        self.assertIn("100m", response.answer)
        
        # Verify why_this_answer contains clear attribution without hidden chain-of-thought
        why_list = response.why_this_answer
        self.assertTrue(any("River extent was obtained from semantic segmentation" in w for w in why_list))
        self.assertTrue(any("Distances were calculated using the GIS spatial-analysis step" in w for w in why_list))
        
        # Verify masks / visualizations
        self.assertGreaterEqual(len(response.masks), 1)
        self.assertIn("seg_mask", response.masks[0])

        # Verify natural language answer without internal headings
        self.assertNotIn("###", response.answer)
        self.assertGreater(len(response.answer), 20)

    def test_compound_containment_evidence_fusion(self):
        """Test compound containment query: SegFormer + YOLO + GIS produces contained count and attribution."""
        query = "How many buildings are located inside the urban area?"
        response = self.agent.analyze(
            query=query,
            images=[self.test_image_path],
        )

        self.assertIsNone(response.error)
        why_list = response.why_this_answer
        self.assertTrue(any("boundaries were delineated using semantic segmentation" in w for w in why_list))
        self.assertTrue(any("Spatial containment was calculated using GIS" in w for w in why_list))

    def test_zero_detections_factual_grounding(self):
        """When 0 objects are detected, response must state 0 objects and eliminate hallucinated counts."""
        tool_results = {
            "yolo": {
                "success": True,
                "model": "YOLO11",
                "results": {"objects": [], "counts": {}},
                "visualization": None,
                "error": None,
            }
        }
        fused = EvidenceAggregator.fuse(
            task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
            tool_results=tool_results,
        )
        self.assertEqual(fused["counts"].get("total_objects"), 0)
        self.assertTrue(any("0 objects were detected" in w for w in fused["why_this_answer"]))

        # Guard enforcement test: raw response claims positive count, Guard must scrub it
        hallucinated_text = "### 1. DIRECT OBSERVATIONS\nThere are 5 buildings visible in this scene."
        guarded = GroundingGuard.enforce(
            raw_response=hallucinated_text,
            task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
            evidence=fused["evidence"],
            confidence=fused["confidence"],
        )
        self.assertNotIn("5 buildings", guarded)
        self.assertIn("No discrete objects were detected", guarded)

    def test_unsupported_modality_clear_communication(self):
        """Unsupported analysis requests (e.g. bathymetry or LiDAR depth) are communicated clearly without hallucination."""
        response = self.agent.analyze(
            query="Measure the bathymetric depth of the water column in meters.",
            images=[self.test_image_path],
        )
        self.assertIsNotNone(response.unsupported_analysis)
        self.assertIn("Bathymetric", response.unsupported_analysis)
        self.assertIn("unsupported by available specialist models", response.answer)
        self.assertTrue(any("Bathymetric water depth measurement" in w for w in response.why_this_answer))

    def test_grounding_guard_preserves_gis_exact_counts(self):
        """GroundingGuard ensures exact GIS buffer count cannot be modified or drifted by the VLM."""
        gis_ev = {
            "success": True,
            "results": {
                "operation": "distance_filter",
                "matches_count": 23,
                "total_evaluated": 30,
                "threshold_meters": 100.0,
                "reference_feature": "segmented river region",
            }
        }
        # If VLM invented "14 buildings"
        raw_vlm_output = "### 1. DIRECT OBSERVATIONS\nFound 14 buildings near the river."
        guarded = GroundingGuard.enforce(
            raw_response=raw_vlm_output,
            task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
            evidence={"gis": gis_ev},
            confidence=0.95,
        )
        self.assertIn("23 detected building(s)/object(s) are within 100m", guarded)

    def test_exact_23_buildings_compound_fusion(self):
        """Test exact prompt scenario: 23 buildings detected by YOLO11 within 100m of river segmented by SegFormer."""
        tool_results = {
            "yolo": {
                "success": True,
                "model": "YOLO11",
                "results": {
                    "counts": {"building": 23},
                    "objects": [{"name": "building", "box": [10, 10, 50, 50], "confidence": 0.88}] * 23,
                },
                "visualization": "/path/to/yolo_overlay.png",
                "error": None,
            },
            "segformer": {
                "success": True,
                "model": "SegFormer",
                "results": {
                    "classes": {
                        "river": {"percentage": 14.5, "polygon": [[256, 100], [256, 400]]},
                        "vegetation": {"percentage": 55.0},
                    },
                    "dominant_class": "vegetation",
                },
                "visualization": "/path/to/river_mask.png",
                "error": None,
            },
            "gis": {
                "success": True,
                "model": "GIS",
                "results": {
                    "operation": "distance_filter",
                    "matches_count": 23,
                    "total_evaluated": 23,
                    "threshold_meters": 100.0,
                    "reference_feature": "segmented river region",
                },
                "visualization": None,
                "error": None,
            },
        }

        fused = EvidenceAggregator.fuse(
            task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
            tool_results=tool_results,
        )

        # 1. Structured numbers
        self.assertEqual(fused["counts"].get("building"), 23)
        self.assertEqual(fused["counts"].get("objects_within_100m"), 23)
        self.assertEqual(fused["percentages"].get("river_coverage_percent"), 14.5)

        # 2. Attribution statements
        why = fused["why_this_answer"]
        self.assertTrue(any("23 building(s) were detected using YOLO11" in w for w in why))
        self.assertTrue(any("River extent was obtained from semantic segmentation" in w for w in why))
        self.assertTrue(any("Distances were calculated using the GIS spatial-analysis step" in w for w in why))

        # 3. Grounded Synthesis
        answer = self.agent.vlm._generate_grounded_synthesis(
            query="How many buildings are within 100m of the river?",
            task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
            evidence=fused["evidence"],
            confidence=fused["confidence"],
            fused_data=fused,
        )

        self.assertIn("23 detected building(s) are within 100m", answer)
        self.assertNotIn("###", answer)
        self.assertGreater(len(answer), 20)


if __name__ == "__main__":
    unittest.main()
