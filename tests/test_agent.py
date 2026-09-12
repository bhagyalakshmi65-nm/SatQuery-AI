"""Comprehensive Unit and Integration Test Suite for SatQuery AI Agent & Orchestrator."""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.agent.agent import SatQueryAgent
from backend.agent.registry import TOOL_REGISTRY, SpecialistTool
from backend.agent.router import TaskRouter
from backend.agent.schemas import TaskType
from backend.main import app


class TestSatQueryAgent(unittest.TestCase):
    """Test suite covering query understanding, input validation, tool routing, evidence, and synthesis."""

    def setUp(self):
        self.agent = SatQueryAgent()
        self.client = TestClient(app)

    # -------------------------------------------------------------------------
    # Test 1: Single Image + Description Query (RS-Qwen Available & Unavailable)
    # -------------------------------------------------------------------------
    def test_single_image_description(self):
        query = "Describe this satellite image and explain what is visible."
        images = ["test_scene.jpg"]

        mock_answer = (
            "### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n"
            "Scene contains residential structures and green vegetation.\n\n"
            "### 2. REMOTE SENSING INTERPRETATION\n"
            "Typical suburban layout with low-density housing."
        )

        with patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=mock_answer) as mock_gen:
            response = self.agent.analyze(query=query, images=images)

        mock_gen.assert_called_once()
        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_DESCRIPTION.value)
        self.assertIn("RS-Qwen", response.execution_trace.tools_used)
        self.assertTrue(any("Synthesized grounded natural-language answer with RS-Qwen" in s for s in response.execution_trace.steps))
        self.assertGreater(response.confidence, 0.0)
        self.assertIn("DIRECT OBSERVATIONS", response.answer)
        self.assertIn("REMOTE SENSING INTERPRETATION", response.answer)

    def test_single_image_description_vlm_unavailable(self):
        query = "Describe this satellite image."
        images = ["test_scene.jpg"]

        with patch.object(self.agent.vlm, "is_available", False):
            response = self.agent.analyze(query=query, images=images)

        self.assertNotIn("RS-Qwen", response.execution_trace.tools_used)
        self.assertTrue(any("RS-Qwen unavailable locally" in s for s in response.execution_trace.steps))
        self.assertIn("RS-Qwen VLM unavailable locally", response.answer)

    def test_yolo_zero_detection_confidence_is_none(self):
        query = "Detect buildings in this image."
        images = ["empty_field.png"]

        mock_empty_yolo = {
            "success": True,
            "model": "YOLO11",
            "image": "empty_field.png",
            "results": {"objects": [], "counts": {}},
            "visualization": None,
            "error": None,
        }

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_empty_yolo):
            response = self.agent.analyze(query=query, images=images)

        self.assertIsNone(response.confidence)
        self.assertIn("No discrete objects were detected", response.answer)
        self.assertIn("N/A (zero detections above threshold)", response.answer)

    # -------------------------------------------------------------------------
    # Test 2: Single Image + Object Detection Query
    # -------------------------------------------------------------------------
    def test_single_image_object_detection(self):
        query = "Detect and count buildings and vehicles in this image."
        images = ["urban_tile.tif"]

        mock_yolo_result = {
            "success": True,
            "model": "YOLO11",
            "image": "urban_tile.tif",
            "results": {
                "objects": [
                    {"class": "building", "confidence": 0.92, "bbox": [10, 10, 50, 50]},
                    {"class": "building", "confidence": 0.88, "bbox": [60, 60, 100, 100]},
                    {"class": "passenger_vehicle", "confidence": 0.85, "bbox": [20, 20, 30, 30]},
                ],
                "counts": {"building": 2, "passenger_vehicle": 1},
            },
            "visualization": "runs/detect/urban_tile.tif",
            "error": None,
        }

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_yolo_result):
            response = self.agent.analyze(query=query, images=images)

        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_OBJECT_DETECTION.value)
        self.assertIn("YOLO11", response.execution_trace.tools_used)
        self.assertIn("object_detection", response.evidence)
        self.assertEqual(response.evidence["object_detection"]["results"]["counts"]["building"], 2)
        self.assertAlmostEqual(response.confidence, 0.8833, places=2)
        self.assertIn("3 object(s) were detected", response.answer)

    # -------------------------------------------------------------------------
    # Test 3: Two Images + Change Query
    # -------------------------------------------------------------------------
    def test_two_images_change_detection(self):
        query = "What changed between these two images?"
        images = ["t1_2022.png", "t2_2024.png"]

        mock_change_result = {
            "success": True,
            "model": "ChangeFormerV6",
            "image_a": "t1_2022.png",
            "image_b": "t2_2024.png",
            "change_map": "ml/change_detection/outputs/change_map.png",
            "total_pixels": 65536,
            "changed_pixels": 8192,
            "unchanged_pixels": 57344,
            "change_percentage": 12.50,
            "error": None,
        }

        with patch.object(TOOL_REGISTRY["change_detection"], "execute", return_value=mock_change_result):
            response = self.agent.analyze(query=query, images=images)

        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.BI_TEMPORAL_CHANGE.value)
        self.assertIn("ChangeFormerV6", response.execution_trace.tools_used)
        self.assertIn("change_detection", response.evidence)
        self.assertEqual(response.evidence["change_detection"]["change_percentage"], 12.50)
        self.assertIn("12.50%", response.answer)

    # -------------------------------------------------------------------------
    # Test 4: Optical + SAR Query
    # -------------------------------------------------------------------------
    def test_optical_sar_query(self):
        query = "Identify built-up and water-covered regions using both optical and SAR sensors."
        images = ["s2_patch.tif", "s1_patch.tif"]

        mock_croma_result = {
            "success": True,
            "model": "CROMA-Base",
            "representations": {
                "sar_encodings_shape": [1, 225, 768],
                "sar_gap_shape": [1, 768],
                "optical_encodings_shape": [1, 225, 768],
                "optical_gap_shape": [1, 768],
                "joint_encodings_shape": [1, 225, 768],
                "joint_gap_shape": [1, 768],
            },
            "summary": "Cross-modal bidirectional fused tokens extracted.",
            "error": None,
        }

        with patch.object(TOOL_REGISTRY["croma"], "execute", return_value=mock_croma_result):
            response = self.agent.analyze(query=query, images=images)

        self.assertIsNone(response.error)
        self.assertEqual(response.task, TaskType.CROSS_MODAL_OPTICAL_SAR.value)
        self.assertIn("CROMA-Base", response.execution_trace.tools_used)
        self.assertIn("optical_sar", response.evidence)
        self.assertIn("joint_encodings_shape", response.evidence["optical_sar"]["representations"])
        self.assertIn("225 patch tokens", response.answer)

    # -------------------------------------------------------------------------
    # Test 5: Missing Second Image for Change Detection
    # -------------------------------------------------------------------------
    def test_missing_second_image_for_change_detection(self):
        query = "Describe the changes between the two dates."
        images = ["single_image.png"]  # Only 1 image provided!

        response = self.agent.analyze(query=query, images=images)

        self.assertIsNotNone(response.error)
        self.assertEqual(response.task, TaskType.BI_TEMPORAL_CHANGE.value)
        self.assertEqual(response.confidence, 0.0)
        self.assertIn("requires exactly two images", response.error)
        self.assertIn("Input Error", response.answer)

    # -------------------------------------------------------------------------
    # Test 6: Missing SAR for Optical + SAR Query
    # -------------------------------------------------------------------------
    def test_missing_sar_for_optical_sar_query(self):
        query = "Use optical and SAR together to analyze this area."
        images = ["only_optical.tif"]  # Missing SAR!

        response = self.agent.analyze(query=query, images=images)

        self.assertIsNotNone(response.error)
        self.assertEqual(response.task, TaskType.CROSS_MODAL_OPTICAL_SAR.value)
        self.assertEqual(response.confidence, 0.0)
        self.assertIn("SAR input is missing", response.error)

    # -------------------------------------------------------------------------
    # Test 7: Unsupported Input Format
    # -------------------------------------------------------------------------
    def test_unsupported_input_format(self):
        query = "What is visible in this file?"
        images = ["satellite_report.pdf"]  # PDF is not a raster format!

        response = self.agent.analyze(query=query, images=images)

        self.assertIsNotNone(response.error)
        self.assertIn("Unsupported input format", response.error)
        self.assertEqual(response.confidence, 0.0)

    # -------------------------------------------------------------------------
    # Test 8: Missing Model / Checkpoint Handling
    # -------------------------------------------------------------------------
    def test_missing_model_checkpoint(self):
        query = "Find all vehicles in this image."
        images = ["aerial.jpg"]

        mock_missing_ckpt = {
            "success": False,
            "model": "YOLO11",
            "image": "aerial.jpg",
            "results": {"objects": [], "counts": {}},
            "visualization": None,
            "error": "YOLO model weights not found: best.pt does not exist",
        }

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_missing_ckpt):
            response = self.agent.analyze(query=query, images=images)

        self.assertEqual(response.task, TaskType.SINGLE_IMAGE_OBJECT_DETECTION.value)
        self.assertFalse(response.evidence["object_detection"]["success"])
        self.assertEqual(response.confidence, 0.0)
        self.assertIn("Object detection could not be completed", response.answer)

    # -------------------------------------------------------------------------
    # Test 9: FastAPI Endpoints Integration
    # -------------------------------------------------------------------------
    def test_fastapi_endpoints(self):
        # Health check
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")
        self.assertIn("yolo", res_health.json()["available_specialists"])

        # Tool listing
        res_tools = self.client.get("/api/agent/tools")
        self.assertEqual(res_tools.status_code, 200)
        self.assertIn("change_detection", res_tools.json())

        # Analyze JSON endpoint
        payload = {
            "query": "Give me a general description of this satellite scene.",
            "images": ["test_scene.png"],
        }
        res_analyze = self.client.post("/api/agent/analyze/json", json=payload)
        self.assertEqual(res_analyze.status_code, 200)
        data = res_analyze.json()
        self.assertEqual(data["task"], TaskType.SINGLE_IMAGE_DESCRIPTION.value)
        self.assertIn("execution_trace", data)
        self.assertIn("answer", data)


    # -------------------------------------------------------------------------
    # Test 10: Specialist Evidence Grounding Context for Object Detection
    # -------------------------------------------------------------------------
    def test_rs_qwen_grounding_context_object_detection(self):
        query = "Identify and count all vehicles in this scene."
        images = ["harbor.tif"]

        mock_yolo = {
            "success": True,
            "model": "YOLO11",
            "image": "harbor.tif",
            "results": {
                "objects": [
                    {"class": "motorboat", "confidence": 0.90, "bbox": [0, 0, 10, 10]},
                    {"class": "motorboat", "confidence": 0.86, "bbox": [20, 20, 30, 30]},
                ],
                "counts": {"motorboat": 2},
            },
            "visualization": "runs/detect/harbor.tif",
            "error": None,
        }

        mock_vlm_answer = "Two motorboats are clearly docked along the pier."

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_yolo), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=mock_vlm_answer) as mock_gen:
            response = self.agent.analyze(query=query, images=images)

        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        self.assertEqual(call_kwargs["image_path"], "harbor.tif")
        prompt_passed = call_kwargs["question"]

        # Check that empirical specialist evidence was injected into RS-Qwen prompt
        self.assertIn("[SPECIALIST MODEL EMPIRICAL EVIDENCE]", prompt_passed)
        self.assertIn("2 object(s) detected: 2 motorboat(s)", prompt_passed)
        self.assertIn("Detection Confidence: 88.0%", prompt_passed)
        self.assertIn("USER QUERY:\nIdentify and count all vehicles in this scene.", prompt_passed)
        self.assertIn("runs/detect/harbor.tif", prompt_passed)

        self.assertEqual(response.answer, mock_vlm_answer)
        self.assertIn("RS-Qwen", response.execution_trace.tools_used)

    # -------------------------------------------------------------------------
    # Test 11: Specialist Evidence Grounding Context for Bi-Temporal Change
    # -------------------------------------------------------------------------
    def test_rs_qwen_grounding_context_change_detection(self):
        query = "Explain what changed between the two dates."
        images = ["pre.png", "post.png"]

        mock_change = {
            "success": True,
            "model": "ChangeFormerV6",
            "image_a": "pre.png",
            "image_b": "post.png",
            "change_map": "outputs/change_map_post.png",
            "total_pixels": 10000,
            "changed_pixels": 1500,
            "unchanged_pixels": 8500,
            "change_percentage": 15.0,
            "error": None,
        }

        mock_vlm_answer = "15% of the surface area experienced structural transformation."

        with patch.object(TOOL_REGISTRY["change_detection"], "execute", return_value=mock_change), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=mock_vlm_answer) as mock_gen:
            response = self.agent.analyze(query=query, images=images)

        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        # For bi-temporal change, target image is post-change image (image_paths[1])
        self.assertEqual(call_kwargs["image_path"], "post.png")
        prompt_passed = call_kwargs["question"]

        self.assertIn("[SPECIALIST MODEL EMPIRICAL EVIDENCE]", prompt_passed)
        self.assertIn("15.00% changed surface area", prompt_passed)
        self.assertIn("1,500 changed / 10,000 total pixels", prompt_passed)
        self.assertIn("outputs/change_map_post.png", prompt_passed)

        self.assertEqual(response.answer, mock_vlm_answer)
        self.assertIn("ChangeFormerV6", response.execution_trace.tools_used)
        self.assertIn("RS-Qwen", response.execution_trace.tools_used)

    # -------------------------------------------------------------------------
    # Test 12: Specialist Evidence Grounding Context for Optical + SAR
    # -------------------------------------------------------------------------
    def test_rs_qwen_grounding_context_optical_sar(self):
        query = "Analyze built-up and water areas using optical and SAR."
        images = ["optical.tif", "sar.tif"]

        mock_croma = {
            "success": True,
            "model": "CROMA-Base",
            "representations": {
                "sar_encodings_shape": [1, 225, 768],
                "sar_gap_shape": [1, 768],
                "optical_encodings_shape": [1, 225, 768],
                "optical_gap_shape": [1, 768],
                "joint_encodings_shape": [1, 225, 768],
                "joint_gap_shape": [1, 768],
            },
            "summary": "Fused multimodal tokens extracted.",
            "error": None,
        }

        mock_vlm_answer = "The combined optical and radar response highlights calm water and dense buildings."

        with patch.object(TOOL_REGISTRY["croma"], "execute", return_value=mock_croma), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=mock_vlm_answer) as mock_gen:
            response = self.agent.analyze(query=query, images=images)

        mock_gen.assert_called_once()
        prompt_passed = mock_gen.call_args.kwargs["question"]

        self.assertIn("[SPECIALIST MODEL EMPIRICAL EVIDENCE]", prompt_passed)
        self.assertIn("CROMA-Base", prompt_passed)
        self.assertIn("Joint tokens [1, 225, 768]", prompt_passed)
        self.assertIn("Pooled scene embedding [1, 768]", prompt_passed)

        self.assertEqual(response.answer, mock_vlm_answer)
        self.assertIn("CROMA-Base", response.execution_trace.tools_used)
        self.assertIn("RS-Qwen", response.execution_trace.tools_used)

    # -------------------------------------------------------------------------
    # Test 13: RS-Qwen Runtime Failure Gracefully Falls Back
    # -------------------------------------------------------------------------
    def test_rs_qwen_failure_gracefully_falls_back(self):
        query = "Describe this satellite scene."
        images = ["test_scene.png"]

        # Simulate generate_response raising an error during inference
        with patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", side_effect=RuntimeError("Weights unreadable")):
            response = self.agent.analyze(query=query, images=images)

        self.assertIsNone(response.error)
        self.assertNotIn("RS-Qwen", response.execution_trace.tools_used)
        self.assertTrue(any("RS-Qwen synthesis failed" in s for s in response.execution_trace.steps))
        # Falls back to structured evidence report
        self.assertIn("DIRECT OBSERVATIONS", response.answer)

    # -------------------------------------------------------------------------
    # Test 14: Regression Test A - YOLO Success + Zero Detections (No Fabricated Count)
    # -------------------------------------------------------------------------
    def test_regression_yolo_zero_detections_no_fabricated_count(self):
        """Verify that when YOLO returns zero detections, RS-Qwen cannot fabricate a numerical count."""
        query = "Detect and count the major discrete objects visible in this remote sensing image."
        images = ["test_empty.png"]

        mock_yolo = {
            "success": True,
            "model": "YOLO11",
            "image": "test_empty.png",
            "results": {
                "objects": [],
                "counts": {},
            },
            "visualization": None,
            "error": None,
        }

        # The exact hallucination observed in real Colab integration
        hallucinated_qwen_output = (
            "There are 10 major discrete objects visible in this remote sensing image."
        )

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_yolo), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=hallucinated_qwen_output):
            response = self.agent.analyze(query=query, images=images)

        # 1. Confidence remains None (no detections above threshold)
        self.assertIsNone(response.confidence)

        # 2. Final answer must NOT contain the fabricated count
        self.assertNotIn("10 major discrete objects", response.answer)
        self.assertNotIn("10 objects", response.answer)
        self.assertNotIn(" 10 ", response.answer)

        # 3. Final answer must explicitly communicate no detections
        self.assertIn("No discrete objects were detected above the configured confidence threshold", response.answer)

        # 4. Tools recorded
        self.assertIn("YOLO11", response.execution_trace.tools_used)
        self.assertIn("RS-Qwen", response.execution_trace.tools_used)

    # -------------------------------------------------------------------------
    # Test 15: Regression Test B - YOLO Success + Real Detections (Count Preserved)
    # -------------------------------------------------------------------------
    def test_regression_yolo_real_detections_count_preserved(self):
        """Verify that when YOLO detects objects, Qwen cannot alter the verified count."""
        query = "Detect and count all buildings and vehicles."
        images = ["urban.tif"]

        mock_yolo = {
            "success": True,
            "model": "YOLO11",
            "image": "urban.tif",
            "results": {
                "objects": [
                    {"class": "building", "confidence": 0.90, "bbox": [0, 0, 10, 10]},
                    {"class": "building", "confidence": 0.86, "bbox": [20, 20, 30, 30]},
                    {"class": "passenger_vehicle", "confidence": 0.85, "bbox": [50, 50, 60, 60]},
                ],
                "counts": {"building": 2, "passenger_vehicle": 1},
            },
            "visualization": "runs/detect/urban.tif",
            "error": None,
        }

        # Qwen tries to alter counts (saying 7 buildings and 4 vehicles instead of 2 and 1)
        altered_qwen_output = "There are 7 buildings and 4 passenger_vehicles visible across the area."

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_yolo), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=altered_qwen_output):
            response = self.agent.analyze(query=query, images=images)

        # Supplied count is preserved and verified
        self.assertIn("2 building(s)", response.answer)
        self.assertIn("1 passenger_vehicle(s)", response.answer)
        self.assertNotIn("7 buildings", response.answer)
        self.assertNotIn("4 passenger_vehicles", response.answer)
        self.assertAlmostEqual(response.confidence, 0.87, places=2)

    # -------------------------------------------------------------------------
    # Test 16: Regression Test C - YOLO Failure/Unavailable (Reports Unavailable, No Fabricated Count)
    # -------------------------------------------------------------------------
    def test_regression_yolo_failure_reports_unavailable_no_fabricated_count(self):
        """Verify that when YOLO fails, detection unavailability is reported and guessed counts are blocked."""
        query = "Find and count all ships in this image."
        images = ["port.png"]

        mock_failed_yolo = {
            "success": False,
            "model": "YOLO11",
            "image": "port.png",
            "results": {"objects": [], "counts": {}},
            "visualization": None,
            "error": "CUDA out of memory during YOLO inference",
        }

        # Qwen tries to guess a count despite specialist failure
        guessed_qwen_output = "I can see 6 ships docked in the bay."

        with patch.object(TOOL_REGISTRY["yolo"], "execute", return_value=mock_failed_yolo), \
             patch.object(self.agent.vlm, "is_available", True), \
             patch("ml.rs_qwen.inference.generate_response", return_value=guessed_qwen_output):
            response = self.agent.analyze(query=query, images=images)

        # Must report unavailable/failed
        self.assertIn("Automated object detection could not be completed", response.answer)
        # Must not adopt the fabricated count
        self.assertNotIn("6 ships", response.answer)
        self.assertEqual(response.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

