"""Unit test suite for Dynamic Model and Tool Selection (AnalysisPlanner & CapabilityRegistry)."""

import unittest
from typing import Optional

from backend.agent.agent import SatQueryAgent
from backend.agent.capability_registry import CapabilityRegistry, CapabilityEntry, CAPABILITY_REGISTRY
from backend.agent.planner import AnalysisPlanner
from backend.agent.query_parser import SemanticQueryParser
from backend.agent.schemas import (
    AnalysisPlan,
    ImageSceneInspection,
    SceneLayout,
    TaskType,
)


class TestAnalysisPlanner(unittest.TestCase):
    """Test dynamic tool selection and capability registry for SatQuery AI."""

    @classmethod
    def setUpClass(cls):
        cls.parser = SemanticQueryParser()
        cls.planner = AnalysisPlanner()
        cls.agent = SatQueryAgent()

    def _make_inspection(
        self,
        scene_count: int = 1,
        file_count: int = 1,
        has_temporal_cues: bool = False,
    ) -> ImageSceneInspection:
        return ImageSceneInspection(
            is_remote_sensing=True,
            file_count=file_count,
            scene_count=scene_count,
            layout=SceneLayout.SIDE_BY_SIDE if scene_count > 1 else SceneLayout.SINGLE,
            has_temporal_cues=has_temporal_cues,
        )

    # -------------------------------------------------------------------------
    # Test 1: Single Specialist Model Queries (YOLO11 & BigEarthNet)
    # -------------------------------------------------------------------------
    def test_single_specialist_model_queries(self):
        """Verify queries requiring exactly one specialist model."""
        # 1. Counting -> YOLO11
        q_count = "How many buildings are visible?"
        qu_count = self.parser.understand(q_count)
        plan_count = self.planner.plan(qu_count)
        self.assertEqual(plan_count.selected_models, ["YOLO11"])
        self.assertIn("RS-Qwen", plan_count.selected_tools)
        self.assertFalse(plan_count.requires_gis)
        self.assertFalse(plan_count.comparison_requested)

        # 2. Localization -> YOLO11
        q_loc = "Where are the buildings?"
        qu_loc = self.parser.understand(q_loc)
        plan_loc = self.planner.plan(qu_loc)
        self.assertEqual(plan_loc.selected_models, ["YOLO11"])
        self.assertIn("RS-Qwen", plan_loc.selected_tools)

        # 3. Land cover classification -> BigEarthNet
        q_lc = "What type of land cover is shown?"
        qu_lc = self.parser.understand(q_lc)
        plan_lc = self.planner.plan(qu_lc)
        self.assertEqual(plan_lc.selected_models, ["BigEarthNet-ResNet50-SeCo-12B"])
        self.assertIn("RS-Qwen", plan_lc.selected_tools)
        self.assertFalse(plan_lc.requires_gis)

    # -------------------------------------------------------------------------
    # Test 2: Specialist Model + GIS Queries (SegFormer + GIS)
    # -------------------------------------------------------------------------
    def test_specialist_plus_gis_queries(self):
        """Verify queries requiring specialist segmentation + GIS calculation."""
        # 1. Vegetation coverage measurement
        q_veg = "How much vegetation is present?"
        qu_veg = self.parser.understand(q_veg)
        plan_veg = self.planner.plan(qu_veg)
        self.assertEqual(plan_veg.selected_models, ["SegFormer"])
        self.assertIn("GIS", plan_veg.selected_tools)
        self.assertIn("RS-Qwen", plan_veg.selected_tools)
        self.assertTrue(plan_veg.requires_gis)

        # 2. Water percentage calculation
        q_water = "What percentage of this area is water?"
        qu_water = self.parser.understand(q_water)
        plan_water = self.planner.plan(qu_water)
        self.assertEqual(plan_water.selected_models, ["SegFormer"])
        self.assertIn("GIS", plan_water.selected_tools)
        self.assertTrue(plan_water.requires_gis)

    # -------------------------------------------------------------------------
    # Test 3: Multiple Specialist Models + GIS Queries
    # -------------------------------------------------------------------------
    def test_multi_specialist_plus_gis_queries(self):
        """Verify compound queries requiring multiple specialist models + GIS."""
        # 1. Distance buffer: "How many buildings are within 100m of the river?"
        # Requires YOLO11 (buildings) + SegFormer (river) + GIS (distance/buffer)
        q_dist = "How many buildings are within 100m of the river?"
        qu_dist = self.parser.understand(q_dist)
        plan_dist = self.planner.plan(qu_dist)
        self.assertIn("YOLO11", plan_dist.selected_models)
        self.assertIn("SegFormer", plan_dist.selected_models)
        self.assertIn("GIS", plan_dist.selected_tools)
        self.assertIn("RS-Qwen", plan_dist.selected_tools)
        self.assertTrue(plan_dist.requires_gis)

        # 2. Spatial containment: "Find agricultural areas and count buildings inside them."
        # Requires SegFormer (agricultural areas) + YOLO11 (buildings) + GIS (containment)
        q_cont = "Find agricultural areas and count buildings inside them."
        qu_cont = self.parser.understand(q_cont)
        plan_cont = self.planner.plan(qu_cont)
        self.assertIn("YOLO11", plan_cont.selected_models)
        self.assertIn("SegFormer", plan_cont.selected_models)
        self.assertIn("GIS", plan_cont.selected_tools)
        self.assertTrue(plan_cont.requires_gis)

    # -------------------------------------------------------------------------
    # Test 4: Zero Specialist Models (VLM Only)
    # -------------------------------------------------------------------------
    def test_zero_specialist_models_queries(self):
        """Verify high-level scene description requires zero specialist models."""
        q_desc = "Describe this image."
        qu_desc = self.parser.understand(q_desc)
        plan_desc = self.planner.plan(qu_desc)
        self.assertEqual(plan_desc.selected_models, [])
        self.assertEqual(plan_desc.selected_tools, ["RS-Qwen"])
        self.assertFalse(plan_desc.requires_gis)
        self.assertFalse(plan_desc.comparison_requested)
        self.assertTrue(any("No specialist models required" in r for r in plan_desc.operational_reasons))

    # -------------------------------------------------------------------------
    # Test 5: Comparison & Temporal Queries
    # -------------------------------------------------------------------------
    def test_comparison_and_temporal_queries(self):
        """Verify comparison and temporal change detection queries."""
        # 1. Feature comparison across scenes: "Did vegetation increase?"
        q_feat = "Did vegetation increase?"
        insp_feat = self._make_inspection(scene_count=2, file_count=2, has_temporal_cues=True)
        qu_feat = self.parser.understand(q_feat, scene_inspection=insp_feat)
        plan_feat = self.planner.plan(qu_feat, scene_inspection=insp_feat)
        self.assertTrue(plan_feat.comparison_requested)
        self.assertEqual(plan_feat.selected_models, ["SegFormer"])
        self.assertIn("GIS", plan_feat.selected_tools)

        # 2. Pixel change detection: "What changed?"
        q_chg = "What changed between these images?"
        insp_chg = self._make_inspection(scene_count=2, file_count=2, has_temporal_cues=True)
        qu_chg = self.parser.understand(q_chg, scene_inspection=insp_chg)
        plan_chg = self.planner.plan(qu_chg, scene_inspection=insp_chg)
        self.assertTrue(plan_chg.comparison_requested)
        self.assertEqual(plan_chg.selected_models, ["ChangeFormerV6"])

    # -------------------------------------------------------------------------
    # Test 6: Extensible Capability Registry & Dynamic Registration
    # -------------------------------------------------------------------------
    def test_dynamic_capability_registration(self):
        """Verify newly registered models are dynamically selected by declared capabilities."""
        custom_registry = CapabilityRegistry(populate_defaults=True)
        custom_registry.register(CapabilityEntry(
            name="ThermalWildfireDetector",
            model_type="specialist_model",
            capabilities=["wildfire_detection", "burn_severity", "thermal_anomaly"],
            tags=["wildfire", "hotspot", "burn"],
            description="Specialized sensor model for wildfire and thermal hotspot detection.",
        ))

        planner = AnalysisPlanner(registry=custom_registry)
        qu = self.parser.understand("Detect wildfire hotspots across the landscape")
        plan = planner.plan(qu)

        self.assertIn("ThermalWildfireDetector", plan.selected_models)
        self.assertIn("RS-Qwen", plan.selected_tools)
        self.assertTrue(any("ThermalWildfireDetector" in r for r in plan.operational_reasons))

    def test_capability_registry_methods(self):
        """Verify registry lookup, listing, filtering, and reset."""
        reg = CapabilityRegistry(populate_defaults=True)
        self.assertIsNotNone(reg.get_entry("YOLO11"))
        self.assertIsNotNone(reg.get_entry("GIS"))
        self.assertIsNotNone(reg.get_entry("RS-Qwen"))

        caps = reg.get_capabilities_for_tool("YOLO11")
        self.assertIn("object_detection", caps)
        self.assertIn("object_counting", caps)

        all_caps = reg.list_capabilities()
        self.assertIn("SegFormer", all_caps)

        models, tools = reg.find_tools_for_capabilities(["object_detection", "distance"])
        self.assertIn("YOLO11", models)
        self.assertIn("GIS", tools)

    # -------------------------------------------------------------------------
    # Test 7: Full Agent Integration & AnalysisPlan in AgentResponse
    # -------------------------------------------------------------------------
    def test_agent_integration_includes_analysis_plan(self):
        """Verify agent.analyze() produces an analysis plan and includes it in the response and trace."""
        # Test counting query on synthetic image
        import numpy as np
        from PIL import Image
        import io

        arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.getvalue()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        try:
            resp = self.agent.analyze(
                query="How many buildings are visible?",
                images=[tmp_path],
            )
            self.assertIsNotNone(resp.analysis_plan)
            self.assertEqual(resp.analysis_plan.selected_models, ["YOLO11"])
            self.assertIn("RS-Qwen", resp.analysis_plan.selected_tools)
            self.assertFalse(resp.analysis_plan.comparison_requested)
            self.assertTrue(len(resp.analysis_plan.operational_reasons) > 0)
            
            # Check execution trace contains Analysis Plan step
            plan_traces = [s for s in resp.execution_trace.steps if "Analysis Plan:" in s]
            self.assertTrue(len(plan_traces) > 0)
        finally:
            import os
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()

