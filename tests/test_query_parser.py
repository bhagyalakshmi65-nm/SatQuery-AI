"""Comprehensive unit test suite for Semantic Query Understanding & Query Decomposition."""

import unittest

from backend.agent.agent import SatQueryAgent
from backend.agent.query_parser import SemanticQueryParser
from backend.agent.schemas import TaskType


class TestSemanticQueryParser(unittest.TestCase):
    """Test suite covering arbitrary query parsing, decomposition, operational reasoning, and agent integration."""

    @classmethod
    def setUpClass(cls):
        cls.parser = SemanticQueryParser()
        cls.agent = SatQueryAgent()

    # -------------------------------------------------------------------------
    # Test 1: Object Counting Queries
    # -------------------------------------------------------------------------
    def test_object_counting_queries(self):
        """Verify counting queries identify discrete objects, counting requirements, and subtasks."""
        res = self.parser.understand("How many buildings are there?")
        self.assertEqual(res.primary_intent, "object_counting")
        self.assertTrue(res.requires_counting)
        self.assertIn("building", res.entities)
        self.assertEqual(len(res.subtasks), 2)
        self.assertEqual(res.subtasks[0].tool_hint, "YOLO11")
        self.assertEqual(res.subtasks[1].tool_hint, "RS-Qwen")

        # Aircraft variant
        res_ac = self.parser.understand("Count all aircraft visible in this satellite scene.")
        self.assertTrue(res_ac.requires_counting)
        self.assertIn("aircraft", res_ac.entities)

    # -------------------------------------------------------------------------
    # Test 2: Object Localization Queries
    # -------------------------------------------------------------------------
    def test_object_localization_queries(self):
        """Verify localization queries identify bounding box / localization needs."""
        res = self.parser.understand("Where are the buildings?")
        self.assertEqual(res.primary_intent, "object_localization")
        self.assertTrue(res.requires_localization)
        self.assertIn("building", res.entities)
        self.assertEqual(len(res.subtasks), 2)
        self.assertEqual(res.subtasks[0].operation, "detect")
        self.assertEqual(res.subtasks[0].tool_hint, "YOLO11")

    # -------------------------------------------------------------------------
    # Test 3: Land Cover & Coverage Measurement Queries
    # -------------------------------------------------------------------------
    def test_coverage_measurement_queries(self):
        """Verify queries asking for percentage, area, or extent of terrain types."""
        # Vegetation extent
        res_veg = self.parser.understand("How much vegetation is present?")
        self.assertEqual(res_veg.primary_intent, "coverage_measurement")
        self.assertIn("vegetation", res_veg.entities)
        self.assertTrue(res_veg.requires_classification)
        self.assertEqual(res_veg.subtasks[0].tool_hint, "BigEarthNet-ResNet50-SeCo-12B")

        # Water percentage
        res_water = self.parser.understand("What percentage of this area is water?")
        self.assertEqual(res_water.primary_intent, "coverage_measurement")
        self.assertIn("percentage", res_water.requested_measurements)
        self.assertIn("water", res_water.entities)
        self.assertTrue(res_water.requires_segmentation)

    # -------------------------------------------------------------------------
    # Test 4: Land Cover Classification Queries
    # -------------------------------------------------------------------------
    def test_landcover_classification_queries(self):
        """Verify classification intent for terrain/environmental queries."""
        res_lc = self.parser.understand("What type of land cover is shown?")
        self.assertEqual(res_lc.primary_intent, "landcover_classification")
        self.assertTrue(res_lc.requires_classification)
        self.assertEqual(res_lc.subtasks[0].tool_hint, "BigEarthNet-ResNet50-SeCo-12B")

        # Urban vs Rural
        res_ur = self.parser.understand("Is this area urban or rural?")
        self.assertIn("urban", res_ur.entities)
        self.assertIn("rural", res_ur.entities)
        self.assertTrue(res_ur.requires_classification)

    # -------------------------------------------------------------------------
    # Test 5: Temporal Change Detection Queries
    # -------------------------------------------------------------------------
    def test_temporal_change_queries(self):
        """Verify bi-temporal change detection queries."""
        res = self.parser.understand("What changed between these images?")
        self.assertEqual(res.primary_intent, "temporal_change_analysis")
        self.assertTrue(res.requires_temporal_reasoning)
        self.assertTrue(res.requires_comparison)
        self.assertEqual(res.subtasks[0].tool_hint, "ChangeFormerV6")
        self.assertEqual(res.subtasks[0].operation, "compare")

        # Difference over time
        res_diff = self.parser.understand("Describe the differences over time between the two dates.")
        self.assertTrue(res_diff.requires_temporal_reasoning)

    # -------------------------------------------------------------------------
    # Test 6: Spatial Directional Queries
    # -------------------------------------------------------------------------
    def test_spatial_directional_queries(self):
        """Verify directional queries (e.g. north of river)."""
        res = self.parser.understand("What is north of the river?")
        self.assertEqual(res.primary_intent, "spatial_object_query")
        self.assertTrue(res.requires_gis)
        self.assertEqual(len(res.spatial_relationships), 1)
        rel = res.spatial_relationships[0]
        self.assertEqual(rel["relation"], "directional")
        self.assertEqual(rel["direction"], "north")
        self.assertEqual(rel["reference"], "river")

        # Should generate 3 subtasks: segment reference, delineate sector with GIS, analyze with RS-Qwen
        self.assertEqual(len(res.subtasks), 3)
        self.assertEqual(res.subtasks[1].tool_hint, "GIS")

    # -------------------------------------------------------------------------
    # Test 7: Compound Spatial Distance Query Decomposition (SPEC REQUIREMENT)
    # -------------------------------------------------------------------------
    def test_compound_spatial_distance_query_decomposition(self):
        """Query: 'How many buildings are within 100m of the river?'

        Must decompose into:
        Task 1: Detect buildings (YOLO11)
        Task 2: Identify/segment river (BigEarthNet)
        Task 3: Calculate building-to-river distances (GIS)
        Task 4: Filter buildings <= 100m (GIS)
        Task 5: Count them (RS-Qwen)
        """
        query = "How many buildings are within 100m of the river?"
        res = self.parser.understand(query)

        self.assertEqual(res.primary_intent, "spatial_object_query")
        self.assertTrue(res.is_compound_query)
        self.assertTrue(res.requires_counting)
        self.assertTrue(res.requires_gis)
        self.assertIn("building", res.entities)
        self.assertIn("river", res.entities)

        # Verify spatial relation struct
        self.assertEqual(len(res.spatial_relationships), 1)
        rel = res.spatial_relationships[0]
        self.assertEqual(rel["relation"], "within_distance")
        self.assertEqual(rel["reference"], "river")
        self.assertEqual(rel["distance_m"], 100.0)
        self.assertEqual(rel["constraint"], "<= 100m")

        # Verify exactly 5 subtasks
        self.assertEqual(len(res.subtasks), 5)

        # Task 1: Detect buildings
        t1 = res.subtasks[0]
        self.assertEqual(t1.step_number, 1)
        self.assertEqual(t1.operation, "detect")
        self.assertEqual(t1.target_entity, "building")
        self.assertEqual(t1.tool_hint, "YOLO11")
        self.assertEqual(t1.operational_reasoning, "YOLO11 is required for building localization.")

        # Task 2: Identify/segment river
        t2 = res.subtasks[1]
        self.assertEqual(t2.step_number, 2)
        self.assertEqual(t2.operation, "segment")
        self.assertEqual(t2.target_entity, "river")
        self.assertEqual(t2.tool_hint, "BigEarthNet-ResNet50-SeCo-12B")
        self.assertEqual(t2.operational_reasoning, "Land cover segmentation is required for river boundary extraction.")

        # Task 3: Calculate distances
        t3 = res.subtasks[2]
        self.assertEqual(t3.step_number, 3)
        self.assertEqual(t3.operation, "distance_calc")
        self.assertEqual(t3.tool_hint, "GIS")
        self.assertEqual(t3.operational_reasoning, "GIS is required for distance calculation.")

        # Task 4: Filter buildings <= 100m
        t4 = res.subtasks[3]
        self.assertEqual(t4.step_number, 4)
        self.assertEqual(t4.operation, "filter")
        self.assertEqual(t4.condition, "<= 100m")
        self.assertEqual(t4.tool_hint, "GIS")
        self.assertEqual(t4.operational_reasoning, "GIS is required for spatial threshold filtering.")

        # Task 5: Count them
        t5 = res.subtasks[4]
        self.assertEqual(t5.step_number, 5)
        self.assertEqual(t5.operation, "count")
        self.assertEqual(t5.tool_hint, "RS-Qwen")
        self.assertEqual(t5.operational_reasoning, "RS-Qwen is required for final answer synthesis and count reporting.")

    # -------------------------------------------------------------------------
    # Test 8: Compound Region Containment Query Decomposition
    # -------------------------------------------------------------------------
    def test_compound_region_containment_query_decomposition(self):
        """Query: 'Find agricultural areas and count buildings inside them.'

        Must decompose into:
        Task 1: Segment agricultural areas
        Task 2: Detect buildings
        Task 3: Spatial intersection inside agricultural areas
        Task 4: Count buildings inside agricultural areas
        """
        query = "Find agricultural areas and count buildings inside them."
        res = self.parser.understand(query)

        self.assertEqual(res.primary_intent, "spatial_object_query")
        self.assertTrue(res.is_compound_query)
        self.assertIn("agriculture", res.entities)
        self.assertIn("building", res.entities)
        self.assertEqual(len(res.subtasks), 4)

        self.assertEqual(res.subtasks[0].operation, "segment")
        self.assertEqual(res.subtasks[0].target_entity, "agriculture")
        self.assertEqual(res.subtasks[0].tool_hint, "BigEarthNet-ResNet50-SeCo-12B")

        self.assertEqual(res.subtasks[1].operation, "detect")
        self.assertEqual(res.subtasks[1].target_entity, "building")
        self.assertEqual(res.subtasks[1].tool_hint, "YOLO11")

        self.assertEqual(res.subtasks[2].operation, "spatial_filter")
        self.assertEqual(res.subtasks[2].tool_hint, "GIS")

        self.assertEqual(res.subtasks[3].operation, "count")
        self.assertEqual(res.subtasks[3].tool_hint, "RS-Qwen")

        # Also test direct phrasing: "How many buildings are inside the vegetated area?"
        res_veg = self.parser.understand("How many buildings are inside the vegetated area?")
        self.assertTrue(res_veg.is_compound_query)
        self.assertIn("vegetation", res_veg.entities)
        self.assertIn("building", res_veg.entities)
        self.assertEqual(len(res_veg.subtasks), 4)

    # -------------------------------------------------------------------------
    # Test 9: Arbitrary / Unanticipated Queries
    # -------------------------------------------------------------------------
    def test_arbitrary_unanticipated_queries(self):
        """Test system robustness on queries not seen during development."""
        # Query A: Solar panels near industrial facilities
        res_solar = self.parser.understand("Are there solar panels located within 200m of the industrial facilities?")
        self.assertEqual(res_solar.primary_intent, "spatial_object_query")
        self.assertIn("solar_panel", res_solar.entities)
        self.assertTrue(res_solar.requires_gis)
        self.assertEqual(len(res_solar.subtasks), 5)
        self.assertEqual(res_solar.subtasks[0].target_entity, "solar_panel")
        self.assertEqual(res_solar.subtasks[0].tool_hint, "YOLO11")

        # Query B: Docked container ships near pier
        res_ships = self.parser.understand("Locate container ships docked adjacent to the pier.")
        self.assertIn("ship", res_ships.entities)
        self.assertIn("pier", res_ships.entities)
        self.assertTrue(res_ships.requires_localization)

        # Query C: Joint SAR & Optical multi-sensor query
        res_sar = self.parser.understand("Fuse synthetic aperture radar with multispectral optical to map flooded regions.")
        self.assertEqual(res_sar.primary_intent, "cross_modal_analysis")
        self.assertEqual(res_sar.subtasks[0].tool_hint, "CROMA-Base")

    # -------------------------------------------------------------------------
    # Test 10: Operational Reasoning Clarity (No Hidden Chain-of-Thought)
    # -------------------------------------------------------------------------
    def test_concise_operational_reasoning_no_hidden_cot(self):
        """Operational reasoning must be concise, auditable justifications with zero internal monologue."""
        res = self.parser.understand("How many buildings are within 100m of the river?")
        for reason in res.operational_reasoning:
            self.assertIsInstance(reason, str)
            self.assertLess(len(reason), 120, "Reasoning statements must be concise")
            # Must not contain CoT buzzwords
            self.assertNotIn("think", reason.lower())
            self.assertNotIn("let's", reason.lower())
            self.assertNotIn("first I will", reason.lower())
            self.assertNotIn("step-by-step", reason.lower())

    # -------------------------------------------------------------------------
    # Test 11: End-to-End SatQueryAgent Integration
    # -------------------------------------------------------------------------
    def test_agent_analyze_includes_query_understanding(self):
        """SatQueryAgent.analyze() must execute query understanding and populate response."""
        response = self.agent.analyze(
            query="How many buildings are within 100m of the river?",
            images=["test_scene.jpg"],
        )

        self.assertIsNotNone(response.query_understanding)
        qu = response.query_understanding
        self.assertEqual(qu.primary_intent, "spatial_object_query")
        self.assertTrue(qu.is_compound_query)
        self.assertEqual(len(qu.subtasks), 5)

        # Execution trace must record the concise operational steps
        trace_str = " ".join(response.execution_trace.steps)
        self.assertIn("Query Understanding:", trace_str)
        self.assertIn("YOLO11 is required for building localization.", trace_str)
        self.assertIn("GIS is required for distance calculation.", trace_str)


if __name__ == "__main__":
    unittest.main()

