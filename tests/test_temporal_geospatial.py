"""Unit and integration tests for Location + Time-Based Earth Observation Analysis in SatQuery AI."""

import os
from pathlib import Path
import unittest

from backend.agent.agent import SatQueryAgent
from backend.agent.query_parser import SemanticQueryParser
from backend.agent.schemas import TaskType
from backend.agent.temporal_geospatial import (
    LocationResolver,
    SatelliteDataCatalog,
    TemporalProgressionAnalyzer,
    TemporalQueryExtractor,
)


class TestTemporalGeospatial(unittest.TestCase):
    """Test suite for location resolution, temporal extraction, catalog matching, and agent workflows."""

    def setUp(self):
        self.parser = SemanticQueryParser()
        self.agent = SatQueryAgent()

    # -------------------------------------------------------------------------
    # 1. Location Resolution & Ambiguity
    # -------------------------------------------------------------------------
    def test_location_resolver_known_cities(self):
        """Verify resolution of benchmark cities and aliases to coordinates and bounding boxes."""
        res_blr = LocationResolver.resolve("What changed in Bengaluru between 2022 and 2026?")
        self.assertTrue(res_blr["resolved"])
        self.assertEqual(res_blr["name"], "Bengaluru")
        self.assertEqual(res_blr["country"], "India")
        self.assertAlmostEqual(res_blr["coordinates"][0], 12.9716, places=2)

        res_alias = LocationResolver.resolve("Show me changes in bangalore since 2020")
        self.assertTrue(res_alias["resolved"])
        self.assertEqual(res_alias["name"], "Bengaluru")

        res_hyd = LocationResolver.resolve("Urban expansion in Hyderabad from 2020 to 2025")
        self.assertTrue(res_hyd["resolved"])
        self.assertEqual(res_hyd["name"], "Hyderabad")

    def test_location_resolver_coordinates(self):
        """Verify explicit latitude and longitude coordinates."""
        res_coords = LocationResolver.resolve("Analyze surface modifications at 12.9716, 77.5946 between 2021 and 2025")
        self.assertTrue(res_coords["resolved"])
        self.assertEqual(res_coords["source"], "coordinates")
        self.assertAlmostEqual(res_coords["coordinates"][0], 12.9716, places=4)

    def test_location_resolver_ambiguity(self):
        """Verify ambiguity detection for locations with multiple global candidates."""
        res_amb = LocationResolver.resolve("What changed in Springfield between 2020 and 2025?")
        self.assertFalse(res_amb["resolved"])
        self.assertTrue(res_amb["ambiguous"])
        self.assertGreater(len(res_amb["ambiguity_options"]), 1)
        self.assertIn("Illinois", res_amb["ambiguity_options"][0])

        res_cambridge = LocationResolver.resolve("Check vegetation growth around Cambridge")
        self.assertFalse(res_cambridge["resolved"])
        self.assertTrue(res_cambridge["ambiguous"])
        self.assertTrue(any("UK" in opt for opt in res_cambridge["ambiguity_options"]))

    # -------------------------------------------------------------------------
    # 2. Temporal Extraction & Intent
    # -------------------------------------------------------------------------
    def test_temporal_extractor_explicit_ranges(self):
        """Verify extraction of year ranges and month-year spans."""
        t1 = TemporalQueryExtractor.extract("What changed in Bengaluru between 2022 and 2026?")
        self.assertTrue(t1["has_temporal_intent"])
        self.assertEqual(t1["start_year"], 2022)
        self.assertEqual(t1["end_year"], 2026)
        self.assertEqual(t1["span_years"], 4)

        t2 = TemporalQueryExtractor.extract("What changes happened around this location between January 2022 and January 2026?")
        self.assertTrue(t2["has_temporal_intent"])
        self.assertEqual(t2["start_year"], 2022)
        self.assertEqual(t2["start_month"], "January")
        self.assertEqual(t2["end_year"], 2026)
        self.assertEqual(t2["end_month"], "January")

    def test_temporal_extractor_relative_periods(self):
        """Verify relative temporal periods (e.g. 'over the last five years', 'since 2020')."""
        t_rel = TemporalQueryExtractor.extract("How has this place changed over the last five years?")
        self.assertTrue(t_rel["has_temporal_intent"])
        self.assertEqual(t_rel["span_years"], 5)
        self.assertEqual(t_rel["temporal_mode"], "relative_past")

        t_since = TemporalQueryExtractor.extract("Has the road network changed in this location since 2020?")
        self.assertTrue(t_since["has_temporal_intent"])
        self.assertEqual(t_since["start_year"], 2020)
        self.assertIn("roads", t_since["requested_features"])

    def test_temporal_target_features(self):
        """Verify extraction of specific target physical features."""
        t_veg = TemporalQueryExtractor.extract("What happened to vegetation in this region between 2021 and 2026?")
        self.assertIn("vegetation", t_veg["requested_features"])

        t_bldg = TemporalQueryExtractor.extract("Did the number of buildings increase in Bengaluru since 2020?")
        self.assertIn("buildings", t_bldg["requested_features"])

        t_urban = TemporalQueryExtractor.extract("Compare urban development in this area between 2019 and 2025.")
        self.assertIn("urban_development", t_urban["requested_features"])

    # -------------------------------------------------------------------------
    # 3. Satellite Data Catalog Verification & Truthfulness
    # -------------------------------------------------------------------------
    def test_catalog_query_successful_benchmark(self):
        """Verify retrieval of genuine Sentinel-2 passes with acquisition dates, low cloud cover, and cached rasters."""
        cat = SatelliteDataCatalog.query(
            location="Bengaluru",
            time_range={"start_year": 2022, "end_year": 2026},
            feature="general_change",
        )
        self.assertTrue(cat["found"])
        self.assertEqual(len(cat["image_paths"]), 2)
        self.assertTrue(Path(cat["image_paths"][0]).exists())
        self.assertTrue(Path(cat["image_paths"][1]).exists())

        ctx = cat["temporal_context"]
        self.assertEqual(ctx["location_name"], "Bengaluru")
        self.assertIn("Sentinel-2", ctx["sensor"])
        self.assertLessEqual(ctx["cloud_cover_a"], 2.0)
        self.assertLessEqual(ctx["cloud_cover_b"], 2.0)
        self.assertIn("timeline", cat)
        self.assertGreaterEqual(len(cat["timeline"]), 2)

    def test_catalog_query_unsupported_location_refusal(self):
        """Verify truthful refusal when satellite data is not available (no fabricated observations)."""
        cat = SatelliteDataCatalog.query(
            location="Atlantis",
            time_range={"start_year": 2020, "end_year": 2025},
            feature="general_change",
        )
        self.assertFalse(cat["found"])
        self.assertIn("not found", cat["message"].lower())

    # -------------------------------------------------------------------------
    # 4. End-to-End SatQueryAgent Zero-Upload Analysis
    # -------------------------------------------------------------------------
    def test_agent_zero_upload_general_change(self):
        """Verify user asks 'What changed in Bengaluru between 2022 and 2026?' without uploading images."""
        resp = self.agent.analyze(
            query="What changed in Bengaluru between 2022 and 2026?",
            images=[],
        )
        self.assertIsNone(resp.error)
        self.assertFalse(resp.clarification_request)
        self.assertIn("bengaluru", resp.answer.lower())
        self.assertTrue(resp.temporal_context)
        self.assertEqual(resp.temporal_context["location_name"], "Bengaluru")
        self.assertEqual(resp.temporal_context["start_year"], 2022)
        self.assertEqual(resp.temporal_context["end_year"], 2026)
        self.assertGreater(len(resp.timeline), 0)

        # Check execution trace
        trace_text = " ".join(resp.execution_trace.steps)
        self.assertIn("Catalog Match", trace_text)
        self.assertIn("ChangeFormerV6", trace_text)

    def test_agent_zero_upload_discrete_building_increase(self):
        """Verify query 'Did the number of buildings increase in Bengaluru since 2020?' routes to YOLO11 + GIS count delta."""
        resp = self.agent.analyze(
            query="Did the number of buildings increase in Bengaluru since 2020?",
            images=[],
        )
        self.assertIsNone(resp.error)
        self.assertTrue(resp.temporal_context)
        self.assertEqual(resp.temporal_context["location_name"], "Bengaluru")

        # Answer should explain building count comparison in normal conversational English
        self.assertIn("building", resp.answer.lower())
        self.assertIn("gis", resp.evidence)
        gis_res = resp.evidence["gis"].get("results", {})
        self.assertEqual(gis_res.get("operation"), "count_delta")
        self.assertIn("delta_count", gis_res)

    def test_agent_zero_upload_vegetation_delta(self):
        """Verify query 'What happened to vegetation in Bengaluru between 2022 and 2026?' routes to SegFormer + GIS area delta."""
        resp = self.agent.analyze(
            query="What happened to vegetation in Bengaluru between 2022 and 2026?",
            images=[],
        )
        self.assertIsNone(resp.error)
        self.assertIn("gis", resp.evidence)
        gis_res = resp.evidence["gis"].get("results", {})
        self.assertEqual(gis_res.get("operation"), "area_delta")
        self.assertEqual(gis_res.get("feature"), "vegetation")
        self.assertIn("percentage_change", gis_res)

    def test_agent_zero_upload_ambiguity_clarification(self):
        """Verify ambiguous location prompts clarification without guessing."""
        resp = self.agent.analyze(
            query="What changed in Springfield between 2020 and 2025?",
            images=[],
        )
        self.assertTrue(resp.clarification_request)
        self.assertGreater(len(resp.ambiguity_options), 0)
        self.assertIn("Which specific region would you like to analyze?", resp.answer)

    def test_agent_zero_upload_unsupported_truthful_refusal(self):
        """Verify unsupported location returns truthful message rather than fabricating imagery."""
        resp = self.agent.analyze(
            query="What changed in Atlantis between 2020 and 2025?",
            images=[],
        )
        self.assertIn("not found", resp.answer.lower())
        self.assertTrue(resp.unsupported_analysis)

    def test_agent_conversation_followup_context_preservation(self):
        """Verify follow-up query inherits active temporal and geographic context."""
        # Turn 1: Primary temporal query
        resp1 = self.agent.analyze(
            query="What changed in Bengaluru between 2022 and 2026?",
            images=[],
        )
        self.assertEqual(resp1.temporal_context["location_name"], "Bengaluru")

        # Turn 2: Follow-up query asking about specific feature without retyping location
        resp2 = self.agent.analyze(
            query="Did the number of buildings increase in this area?",
            images=[],
            metadata={"active_temporal_context": resp1.temporal_context},
        )
        self.assertIsNone(resp2.error)
        self.assertEqual(resp2.temporal_context["location_name"], "Bengaluru")
        self.assertIn("building", resp2.answer.lower())
        self.assertIn("gis", resp2.evidence)

    def test_agent_zero_upload_bengaluru_2024_2025(self):
        """Verify 2024-2025 Bengaluru temporal query returns exact dates, base64 data URLs, and grounded change explanation."""
        resp = self.agent.analyze(
            query="What changes happened between 2024 and 2025 in Bengaluru?",
            images=[],
        )
        self.assertIsNone(resp.error)
        self.assertEqual(resp.task, "bi_temporal_change")
        self.assertEqual(resp.temporal_context["start_year"], 2024)
        self.assertEqual(resp.temporal_context["end_year"], 2025)
        self.assertEqual(len(resp.observations), 2)
        self.assertIn("2024", resp.observations[0]["formatted_date"])
        self.assertIn("2025", resp.observations[1]["formatted_date"])
        self.assertTrue(resp.observations[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(resp.observations[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertGreater(len(resp.visualizations), 0)
        self.assertTrue(resp.visualizations[0].startswith("data:image/png;base64,"))
        self.assertNotIn("0.0% surface change", resp.answer)
        self.assertNotIn("essentially identical", resp.answer)
        self.assertIn("Bengaluru", resp.answer)

        # Verify structured change regions and map grounding
        self.assertGreater(len(resp.change_regions), 0)
        first_region = resp.change_regions[0]
        self.assertIn("direction", first_region)
        self.assertIn("direction_label", first_region)
        self.assertIn("label", first_region)
        self.assertGreater(first_region["pixel_count"], 0)
        self.assertGreater(first_region["area_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
