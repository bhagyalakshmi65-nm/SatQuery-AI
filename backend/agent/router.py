"""Deterministic and explainable router for SatQuery AI Agent."""

import re
from typing import Any, Dict, List, Optional
from backend.agent.schemas import AnalysisPlan, ImageSceneInspection, QueryUnderstanding, RoutingDecision, TaskType


# Semantic keywords
CHANGE_KEYWORDS = [
    r"\bchange\b", r"\bchanges\b", r"\bchanged\b", r"\bchanging\b",
    r"\bdifference\b", r"\bdifferences\b", r"\bdiffer\b",
    r"\bincrease\b", r"\bincreased\b", r"\bdecrease\b", r"\bdecreased\b",
    r"\bbefore\s+and\s+after\b", r"\bbetween\s+(the\s+)?two\b", r"\btwo\s+dates\b",
    r"\bover\s+time\b", r"\bexpanded\b", r"\bexpansion\b",
    r"\bdemolished\b", r"\bconstruction\b", r"\bnew\s+buildings?\b",
]

SAR_KEYWORDS = [
    r"\bsar\b", r"\bradar\b", r"\bsentinel[- ]?1\b", r"\bdual[- ]?pol\b",
    r"\bcroma\b", r"\bmicrowaves?\b", r"\bbackscatter\b"
]

OPTICAL_SAR_KEYWORDS = [
    r"\boptical\s+(and|\+)\s+sar\b", r"\bsar\s+(and|\+)\s+optical\b",
    r"\bboth\s+sensors\b", r"\bcross[- ]modal\b", r"\btogether\b"
]

OBJECT_DETECTION_KEYWORDS = [
    r"\bdetect\b", r"\bfind\b", r"\bcount\b", r"\blocate\b",
    r"\bbounding\s*box(es)?\b", r"\bobjects?\b",
    r"\bbuildings?\b", r"\baircraft\b", r"\bairplanes?\b", r"\bplanes?\b",
    r"\bvehicles?\b", r"\bcars?\b", r"\btrucks?\b", r"\bbus(es)?\b",
    r"\bships?\b", r"\bvessels?\b", r"\bboats?\b", r"\bhelicopters?\b",
    r"\bstorage\s*tanks?\b", r"\bcranes?\b"
]

LANDCOVER_KEYWORDS = [
    r"\bland\s*cover\b", r"\blandcover\b", r"\bvegetation\b",
    r"\barable\b", r"\bcrops?\b", r"\bpastures?\b", r"\bforest\b",
    r"\bwetlands?\b", r"\burban\s*fabric\b", r"\bwater\s*bodies\b",
    r"\bbigearthnet\b", r"\b12[- ]band\b", r"\bmultispectral\b"
]


class TaskRouter:
    """Determines the appropriate specialist tools and task type based on query, capabilities, and inputs."""

    @staticmethod
    def route(
        query: str,
        image_count: int,
        sensor_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        query_understanding: Optional[QueryUnderstanding] = None,
        scene_inspection: Optional[ImageSceneInspection] = None,
        analysis_plan: Optional[AnalysisPlan] = None,
    ) -> RoutingDecision:
        """Route user query and inputs to appropriate task and tools using semantic understanding."""
        q = query.lower().strip()
        metadata = metadata or {}
        explicit_sensor = (sensor_type or metadata.get("sensor_type", "")).lower()

        # 1. Optical + SAR Cross-Modal Routing
        has_sar_kw = any(re.search(p, q) for p in SAR_KEYWORDS)
        has_opt_sar_combo = any(re.search(p, q) for p in OPTICAL_SAR_KEYWORDS)
        is_opt_sar_sensor = explicit_sensor in ["optical+sar", "optical_sar", "sentinel1+sentinel2", "s1+s2", "sar"]
        is_opt_sar_semantic = query_understanding and query_understanding.primary_intent == "cross_modal_analysis"

        if is_opt_sar_sensor or is_opt_sar_semantic or (has_sar_kw and (has_opt_sar_combo or "optical" in q or "both" in q)):
            return RoutingDecision(
                task=TaskType.CROSS_MODAL_OPTICAL_SAR,
                tools_used=["CROMA-Base", "RS-Qwen"],
                rationale="Query or sensor metadata specifies joint Optical and SAR cross-modal analysis.",
                extracted_intent="cross_modal_analysis"
            )

        # 2. Bi-Temporal Change Detection
        # CRITICAL RULE: Change detection must NOT be selected merely because 2 files or 2 panels exist.
        # The USER'S QUERY must semantically require comparison or change analysis.
        has_change_kw = any(re.search(p, q) for p in CHANGE_KEYWORDS)
        query_requests_change = (
            (query_understanding is not None and (
                query_understanding.requires_temporal_reasoning or
                query_understanding.requires_comparison or
                query_understanding.primary_intent in ["temporal_change_analysis", "temporal_geospatial_analysis"] or
                query_understanding.is_temporal_geospatial
            )) or
            has_change_kw or
            any(w in q for w in ["compare", "comparison", "differ", "change", "increased", "decreased"])
        )

        if query_requests_change:
            tools_used = ["ChangeFormerV6", "RS-Qwen"]
            if analysis_plan and (analysis_plan.selected_models or analysis_plan.selected_tools):
                tools_used = list(dict.fromkeys(analysis_plan.selected_models + analysis_plan.selected_tools))
            return RoutingDecision(
                task=TaskType.BI_TEMPORAL_CHANGE,
                tools_used=tools_used,
                rationale="User query semantically requires change detection or temporal/cross-scene comparison.",
                extracted_intent="change_detection"
            )

        # 3. Object Detection & Spatial Object Queries
        has_obj_kw = any(re.search(p, q) for p in OBJECT_DETECTION_KEYWORDS)
        has_semantic_obj = query_understanding and (
            query_understanding.requires_localization or
            query_understanding.requires_counting or
            query_understanding.primary_intent in ["object_detection", "object_counting", "spatial_object_query"]
        ) and any(e in ["building", "vehicle", "aircraft", "ship", "storage_tank", "bridge", "crane", "solar_panel", "road", "runway", "pier"] for e in query_understanding.entities)

        if has_semantic_obj or (has_obj_kw and not any(re.search(p, q) for p in LANDCOVER_KEYWORDS)):
            return RoutingDecision(
                task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
                tools_used=["YOLO11", "RS-Qwen"],
                rationale="Query requests detection, location, or counting of discrete objects (e.g. buildings, vehicles, aircraft, ships).",
                extracted_intent="object_detection"
            )

        # 4. Land Cover Classification & Coverage Measurement
        has_lc_kw = any(re.search(p, q) for p in LANDCOVER_KEYWORDS)
        has_semantic_lc = query_understanding and (
            query_understanding.requires_classification or
            query_understanding.primary_intent in ["landcover_classification", "coverage_measurement"]
        )
        if has_semantic_lc or has_lc_kw:
            is_12band = False
            if scene_inspection and scene_inspection.dimensions:
                is_12band = any(d.get("channels") == 12 for d in scene_inspection.dimensions)
            elif metadata:
                is_12band = metadata.get("channels") == 12 or metadata.get("is_multispectral_native", False)

            planned_models = analysis_plan.selected_models if analysis_plan else []
            if "BigEarthNet-ResNet50-SeCo-12B" in planned_models or is_12band:
                tools = ["BigEarthNet-ResNet50-SeCo-12B", "RS-Qwen"]
                rationale = "Query requests land-cover classification across 12-band multispectral Sentinel-2 surface reflectance."
            else:
                tools = ["SegFormer", "RS-Qwen"]
                rationale = "Input is 3-channel optical RGB. Routed to SegFormer for empirical surface segmentation and land-cover class area measurement."

            return RoutingDecision(
                task=TaskType.SINGLE_IMAGE_LANDCOVER,
                tools_used=tools,
                rationale=rationale,
                extracted_intent="land_cover_classification"
            )

        # 5. Default: Single-Image Description / Visual Understanding
        return RoutingDecision(
            task=TaskType.SINGLE_IMAGE_DESCRIPTION,
            tools_used=["RS-Qwen"],
            rationale="Query asks for general image description, scene overview, or visual reasoning.",
            extracted_intent="visual_description"
        )

