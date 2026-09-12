"""Deterministic and explainable router for SatQuery AI Agent."""

import re
from typing import Any, Dict, List, Optional
from backend.agent.schemas import RoutingDecision, TaskType


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
    """Determines the appropriate specialist tools and task type based on query and inputs."""

    @staticmethod
    def route(
        query: str,
        image_count: int,
        sensor_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route user query and inputs to appropriate task and tools."""
        q = query.lower().strip()
        metadata = metadata or {}
        explicit_sensor = (sensor_type or metadata.get("sensor_type", "")).lower()

        # 1. Optical + SAR Cross-Modal Routing
        has_sar_kw = any(re.search(p, q) for p in SAR_KEYWORDS)
        has_opt_sar_combo = any(re.search(p, q) for p in OPTICAL_SAR_KEYWORDS)
        is_opt_sar_sensor = explicit_sensor in ["optical+sar", "optical_sar", "sentinel1+sentinel2", "s1+s2", "sar"]

        if is_opt_sar_sensor or (has_sar_kw and (has_opt_sar_combo or "optical" in q or "both" in q)):
            return RoutingDecision(
                task=TaskType.CROSS_MODAL_OPTICAL_SAR,
                tools_used=["CROMA-Base", "RS-Qwen"],
                rationale="Query or sensor metadata specifies joint Optical and SAR cross-modal analysis.",
                extracted_intent="cross_modal_analysis"
            )

        # 2. Bi-Temporal Change Detection
        has_change_kw = any(re.search(p, q) for p in CHANGE_KEYWORDS)
        # If 2 images provided or change keywords present with 2+ images, or user explicitly asks about change
        if has_change_kw or image_count == 2:
            return RoutingDecision(
                task=TaskType.BI_TEMPORAL_CHANGE,
                tools_used=["ChangeFormerV6", "RS-Qwen"],
                rationale="Query or image count indicates bi-temporal change detection across two dates/images.",
                extracted_intent="change_detection"
            )

        # 3. Object Detection
        has_obj_kw = any(re.search(p, q) for p in OBJECT_DETECTION_KEYWORDS)
        if has_obj_kw and not any(re.search(p, q) for p in LANDCOVER_KEYWORDS):
            return RoutingDecision(
                task=TaskType.SINGLE_IMAGE_OBJECT_DETECTION,
                tools_used=["YOLO11", "RS-Qwen"],
                rationale="Query requests detection, location, or counting of discrete objects (e.g. buildings, vehicles, aircraft, ships).",
                extracted_intent="object_detection"
            )

        # 4. Land Cover Classification
        has_lc_kw = any(re.search(p, q) for p in LANDCOVER_KEYWORDS)
        if has_lc_kw:
            return RoutingDecision(
                task=TaskType.SINGLE_IMAGE_LANDCOVER,
                tools_used=["BigEarthNet-ResNet50-SeCo-12B", "RS-Qwen"],
                rationale="Query requests land-cover/vegetation classification across multispectral Sentinel-2 classes.",
                extracted_intent="land_cover_classification"
            )

        # 5. Default: Single-Image Description / Visual Understanding
        return RoutingDecision(
            task=TaskType.SINGLE_IMAGE_DESCRIPTION,
            tools_used=["RS-Qwen"],
            rationale="Query asks for general image description, scene overview, or visual reasoning.",
            extracted_intent="visual_description"
        )

