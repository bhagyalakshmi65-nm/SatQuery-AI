"""Dynamic Analysis Planner for SatQuery AI.

Connects Image Understanding (scene composition) and Semantic Query Understanding
(intents, entities, spatial relations) to an extensible capability registry to
dynamically select appropriate specialist model(s) and tool(s).

Enforces:
- Zero specialist models (VLM only)
- One specialist model
- Multiple specialist models
- Specialist models + GIS
- Dynamic capability matching without hardcoded queries
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.agent.capability_registry import CAPABILITY_REGISTRY, CapabilityRegistry
from backend.agent.schemas import (
    AnalysisPlan,
    ImageSceneInspection,
    QueryUnderstanding,
)

logger = logging.getLogger("agent.planner")


class AnalysisPlanner:
    """Dynamically plans model and tool execution based on query requirements and scene structure."""

    def __init__(self, registry: Optional[CapabilityRegistry] = None):
        self.registry = registry or CAPABILITY_REGISTRY

    def plan(
        self,
        query_understanding: QueryUnderstanding,
        scene_inspection: Optional[ImageSceneInspection] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalysisPlan:
        """Construct a structured analysis plan selecting models and tools based on capabilities."""
        metadata = metadata or {}
        scene_count = scene_inspection.scene_count if scene_inspection else 1
        file_count = scene_inspection.file_count if scene_inspection else 1

        selected_models: List[str] = []
        selected_tools: List[str] = []
        operational_reasons: List[str] = []

        q_lower = query_understanding.raw_query.lower()
        entities = set(query_understanding.entities)
        spatial_rels = query_understanding.spatial_relationships

        # Discrete object entities vs environmental/landcover entities
        discrete_set = {"building", "vehicle", "aircraft", "ship", "storage_tank", "crane", "bridge", "solar_panel", "road", "runway", "pier"}
        landcover_set = {"water", "river", "lake", "ocean", "vegetation", "forest", "agriculture", "urban", "rural", "wetland", "bare_soil"}

        has_discrete = bool(entities & discrete_set)
        has_landcover = bool(entities & landcover_set)

        # ---------------------------------------------------------------------
        # 1. Comparison & Temporal Determination
        # ---------------------------------------------------------------------
        comparison_requested = (
            query_understanding.requires_temporal_reasoning or
            query_understanding.requires_comparison or
            (scene_inspection is not None and scene_inspection.has_temporal_cues and scene_count >= 2 and "change" in q_lower)
        )

        # Check for specific feature increase/decrease comparison (e.g. "Did vegetation increase?")
        is_feature_comparison = comparison_requested and (has_landcover or has_discrete) and any(w in q_lower for w in ["increase", "decrease", "more", "less", "expand", "shrink"])

        # ---------------------------------------------------------------------
        # 2. Dynamic Capability Matching & Model Selection
        # ---------------------------------------------------------------------

        # A. Cross-Modal Optical + SAR
        if query_understanding.primary_intent == "cross_modal_analysis":
            selected_models.append("CROMA-Base")
            operational_reasons.append("CROMA-Base selected for multi-sensor Optical + SAR joint representation.")

            selected_models.append("SAR-Backscatter-Analyzer")
            operational_reasons.append("SAR-Backscatter-Analyzer selected for radar surface roughness and double-bounce structural analysis.")

            if has_discrete or any(w in q_lower for w in ["building", "buildings", "structure", "structures", "objects", "cars", "vehicles"]):
                selected_models.append("YOLO11")
                operational_reasons.append("YOLO11 selected for discrete building and structural detection.")

            if has_landcover or any(w in q_lower for w in ["vegetation", "water", "road", "roads", "land", "forest", "urban"]):
                selected_models.append("SegFormer")
                operational_reasons.append("SegFormer selected for semantic segmentation of vegetation, water, and road corridors.")

        # B. Specific Feature Change Comparison (e.g. "Did vegetation increase?", "Did buildings increase?")
        elif is_feature_comparison or (comparison_requested and bool(getattr(query_understanding, "requested_features", [])) and "general_change" not in getattr(query_understanding, "requested_features", [])):
            req_feats = getattr(query_understanding, "requested_features", [])
            is_discrete = has_discrete or any(f in ["buildings", "structures", "vehicles"] for f in req_feats)
            is_lc = has_landcover or any(f in ["vegetation", "water", "urban_development", "land_cover"] for f in req_feats)

            if is_discrete and not is_lc:
                selected_models.append("YOLO11")
                operational_reasons.append("YOLO11 selected for bi-temporal discrete object detection and count delta comparison.")
            else:
                selected_models.append("SegFormer")
                operational_reasons.append("SegFormer selected for multi-scene region segmentation and area delta estimation.")

            if "GIS" not in selected_tools:
                selected_tools.append("GIS")
                operational_reasons.append("GIS selected for bi-temporal metric delta calculation.")

        # C. Generic Change Detection (e.g. "What changed?")
        elif comparison_requested and query_understanding.primary_intent in ["temporal_change_analysis", "temporal_geospatial_analysis"]:
            selected_models.append("ChangeFormerV6")
            operational_reasons.append("ChangeFormerV6 selected for pixel-level bi-temporal change comparison.")

        # D. Discrete Objects (Detection / Counting / Localization)
        # E.g. "How many buildings are visible?", "Where are the buildings?"
        elif query_understanding.primary_intent in ["object_counting", "object_localization"] or (has_discrete and not has_landcover and not spatial_rels):
            selected_models.append("YOLO11")
            operational_reasons.append("YOLO11 selected for discrete object detection, localization, and counting.")

        # E. Multi-Model Queries & Spatial Constraints (Compound)
        # E.g. "How many buildings are within 100m of the river?" or "Find agricultural areas and count buildings inside them"
        elif query_understanding.is_compound_query or query_understanding.requires_gis or spatial_rels:
            # Need YOLO11 if discrete objects exist
            if has_discrete or "building" in q_lower or "vehicle" in q_lower or query_understanding.requires_counting:
                selected_models.append("YOLO11")
                operational_reasons.append("YOLO11 selected for discrete object localization.")

            # Need SegFormer if land-cover or natural feature needs segmentation/boundary
            if has_landcover or any(r.get("reference") in landcover_set for r in spatial_rels) or "river" in q_lower or "area" in q_lower:
                selected_models.append("SegFormer")
                operational_reasons.append("SegFormer selected for reference region extraction and semantic segmentation.")

            # Need GIS for spatial distance, containment, intersection, buffers
            if query_understanding.requires_gis or spatial_rels or "within" in q_lower or "inside" in q_lower:
                selected_tools.append("GIS")
                operational_reasons.append("GIS selected for geometric distance calculation and spatial containment filtering.")

        # F. Coverage & Extent Estimation
        # E.g. "How much vegetation is present?", "What percentage of this area is water?"
        elif query_understanding.primary_intent == "coverage_measurement" or "percentage" in query_understanding.requested_measurements:
            selected_models.append("SegFormer")
            operational_reasons.append("SegFormer selected for semantic segmentation and pixel area extraction.")
            selected_tools.append("GIS")
            operational_reasons.append("GIS selected for coverage percentage calculation.")

        # G. Land-Cover Scene Classification
        # E.g. "What type of land cover is shown?", "Is this area urban or rural?"
        elif query_understanding.requires_classification or query_understanding.primary_intent == "landcover_classification":
            is_explicit_rgb = False
            if scene_inspection and scene_inspection.dimensions:
                is_explicit_rgb = all(d.get("channels") in [1, 3, 4] for d in scene_inspection.dimensions)
            elif metadata and metadata.get("channels") in [1, 3, 4]:
                is_explicit_rgb = True

            if is_explicit_rgb:
                selected_models.append("SegFormer")
                operational_reasons.append("SegFormer selected for RGB optical surface segmentation and land-cover class area measurement.")
                if "GIS" not in selected_tools:
                    selected_tools.append("GIS")
                    operational_reasons.append("GIS selected for coverage percentage calculation.")
            else:
                selected_models.append("BigEarthNet-ResNet50-SeCo-12B")
                operational_reasons.append("BigEarthNet-ResNet50-SeCo-12B selected for 12-band Sentinel-2 multispectral land-cover classification.")

        # H. Dynamic Registry Lookup / Custom Capability Matching
        else:
            # Check if any registered specialist model or tool matches intent or subtask capabilities
            candidate_caps = [query_understanding.primary_intent]
            for st in query_understanding.subtasks:
                if st.operation:
                    candidate_caps.append(st.operation)
                if st.target_entity:
                    candidate_caps.append(st.target_entity)
            for ent in query_understanding.entities:
                candidate_caps.append(ent)
            for cls in query_understanding.requested_classes:
                candidate_caps.append(cls)

            # Match capabilities or tags of registered tools occurring in query
            for tool_name, entry in self.registry.list_entries().items():
                for cap in entry.capabilities:
                    cap_clean = cap.replace("_", " ").lower()
                    if cap_clean in q_lower or cap.lower() in q_lower:
                        candidate_caps.append(cap)
                for tag in entry.tags:
                    tag_clean = tag.replace("_", " ").lower()
                    if tag_clean in q_lower or tag.lower() in q_lower:
                        candidate_caps.append(entry.capabilities[0] if entry.capabilities else tag)
            
            dyn_models, dyn_tools = self.registry.find_tools_for_capabilities(candidate_caps)
            for m in dyn_models:
                if m not in selected_models:
                    selected_models.append(m)
                    operational_reasons.append(f"{m} dynamically selected via declared capability matching.")
            for t in dyn_tools:
                if t not in selected_tools:
                    selected_tools.append(t)
                    operational_reasons.append(f"{t} dynamically selected via declared capability matching.")

            if not selected_models:
                # Zero specialist models required
                operational_reasons.append("No specialist models required; query can be answered via foundation vision-language reasoning.")

        # Always include VLM tool for natural-language interpretation & explanation
        if "RS-Qwen" not in selected_tools:
            selected_tools.append("RS-Qwen")

        requires_gis = "GIS" in selected_tools

        # Build concise planner summary
        models_str = f"[{', '.join(selected_models)}]" if selected_models else "[None (VLM only)]"
        tools_str = f"[{', '.join(selected_tools)}]"
        summary = (
            f"Analysis Plan: Selected models: {models_str}, tools: {tools_str} "
            f"for {scene_count} scene(s). Comparison requested: {comparison_requested}."
        )

        return AnalysisPlan(
            intent=query_understanding.primary_intent,
            scene_count=scene_count,
            file_count=file_count,
            subtasks=query_understanding.subtasks,
            selected_models=selected_models,
            selected_tools=selected_tools,
            comparison_requested=comparison_requested,
            requires_gis=requires_gis,
            operational_reasons=operational_reasons,
            planner_summary=summary,
        )
