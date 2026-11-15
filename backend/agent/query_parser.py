"""Semantic Query Understanding and Query Decomposition for SatQuery AI.

Analyzes natural-language remote-sensing questions into a structured semantic
representation without relying on fixed question lists or fragile keyword matching.
Decomposes complex compound questions into ordered execution subtasks with concise
operational reasoning.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.agent.schemas import (
    ImageSceneInspection,
    QuerySubtask,
    QueryUnderstanding,
)
from backend.agent.temporal_geospatial import LocationResolver, TemporalQueryExtractor

logger = logging.getLogger("agent.query_parser")

# Domain Entity Lexicons (normalized to canonical categories)
DISCRETE_OBJECT_MAP: Dict[str, str] = {
    "building": "building",
    "buildings": "building",
    "house": "building",
    "houses": "building",
    "structure": "building",
    "structures": "building",
    "facility": "building",
    "facilities": "building",
    "dwelling": "building",
    "dwellings": "building",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "car": "vehicle",
    "cars": "vehicle",
    "truck": "vehicle",
    "trucks": "vehicle",
    "bus": "vehicle",
    "buses": "vehicle",
    "aircraft": "aircraft",
    "airplane": "aircraft",
    "airplanes": "aircraft",
    "plane": "aircraft",
    "planes": "aircraft",
    "helicopter": "aircraft",
    "helicopters": "aircraft",
    "ship": "ship",
    "ships": "ship",
    "vessel": "ship",
    "vessels": "ship",
    "boat": "ship",
    "boats": "ship",
    "tanker": "ship",
    "tankers": "ship",
    "storage tank": "storage_tank",
    "storage tanks": "storage_tank",
    "oil tank": "storage_tank",
    "fuel tank": "storage_tank",
    "silo": "storage_tank",
    "silos": "storage_tank",
    "crane": "crane",
    "cranes": "crane",
    "bridge": "bridge",
    "bridges": "bridge",
    "solar panel": "solar_panel",
    "solar panels": "solar_panel",
    "solar array": "solar_panel",
    "solar farm": "solar_panel",
    "runway": "runway",
    "runways": "runway",
    "road": "road",
    "roads": "road",
    "pier": "pier",
    "piers": "pier",
    "dock": "pier",
    "docks": "pier",
}

LANDCOVER_FEATURE_MAP: Dict[str, str] = {
    "water": "water",
    "river": "river",
    "rivers": "river",
    "lake": "lake",
    "lakes": "lake",
    "ocean": "ocean",
    "sea": "sea",
    "stream": "river",
    "streams": "river",
    "canal": "canal",
    "canals": "canal",
    "reservoir": "water",
    "water body": "water",
    "water bodies": "water",
    "waterway": "river",
    "waterways": "river",
    "vegetation": "vegetation",
    "vegetated area": "vegetation",
    "vegetated areas": "vegetation",
    "forest": "forest",
    "forests": "forest",
    "forested": "forest",
    "tree": "forest",
    "trees": "forest",
    "woodland": "forest",
    "greenery": "vegetation",
    "green area": "vegetation",
    "agriculture": "agriculture",
    "agricultural area": "agriculture",
    "agricultural areas": "agriculture",
    "farmland": "agriculture",
    "crops": "agriculture",
    "cropland": "agriculture",
    "arable land": "agriculture",
    "pasture": "pasture",
    "pastures": "pasture",
    "urban": "urban",
    "urban area": "urban",
    "urban fabric": "urban",
    "built-up": "urban",
    "city": "urban",
    "rural": "rural",
    "rural area": "rural",
    "countryside": "rural",
    "wetland": "wetland",
    "wetlands": "wetland",
    "bare soil": "bare_soil",
    "soil": "bare_soil",
    "sand": "bare_soil",
}

# Syntactic measurement indicators
MEASUREMENT_PATTERNS: Dict[str, List[str]] = {
    "count": [r"\bhow\s+many\b", r"\bcount\b", r"\bnumber\s+of\b", r"\bquantity\b"],
    "percentage": [r"\bpercentage\b", r"\bpercent\b", r"%", r"\bproportion\b", r"\bfraction\b", r"\bratio\b"],
    "area": [r"\bhow\s+much\b", r"\bextent\b", r"\barea\b", r"\bhectares\b", r"\bsq\s*km\b", r"\bsquare\s+meters\b", r"\bcoverage\b"],
    "distance": [r"\bdistance\b", r"\bhow\s+far\b", r"\bmeters?\b", r"\bkm\b", r"\bkilometers?\b", r"\bradius\b"],
    "localization": [r"\bwhere\b", r"\blocate\b", r"\bfind\b", r"\bbounding\s*box(es)?\b", r"\bcoordinates\b", r"\bpositions?\b"],
}

# Change & Temporal indicators
TEMPORAL_PATTERNS = [
    r"\bchange[sd]?\b", r"\bchanging\b", r"\bdifference[s]?\b", r"\bdiffer\b",
    r"\bbefore\s+and\s+after\b", r"\bbetween\s+(the\s+)?(two|images|dates)\b",
    r"\bover\s+time\b", r"\btemporal\b", r"\btwo\s+dates\b", r"\bnewly\s+built\b",
    r"\bdemolished\b", r"\bexpansion\b", r"\bevolution\b",
    r"\bincrease[sd]?\b", r"\bdecreas[ed]?\b", r"\bgrowth\b", r"\bshrink\b"
]

# Spatial Relation regex
DISTANCE_REL_REGEX = re.compile(
    r"\b(?:within|closer\s+than|less\s+than)\s+(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)?\s+of\s+(?:the\s+)?([a-zA-Z_\s]+)",
    re.IGNORECASE,
)
CONTAINMENT_REL_REGEX = re.compile(
    r"\b(?:inside|within|contained\s+in|in)\s+(?:the\s+)?([a-zA-Z_\s]+?(?:area|areas|zone|zones|region|regions|fabric)?)",
    re.IGNORECASE,
)
DIRECTIONAL_REL_REGEX = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\s+of\s+(?:the\s+)?([a-zA-Z_\s]+)",
    re.IGNORECASE,
)
PROXIMITY_REL_REGEX = re.compile(
    r"\b(?:adjacent\s+to|next\s+to|along|near|beside|close\s+to)\s+(?:the\s+)?([a-zA-Z_\s]+)",
    re.IGNORECASE,
)


class SemanticQueryParser:
    """Natural-language semantic parser for remote sensing and satellite analytics queries."""

    def __init__(self):
        pass

    def understand(
        self,
        query: str,
        scene_inspection: Optional[ImageSceneInspection] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryUnderstanding:
        """Parse natural-language query into structured semantic intent, entities, and decomposed subtasks."""
        q = query.strip()
        q_lower = q.lower()
        metadata = metadata or {}

        # 1. Extract Entities & Classes
        discrete_objs, landcover_feats = self._extract_entities(q_lower)
        all_entities = sorted(list(set(discrete_objs + landcover_feats)))
        requested_classes = all_entities.copy()

        # 2. Extract Requested Measurements
        measurements = self._extract_measurements(q_lower)

        # 3. Extract Spatial Relationships & GIS Constraints
        spatial_rels = self._extract_spatial_relations(q_lower, all_entities)

        # 4. Infer Requirement Flags
        req_counting = "count" in measurements or any(re.search(p, q_lower) for p in MEASUREMENT_PATTERNS["count"])
        req_localization = "localization" in measurements or any(re.search(p, q_lower) for p in MEASUREMENT_PATTERNS["localization"])
        req_segmentation = (
            bool(landcover_feats) or "water" in all_entities or "river" in all_entities or
            bool(spatial_rels) or "percentage" in measurements or "area" in measurements
        )
        req_classification = (
            bool(landcover_feats) or "type of" in q_lower or "urban or rural" in q_lower or
            "land cover" in q_lower or "landcover" in q_lower
        )
        # 4. Location & Temporal Understanding
        loc_res = LocationResolver.resolve(q)
        temporal_res = TemporalQueryExtractor.extract(q)
        
        has_temporal = temporal_res.get("has_temporal_intent", False) or any(re.search(p, q_lower) for p in TEMPORAL_PATTERNS)
        has_named_location = (loc_res.get("resolved", False) and loc_res.get("source") != "relative_reference") or loc_res.get("ambiguous", False)
        has_relative_ref = loc_res.get("source") == "relative_reference"

        is_temp_geo = has_temporal or has_named_location or (has_relative_ref and (has_temporal or any(w in q_lower for w in ["change", "changed", "grow", "growth", "increase", "decrease", "differ"])))
        req_temporal = has_temporal
        req_gis = len(spatial_rels) > 0 or "distance" in measurements

        # Check cross-modal requirements
        cross_modal_patterns = [
            r"\boptical\s*(?:and|\+)\s*sar\b",
            r"\bsar\s*(?:and|\+)\s*optical\b",
            r"\boptical\s*(?:and|\+)\s*radar\b",
            r"\bradar\s*(?:and|\+)\s*optical\b",
            r"\bsentinel[- ]?1\s*(?:and|\+)\s*sentinel[- ]?2\b",
            r"\bs1\s*(?:and|\+)\s*s2\b",
            r"\bmultispectral\s*(?:and|\+)\s*radar\b",
            r"\bcross[- ]modal\b",
            r"\bmultimodal\s+satellite\b",
            r"\bboth\s+sensors\b",
        ]
        has_cross_pattern = any(re.search(p, q_lower) for p in cross_modal_patterns)
        req_cross_modal = (
            has_cross_pattern or
            (
                ("sar" in q_lower or "radar" in q_lower) and
                ("optical" in q_lower or "both" in q_lower or "croma" in q_lower or metadata.get("sensor_type") in ["optical+sar", "optical_sar"])
            )
        )

        # Comparison requirement
        req_comparison = (
            req_temporal
            or temporal_res.get("span_years", 0) > 0
            or "compare" in q_lower
            or "difference" in q_lower
            or (bool(re.search(r"\bbetween\s+\d{4}\b", q_lower)))
            or "growth" in q_lower
            or "expansion" in q_lower
            or "changed" in q_lower
            or "increase" in q_lower
        )

        # Compound Query Flag
        is_compound = (
            req_gis or
            (bool(discrete_objs) and bool(landcover_feats)) or
            (req_counting and bool(spatial_rels)) or
            (req_localization and bool(spatial_rels)) or
            (" and " in q_lower and ("count" in q_lower or "find" in q_lower or "detect" in q_lower))
        )

        # 5. Classify Primary Intent
        primary_intent = self._classify_primary_intent(
            q_lower=q_lower,
            discrete_objs=discrete_objs,
            landcover_feats=landcover_feats,
            measurements=measurements,
            spatial_rels=spatial_rels,
            req_temporal=req_temporal,
            req_cross_modal=req_cross_modal,
            req_gis=req_gis,
            is_compound=is_compound,
        )

        # 4b. Continuous Monitoring Intent Check
        monitoring_patterns = [
            r"\balert\s+me\b",
            r"\blet\s+me\s+know\s+when\b",
            r"\bmonitor\s+(?:this|the|an?)\b",
            r"\bwatch\s+(?:this|the|an?)\b",
            r"\bnotify\s+me\b",
            r"\btell\s+me\s+if\b",
            r"\bkeep\s+watching\b",
            r"\bcontinuously\s+monitor\b",
        ]
        is_monitoring = any(re.search(p, q_lower) for p in monitoring_patterns)
        monitoring_cond = None
        monitoring_thresh = None
        monitoring_thresh_unit = "percentage_points"

        if is_monitoring:
            if any(w in q_lower for w in ["building", "buildings", "structure", "structures", "house", "construction"]):
                monitoring_cond = "new_buildings"
                monitoring_thresh = 1.0
                monitoring_thresh_unit = "count"
            elif any(w in q_lower for w in ["vegetation", "forest", "tree", "greenery"]) and any(w in q_lower for w in ["decrease", "decline", "drop", "loss", "reduced", "disappear"]):
                monitoring_cond = "vegetation_decrease"
                monitoring_thresh = 5.0
                monitoring_thresh_unit = "percentage_points"
            elif any(w in q_lower for w in ["flood", "flooding", "inundat", "water surge", "water rise"]):
                monitoring_cond = "flooding"
                monitoring_thresh = 3.0
                monitoring_thresh_unit = "percentage_points"
            elif any(w in q_lower for w in ["road", "highway", "expressway", "street"]) and any(w in q_lower for w in ["construct", "new", "built", "appear"]):
                monitoring_cond = "new_road"
                monitoring_thresh = 1.0
                monitoring_thresh_unit = "count"
            elif req_cross_modal:
                monitoring_cond = "optical_sar_change"
                monitoring_thresh = 5.0
                monitoring_thresh_unit = "percentage_points"
            elif any(w in q_lower for w in ["land cover", "landcover"]):
                monitoring_cond = "landcover_change"
                monitoring_thresh = 5.0
                monitoring_thresh_unit = "percentage_points"
            else:
                monitoring_cond = "significant_general_change"
                monitoring_thresh = 5.0
                monitoring_thresh_unit = "percentage_points"

            pct_match = re.search(r"(?:by\s+)?(?:more\s+than\s+|at\s+least\s+)?(\d+(?:\.\d+)?)\s*%", q_lower)
            if pct_match:
                monitoring_thresh = float(pct_match.group(1))
                monitoring_thresh_unit = "percentage_points"
            elif monitoring_thresh_unit == "count":
                count_match = re.search(r"(?:more\s+than\s+|at\s+least\s+)(\d+)\b", q_lower)
                if count_match:
                    monitoring_thresh = float(count_match.group(1))

            primary_intent = "satellite_monitoring"

        if is_temp_geo and req_comparison and primary_intent in ["scene_description", "unknown"]:
            primary_intent = "temporal_change_analysis"

        # 6. Decompose Query into Structured Subtasks with Concise Operational Reasoning
        subtasks, operational_reasons = self._decompose_query(
            raw_query=q,
            q_lower=q_lower,
            primary_intent=primary_intent,
            discrete_objs=discrete_objs,
            landcover_feats=landcover_feats,
            measurements=measurements,
            spatial_rels=spatial_rels,
            req_counting=req_counting,
            req_localization=req_localization,
            req_temporal=req_temporal,
            req_cross_modal=req_cross_modal,
            req_gis=req_gis,
            is_compound=is_compound,
        )

        if loc_res.get("resolved") and loc_res.get("name"):
            operational_reasons.append(f"Geographic target resolved to '{loc_res['name']}'")
        if temporal_res.get("has_temporal_intent"):
            span_desc = f"{temporal_res['start_year']} to {temporal_res['end_year']}" if temporal_res.get('end_year') != temporal_res.get('start_year') else f"{temporal_res.get('start_year')}"
            operational_reasons.append(f"Temporal window extracted as {span_desc} ({temporal_res.get('temporal_mode')})")

        return QueryUnderstanding(
            raw_query=q,
            primary_intent=primary_intent,
            entities=all_entities,
            requested_classes=requested_classes,
            requested_measurements=measurements,
            spatial_relationships=spatial_rels,
            requires_counting=req_counting,
            requires_localization=req_localization,
            requires_segmentation=req_segmentation,
            requires_classification=req_classification,
            requires_comparison=req_comparison,
            requires_temporal_reasoning=req_temporal,
            requires_gis=req_gis,
            requires_cross_modal=req_cross_modal,
            is_compound_query=is_compound,
            is_temporal_geospatial=is_temp_geo,
            is_monitoring=is_monitoring,
            monitoring_condition=monitoring_cond,
            monitoring_threshold=monitoring_thresh,
            monitoring_threshold_unit=monitoring_thresh_unit,
            location_name=loc_res.get("name"),
            coordinates=loc_res.get("coordinates"),
            time_range=temporal_res if temporal_res.get("has_temporal_intent") else None,
            temporal_intent="multi_year_progression" if temporal_res.get("is_multi_year") else ("bitemporal_comparison" if req_comparison else ("single_date" if temporal_res.get("start_year") else None)),
            requested_features=temporal_res.get("requested_features", []),
            is_ambiguous_location=loc_res.get("ambiguous", False),
            ambiguity_options=loc_res.get("ambiguity_options", []),
            subtasks=subtasks,
            operational_reasoning=operational_reasons,
        )

    def _extract_entities(self, q_lower: str) -> Tuple[List[str], List[str]]:
        """Extract canonical discrete physical objects and landcover features from query text in order of appearance."""
        discrete_found: Set[str] = set()
        landcover_found: Set[str] = set()

        # Match discrete objects (check multi-word terms first)
        for term, canonical in sorted(DISCRETE_OBJECT_MAP.items(), key=lambda x: -len(x[0])):
            pattern = rf"(?<![a-zA-Z0-9]){re.escape(term)}(?![a-zA-Z0-9])"
            if re.search(pattern, q_lower):
                discrete_found.add(canonical)

        # Match landcover features
        for term, canonical in sorted(LANDCOVER_FEATURE_MAP.items(), key=lambda x: -len(x[0])):
            pattern = rf"(?<![a-zA-Z0-9]){re.escape(term)}(?![a-zA-Z0-9])"
            if re.search(pattern, q_lower):
                landcover_found.add(canonical)

        # Sort entities by their first occurrence index in query
        discrete_ordered = sorted(
            list(discrete_found),
            key=lambda x: min([q_lower.find(t) for t, c in DISCRETE_OBJECT_MAP.items() if c == x and q_lower.find(t) != -1] or [9999]),
        )
        landcover_ordered = sorted(
            list(landcover_found),
            key=lambda x: min([q_lower.find(t) for t, c in LANDCOVER_FEATURE_MAP.items() if c == x and q_lower.find(t) != -1] or [9999]),
        )

        return discrete_ordered, landcover_ordered

    def _extract_measurements(self, q_lower: str) -> List[str]:
        """Identify requested metrics and measurements."""
        found: List[str] = []
        for m_type, patterns in MEASUREMENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_lower):
                    found.append(m_type)
                    break
        return found

    def _extract_spatial_relations(self, q_lower: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Extract spatial relationships (distance buffers, containment, directional, proximity)."""
        relations: List[Dict[str, Any]] = []

        # 1. Distance Buffer: "within 100m of the river"
        m_dist = DISTANCE_REL_REGEX.search(q_lower)
        if m_dist:
            val_str = m_dist.group(1)
            unit_str = (m_dist.group(2) or "m").lower()
            ref_raw = m_dist.group(3).strip()
            
            ref_norm = self._resolve_entity_name(ref_raw)
            try:
                val = float(val_str)
                meters = val * 1000 if "km" in unit_str else val
            except ValueError:
                meters = 100.0

            relations.append({
                "relation": "within_distance",
                "reference": ref_norm,
                "distance_m": meters,
                "constraint": f"<= {int(meters)}m",
                "raw_reference": ref_raw,
            })

        # 2. Directional: "north of the river"
        m_dir = DIRECTIONAL_REL_REGEX.search(q_lower)
        if m_dir:
            direction = m_dir.group(1).lower()
            ref_raw = m_dir.group(2).strip()
            ref_norm = self._resolve_entity_name(ref_raw)
            relations.append({
                "relation": "directional",
                "direction": direction,
                "reference": ref_norm,
                "raw_reference": ref_raw,
            })

        # 3. Containment & Anaphora: "inside the vegetated area", "inside them"
        if not m_dist:
            # Check anaphora like "inside them" or "within it"
            m_anaphora = re.search(r"\b(?:inside|within|contained\s+in)\s+(?:them|it)\b", q_lower)
            if m_anaphora:
                # Resolve to preceding landcover entity if present
                for ent in entities:
                    if ent in LANDCOVER_FEATURE_MAP.values():
                        relations.append({
                            "relation": "inside",
                            "reference": ent,
                            "raw_reference": ent,
                        })
                        break
            else:
                m_cont = re.search(r"\b(?:inside|contained\s+in|within)\s+(?:the\s+)?([a-zA-Z_\s]+?)(?:\s*\?|\s*\.|\s*$|\s+and\b)", q_lower)
                if m_cont:
                    ref_raw = m_cont.group(1).strip()
                    ref_norm = self._resolve_entity_name(ref_raw)
                    if ref_norm in LANDCOVER_FEATURE_MAP.values() or ref_norm in DISCRETE_OBJECT_MAP.values():
                        relations.append({
                            "relation": "inside",
                            "reference": ref_norm,
                            "raw_reference": ref_raw,
                        })

        # 4. Proximity: "adjacent to the pier", "near the runway"
        m_prox = PROXIMITY_REL_REGEX.search(q_lower)
        if m_prox and not relations:
            ref_raw = m_prox.group(1).strip()
            ref_norm = self._resolve_entity_name(ref_raw)
            relations.append({
                "relation": "proximity",
                "reference": ref_norm,
                "raw_reference": ref_raw,
            })

        return relations

    def _resolve_entity_name(self, raw_text: str) -> str:
        """Resolve a raw phrase to a known canonical entity name if possible."""
        text = raw_text.strip().lower()
        if text in DISCRETE_OBJECT_MAP:
            return DISCRETE_OBJECT_MAP[text]
        if text in LANDCOVER_FEATURE_MAP:
            return LANDCOVER_FEATURE_MAP[text]
        for term, canonical in DISCRETE_OBJECT_MAP.items():
            if term in text:
                return canonical
        for term, canonical in LANDCOVER_FEATURE_MAP.items():
            if term in text:
                return canonical
        # Clean trailing noise words and re-check
        cleaned = re.sub(r"\b(area|zone|region|facility|facilities)\b", "", text).strip()
        if cleaned in DISCRETE_OBJECT_MAP:
            return DISCRETE_OBJECT_MAP[cleaned]
        if cleaned in LANDCOVER_FEATURE_MAP:
            return LANDCOVER_FEATURE_MAP[cleaned]
        return text

    def _classify_primary_intent(
        self,
        q_lower: str,
        discrete_objs: List[str],
        landcover_feats: List[str],
        measurements: List[str],
        spatial_rels: List[Dict[str, Any]],
        req_temporal: bool,
        req_cross_modal: bool,
        req_gis: bool,
        is_compound: bool,
    ) -> str:
        """Determine primary semantic intent category."""
        if req_cross_modal:
            return "cross_modal_analysis"
        if req_temporal:
            return "temporal_change_analysis"
        if is_compound or req_gis:
            return "spatial_object_query"
        if "count" in measurements or "how many" in q_lower:
            return "object_counting"
        if "localization" in measurements or "where" in q_lower:
            return "object_localization"
        if "percentage" in measurements or ("area" in measurements and landcover_feats):
            return "coverage_measurement"
        if landcover_feats or "land cover" in q_lower or "urban or rural" in q_lower:
            return "landcover_classification"
        if discrete_objs:
            return "object_detection"
        return "scene_description"

    def _decompose_query(
        self,
        raw_query: str,
        q_lower: str,
        primary_intent: str,
        discrete_objs: List[str],
        landcover_feats: List[str],
        measurements: List[str],
        spatial_rels: List[Dict[str, Any]],
        req_counting: bool,
        req_localization: bool,
        req_temporal: bool,
        req_cross_modal: bool,
        req_gis: bool,
        is_compound: bool,
    ) -> Tuple[List[QuerySubtask], List[str]]:
        """Decompose query into ordered atomic subtasks with concise operational reasoning."""
        subtasks: List[QuerySubtask] = []
        reasons: List[str] = []

        # Target object (default to first discrete object or 'objects')
        target_obj = discrete_objs[0] if discrete_objs else "objects"
        target_obj_plural = f"{target_obj}s" if not target_obj.endswith("s") else target_obj

        # ---------------------------------------------------------------------
        # Case 1: Spatial Object Query with Distance Filter
        # Example: "How many buildings are within 100m of the river?"
        # ---------------------------------------------------------------------
        dist_rel = next((r for r in spatial_rels if r.get("relation") == "within_distance"), None)
        if dist_rel:
            ref_entity = dist_rel["reference"]
            dist_val = int(dist_rel["distance_m"])

            # Step 1: Detect target objects
            subtasks.append(QuerySubtask(
                step_number=1,
                description=f"Detect {target_obj_plural}",
                operation="detect",
                target_entity=target_obj,
                tool_hint="YOLO11",
                operational_reasoning=f"YOLO11 is required for {target_obj} localization.",
            ))
            reasons.append(f"YOLO11 is required for {target_obj} localization.")

            # Step 2: Segment/Identify reference feature
            ref_tool = "BigEarthNet-ResNet50-SeCo-12B" if ref_entity in ["water", "river", "lake", "vegetation", "forest", "agriculture"] else "YOLO11"
            subtasks.append(QuerySubtask(
                step_number=2,
                description=f"Identify/segment {ref_entity}",
                operation="segment",
                target_entity=ref_entity,
                tool_hint=ref_tool,
                operational_reasoning=f"Land cover segmentation is required for {ref_entity} boundary extraction.",
            ))
            reasons.append(f"Land cover segmentation is required for {ref_entity} boundary extraction.")

            # Step 3: Calculate distances
            subtasks.append(QuerySubtask(
                step_number=3,
                description=f"Calculate {target_obj}-to-{ref_entity} distances",
                operation="distance_calc",
                target_entity=target_obj,
                condition=f"to_{ref_entity}",
                tool_hint="GIS",
                operational_reasoning="GIS is required for distance calculation.",
            ))
            reasons.append("GIS is required for distance calculation.")

            # Step 4: Filter distance <= threshold
            subtasks.append(QuerySubtask(
                step_number=4,
                description=f"Filter {target_obj_plural} <= {dist_val}m",
                operation="filter",
                target_entity=target_obj,
                condition=f"<= {dist_val}m",
                tool_hint="GIS",
                operational_reasoning="GIS is required for spatial threshold filtering.",
            ))
            reasons.append("GIS is required for spatial threshold filtering.")

            # Step 5: Count or localize
            final_op = "count" if req_counting else "localize"
            subtasks.append(QuerySubtask(
                step_number=5,
                description=f"Count them" if req_counting else f"Localize filtered {target_obj_plural}",
                operation=final_op,
                target_entity=target_obj,
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for final answer synthesis and count reporting.",
            ))
            reasons.append("RS-Qwen is required for final answer synthesis and count reporting.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 2: Spatial Object Query with Containment
        # Example: "Find agricultural areas and count buildings inside them."
        # ---------------------------------------------------------------------
        inside_rel = next((r for r in spatial_rels if r.get("relation") == "inside"), None)
        if inside_rel or ("inside" in q_lower and discrete_objs and landcover_feats):
            ref_entity = inside_rel["reference"] if inside_rel else landcover_feats[0]

            # Step 1: Segment reference area
            subtasks.append(QuerySubtask(
                step_number=1,
                description=f"Identify and segment {ref_entity} areas",
                operation="segment",
                target_entity=ref_entity,
                tool_hint="BigEarthNet-ResNet50-SeCo-12B",
                operational_reasoning=f"BigEarthNet classifier is required for {ref_entity} boundary segmentation.",
            ))
            reasons.append(f"BigEarthNet classifier is required for {ref_entity} boundary segmentation.")

            # Step 2: Detect discrete objects
            subtasks.append(QuerySubtask(
                step_number=2,
                description=f"Detect {target_obj_plural}",
                operation="detect",
                target_entity=target_obj,
                tool_hint="YOLO11",
                operational_reasoning=f"YOLO11 is required for {target_obj} localization.",
            ))
            reasons.append(f"YOLO11 is required for {target_obj} localization.")

            # Step 3: Spatial intersection
            subtasks.append(QuerySubtask(
                step_number=3,
                description=f"Perform spatial intersection inside {ref_entity}",
                operation="spatial_filter",
                target_entity=target_obj,
                condition=f"inside {ref_entity}",
                tool_hint="GIS",
                operational_reasoning="GIS is required for point-in-polygon containment filtering.",
            ))
            reasons.append("GIS is required for point-in-polygon containment filtering.")

            # Step 4: Count / Report
            subtasks.append(QuerySubtask(
                step_number=4,
                description=f"Count {target_obj_plural} inside {ref_entity}",
                operation="count",
                target_entity=target_obj,
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for domain reporting and verified evidence synthesis.",
            ))
            reasons.append("RS-Qwen is required for domain reporting and verified evidence synthesis.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 3: Directional Spatial Query
        # Example: "What is north of the river?"
        # ---------------------------------------------------------------------
        dir_rel = next((r for r in spatial_rels if r.get("relation") == "directional"), None)
        if dir_rel:
            direction = dir_rel["direction"]
            ref_entity = dir_rel["reference"]

            subtasks.append(QuerySubtask(
                step_number=1,
                description=f"Segment/localize reference feature '{ref_entity}'",
                operation="segment",
                target_entity=ref_entity,
                tool_hint="BigEarthNet-ResNet50-SeCo-12B",
                operational_reasoning=f"Segmentation is required to locate reference {ref_entity}.",
            ))
            reasons.append(f"Segmentation is required to locate reference {ref_entity}.")

            subtasks.append(QuerySubtask(
                step_number=2,
                description=f"Delineate {direction} spatial bounding sector",
                operation="spatial_filter",
                condition=f"{direction} of {ref_entity}",
                tool_hint="GIS",
                operational_reasoning="GIS is required for directional bounding delineation.",
            ))
            reasons.append("GIS is required for directional bounding delineation.")

            subtasks.append(QuerySubtask(
                step_number=3,
                description=f"Analyze and identify features {direction} of {ref_entity}",
                operation="describe",
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for natural-language spatial scene analysis.",
            ))
            reasons.append("RS-Qwen is required for natural-language spatial scene analysis.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 4: Temporal Change Analysis
        # Example: "What changed between these images?"
        # ---------------------------------------------------------------------
        if req_temporal:
            subtasks.append(QuerySubtask(
                step_number=1,
                description="Execute bi-temporal feature comparison",
                operation="compare",
                tool_hint="ChangeFormerV6",
                operational_reasoning="ChangeFormerV6 is required for bi-temporal pixel change detection.",
            ))
            reasons.append("ChangeFormerV6 is required for bi-temporal pixel change detection.")

            subtasks.append(QuerySubtask(
                step_number=2,
                description="Synthesize change analysis and geospatial interpretation",
                operation="describe",
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for grounded change interpretation.",
            ))
            reasons.append("RS-Qwen is required for grounded change interpretation.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 5: Optical + SAR Cross-Modal Analysis
        # Example: "Identify built-up and water areas using optical and SAR."
        # ---------------------------------------------------------------------
        if req_cross_modal:
            subtasks.append(QuerySubtask(
                step_number=1,
                description="Extract cross-modal Optical + SAR feature representations",
                operation="fuse",
                tool_hint="CROMA-Base",
                operational_reasoning="CROMA-Base is required for multi-sensor Optical + SAR feature fusion.",
            ))
            reasons.append("CROMA-Base is required for multi-sensor Optical + SAR feature fusion.")

            subtasks.append(QuerySubtask(
                step_number=2,
                description="Synthesize multi-sensor analysis",
                operation="describe",
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for joint sensor reasoning.",
            ))
            reasons.append("RS-Qwen is required for joint sensor reasoning.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 6: Object Detection / Counting
        # Example: "How many buildings are there?" / "Where are the buildings?"
        # ---------------------------------------------------------------------
        if discrete_objs or req_counting or req_localization:
            subtasks.append(QuerySubtask(
                step_number=1,
                description=f"Detect and localize {target_obj_plural}",
                operation="detect",
                target_entity=target_obj,
                tool_hint="YOLO11",
                operational_reasoning=f"YOLO11 is required for discrete {target_obj} localization and bounding boxes.",
            ))
            reasons.append(f"YOLO11 is required for discrete {target_obj} localization and bounding boxes.")

            report_desc = f"Count and report {target_obj_plural}" if req_counting else f"Report {target_obj} locations"
            subtasks.append(QuerySubtask(
                step_number=2,
                description=report_desc,
                operation="count" if req_counting else "describe",
                target_entity=target_obj,
                tool_hint="RS-Qwen",
                operational_reasoning=f"RS-Qwen is required for grounded interpretation of {target_obj} telemetry.",
            ))
            reasons.append(f"RS-Qwen is required for grounded interpretation of {target_obj} telemetry.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 7: Land Cover Classification & Coverage Measurement
        # Example: "How much vegetation is present?" / "What percentage is water?"
        # ---------------------------------------------------------------------
        if landcover_feats or "percentage" in measurements or "area" in measurements or "land cover" in q_lower or "urban or rural" in q_lower:
            feat_desc = ", ".join(landcover_feats) if landcover_feats else "land cover"
            subtasks.append(QuerySubtask(
                step_number=1,
                description=f"Classify multispectral land cover classes ({feat_desc})",
                operation="classify",
                tool_hint="BigEarthNet-ResNet50-SeCo-12B",
                operational_reasoning="BigEarthNet-ResNet50-SeCo-12B is required for 12-band multispectral classification.",
            ))
            reasons.append("BigEarthNet-ResNet50-SeCo-12B is required for 12-band multispectral classification.")

            subtasks.append(QuerySubtask(
                step_number=2,
                description=f"Calculate coverage proportion and synthesize report",
                operation="measure",
                tool_hint="RS-Qwen",
                operational_reasoning="RS-Qwen is required for class proportion reporting and domain interpretation.",
            ))
            reasons.append("RS-Qwen is required for class proportion reporting and domain interpretation.")
            return subtasks, reasons

        # ---------------------------------------------------------------------
        # Case 8: General Scene Description / VLM Reasoning
        # Example: "Analyze this image tell about it?" / "Describe this scene"
        # ---------------------------------------------------------------------
        subtasks.append(QuerySubtask(
            step_number=1,
            description="Analyze scene features and synthesize domain description",
            operation="describe",
            tool_hint="RS-Qwen",
            operational_reasoning="RS-Qwen is required for comprehensive remote-sensing visual understanding.",
        ))
        reasons.append("RS-Qwen is required for comprehensive remote-sensing visual understanding.")
        return subtasks, reasons
