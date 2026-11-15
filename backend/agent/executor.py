"""Specialist Execution Engine for SatQuery AI.

Executes models and tools according to the AnalysisPlan produced by the planner.
Handles:
- Dependency resolution for compound multi-stage queries
  (e.g., YOLO11 -> SegFormer -> GIS -> Threshold Filter -> Count)
- Standardized output envelopes:
  { "success": True, "model": "...", "results": {}, "visualization": "...", "error": None }
- Resilient error handling (partial tool failures do not crash entire analysis)
- Comprehensive execution audit logging (timestamps, durations, locations, success/failure)
- Precise quantitative evidence integration
"""

import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.agent.registry import get_tool
from backend.agent.schemas import AnalysisPlan, QueryUnderstanding

logger = logging.getLogger("agent.executor")


class AnalysisExecutor:
    """Orchestrates topological execution of specialist models and GIS operations."""

    def __init__(self):
        pass

    def execute(
        self,
        plan: AnalysisPlan,
        query_understanding: QueryUnderstanding,
        images: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Execute selected models in correct dependency order.

        Args:
            plan: The structured AnalysisPlan from AnalysisPlanner.
            query_understanding: The structured QueryUnderstanding.
            images: List of input images (files or extracted panels).
            metadata: Optional scene/sensor metadata.

        Returns:
            Tuple of (tool_results_dict, execution_trace_steps).
        """
        metadata = metadata or {}
        tool_results: Dict[str, Any] = {}
        trace_steps: List[str] = []

        q_lower = query_understanding.raw_query.lower()
        selected_models = plan.selected_models
        selected_tools = plan.selected_tools

        if not selected_models and "GIS" not in selected_tools:
            trace_steps.append("Zero specialist models planned; proceeding directly to foundation vision reasoning.")
            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case A: Compound Spatial Distance Query
        # e.g., "How many buildings are within 100m of the river?"
        # Dependencies: YOLO11 (buildings) -> SegFormer (river) -> GIS (distance) -> Filter -> Count
        # ---------------------------------------------------------------------
        is_distance_query = (
            ("YOLO11" in selected_models) and
            ("SegFormer" in selected_models) and
            ("GIS" in selected_tools) and
            any(
                r.get("relation") in ["within_distance", "distance", "within", "proximity"]
                or "distance_m" in r
                or "distance" in r
                for r in query_understanding.spatial_relationships
            )
        )

        if is_distance_query and images:
            trace_steps.append("Executing compound spatial distance pipeline: [YOLO11 -> SegFormer -> GIS Distance Filter -> Count]")
            
            # 1. YOLO11: Building Detections
            yolo_tool = get_tool("yolo")
            if yolo_tool:
                t0 = time.time()
                trace_steps.append(f"Starting {yolo_tool.model_id} object localization...")
                yolo_res = yolo_tool.execute(images[0])
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["yolo"] = yolo_res
                if yolo_res.get("success"):
                    obj_count = len(yolo_res.get("results", {}).get("objects", []))
                    trace_steps.append(f"Specialist {yolo_tool.model_id} completed in {duration_ms}ms: {obj_count} candidate object(s) detected.")
                else:
                    trace_steps.append(f"Specialist {yolo_tool.model_id} failed in {duration_ms}ms: {yolo_res.get('error')}")

            # 2. SegFormer: River / Water Region Extraction
            seg_tool = get_tool("segformer")
            if seg_tool:
                t0 = time.time()
                trace_steps.append(f"Starting {seg_tool.model_id} reference boundary extraction...")
                seg_res = seg_tool.execute(images[0])
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["segformer"] = seg_res
                if seg_res.get("success"):
                    trace_steps.append(f"Specialist {seg_tool.model_id} completed in {duration_ms}ms: Extracted reference feature boundaries.")
                else:
                    trace_steps.append(f"Specialist {seg_tool.model_id} failed in {duration_ms}ms: {seg_res.get('error')}")

            # 3. GIS: Distance Calculations & Filtering
            gis_tool = get_tool("gis")
            if gis_tool:
                t0 = time.time()
                trace_steps.append("Starting GIS topological distance calculations...")
                
                # Extract objects from YOLO
                detected_objs = tool_results.get("yolo", {}).get("results", {}).get("objects", [])
                
                # Extract threshold distance from query (e.g. 100m)
                thresh_val = 100.0
                for r in query_understanding.spatial_relationships:
                    if "distance_m" in r:
                        thresh_val = float(r["distance_m"])
                        break
                    dist_str = str(r.get("distance", ""))
                    m = re.search(r"(\d+(?:\.\d+)?)", dist_str)
                    if m:
                        thresh_val = float(m.group(1))
                        break

                # Extract reference polygon from SegFormer (river or water)
                ref_poly = []
                if "segformer" in tool_results and tool_results["segformer"].get("success"):
                    seg_classes = tool_results["segformer"].get("results", {}).get("classes", {})
                    if "river" in seg_classes and seg_classes["river"].get("polygon"):
                        ref_poly = seg_classes["river"]["polygon"]
                    elif "water" in seg_classes and seg_classes["water"].get("polygon"):
                        ref_poly = seg_classes["water"]["polygon"]

                gis_res = gis_tool.execute(
                    operation="distance_filter",
                    objects=detected_objs,
                    reference_polygon=ref_poly,
                    threshold_meters=thresh_val,
                )
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["gis"] = gis_res

                if gis_res.get("success"):
                    m_count = gis_res.get("results", {}).get("matches_count", 0)
                    trace_steps.append(
                        f"Specialist GIS completed in {duration_ms}ms: Filtered {m_count} object(s) within {thresh_val}m buffer."
                    )
                else:
                    trace_steps.append(f"Specialist GIS failed in {duration_ms}ms: {gis_res.get('error')}")

            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case B: Compound Spatial Containment Query
        # e.g., "Find agricultural areas and count buildings inside them."
        # Dependencies: SegFormer (agriculture) -> YOLO11 (buildings) -> GIS (containment) -> Count
        # ---------------------------------------------------------------------
        is_containment_query = (
            ("SegFormer" in selected_models) and
            ("YOLO11" in selected_models) and
            ("GIS" in selected_tools)
        )

        if is_containment_query and images:
            trace_steps.append("Executing compound spatial containment pipeline: [SegFormer -> YOLO11 -> GIS Containment -> Count]")

            # 1. SegFormer: Region segmentation
            seg_tool = get_tool("segformer")
            if seg_tool:
                t0 = time.time()
                trace_steps.append(f"Starting {seg_tool.model_id} parcel segmentation...")
                seg_res = seg_tool.execute(images[0])
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["segformer"] = seg_res
                if seg_res.get("success"):
                    trace_steps.append(f"Specialist {seg_tool.model_id} completed in {duration_ms}ms.")
                else:
                    trace_steps.append(f"Specialist {seg_tool.model_id} failed in {duration_ms}ms: {seg_res.get('error')}")

            # 2. YOLO11: Discrete object detection
            yolo_tool = get_tool("yolo")
            if yolo_tool:
                t0 = time.time()
                trace_steps.append(f"Starting {yolo_tool.model_id} object localization...")
                yolo_res = yolo_tool.execute(images[0])
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["yolo"] = yolo_res
                if yolo_res.get("success"):
                    trace_steps.append(f"Specialist {yolo_tool.model_id} completed in {duration_ms}ms.")
                else:
                    trace_steps.append(f"Specialist {yolo_tool.model_id} failed in {duration_ms}ms: {yolo_res.get('error')}")

            # 3. GIS: Containment
            gis_tool = get_tool("gis")
            if gis_tool:
                t0 = time.time()
                trace_steps.append("Starting GIS spatial containment intersection...")
                detected_objs = tool_results.get("yolo", {}).get("results", {}).get("objects", [])
                
                # Determine container name from spatial relationships or entities
                target_container = "urban"
                for r in query_understanding.spatial_relationships:
                    if r.get("relation") in ["inside", "containment", "contains"]:
                        target_container = r.get("reference", target_container)
                        break
                if not target_container and query_understanding.entities:
                    for ent in query_understanding.entities:
                        if ent in ["urban", "built_up", "vegetation", "agriculture", "forest", "water"]:
                            target_container = ent
                            break

                # Container polygon from SegFormer
                cont_poly = []
                if "segformer" in tool_results and tool_results["segformer"].get("success"):
                    classes = tool_results["segformer"].get("results", {}).get("classes", {})
                    if target_container in classes and classes[target_container].get("polygon"):
                        cont_poly = classes[target_container]["polygon"]
                    elif "built_up" in classes and target_container in ["urban", "city", "built_up"] and classes["built_up"].get("polygon"):
                        cont_poly = classes["built_up"]["polygon"]
                    else:
                        for c_name, c_data in classes.items():
                            if c_data.get("polygon"):
                                cont_poly = c_data["polygon"]
                                target_container = c_name
                                break

                gis_res = gis_tool.execute(
                    operation="containment_filter",
                    objects=detected_objs,
                    container_polygon=cont_poly,
                    container_name=target_container,
                )
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["gis"] = gis_res

                if gis_res.get("success"):
                    c_count = gis_res.get("results", {}).get("contained_count", 0)
                    trace_steps.append(
                        f"Specialist GIS completed in {duration_ms}ms: Confirmed {c_count} object(s) within {target_container} bounds."
                    )
                else:
                    trace_steps.append(f"Specialist GIS failed in {duration_ms}ms: {gis_res.get('error')}")

            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case C: Feature Change Comparison Across Scenes / Panels
        # e.g., "Did vegetation increase?"
        # Dependencies: SegFormer(Panel A) -> SegFormer(Panel B) -> GIS Area Delta
        # ---------------------------------------------------------------------
        is_feature_comparison = (
            plan.comparison_requested and
            "SegFormer" in selected_models and
            len(images) >= 2
        )

        if is_feature_comparison:
            trace_steps.append("Executing bi-temporal feature comparison: [SegFormer(Panel A) -> SegFormer(Panel B) -> GIS Area Delta]")
            
            seg_tool = get_tool("segformer")
            gis_tool = get_tool("gis")

            if seg_tool:
                # Panel A
                t0 = time.time()
                res_a = seg_tool.execute(images[0])
                dur_a = round((time.time() - t0) * 1000, 1)
                
                # Panel B
                t1 = time.time()
                res_b = seg_tool.execute(images[1])
                dur_b = round((time.time() - t1) * 1000, 1)

                tool_results["segformer"] = res_b  # primary latest
                tool_results["segformer_panel_a"] = res_a
                tool_results["segformer_panel_b"] = res_b

                trace_steps.append(f"Specialist {seg_tool.model_id} evaluated Panel A in {dur_a}ms and Panel B in {dur_b}ms.")

                # Target feature
                feature = "vegetation"
                if "water" in q_lower:
                    feature = "water"
                elif "built" in q_lower or "building" in q_lower:
                    feature = "built_up"

                area_a = res_a.get("results", {}).get("classes", {}).get(feature, {}).get("area_sq_m", 0.0)
                area_b = res_b.get("results", {}).get("classes", {}).get(feature, {}).get("area_sq_m", 0.0)

                if gis_tool:
                    t_gis = time.time()
                    gis_res = gis_tool.execute(
                        operation="area_delta",
                        feature=feature,
                        area_a=area_a,
                        area_b=area_b,
                    )
                    dur_gis = round((time.time() - t_gis) * 1000, 1)
                    tool_results["gis"] = gis_res
                    if gis_res.get("success"):
                        trend = gis_res.get("results", {}).get("trend", "")
                        pct_chg = gis_res.get("results", {}).get("percentage_change", 0.0)
                        trace_steps.append(
                            f"Specialist GIS completed in {dur_gis}ms: Verified {trend} of {abs(pct_chg)}% in {feature}."
                        )

            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case C2: Discrete Object Temporal Comparison Across Scenes / Panels
        # e.g., "Did the number of buildings increase since 2020?", "How many new buildings were constructed?"
        # Dependencies: YOLO11(Panel A) -> YOLO11(Panel B) -> GIS Count Delta
        # ---------------------------------------------------------------------
        is_discrete_comparison = (
            plan.comparison_requested and
            "YOLO11" in selected_models and
            len(images) >= 2
        )

        if is_discrete_comparison:
            trace_steps.append("Executing bi-temporal discrete object comparison: [YOLO11(Panel A) -> YOLO11(Panel B) -> GIS Count Delta]")

            yolo_tool = get_tool("yolo")
            gis_tool = get_tool("gis")

            if yolo_tool:
                # Panel A
                t0 = time.time()
                res_a = yolo_tool.execute(images[0])
                dur_a = round((time.time() - t0) * 1000, 1)

                # Panel B
                t1 = time.time()
                res_b = yolo_tool.execute(images[1])
                dur_b = round((time.time() - t1) * 1000, 1)

                tool_results["yolo"] = res_b  # primary latest
                tool_results["yolo_panel_a"] = res_a
                tool_results["yolo_panel_b"] = res_b

                trace_steps.append(f"Specialist {yolo_tool.model_id} evaluated Panel A in {dur_a}ms and Panel B in {dur_b}ms.")

                # Extract object counts
                cnt_a = len(res_a.get("results", {}).get("objects", []))
                cnt_b = len(res_b.get("results", {}).get("objects", []))

                # Identify target discrete class (default: buildings or structures)
                feature = "buildings"
                if "vehicle" in q_lower:
                    feature = "vehicles"
                elif "aircraft" in q_lower or "plane" in q_lower:
                    feature = "aircraft"
                elif "vessel" in q_lower or "ship" in q_lower:
                    feature = "vessels"

                if gis_tool:
                    t_gis = time.time()
                    gis_res = gis_tool.execute(
                        operation="count_delta",
                        feature=feature,
                        count_a=cnt_a,
                        count_b=cnt_b,
                    )
                    dur_gis = round((time.time() - t_gis) * 1000, 1)
                    tool_results["gis"] = gis_res
                    if gis_res.get("success"):
                        trend = gis_res.get("results", {}).get("trend", "")
                        delta_cnt = gis_res.get("results", {}).get("delta_count", 0)
                        pct_chg = gis_res.get("results", {}).get("percentage_change", 0.0)
                        trace_steps.append(
                            f"Specialist GIS completed in {dur_gis}ms: Verified {trend} of {delta_cnt} {feature} ({pct_chg:+.1f}%)."
                        )

            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case D: Pixel Change Detection (ChangeFormerV6)
        # e.g., "What changed?"
        # ---------------------------------------------------------------------
        if "ChangeFormerV6" in selected_models:
            cd_tool = get_tool("change_detection")
            if cd_tool:
                t0 = time.time()
                trace_steps.append(f"Executing specialist model: {cd_tool.model_id}")
                if len(images) >= 2:
                    cd_res = cd_tool.execute(images[0], images[1])
                else:
                    cd_res = {
                        "success": False,
                        "model": cd_tool.model_id,
                        "results": {},
                        "visualization": None,
                        "error": f"Change detection requires 2 scenes, received {len(images)}.",
                    }
                duration_ms = round((time.time() - t0) * 1000, 1)
                tool_results["change_detection"] = cd_res
                if cd_res.get("success"):
                    pct = cd_res.get("change_percentage", cd_res.get("results", {}).get("change_percentage", 0.0))
                    trace_steps.append(f"Specialist {cd_tool.model_id} completed in {duration_ms}ms: {pct:.2f}% surface change detected.")
                else:
                    trace_steps.append(f"Specialist {cd_tool.model_id} failed in {duration_ms}ms: {cd_res.get('error')}")

            return tool_results, trace_steps

        # ---------------------------------------------------------------------
        # Case E: Multi-Model & Single-Model Specialist Execution
        # (YOLO11, SegFormer, BigEarthNet, CROMA, SAR-Backscatter-Analyzer)
        # ---------------------------------------------------------------------
        # Resolve optical vs SAR image targets using modality inspection
        from backend.agent.modality import ModalityInspector

        opt_target = None
        sar_target = None
        if images:
            for img in images:
                m_info = ModalityInspector.inspect(img)
                if m_info.probable_modality == "optical" and opt_target is None:
                    opt_target = img
                elif m_info.probable_modality == "sar" and sar_target is None:
                    sar_target = img

            if opt_target is None:
                opt_target = images[0]
            if sar_target is None:
                sar_target = images[1] if len(images) > 1 and images[1] != opt_target else (metadata.get("sar_image") or images[0])

        for model_name in selected_models:
            if model_name == "YOLO11" and "yolo" not in tool_results:
                tool = get_tool("yolo")
                target_img = opt_target or (images[0] if images else None)
                if tool and target_img:
                    t0 = time.time()
                    trace_steps.append(f"Executing specialist model: {tool.model_id} on optical target")
                    res = tool.execute(target_img)
                    dur = round((time.time() - t0) * 1000, 1)
                    tool_results["yolo"] = res
                    if res.get("success"):
                        cnt = len(res.get("results", {}).get("objects", []))
                        trace_steps.append(f"Specialist {tool.model_id} completed in {dur}ms: {cnt} object(s) detected.")
                    else:
                        trace_steps.append(f"Specialist {tool.model_id} failed in {dur}ms: {res.get('error')}")

            elif model_name == "SegFormer" and "segformer" not in tool_results:
                tool = get_tool("segformer")
                target_img = opt_target or (images[0] if images else None)
                if tool and target_img:
                    t0 = time.time()
                    trace_steps.append(f"Executing specialist model: {tool.model_id} on optical target")
                    res = tool.execute(target_img)
                    dur = round((time.time() - t0) * 1000, 1)
                    tool_results["segformer"] = res
                    if res.get("success"):
                        dom = res.get("results", {}).get("dominant_class", "landcover")
                        trace_steps.append(f"Specialist {tool.model_id} completed in {dur}ms: dominant class '{dom}'.")
                    else:
                        trace_steps.append(f"Specialist {tool.model_id} failed in {dur}ms: {res.get('error')}")

            elif model_name == "SAR-Backscatter-Analyzer" and "sar_analyzer" not in tool_results:
                tool = get_tool("sar_analyzer")
                target_img = sar_target or (images[0] if images else None)
                if tool and target_img:
                    t0 = time.time()
                    trace_steps.append(f"Executing specialist model: {tool.model_id} on SAR target")
                    res = tool.execute(target_img)
                    dur = round((time.time() - t0) * 1000, 1)
                    tool_results["sar_analyzer"] = res
                    if res.get("success"):
                        trace_steps.append(f"Specialist {tool.model_id} completed in {dur}ms: Backscatter distribution characterized.")
                    else:
                        trace_steps.append(f"Specialist {tool.model_id} failed in {dur}ms: {res.get('error')}")

            elif model_name == "BigEarthNet-ResNet50-SeCo-12B" and "landcover" not in tool_results:
                tool = get_tool("landcover")
                target_img = opt_target or (images[0] if images else None)
                if tool and target_img:
                    t0 = time.time()
                    trace_steps.append(f"Executing specialist model: {tool.model_id}")
                    res = tool.execute(target_img)
                    dur = round((time.time() - t0) * 1000, 1)
                    tool_results["landcover"] = res
                    if res.get("success"):
                        trace_steps.append(f"Specialist {tool.model_id} completed in {dur}ms.")
                    else:
                        trace_steps.append(f"Specialist {tool.model_id} failed in {dur}ms: {res.get('error')}")

            elif model_name == "CROMA-Base" and "croma" not in tool_results:
                tool = get_tool("croma")
                if tool and images:
                    t0 = time.time()
                    trace_steps.append(f"Executing specialist model: {tool.model_id} (Optical + SAR)")
                    res = tool.execute(optical=opt_target, sar=sar_target)
                    dur = round((time.time() - t0) * 1000, 1)
                    tool_results["croma"] = res
                    if res.get("success"):
                        trace_steps.append(f"Specialist {tool.model_id} completed in {dur}ms: Cross-modal feature tokens fused.")
                    else:
                        trace_steps.append(f"Specialist {tool.model_id} failed in {dur}ms: {res.get('error')}")

        return tool_results, trace_steps
