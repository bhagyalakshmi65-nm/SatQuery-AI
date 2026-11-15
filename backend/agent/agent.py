"""Core SatQuery AI Agent and Orchestrator."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from backend.agent.evidence import EvidenceAggregator
from backend.agent.executor import AnalysisExecutor
from backend.agent.inspector import ImageInspector
from backend.agent.planner import AnalysisPlanner
from backend.agent.query_parser import SemanticQueryParser
from backend.agent.registry import TOOL_REGISTRY, get_tool
from backend.agent.router import TaskRouter
from backend.agent.schemas import (
    AgentResponse,
    AnalysisPlan,
    ExecutionTrace,
    ImageSceneInspection,
    QueryUnderstanding,
    RoutingDecision,
    TaskType,
)
from backend.agent.temporal_geospatial import SatelliteDataCatalog
from backend.agent.vlm import RSQwenEngine

logger = logging.getLogger("agent.orchestrator")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".pt", ".npz"}


class SatQueryAgent:
    """Central AI Agent for SatQuery AI remote sensing analysis."""

    def __init__(
        self,
        vlm_engine: Optional[RSQwenEngine] = None,
        inspector: Optional[ImageInspector] = None,
        query_parser: Optional[SemanticQueryParser] = None,
        planner: Optional[AnalysisPlanner] = None,
        executor: Optional[AnalysisExecutor] = None,
    ):
        self.vlm = vlm_engine or RSQwenEngine()
        self.inspector = inspector or ImageInspector()
        self.query_parser = query_parser or SemanticQueryParser()
        self.planner = planner or AnalysisPlanner()
        self.executor = executor or AnalysisExecutor()

    def validate_inputs(
        self,
        images: List[Union[str, Path]],
        task: TaskType,
        sensor_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scene_inspection: Optional[ImageSceneInspection] = None,
    ) -> Optional[str]:
        """Validate input images and sensor specifications.

        Returns:
            Error message string if invalid, or None if valid.
        """
        metadata = metadata or {}

        # Check file extensions
        for img in images:
            p = Path(img)
            ext = p.suffix.lower()
            if ext and ext not in SUPPORTED_EXTENSIONS:
                return (
                    f"Unsupported input format: '{ext}' for file '{p.name}'. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
                )

        # 1. Bi-Temporal Change Detection requires exactly two images or a separable multi-panel collage
        if task == TaskType.BI_TEMPORAL_CHANGE:
            can_separate = False
            if scene_inspection and scene_inspection.can_separate_panels and scene_inspection.scene_count >= 2:
                can_separate = True
            elif metadata and (metadata.get("can_separate_panels") or metadata.get("scene_count", 0) >= 2):
                can_separate = True

            if len(images) != 2 and not (len(images) == 1 and can_separate):
                return "I couldn't complete the requested comparison because I couldn't identify two analyzable scenes."

        # 2. Optical + SAR Cross-Modal analysis requires both modalities
        if task == TaskType.CROSS_MODAL_OPTICAL_SAR:
            from backend.agent.modality import ModalityInspector, CollageDetector
            if len(images) == 1:
                name_lower = Path(images[0]).name.lower()
                is_14 = "14" in name_lower or "dfc" in name_lower or "croma" in name_lower
                collage_res = CollageDetector.detect_and_split(images[0])
                if not is_14 and not (collage_res and collage_res.get("is_collage")):
                    return "Optical and SAR cross-modal analysis requires both an optical (Sentinel-2) and a SAR (Sentinel-1) image. If using a single file, please ensure it contains both panels or 14 channels."
            elif len(images) >= 2:
                info_0 = ModalityInspector.inspect(images[0], metadata=metadata)
                info_1 = ModalityInspector.inspect(images[1], metadata=metadata)
                mod_0 = info_0.probable_modality
                mod_1 = info_1.probable_modality

                if mod_0 == "optical" and mod_1 == "optical":
                    return "Two optical images were detected. Optical and SAR cross-modal analysis requires both an optical (Sentinel-2) and a SAR (Sentinel-1) image. Please provide a SAR radar image."
                elif mod_0 == "sar" and mod_1 == "sar":
                    return "Two SAR radar images were detected. Optical and SAR cross-modal analysis requires both an optical (Sentinel-2) and a SAR (Sentinel-1) image. Please provide an optical image."
                elif mod_0 == "unknown" and mod_1 == "unknown":
                    return "The modalities of the provided images could not be determined. Please provide verified Sentinel-1 (SAR) and Sentinel-2 (optical) images or specify sensor metadata."

        # 3. Single-image tasks require at least one image
        if task in [TaskType.SINGLE_IMAGE_DESCRIPTION, TaskType.SINGLE_IMAGE_OBJECT_DETECTION, TaskType.SINGLE_IMAGE_LANDCOVER]:
            if len(images) < 1:
                return "Please provide at least one satellite image to analyze."

        return None

    def analyze(
        self,
        query: str,
        images: Optional[List[Union[str, Path]]] = None,
        sensor_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Process user query and images through the AI agent pipeline."""
        start_time = time.time()
        images = images or []
        metadata = metadata or {}
        trace_steps: List[str] = []

        logger.info("Received query: '%s' with %d image(s)", query, len(images))

        temporal_context = None
        timeline = None
        observations: List[Dict[str, Any]] = []
        retrieved_images: List[str] = []

        # 0. Zero-Upload Temporal Geospatial Handling
        if not images:
            pre_understanding = self.query_parser.understand(
                query=query,
                scene_inspection=None,
                metadata=metadata,
            )

            active_ctx = metadata.get("active_temporal_context") or {}
            is_temporal = pre_understanding.is_temporal_geospatial or bool(active_ctx.get("location_name"))

            if is_temporal:
                # A. Ambiguous Location Check
                if pre_understanding.is_ambiguous_location:
                    loc_name = pre_understanding.location_name
                    opts = pre_understanding.ambiguity_options
                    opts_str = ", ".join(opts)
                    clarification_text = (
                        f"The location '{loc_name}' could refer to multiple places ({opts_str}). "
                        f"Which specific region would you like to analyze?"
                    )
                    exec_time = round((time.time() - start_time) * 1000, 2)
                    trace_steps.append(f"Location Ambiguity: '{loc_name}' matches multiple known places: {opts_str}")
                    return AgentResponse(
                        answer=clarification_text,
                        task="temporal_geospatial_clarification",
                        evidence={},
                        confidence=1.0,
                        execution_trace=ExecutionTrace(
                            task="temporal_geospatial_clarification",
                            tools_used=["RS-Qwen"],
                            steps=trace_steps,
                            execution_time_ms=exec_time,
                        ),
                        clarification_request=True,
                        ambiguity_options=opts,
                        temporal_context={
                            "location_name": loc_name,
                            "is_ambiguous": True,
                            "ambiguity_options": opts,
                        },
                        query_understanding=pre_understanding,
                        analysis={
                            "intent": "temporal_geospatial_clarification",
                            "scene_count": 0,
                            "comparison_requested": False,
                        },
                    )

                # B. Determine Location and Temporal Range
                loc_name = pre_understanding.location_name
                if not loc_name or loc_name == "Target Study Area":
                    loc_name = active_ctx.get("location_name") or loc_name
                time_range = dict(pre_understanding.time_range or {})
                if not time_range.get("start_year") and active_ctx.get("start_year"):
                    time_range["start_year"] = active_ctx.get("start_year")
                if not time_range.get("end_year") and active_ctx.get("end_year"):
                    time_range["end_year"] = active_ctx.get("end_year")
                if not time_range and active_ctx.get("time_range"):
                    time_range = dict(active_ctx.get("time_range"))
                req_features = pre_understanding.requested_features or active_ctx.get("requested_features", ["general_change"])

                if not loc_name:
                    exec_time = round((time.time() - start_time) * 1000, 2)
                    trace_steps.append("Zero-upload query missing location specification.")
                    return AgentResponse(
                        answer="Please specify a location or upload satellite imagery to begin your analysis.",
                        task="unknown",
                        evidence={},
                        confidence=0.0,
                        execution_trace=ExecutionTrace(
                            task="unknown",
                            tools_used=[],
                            steps=trace_steps,
                            execution_time_ms=exec_time,
                        ),
                        error="Please specify a location or upload satellite imagery to begin your analysis.",
                        query_understanding=pre_understanding,
                    )

                # C. Check for Explicit Comparison Time Period
                start_yr = time_range.get("start_year")
                end_yr = time_range.get("end_year")
                is_change_or_timeline = (
                    pre_understanding.requires_comparison
                    or pre_understanding.requires_temporal_reasoning
                    or any(w in query.lower() for w in ["change", "changed", "changes", "over time", "evolution", "growth", "expansion", "progress", "differ", "difference", "happened"])
                )

                if (not start_yr or not end_yr) and is_change_or_timeline:
                    exec_time = round((time.time() - start_time) * 1000, 2)
                    trace_steps.append(f"Temporal Query Clarification: Missing explicit time period for {loc_name}.")
                    clarification_answer = (
                        "What time period would you like me to compare?\n\n"
                        "For example:\n"
                        "• 2023 to 2024\n"
                        "• 2024 to 2025\n"
                        "• 2020 to 2025"
                    )
                    return AgentResponse(
                        answer=clarification_answer,
                        task="temporal_date_clarification",
                        evidence={},
                        confidence=1.0,
                        execution_trace=ExecutionTrace(
                            task="temporal_date_clarification",
                            tools_used=["QueryParser"],
                            steps=trace_steps,
                            execution_time_ms=exec_time,
                        ),
                        clarification_request=True,
                        ambiguity_options=["2023 to 2024", "2024 to 2025", "2020 to 2025"],
                        temporal_context={
                            "location_name": loc_name,
                            "needs_date_range": True,
                        },
                        query_understanding=pre_understanding,
                        analysis={
                            "intent": "temporal_date_clarification",
                            "scene_count": 0,
                            "comparison_requested": False,
                        },
                    )

                start_yr = start_yr or 2020
                end_yr = end_yr or 2025
                trace_steps.append(f"Catalog Search: Searching Sentinel-2 cloud-free passes for {loc_name} ({start_yr}-{end_yr})")
                cat_feature = req_features[0] if req_features else "general_change"
                catalog_res = SatelliteDataCatalog.query(
                    location=loc_name,
                    time_range=time_range,
                    feature=cat_feature,
                )

                if not catalog_res.get("found"):
                    exec_time = round((time.time() - start_time) * 1000, 2)
                    msg = catalog_res.get(
                        "message",
                        f"I couldn't retrieve suitable satellite imagery for '{loc_name}' across the requested comparison ({start_yr}-{end_yr})."
                    )
                    trace_steps.append(f"Catalog Refusal: {msg}")
                    return AgentResponse(
                        answer=msg,
                        task="unsupported_temporal_query",
                        evidence={},
                        confidence=0.0,
                        execution_trace=ExecutionTrace(
                            task="unsupported_temporal_query",
                            tools_used=["Sentinel-2-Catalog"],
                            steps=trace_steps,
                            execution_time_ms=exec_time,
                        ),
                        unsupported_analysis=msg,
                        temporal_context=catalog_res.get("temporal_context"),
                        query_understanding=pre_understanding,
                        visualizations=[],
                        masks=[],
                    )

                # D. Retrieved Authentic Observation Scenes
                images = catalog_res["image_paths"]
                temporal_context = catalog_res["temporal_context"]
                timeline = catalog_res.get("timeline")
                observations = catalog_res.get("observations", [])
                retrieved_images = catalog_res.get("image_urls", [])
                trace_steps.append(
                    f"Catalog Match: Retrieved {len(images)} Sentinel-2 L2A scene(s) for {loc_name} "
                    f"(Acquired {temporal_context.get('scene_a_date')} & {temporal_context.get('scene_b_date')}, cloud cover <2%)"
                )

        # 0. Image Understanding & Scene Inspection
        scene_inspection: Optional[ImageSceneInspection] = None
        if images:
            scene_inspection = self.inspector.inspect(images=images, metadata=metadata)
            trace_steps.append(f"Image Understanding: {scene_inspection.inspection_summary}")
            logger.info("Image inspection: %s", scene_inspection.inspection_summary)

        # 1. Semantic Query Understanding & Subtask Decomposition
        query_understanding: QueryUnderstanding = self.query_parser.understand(
            query=query,
            scene_inspection=scene_inspection,
            metadata=metadata,
        )
        if query_understanding.operational_reasoning:
            for r_step in query_understanding.operational_reasoning:
                trace_steps.append(f"Query Understanding: {r_step}")
        else:
            trace_steps.append(f"Query Understanding: Intent classified as '{query_understanding.primary_intent}'")
        logger.info("Query understanding: intent=%s, %d subtask(s)", query_understanding.primary_intent, len(query_understanding.subtasks))

        # 1b. Natural-Language Monitoring Intent Interception
        if query_understanding.is_monitoring:
            from backend.monitoring.models import MonitoringJobRepository
            cond = query_understanding.monitoring_condition or "significant_general_change"
            thresh = query_understanding.monitoring_threshold or 5.0
            unit = query_understanding.monitoring_threshold_unit or "percentage_points"

            area_name = query_understanding.location_name
            baseline_id = None
            baseline_date = None
            baseline_sensor = None
            baseline_img = None

            if images:
                p = Path(images[0])
                area_name = area_name or p.stem
                baseline_img = str(p)
                baseline_id = p.name
                baseline_sensor = sensor_type or "Uploaded Imagery"
            elif not area_name or area_name == "Target Study Area":
                return AgentResponse(
                    answer="To configure continuous satellite monitoring and alerts, please specify a geographic area (e.g. 'in Bengaluru') or upload a baseline satellite image.",
                    task="satellite_monitoring",
                    evidence={},
                    confidence=1.0,
                    execution_trace=ExecutionTrace(
                        task="satellite_monitoring",
                        tools_used=[],
                        steps=trace_steps + ["Monitoring job creation requires a target location or baseline image."],
                        execution_time_ms=round((time.time() - start_time) * 1000, 2),
                    ),
                    clarification_request=True,
                )

            # Auto-resolve baseline if known city
            loc = (area_name or "").strip()
            for k in SatelliteDataCatalog.CATALOG:
                if k.lower() in loc.lower() or loc.lower() in k.lower():
                    loc = k
                    break
            if loc in SatelliteDataCatalog.CATALOG and not baseline_date:
                earliest_yr = min(SatelliteDataCatalog.CATALOG[loc].keys())
                m = SatelliteDataCatalog.CATALOG[loc][earliest_yr]
                baseline_id = f"{loc.lower()}_{earliest_yr}"
                baseline_date = m["date"]
                baseline_sensor = m["sensor"]
                baseline_img = SatelliteDataCatalog._ensure_cache_image(loc, earliest_yr, m)

            job = MonitoringJobRepository.create(
                name=f"{area_name} {cond.replace('_', ' ').title()} Alert",
                area_name=area_name,
                condition_text=query,
                condition_type=cond,
                threshold=thresh,
                threshold_unit=unit,
                sensor_preference="optical_sar" if query_understanding.requires_cross_modal else "auto",
                session_id=metadata.get("session_id"),
                baseline_scene_id=baseline_id,
                baseline_date=baseline_date,
                baseline_sensor=baseline_sensor,
                baseline_image_path=baseline_img,
            )

            human_cond = cond.replace("_", " ").title()
            sensor_str = "Sentinel-2 (Optical) + Sentinel-1 (SAR)" if query_understanding.requires_cross_modal else "Automatic (Sentinel-2 / Sentinel-1)"
            thresh_str = f"{thresh:.0f} {unit}" if unit == "count" else f"{thresh:.1f}% ({unit})"

            answer_msg = (
                f"I have established an active satellite monitoring job for **{area_name}**.\n\n"
                f"• **Condition:** Alert when {human_cond} occurs\n"
                f"• **Trigger Threshold:** {thresh_str}\n"
                f"• **Sensors:** {sensor_str}\n"
                f"• **Reference Baseline:** {baseline_date or 'Uploaded scene'} ({baseline_sensor or 'Sentinel-2 L2A'})\n"
                f"• **Status:** Active (Job ID: `{job['id']}`)\n\n"
                f"SATQuery AI will monitor new satellite passes over {area_name}. When a confirmed change meeting this threshold is detected, an alert with visual evidence will appear in your **Monitoring** dashboard and notification bell."
            )

            trace_steps.append(f"Monitoring Setup: Created persistent monitoring job {job['id']} for {area_name}")
            return AgentResponse(
                answer=answer_msg,
                task="satellite_monitoring",
                evidence={"monitoring_job": job},
                confidence=1.0,
                execution_trace=ExecutionTrace(
                    task="satellite_monitoring",
                    tools_used=["MonitoringEngine"],
                    steps=trace_steps,
                    execution_time_ms=round((time.time() - start_time) * 1000, 2),
                ),
                monitoring_proposal=job,
            )

        # 2. Dynamic Analysis Planning
        analysis_plan: AnalysisPlan = self.planner.plan(
            query_understanding=query_understanding,
            scene_inspection=scene_inspection,
            metadata=metadata,
        )
        trace_steps.append(f"Analysis Plan: {analysis_plan.planner_summary}")
        logger.info("Generated analysis plan: models=%s, tools=%s", analysis_plan.selected_models, analysis_plan.selected_tools)

        # 3. Routing
        routing: RoutingDecision = TaskRouter.route(
            query=query,
            image_count=len(images),
            sensor_type=sensor_type,
            metadata=metadata,
            query_understanding=query_understanding,
            scene_inspection=scene_inspection,
            analysis_plan=analysis_plan,
        )
        trace_steps.append(f"Classified task as '{routing.task.value}': {routing.rationale}")
        logger.info("Task routed to %s: %s", routing.task.value, routing.tools_used)

        # Multi-panel extraction if comparative analysis or optical+sar requested on a single image
        if (routing.task in [TaskType.BI_TEMPORAL_CHANGE, TaskType.CROSS_MODAL_OPTICAL_SAR] or analysis_plan.comparison_requested) and len(images) == 1:
            from backend.agent.modality import CollageDetector
            c_res = CollageDetector.detect_and_split(images[0])
            if c_res and c_res.get("is_collage"):
                images = [c_res["optical_path"], c_res["sar_path"]]
                trace_steps.append(
                    f"Multi-Panel Extraction: Detected and segmented Optical + SAR collage panels from single uploaded image ({c_res.get('split_mode')} split)"
                )
                logger.info("Extracted Optical (%s) and SAR (%s) panels from collage", c_res["optical_path"], c_res["sar_path"])
            elif scene_inspection and scene_inspection.can_separate_panels and scene_inspection.scene_count >= 2:
                try:
                    extracted = self.inspector.extract_panels(images[0], scene_inspection)
                    if len(extracted) >= 2:
                        panel_names = [p.name for p in extracted[:2]]
                        trace_steps.append(
                            f"Multi-Panel Extraction: Extracted {len(extracted)} panels ({', '.join(panel_names)}) for comparative analysis"
                        )
                        logger.info("Extracted %d panels from collage %s", len(extracted), images[0])
                        images = [str(p) for p in extracted[:2]]
                except Exception as e:
                    logger.warning("Failed to extract panels from collage: %s", e)

        # 4. Input Validation
        val_error = self.validate_inputs(
            images=images,
            task=routing.task,
            sensor_type=sensor_type,
            metadata=metadata,
            scene_inspection=scene_inspection,
        )
        if val_error:
            logger.warning("Input validation failed: %s", val_error)
            trace_steps.append(f"Input validation rejected: {val_error}")
            exec_time = round((time.time() - start_time) * 1000, 2)
            return AgentResponse(
                answer=val_error,
                task=routing.task.value,
                evidence={},
                confidence=0.0,
                execution_trace=ExecutionTrace(
                    task=routing.task.value,
                    tools_used=routing.tools_used,
                    steps=trace_steps,
                    execution_time_ms=exec_time,
                ),
                error=val_error,
                scene_inspection=scene_inspection,
                query_understanding=query_understanding,
                analysis_plan=analysis_plan,
                analysis={
                    "intent": query_understanding.primary_intent if query_understanding else "unknown",
                    "scene_count": scene_inspection.scene_count if scene_inspection else len(images),
                    "comparison_requested": analysis_plan.comparison_requested if analysis_plan else False,
                },
                technical={
                    "models_used": routing.tools_used,
                    "execution_time_ms": exec_time,
                },
                success=False,
            )

        # Inspect modality of all provided inputs
        inspected_inputs: List[Dict[str, Any]] = []
        from backend.agent.modality import ModalityInspector, SceneCoLocationChecker
        if images:
            for img_p in images:
                m_info = ModalityInspector.inspect(img_p, metadata=metadata)
                inspected_inputs.append(m_info.to_dict())
                trace_steps.append(f"Modality Inspection: '{Path(img_p).name}' -> {m_info.display_label} ({m_info.reasoning})")

        # Co-location verification for multi-image tasks
        colocation_info = {"performed": False, "same_area_confidence": 1.0}
        if routing.task == TaskType.CROSS_MODAL_OPTICAL_SAR and len(images) >= 2:
            is_coloc, coloc_conf, coloc_reason = SceneCoLocationChecker.check_co_location(images[0], images[1], metadata=metadata)
            colocation_info = {"performed": True, "same_area_confidence": coloc_conf, "reason": coloc_reason}
            trace_steps.append(f"Scene Co-Location: {coloc_reason} (confidence {coloc_conf})")
            if not is_coloc:
                exec_time = round((time.time() - start_time) * 1000, 2)
                explanation = "The two images appear to represent different geographic areas, so I did not combine them. Please provide optical and SAR images covering the same area."
                return AgentResponse(
                    answer=explanation,
                    task=routing.task.value,
                    evidence={},
                    confidence=coloc_conf,
                    execution_trace=ExecutionTrace(
                        task=routing.task.value,
                        tools_used=routing.tools_used,
                        steps=trace_steps,
                        execution_time_ms=exec_time,
                    ),
                    error=explanation,
                    inputs=inspected_inputs,
                    alignment=colocation_info,
                    success=False,
                )

        trace_steps.append(f"Validated {len(images)} input image(s)")

        # 5. Specialist Execution via Dynamic Analysis Executor
        actual_tools_used: List[str] = []
        tool_results, exec_steps = self.executor.execute(
            plan=analysis_plan,
            query_understanding=query_understanding,
            images=images,
            metadata=metadata,
        )
        trace_steps.extend(exec_steps)
        for res in tool_results.values():
            if isinstance(res, dict) and res.get("success") and res.get("model") and res.get("model") not in actual_tools_used:
                actual_tools_used.append(res["model"])

        # 4. Evidence Fusion & Confidence Computation
        fused = EvidenceAggregator.fuse(
            task=routing.task,
            tool_results=tool_results,
            query_understanding=query_understanding,
            analysis_plan=analysis_plan,
        )
        if temporal_context:
            fused["temporal_context"] = temporal_context
        if timeline:
            fused["timeline"] = timeline

        evidence = fused["evidence"]
        confidence = fused["confidence"]
        counts = fused["counts"]
        percentages = fused["percentages"]
        detected_regions = fused["detected_regions"]
        bounding_boxes = fused["bounding_boxes"]
        visualizations = fused["visualizations"]
        masks = fused["masks"]
        why_this_answer = fused["why_this_answer"]
        unsupported_analysis = fused["unsupported_analysis"]

        conf_str = f"{confidence:.4f}" if confidence is not None else "None (no detections above threshold)"
        trace_steps.append(f"Fused multi-source evidence package with empirical confidence {conf_str}")
        for why_item in why_this_answer:
            trace_steps.append(f"Evidence Attribution: {why_item}")

        # 5. RS-Qwen Grounded Reasoning
        if self.vlm.is_available and images:
            try:
                answer = self.vlm.synthesize_response(
                    query=query,
                    task=routing.task,
                    evidence=evidence,
                    image_paths=images,
                    confidence=confidence,
                    fused_data=fused,
                )
                if "RS-Qwen" not in actual_tools_used:
                    actual_tools_used.append("RS-Qwen")
                trace_steps.append("Synthesized grounded natural-language answer with RS-Qwen")
            except Exception as e:
                logger.warning("RS-Qwen synthesis failed (%s); falling back to structured evidence report.", e)
                trace_steps.append(f"RS-Qwen synthesis failed ({e}); final VLM synthesis skipped. Generated structured specialist evidence report.")
                answer = self.vlm._generate_grounded_synthesis(query, routing.task, evidence, confidence, fused_data=fused, image_paths=images)
        else:
            trace_steps.append("RS-Qwen unavailable locally; final VLM synthesis skipped. Generated structured specialist evidence report.")
            answer = self.vlm._generate_grounded_synthesis(query, routing.task, evidence, confidence, fused_data=fused, image_paths=images)

        exec_time = round((time.time() - start_time) * 1000, 2)
        trace_steps.append(f"Agent pipeline finished in {exec_time}ms")

        change_metrics = fused.get("change_metrics", {})
        change_regions = fused.get("change_regions", [])

        # Convert all visualization and mask disk paths into base64 Data URLs for safe browser rendering
        from backend.agent.semantic_vision import path_to_data_url
        data_visualizations = [path_to_data_url(v) or v for v in visualizations if v]
        data_masks = [path_to_data_url(m) or m for m in masks if m]

        # Ensure change_overlay in evidence is also available as data URL
        if "change_detection" in evidence and isinstance(evidence["change_detection"], dict):
            co = evidence["change_detection"].get("change_overlay")
            if co:
                evidence["change_detection"]["change_overlay_url"] = path_to_data_url(co) or co

        # For zero-upload queries with retrieved imagery, ensure visualizations contains data URLs
        if not data_visualizations and retrieved_images:
            data_visualizations = list(retrieved_images)

        return AgentResponse(
            answer=answer,
            task=routing.task.value,
            evidence=evidence,
            confidence=confidence,
            execution_trace=ExecutionTrace(
                task=routing.task.value,
                tools_used=actual_tools_used,
                steps=trace_steps,
                execution_time_ms=exec_time,
            ),
            error=None,
            scene_inspection=scene_inspection,
            query_understanding=query_understanding,
            analysis_plan=analysis_plan,
            counts=counts,
            percentages=percentages,
            detected_regions=detected_regions,
            bounding_boxes=bounding_boxes,
            visualizations=data_visualizations,
            masks=masks,
            why_this_answer=why_this_answer,
            unsupported_analysis=unsupported_analysis,
            change_metrics=change_metrics,
            change_regions=change_regions,
            temporal_context=temporal_context,
            timeline=timeline,
            retrieved_images=retrieved_images,
            observations=observations,
            analysis={
                "intent": query_understanding.primary_intent if query_understanding else "unknown",
                "scene_count": scene_inspection.scene_count if scene_inspection else len(images),
                "comparison_requested": analysis_plan.comparison_requested if analysis_plan else False,
            },
            technical={
                "models_used": actual_tools_used,
                "execution_time_ms": exec_time,
                "change_metrics": change_metrics,
            },
            success=True,
            inputs=inspected_inputs,
            alignment=colocation_info,
            warnings=[],
        )

