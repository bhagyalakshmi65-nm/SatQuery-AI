"""Core SatQuery AI Agent and Orchestrator."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from backend.agent.evidence import EvidenceAggregator
from backend.agent.registry import TOOL_REGISTRY, get_tool
from backend.agent.router import TaskRouter
from backend.agent.schemas import AgentResponse, ExecutionTrace, RoutingDecision, TaskType
from backend.agent.vlm import RSQwenEngine

logger = logging.getLogger("agent.orchestrator")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".pt", ".npz"}


class SatQueryAgent:
    """Central AI Agent for SatQuery AI remote sensing analysis."""

    def __init__(self, vlm_engine: Optional[RSQwenEngine] = None):
        self.vlm = vlm_engine or RSQwenEngine()

    def validate_inputs(
        self,
        images: List[Union[str, Path]],
        task: TaskType,
        sensor_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        # 1. Bi-Temporal Change Detection requires exactly two images
        if task == TaskType.BI_TEMPORAL_CHANGE:
            if len(images) != 2:
                return (
                    f"Bi-temporal change detection requires exactly two images (earlier and later dates). "
                    f"Received {len(images)} image(s)."
                )

        # 2. Optical + SAR Cross-Modal analysis requires both modalities
        if task == TaskType.CROSS_MODAL_OPTICAL_SAR:
            has_sar_file = False
            # Check metadata or filenames
            sar_path = metadata.get("sar_image") or metadata.get("sar_path")
            if sar_path or len(images) >= 2:
                has_sar_file = True
            elif len(images) == 1:
                # If a single file is provided, check if it contains 14 channels
                name_lower = Path(images[0]).name.lower()
                if "14" in name_lower or "dfc" in name_lower or "croma" in name_lower:
                    has_sar_file = True

            if not has_sar_file:
                return (
                    "Optical + SAR cross-modal analysis requires both Optical (Sentinel-2) and "
                    "SAR (Sentinel-1) inputs. SAR input is missing."
                )

        # 3. Single-image tasks require at least one image
        if task in [TaskType.SINGLE_IMAGE_DESCRIPTION, TaskType.SINGLE_IMAGE_OBJECT_DETECTION, TaskType.SINGLE_IMAGE_LANDCOVER]:
            if len(images) < 1:
                return f"Task '{task.value}' requires at least one satellite image."

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

        # 1. Routing
        routing: RoutingDecision = TaskRouter.route(
            query=query,
            image_count=len(images),
            sensor_type=sensor_type,
            metadata=metadata,
        )
        trace_steps.append(f"Classified task as '{routing.task.value}': {routing.rationale}")
        logger.info("Task routed to %s: %s", routing.task.value, routing.tools_used)

        # 2. Input Validation
        val_error = self.validate_inputs(
            images=images,
            task=routing.task,
            sensor_type=sensor_type,
            metadata=metadata,
        )
        if val_error:
            logger.warning("Input validation failed: %s", val_error)
            trace_steps.append(f"Input validation rejected: {val_error}")
            exec_time = round((time.time() - start_time) * 1000, 2)
            return AgentResponse(
                answer=f"Input Error: {val_error}",
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
            )

        trace_steps.append(f"Validated {len(images)} input image(s)")

        # 3. Specialist Execution
        tool_results: Dict[str, Any] = {}
        actual_tools_used: List[str] = []

        if routing.task == TaskType.SINGLE_IMAGE_OBJECT_DETECTION:
            yolo_tool = get_tool("yolo")
            if yolo_tool:
                actual_tools_used.append(yolo_tool.model_id)
                trace_steps.append(f"Executing specialist model: {yolo_tool.model_id}")
                t0 = time.time()
                tool_results["yolo"] = yolo_tool.execute(images[0])
                trace_steps.append(f"Completed {yolo_tool.model_id} in {round((time.time() - t0)*1000, 1)}ms")

        elif routing.task == TaskType.BI_TEMPORAL_CHANGE:
            cd_tool = get_tool("change_detection")
            if cd_tool:
                actual_tools_used.append(cd_tool.model_id)
                trace_steps.append(f"Executing specialist model: {cd_tool.model_id}")
                t0 = time.time()
                tool_results["change_detection"] = cd_tool.execute(images[0], images[1])
                trace_steps.append(f"Completed {cd_tool.model_id} in {round((time.time() - t0)*1000, 1)}ms")

        elif routing.task == TaskType.SINGLE_IMAGE_LANDCOVER:
            lc_tool = get_tool("landcover")
            if lc_tool:
                actual_tools_used.append(lc_tool.model_id)
                trace_steps.append(f"Executing specialist model: {lc_tool.model_id}")
                t0 = time.time()
                tool_results["landcover"] = lc_tool.execute(images[0])
                trace_steps.append(f"Completed {lc_tool.model_id} in {round((time.time() - t0)*1000, 1)}ms")

        elif routing.task == TaskType.CROSS_MODAL_OPTICAL_SAR:
            croma_tool = get_tool("croma")
            if croma_tool:
                actual_tools_used.append(croma_tool.model_id)
                trace_steps.append(f"Executing specialist model: {croma_tool.model_id}")
                t0 = time.time()
                if len(images) >= 2:
                    tool_results["croma"] = croma_tool.execute(optical=images[0], sar=images[1])
                elif metadata.get("sar_image"):
                    tool_results["croma"] = croma_tool.execute(optical=images[0], sar=metadata["sar_image"])
                else:
                    # Single 14-channel file
                    tool_results["croma"] = croma_tool.execute(optical=images[0], sar=None)
                trace_steps.append(f"Completed {croma_tool.model_id} in {round((time.time() - t0)*1000, 1)}ms")

        # 4. Evidence Aggregation & Confidence Computation
        evidence, confidence = EvidenceAggregator.aggregate(routing.task, tool_results)
        conf_str = f"{confidence:.4f}" if confidence is not None else "None (no detections above threshold)"
        trace_steps.append(f"Aggregated structured evidence with empirical confidence {conf_str}")

        # 5. RS-Qwen Grounded Reasoning
        if self.vlm.is_available and images:
            try:
                answer = self.vlm.synthesize_response(
                    query=query,
                    task=routing.task,
                    evidence=evidence,
                    image_paths=images,
                    confidence=confidence,
                )
                actual_tools_used.append("RS-Qwen")
                trace_steps.append("Synthesized grounded natural-language answer with RS-Qwen")
            except Exception as e:
                logger.warning("RS-Qwen synthesis failed (%s); falling back to structured evidence report.", e)
                trace_steps.append(f"RS-Qwen synthesis failed ({e}); final VLM synthesis skipped. Generated structured specialist evidence report.")
                answer = self.vlm._generate_grounded_synthesis(query, routing.task, evidence, confidence)
        else:
            trace_steps.append("RS-Qwen unavailable locally; final VLM synthesis skipped. Generated structured specialist evidence report.")
            answer = self.vlm._generate_grounded_synthesis(query, routing.task, evidence, confidence)

        exec_time = round((time.time() - start_time) * 1000, 2)
        trace_steps.append(f"Agent pipeline finished in {exec_time}ms")

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
        )

