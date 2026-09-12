"""Pydantic schemas and data models for SatQuery AI Agent & Orchestration layer."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported task types in SatQuery AI."""
    SINGLE_IMAGE_DESCRIPTION = "single_image_description"
    SINGLE_IMAGE_OBJECT_DETECTION = "single_image_object_detection"
    SINGLE_IMAGE_LANDCOVER = "single_image_landcover"
    BI_TEMPORAL_CHANGE = "bi_temporal_change"
    CROSS_MODAL_OPTICAL_SAR = "cross_modal_optical_sar"


class RoutingDecision(BaseModel):
    """Deterministic routing decision output."""
    task: TaskType
    tools_used: List[str]
    rationale: str
    extracted_intent: str


class ExecutionTrace(BaseModel):
    """Auditable execution trace containing tools and sequential steps."""
    task: str
    tools_used: List[str]
    steps: List[str]
    execution_time_ms: Optional[float] = None


class StructuredEvidence(BaseModel):
    """Aggregated structured empirical evidence from specialist models."""
    object_detection: Optional[Dict[str, Any]] = None
    change_detection: Optional[Dict[str, Any]] = None
    landcover: Optional[Dict[str, Any]] = None
    optical_sar: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Unified API response returned by SatQuery AI Agent."""
    answer: str
    task: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    execution_trace: ExecutionTrace
    error: Optional[str] = None

