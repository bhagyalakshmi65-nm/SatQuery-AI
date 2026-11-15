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
    segmentation: Optional[Dict[str, Any]] = None
    gis: Optional[Dict[str, Any]] = None



class SceneLayout(str, Enum):
    """Layout structure of uploaded imagery."""
    SINGLE = "single"
    SIDE_BY_SIDE = "side_by_side"
    VERTICAL_SPLIT = "vertical_split"
    GRID_COLLAGE = "grid_collage"
    MULTI_FILE = "multi_file"
    UNKNOWN = "unknown"


class PanelInfo(BaseModel):
    """Information regarding an individual detected scene or panel."""
    panel_id: str
    file_index: int = 0
    source_file: str
    bbox: List[int] = Field(description="[ymin, xmin, ymax, xmax] in pixel coordinates")
    normalized_bbox: List[float] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0.0-1.0")
    width: int
    height: int
    aspect_ratio: float
    visible_labels: List[str] = Field(default_factory=list)
    detected_date: Optional[str] = None
    display_name: str = Field(default="", description="Display label, e.g. 'Panel A', 'Panel B' or explicit date")
    can_separate: bool = True


class ImageSceneInspection(BaseModel):
    """Structured inspection result describing uploaded imagery prior to tool selection."""
    file_count: int
    scene_count: int
    layout: SceneLayout
    is_remote_sensing: bool
    remote_sensing_confidence: float = 1.0
    remote_sensing_rationale: str = ""
    panels: List[PanelInfo] = Field(default_factory=list)
    dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    visible_labels: List[str] = Field(default_factory=list)
    detected_dates: List[str] = Field(default_factory=list)
    has_temporal_cues: bool = False
    can_separate_panels: bool = False
    available_metadata: Dict[str, Any] = Field(default_factory=dict)
    inspection_summary: str = ""


class QuerySubtask(BaseModel):
    """An atomic operational step in a decomposed query execution plan."""
    step_number: int
    description: str
    operation: str = Field(description="Canonical operation, e.g. detect, segment, classify, distance_calc, filter, count, compare, describe")
    target_entity: Optional[str] = None
    condition: Optional[str] = None
    tool_hint: Optional[str] = None
    operational_reasoning: str = Field(description="Concise operational reason for this subtask")


class QueryUnderstanding(BaseModel):
    """Structured semantic representation and decomposition of an arbitrary user query."""
    raw_query: str
    primary_intent: str = Field(description="Primary classified intent of query")
    entities: List[str] = Field(default_factory=list, description="Extracted physical objects and environmental entities")
    requested_classes: List[str] = Field(default_factory=list)
    requested_measurements: List[str] = Field(default_factory=list, description="e.g. count, percentage, distance, area")
    spatial_relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Distance, containment, directional relations")
    requires_counting: bool = False
    requires_localization: bool = False
    requires_segmentation: bool = False
    requires_classification: bool = False
    requires_comparison: bool = False
    requires_temporal_reasoning: bool = False
    requires_gis: bool = False
    requires_cross_modal: bool = False
    is_compound_query: bool = False
    is_temporal_geospatial: bool = False
    is_monitoring: bool = False
    monitoring_condition: Optional[str] = None
    monitoring_threshold: Optional[float] = None
    monitoring_threshold_unit: Optional[str] = None
    monitoring_proposal: Optional[Dict[str, Any]] = None
    location_name: Optional[str] = None
    coordinates: Optional[tuple] = None
    time_range: Optional[Dict[str, Any]] = None
    temporal_intent: Optional[str] = None
    requested_features: List[str] = Field(default_factory=list)
    is_ambiguous_location: bool = False
    ambiguity_options: List[str] = Field(default_factory=list)
    subtasks: List[QuerySubtask] = Field(default_factory=list)
    operational_reasoning: List[str] = Field(default_factory=list, description="Concise operational justifications")


class CapabilityEntry(BaseModel):
    """Declared capabilities and metadata for a tool or specialist model."""
    name: str
    model_type: str = Field(description="specialist_model, gis_tool, or vlm_foundation")
    capabilities: List[str] = Field(description="List of declared capabilities")
    description: str = ""
    tags: List[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    """Structured plan dynamically generated by the planner selecting required models and tools."""
    intent: str
    scene_count: int = 1
    file_count: int = 1
    subtasks: List[QuerySubtask] = Field(default_factory=list)
    selected_models: List[str] = Field(default_factory=list, description="Specialist models selected (e.g. YOLO11, SegFormer)")
    selected_tools: List[str] = Field(default_factory=list, description="Tools selected (e.g. GIS, RS-Qwen)")
    comparison_requested: bool = False
    requires_gis: bool = False
    operational_reasons: List[str] = Field(default_factory=list)
    planner_summary: str = ""


class AgentResponse(BaseModel):
    """Unified API response returned by SatQuery AI Agent."""
    answer: str
    task: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    execution_trace: ExecutionTrace
    error: Optional[str] = None
    scene_inspection: Optional[ImageSceneInspection] = None
    query_understanding: Optional[QueryUnderstanding] = None
    analysis_plan: Optional[AnalysisPlan] = None
    counts: Dict[str, int] = Field(default_factory=dict, description="Consolidated object or feature counts")
    percentages: Dict[str, float] = Field(default_factory=dict, description="Class area percentages or change ratios")
    detected_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Delineated regions or polygons")
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list, description="Normalized or pixel bounding boxes")
    visualizations: List[str] = Field(default_factory=list, description="Saved overlay/visualization image paths")
    masks: List[str] = Field(default_factory=list, description="Saved segmentation mask or change map paths")
    why_this_answer: List[str] = Field(default_factory=list, description="Concise evidence explanations of why the answer was produced")
    unsupported_analysis: Optional[str] = Field(default=None, description="Clear notice if an inquiry is unsupported by available specialist models")
    change_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Bi-temporal change evaluation metrics")
    change_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Structured change regions [id, bbox, direction, semantic_type, confidence]")
    temporal_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Location, date range, satellite sensor metadata, and observation scenes")
    timeline: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Chronological multi-year progression observations")
    retrieved_images: List[str] = Field(default_factory=list, description="Base64 data URLs or preview URLs of retrieved satellite scenes")
    observations: List[Dict[str, Any]] = Field(default_factory=list, description="Structured satellite observations used in the analysis")
    ambiguity_options: List[str] = Field(default_factory=list, description="Candidate options if a location or intent is ambiguous")
    analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="High-level analysis parameters (intent, scene_count, comparison_requested)")
    technical: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Internal technical telemetry (models_used, execution_time_ms)")
    success: bool = Field(default=True, description="Whether the request succeeded")
    inputs: List[Dict[str, Any]] = Field(default_factory=list, description="Inspected image input metadata with detected modalities and confidence")
    alignment: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Spatial alignment and co-location verification results")
    clarification_request: bool = Field(default=False, description="Whether clarification is requested from the user")
    monitoring_proposal: Optional[Dict[str, Any]] = Field(default=None, description="Structured proposal when query initiates a persistent satellite monitoring task")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or modality uncertainty notices")

