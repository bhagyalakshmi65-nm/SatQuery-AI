"""Pydantic schemas for the SatQuery AI Monitoring API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MonitoringJobCreate(BaseModel):
    name: str = Field(..., description="Human-readable title for the monitor")
    area_name: str = Field(..., description="Target area name or place identifier")
    condition_text: str = Field(..., description="Natural language alert condition")
    condition_type: str = Field(
        ...,
        description="Structured condition: new_buildings, vegetation_decrease, flooding, new_road, landcover_change, optical_sar_change, significant_general_change"
    )
    area_geometry: Optional[Dict[str, Any]] = Field(default=None, description="GeoJSON geometry or bounding box")
    threshold: float = Field(default=5.0, description="Numerical trigger threshold")
    threshold_unit: str = Field(default="percentage_points", description="e.g. percentage_points, count, area_m2")
    sensor_preference: str = Field(default="auto", description="auto, optical, sar, optical_sar")
    session_id: Optional[str] = None
    baseline_scene_id: Optional[str] = None
    baseline_date: Optional[str] = None
    baseline_sensor: Optional[str] = None
    baseline_image_path: Optional[str] = None
    baseline_metadata: Optional[Dict[str, Any]] = None


class MonitoringJobUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None  # active, paused
    threshold: Optional[float] = None
    threshold_unit: Optional[str] = None
    sensor_preference: Optional[str] = None


class MonitoringJobResponse(BaseModel):
    id: str
    session_id: Optional[str] = None
    name: str
    area_name: str
    area_geometry: Optional[Dict[str, Any]] = None
    condition_text: str
    condition_type: str
    threshold: float
    threshold_unit: str
    sensor_preference: str
    baseline_scene_id: Optional[str] = None
    baseline_date: Optional[str] = None
    baseline_sensor: Optional[str] = None
    baseline_image_path: Optional[str] = None
    baseline_metadata: Optional[Dict[str, Any]] = None
    last_scene_id: Optional[str] = None
    last_scene_date: Optional[str] = None
    last_checked_at: Optional[str] = None
    status: str
    created_at: str
    updated_at: str
    latest_runs: Optional[List[Dict[str, Any]]] = None


class MonitoringRunResponse(BaseModel):
    id: str
    monitor_id: str
    scene_id: Optional[str] = None
    scene_date: Optional[str] = None
    sensor: Optional[str] = None
    status: str  # queued, running, completed, failed, no_new_scene, change_not_confirmed, alert_triggered
    analysis_result: Optional[Dict[str, Any]] = None
    condition_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class CheckNowResponse(BaseModel):
    job_id: str = Field(..., description="Monitoring run tracking ID")
    monitor_id: str
    status: str = Field(default="queued")
    message: str = "Monitoring check queued for background analysis."


class AlertResponse(BaseModel):
    id: str
    monitor_id: str
    run_id: Optional[str] = None
    title: str
    message: str
    severity: str
    evidence: Optional[Dict[str, Any]] = None
    read: bool
    status: str
    created_at: str
    monitor_name: Optional[str] = None


class UnreadCountResponse(BaseModel):
    unread_count: int
