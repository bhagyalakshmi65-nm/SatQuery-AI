"""FastAPI endpoints for SatQuery AI Monitoring and Change Alert System."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.monitoring.models import (
    AlertRepository,
    MonitoringJobRepository,
    MonitoringRunRepository,
)
from backend.monitoring.schemas import (
    AlertResponse,
    CheckNowResponse,
    MonitoringJobCreate,
    MonitoringJobResponse,
    MonitoringJobUpdate,
    MonitoringRunResponse,
    UnreadCountResponse,
)

logger = logging.getLogger("backend.monitoring.routes")

router = APIRouter(prefix="/api", tags=["monitoring"])


# -----------------------------------------------------------------------------
# 1. Monitoring Jobs
# -----------------------------------------------------------------------------
@router.post("/monitoring", response_model=MonitoringJobResponse, status_code=201)
def create_monitoring_job(job_in: MonitoringJobCreate):
    """Create a persistent satellite monitoring task."""
    try:
        # If baseline scene is not specified, attempt to resolve from known catalog
        baseline_scene_id = job_in.baseline_scene_id
        baseline_date = job_in.baseline_date
        baseline_sensor = job_in.baseline_sensor
        baseline_img = job_in.baseline_image_path

        if not baseline_date:
            from backend.agent.temporal_geospatial import SatelliteDataCatalog
            loc = job_in.area_name.strip()
            if loc in SatelliteDataCatalog.CATALOG:
                earliest_yr = min(SatelliteDataCatalog.CATALOG[loc].keys())
                m = SatelliteDataCatalog.CATALOG[loc][earliest_yr]
                baseline_scene_id = f"{loc.lower()}_{earliest_yr}"
                baseline_date = m["date"]
                baseline_sensor = m["sensor"]
                baseline_img = SatelliteDataCatalog._ensure_cache_image(loc, earliest_yr, m)

        job = MonitoringJobRepository.create(
            name=job_in.name,
            area_name=job_in.area_name,
            condition_text=job_in.condition_text,
            condition_type=job_in.condition_type,
            area_geometry=job_in.area_geometry,
            threshold=job_in.threshold,
            threshold_unit=job_in.threshold_unit,
            sensor_preference=job_in.sensor_preference,
            session_id=job_in.session_id,
            baseline_scene_id=baseline_scene_id,
            baseline_date=baseline_date,
            baseline_sensor=baseline_sensor,
            baseline_image_path=baseline_img,
            baseline_metadata=job_in.baseline_metadata,
        )
        return job
    except Exception as e:
        logger.error("Failed to create monitoring job: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create monitoring job: {str(e)}")


@router.get("/monitoring", response_model=List[MonitoringJobResponse])
def list_monitoring_jobs(
    status: Optional[str] = Query(None, description="Filter by status: active, paused"),
    session_id: Optional[str] = Query(None, description="Filter by user session"),
):
    """List all configured monitoring jobs."""
    try:
        jobs = MonitoringJobRepository.list_all(status=status, session_id=session_id)
        return jobs
    except Exception as e:
        logger.error("Failed to list monitoring jobs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/{monitor_id}", response_model=MonitoringJobResponse)
def get_monitoring_job(monitor_id: str):
    """Retrieve a single monitoring job with recent audit runs."""
    job = MonitoringJobRepository.get_by_id(monitor_id)
    if not job:
        raise HTTPException(status_code=404, detail="Monitoring job not found")

    # Attach latest 10 runs
    job["latest_runs"] = MonitoringRunRepository.list_by_monitor(monitor_id, limit=10)
    return job


@router.patch("/monitoring/{monitor_id}", response_model=MonitoringJobResponse)
def update_monitoring_job(monitor_id: str, updates: MonitoringJobUpdate):
    """Update monitoring parameters or pause/resume a monitoring task."""
    job = MonitoringJobRepository.get_by_id(monitor_id)
    if not job:
        raise HTTPException(status_code=404, detail="Monitoring job not found")

    patch_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
    updated = MonitoringJobRepository.update(monitor_id, patch_dict)
    return updated


@router.delete("/monitoring/{monitor_id}")
def delete_monitoring_job(monitor_id: str):
    """Delete a monitoring task and cascade its runs and alerts."""
    deleted = MonitoringJobRepository.delete(monitor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Monitoring job not found")
    return {"success": True, "message": f"Monitoring job '{monitor_id}' deleted successfully"}


# -----------------------------------------------------------------------------
# 2. Check Now & Run Tracking
# -----------------------------------------------------------------------------
@router.post("/monitoring/{monitor_id}/check-now", response_model=CheckNowResponse)
def trigger_check_now(monitor_id: str):
    """Asynchronously enqueue an immediate satellite imagery check for the given monitor."""
    job = MonitoringJobRepository.get_by_id(monitor_id)
    if not job:
        raise HTTPException(status_code=404, detail="Monitoring job not found")

    if job.get("status") == "paused":
        raise HTTPException(status_code=400, detail="Cannot check a paused monitoring job. Resume it first.")

    # Create run record in queued state
    run = MonitoringRunRepository.create(
        monitor_id=monitor_id,
        status="queued",
    )

    # Trigger asynchronous background execution
    try:
        from backend.monitoring.worker import enqueue_check
        enqueue_check(run["id"], monitor_id)
    except ImportError:
        logger.warning("Worker module not yet initialized; run queued in DB.")

    return {
        "job_id": run["id"],
        "monitor_id": monitor_id,
        "status": "queued",
        "message": "Monitoring check queued for background analysis.",
    }


@router.get("/monitoring/jobs/{job_id}", response_model=MonitoringRunResponse)
def get_monitoring_run_status(job_id: str):
    """Poll the status of an ongoing or completed monitoring run."""
    run = MonitoringRunRepository.get_by_id(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Monitoring run not found")
    return run


# -----------------------------------------------------------------------------
# 3. Alerts
# -----------------------------------------------------------------------------
@router.get("/alerts", response_model=List[AlertResponse])
def list_alerts(
    unread_only: bool = Query(False, description="Filter for unread alerts only"),
    monitor_id: Optional[str] = Query(None, description="Filter by specific monitor"),
    limit: int = Query(30, ge=1, le=100),
):
    """List generated change alerts with evidence and severity."""
    try:
        alerts = AlertRepository.list_all(unread_only=unread_only, monitor_id=monitor_id, limit=limit)
        # Attach monitor name
        for alt in alerts:
            job = MonitoringJobRepository.get_by_id(alt["monitor_id"])
            alt["monitor_name"] = job["name"] if job else "Monitored Area"
        return alerts
    except Exception as e:
        logger.error("Failed to list alerts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/unread-count", response_model=UnreadCountResponse)
def get_unread_alerts_count():
    """Return count of unread active alerts for bell badges."""
    count = AlertRepository.count_unread()
    return {"unread_count": count}


@router.patch("/alerts/{alert_id}/read", response_model=AlertResponse)
def mark_alert_as_read(alert_id: str):
    """Mark an alert as read."""
    alert = AlertRepository.mark_read(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    job = MonitoringJobRepository.get_by_id(alert["monitor_id"])
    alert["monitor_name"] = job["name"] if job else "Monitored Area"
    return alert


@router.delete("/alerts/{alert_id}")
def dismiss_alert(alert_id: str):
    """Dismiss/delete an alert."""
    deleted = AlertRepository.delete(alert_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True, "message": "Alert dismissed"}
