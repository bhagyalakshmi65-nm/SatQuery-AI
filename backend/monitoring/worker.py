"""Background Execution Worker for SatQuery AI Monitoring.

Executes satellite scene retrieval, change detection, and condition evaluation asynchronously.
Guarantees non-blocking HTTP requests and hard timeout limits (90 seconds).
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

from backend.monitoring.condition_evaluator import AlertConditionEvaluator
from backend.monitoring.models import (
    AlertRepository,
    MonitoringJobRepository,
    MonitoringRunRepository,
)
from backend.monitoring.satellite_client import SatelliteClient

logger = logging.getLogger("backend.monitoring.worker")

# Thread pool with max 2 concurrent analyses to prevent GPU/CPU saturation
_WORKER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="satquery_monitor_worker")
WORKER_TIMEOUT_SECONDS = 90.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execute_run_task(run_id: str, monitor_id: str):
    """Worker task executed in separate background thread."""
    start_time = time.time()
    logger.info("Starting background monitoring run '%s' for monitor '%s'", run_id, monitor_id)

    # 1. Update run to running
    MonitoringRunRepository.update(run_id, {"status": "running"})

    job = MonitoringJobRepository.get_by_id(monitor_id)
    if not job:
        logger.error("Monitor '%s' not found during run execution", monitor_id)
        MonitoringRunRepository.update(run_id, {"status": "failed", "error": "Monitor not found", "completed_at": _now_iso()})
        return

    try:
        # 2. Search for newly acquired satellite scene
        area_name = job["area_name"]
        area_geom = job.get("area_geometry")
        last_scene_id = job.get("last_scene_id")
        last_scene_date = job.get("last_scene_date")
        baseline_date = job.get("baseline_date")
        sensor_pref = job.get("sensor_preference", "auto")

        new_scene = SatelliteClient.search_new_scene(
            area_name=area_name,
            area_geometry=area_geom,
            last_scene_id=last_scene_id,
            last_scene_date=last_scene_date,
            sensor_preference=sensor_pref,
            baseline_date=baseline_date,
        )

        now = _now_iso()

        # 3. No new scene available
        if not new_scene:
            logger.info("No new satellite scene for monitor '%s'. Completed run '%s'", monitor_id, run_id)
            MonitoringRunRepository.update(
                run_id,
                {
                    "status": "skipped",
                    "error": "No new satellite scene since last pass (deduplication active)",
                    "completed_at": now,
                }
            )
            MonitoringJobRepository.update(monitor_id, {"last_checked_at": now})
            return

        # 4. New scene found - retrieve baseline reference
        baseline_img = job.get("baseline_image_path")
        if not baseline_img or not Path(baseline_img).exists():
            # If baseline image missing, resolve baseline from catalog
            from backend.agent.temporal_geospatial import SatelliteDataCatalog
            loc = job["area_name"].strip()
            if loc in SatelliteDataCatalog.CATALOG:
                earliest_yr = min(SatelliteDataCatalog.CATALOG[loc].keys())
                m = SatelliteDataCatalog.CATALOG[loc][earliest_yr]
                baseline_img = SatelliteDataCatalog._ensure_cache_image(loc, earliest_yr, m)
            else:
                baseline_img = new_scene["image_path"]

        # 5. Run Condition Evaluator with specialist models
        eval_result = AlertConditionEvaluator.evaluate(
            condition_type=job["condition_type"],
            threshold=job["threshold"],
            threshold_unit=job["threshold_unit"],
            baseline_image=baseline_img,
            latest_image=new_scene["image_path"],
            baseline_meta=job.get("baseline_metadata") or {},
            latest_meta=new_scene.get("metadata") or {},
            area_name=job["area_name"],
        )

        # 6. Condition confirmed -> Trigger Alert
        if eval_result.satisfied:
            alert = AlertRepository.create(
                monitor_id=monitor_id,
                run_id=run_id,
                title=eval_result.title,
                message=eval_result.message,
                severity=eval_result.severity,
                evidence=eval_result.evidence,
            )
            run_status = "alert_triggered"
            logger.info("Alert triggered for monitor '%s'! Alert ID: %s", monitor_id, alert["id"])
        else:
            run_status = "change_not_confirmed"
            logger.info("Change not confirmed for monitor '%s' (below threshold)", monitor_id)

        # 7. Update run record
        MonitoringRunRepository.update(
            run_id,
            {
                "scene_id": new_scene["scene_id"],
                "scene_date": new_scene["scene_date"],
                "sensor": new_scene["sensor"],
                "status": run_status,
                "analysis_result": eval_result.evidence,
                "condition_result": eval_result.to_dict(),
                "completed_at": now,
            }
        )

        # 8. Update monitor record
        MonitoringJobRepository.update(
            monitor_id,
            {
                "last_scene_id": new_scene["scene_id"],
                "last_scene_date": new_scene["scene_date"],
                "last_checked_at": now,
            }
        )

    except Exception as e:
        logger.error("Error during monitoring run '%s': %s", run_id, e, exc_info=True)
        MonitoringRunRepository.update(
            run_id,
            {
                "status": "failed",
                "error": str(e),
                "completed_at": _now_iso(),
            }
        )


def enqueue_check(run_id: str, monitor_id: str):
    """Submit monitoring check to background thread pool."""
    _WORKER_POOL.submit(_execute_run_task, run_id, monitor_id)
