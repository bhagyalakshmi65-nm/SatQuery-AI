"""Data models and database repository for SatQuery AI Monitoring System."""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from backend.monitoring.db import get_db_connection

logger = logging.getLogger("backend.monitoring.models")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    def _default(o):
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=_default)


class MonitoringJobRepository:
    """CRUD repository for monitoring jobs."""

    @staticmethod
    def create(
        name: str,
        area_name: str,
        condition_text: str,
        condition_type: str,
        area_geometry: Optional[Dict[str, Any]] = None,
        threshold: float = 5.0,
        threshold_unit: str = "percentage_points",
        sensor_preference: str = "auto",
        session_id: Optional[str] = None,
        baseline_scene_id: Optional[str] = None,
        baseline_date: Optional[str] = None,
        baseline_sensor: Optional[str] = None,
        baseline_image_path: Optional[str] = None,
        baseline_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job_id = f"mon_{uuid.uuid4().hex[:10]}"
        now = _now_iso()

        geom_str = _json_dumps(area_geometry) if area_geometry else None
        meta_str = _json_dumps(baseline_metadata) if baseline_metadata else None

        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO monitoring_jobs (
                        id, session_id, name, area_name, area_geometry,
                        condition_text, condition_type, threshold, threshold_unit,
                        sensor_preference, baseline_scene_id, baseline_date,
                        baseline_sensor, baseline_image_path, baseline_metadata,
                        last_scene_id, last_scene_date, last_checked_at,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, session_id, name, area_name, geom_str,
                        condition_text, condition_type, threshold, threshold_unit,
                        sensor_preference, baseline_scene_id, baseline_date,
                        baseline_sensor, baseline_image_path, meta_str,
                        baseline_scene_id, baseline_date, None,
                        "active", now, now
                    )
                )
            return MonitoringJobRepository.get_by_id(job_id)
        finally:
            conn.close()

    @staticmethod
    def get_by_id(job_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT * FROM monitoring_jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("area_geometry"):
                try:
                    data["area_geometry"] = json.loads(data["area_geometry"])
                except Exception:
                    pass
            if data.get("baseline_metadata"):
                try:
                    data["baseline_metadata"] = json.loads(data["baseline_metadata"])
                except Exception:
                    pass
            return data
        finally:
            conn.close()

    @staticmethod
    def list_all(status: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            query = "SELECT * FROM monitoring_jobs"
            params = []
            conditions = []
            if status:
                conditions.append("status = ?")
                params.append(status)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC"

            cur = conn.execute(query, params)
            results = []
            for row in cur.fetchall():
                item = dict(row)
                if item.get("area_geometry"):
                    try:
                        item["area_geometry"] = json.loads(item["area_geometry"])
                    except Exception:
                        pass
                if item.get("baseline_metadata"):
                    try:
                        item["baseline_metadata"] = json.loads(item["baseline_metadata"])
                    except Exception:
                        pass
                results.append(item)
            return results
        finally:
            conn.close()

    @staticmethod
    def update(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = [
            "name", "status", "threshold", "threshold_unit", "sensor_preference",
            "last_scene_id", "last_scene_date", "last_checked_at"
        ]
        fields = []
        params = []
        for k, v in updates.items():
            if k in allowed:
                fields.append(f"{k} = ?")
                params.append(v)

        if not fields:
            return MonitoringJobRepository.get_by_id(job_id)

        fields.append("updated_at = ?")
        params.append(_now_iso())
        params.append(job_id)

        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    f"UPDATE monitoring_jobs SET {', '.join(fields)} WHERE id = ?",
                    params
                )
            return MonitoringJobRepository.get_by_id(job_id)
        finally:
            conn.close()

    @staticmethod
    def delete(job_id: str) -> bool:
        conn = get_db_connection()
        try:
            with conn:
                cur = conn.execute("DELETE FROM monitoring_jobs WHERE id = ?", (job_id,))
                return cur.rowcount > 0
        finally:
            conn.close()


class MonitoringRunRepository:
    """CRUD repository for audit runs."""

    @staticmethod
    def create(
        monitor_id: str,
        scene_id: Optional[str] = None,
        scene_date: Optional[str] = None,
        sensor: Optional[str] = None,
        status: str = "queued",
    ) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        now = _now_iso()

        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO monitoring_runs (
                        id, monitor_id, scene_id, scene_date, sensor,
                        status, analysis_result, condition_result, error,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, monitor_id, scene_id, scene_date, sensor, status, None, None, None, now, None)
                )
            return MonitoringRunRepository.get_by_id(run_id)
        finally:
            conn.close()

    @staticmethod
    def get_by_id(run_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT * FROM monitoring_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("analysis_result"):
                try:
                    data["analysis_result"] = json.loads(data["analysis_result"])
                except Exception:
                    pass
            if data.get("condition_result"):
                try:
                    data["condition_result"] = json.loads(data["condition_result"])
                except Exception:
                    pass
            return data
        finally:
            conn.close()

    @staticmethod
    def list_by_monitor(monitor_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM monitoring_runs WHERE monitor_id = ? ORDER BY created_at DESC LIMIT ?",
                (monitor_id, limit)
            )
            runs = []
            for row in cur.fetchall():
                item = dict(row)
                if item.get("analysis_result"):
                    try:
                        item["analysis_result"] = json.loads(item["analysis_result"])
                    except Exception:
                        pass
                if item.get("condition_result"):
                    try:
                        item["condition_result"] = json.loads(item["condition_result"])
                    except Exception:
                        pass
                runs.append(item)
            return runs
        finally:
            conn.close()

    @staticmethod
    def update(run_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = [
            "scene_id", "scene_date", "sensor", "status",
            "analysis_result", "condition_result", "error", "completed_at"
        ]
        fields = []
        params = []
        for k, v in updates.items():
            if k in allowed:
                fields.append(f"{k} = ?")
                if k in ["analysis_result", "condition_result"] and isinstance(v, (dict, list)):
                    params.append(_json_dumps(v))
                else:
                    params.append(v)

        if not fields:
            return MonitoringRunRepository.get_by_id(run_id)

        params.append(run_id)
        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    f"UPDATE monitoring_runs SET {', '.join(fields)} WHERE id = ?",
                    params
                )
            return MonitoringRunRepository.get_by_id(run_id)
        finally:
            conn.close()


class AlertRepository:
    """CRUD repository for in-app alerts."""

    @staticmethod
    def create(
        monitor_id: str,
        title: str,
        message: str,
        run_id: Optional[str] = None,
        severity: str = "warning",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        alert_id = f"alt_{uuid.uuid4().hex[:10]}"
        now = _now_iso()
        ev_str = _json_dumps(evidence) if evidence else None

        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO alerts (
                        id, monitor_id, run_id, title, message,
                        severity, evidence, read, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', ?)
                    """,
                    (alert_id, monitor_id, run_id, title, message, severity, ev_str, now)
                )
            return AlertRepository.get_by_id(alert_id)
        finally:
            conn.close()

    @staticmethod
    def get_by_id(alert_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            data["read"] = bool(data["read"])
            if data.get("evidence"):
                try:
                    data["evidence"] = json.loads(data["evidence"])
                except Exception:
                    pass
            return data
        finally:
            conn.close()

    @staticmethod
    def list_all(
        unread_only: bool = False,
        monitor_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            query = "SELECT * FROM alerts"
            conditions = []
            params = []
            if unread_only:
                conditions.append("read = 0")
            if monitor_id:
                conditions.append("monitor_id = ?")
                params.append(monitor_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, params)
            alerts = []
            for row in cur.fetchall():
                item = dict(row)
                item["read"] = bool(item["read"])
                if item.get("evidence"):
                    try:
                        item["evidence"] = json.loads(item["evidence"])
                    except Exception:
                        pass
                alerts.append(item)
            return alerts
        finally:
            conn.close()

    @staticmethod
    def count_unread() -> int:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM alerts WHERE read = 0 AND status = 'active'")
            return cur.fetchone()[0]
        finally:
            conn.close()

    @staticmethod
    def mark_read(alert_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("UPDATE alerts SET read = 1 WHERE id = ?", (alert_id,))
            return AlertRepository.get_by_id(alert_id)
        finally:
            conn.close()

    @staticmethod
    def delete(alert_id: str) -> bool:
        conn = get_db_connection()
        try:
            with conn:
                cur = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                return cur.rowcount > 0
        finally:
            conn.close()
