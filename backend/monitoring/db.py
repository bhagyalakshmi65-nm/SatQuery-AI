"""Database management for SatQuery AI Monitoring and Change Alert System.

Provides SQLite persistence for:
- monitoring_jobs: Persistent monitoring tasks with conditions and area geometries.
- monitoring_runs: Historical audit logs of satellite scene evaluations.
- alerts: Grounded change alerts with visual evidence overlays.
"""

import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger("backend.monitoring.db")

DB_DIR = Path("backend/data")
DB_PATH = DB_DIR / "satquery_monitoring.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish a thread-safe connection to the SQLite database with dict row factory."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Create monitoring tables and indices if they do not already exist."""
    conn = get_db_connection()
    try:
        with conn:
            # 1. Monitoring Jobs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    name TEXT NOT NULL,
                    area_name TEXT NOT NULL,
                    area_geometry TEXT,
                    condition_text TEXT NOT NULL,
                    condition_type TEXT NOT NULL,
                    threshold REAL DEFAULT 5.0,
                    threshold_unit TEXT DEFAULT 'percentage_points',
                    sensor_preference TEXT DEFAULT 'auto',
                    baseline_scene_id TEXT,
                    baseline_date TEXT,
                    baseline_sensor TEXT,
                    baseline_image_path TEXT,
                    baseline_metadata TEXT,
                    last_scene_id TEXT,
                    last_scene_date TEXT,
                    last_checked_at TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 2. Monitoring Runs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    scene_id TEXT,
                    scene_date TEXT,
                    sensor TEXT,
                    status TEXT DEFAULT 'queued',
                    analysis_result TEXT,
                    condition_result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (monitor_id) REFERENCES monitoring_jobs (id) ON DELETE CASCADE
                );
            """)

            # 3. Alerts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    run_id TEXT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    evidence TEXT,
                    read INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (monitor_id) REFERENCES monitoring_jobs (id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id) REFERENCES monitoring_runs (id) ON DELETE SET NULL
                );
            """)

            # Indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_status ON monitoring_jobs (status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_monitor_id ON monitoring_runs (monitor_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON monitoring_runs (status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts (read);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_monitor_id ON alerts (monitor_id);")

        logger.info("Initialized monitoring database at %s", DB_PATH)
    finally:
        conn.close()
