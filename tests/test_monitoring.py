"""Tests for SatQuery AI Smart Satellite Monitoring and Alert System."""

import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.monitoring.db import init_db
from backend.monitoring.models import (
    AlertRepository,
    MonitoringJobRepository,
    MonitoringRunRepository,
)


class TestMonitoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        # Clean test entries
        jobs = MonitoringJobRepository.list_all()
        for j in jobs:
            MonitoringJobRepository.delete(j["id"])

    def test_01_create_monitoring_job(self):
        payload = {
            "name": "Bengaluru Urban Growth Watch",
            "area_name": "Bengaluru",
            "condition_text": "Alert me if new buildings appear here",
            "condition_type": "new_buildings",
            "threshold": 3.0,
            "threshold_unit": "count",
            "sensor_preference": "optical",
        }
        res = self.client.post("/api/monitoring", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("id", data)
        self.assertEqual(data["name"], "Bengaluru Urban Growth Watch")
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["condition_type"], "new_buildings")
        # Should have auto-populated baseline from catalog
        self.assertIsNotNone(data["baseline_date"])

    def test_02_list_and_get_job(self):
        # Create job
        job = MonitoringJobRepository.create(
            name="Hyderabad Flood Alert",
            area_name="Hyderabad",
            condition_text="Alert if flooding occurs",
            condition_type="flooding",
            threshold=5.0,
        )
        res = self.client.get("/api/monitoring")
        self.assertEqual(res.status_code, 200)
        jobs = res.json()
        self.assertGreaterEqual(len(jobs), 1)

        # Get specific
        res_get = self.client.get(f"/api/monitoring/{job['id']}")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["id"], job["id"])
        self.assertIn("latest_runs", res_get.json())

    def test_03_update_and_delete_job(self):
        job = MonitoringJobRepository.create(
            name="Delhi Vegetation Watch",
            area_name="New Delhi",
            condition_text="Alert if vegetation decreases",
            condition_type="vegetation_decrease",
            threshold=10.0,
        )

        # Pause
        res_patch = self.client.patch(
            f"/api/monitoring/{job['id']}",
            json={"status": "paused", "threshold": 12.0},
        )
        self.assertEqual(res_patch.status_code, 200)
        self.assertEqual(res_patch.json()["status"], "paused")
        self.assertEqual(res_patch.json()["threshold"], 12.0)

        # Delete
        res_del = self.client.delete(f"/api/monitoring/{job['id']}")
        self.assertEqual(res_del.status_code, 200)

        res_404 = self.client.get(f"/api/monitoring/{job['id']}")
        self.assertEqual(res_404.status_code, 404)

    def test_04_check_now_flow(self):
        job = MonitoringJobRepository.create(
            name="Active Monitor",
            area_name="Bengaluru",
            condition_text="Alert on change",
            condition_type="significant_general_change",
        )

        res = self.client.post(f"/api/monitoring/{job['id']}/check-now")
        self.assertEqual(res.status_code, 200)
        check_data = res.json()
        self.assertIn("job_id", check_data)
        self.assertEqual(check_data["status"], "queued")

        # Poll status
        run_res = self.client.get(f"/api/monitoring/jobs/{check_data['job_id']}")
        self.assertEqual(run_res.status_code, 200)
        self.assertEqual(run_res.json()["monitor_id"], job["id"])

    def test_05_alerts_crud_and_unread_count(self):
        job = MonitoringJobRepository.create(
            name="Alert Test Monitor",
            area_name="Chennai",
            condition_text="Watch area",
            condition_type="new_buildings",
        )

        # Create alert
        alert = AlertRepository.create(
            monitor_id=job["id"],
            title="New Buildings Detected",
            message="3 new building footprints were confirmed.",
            severity="warning",
            evidence={"new_buildings_count": 3},
        )

        # Unread count should be at least 1
        res_count = self.client.get("/api/alerts/unread-count")
        self.assertEqual(res_count.status_code, 200)
        self.assertGreaterEqual(res_count.json()["unread_count"], 1)

        # List alerts
        res_list = self.client.get("/api/alerts?unread_only=true")
        self.assertEqual(res_list.status_code, 200)
        alerts = res_list.json()
        self.assertTrue(any(a["id"] == alert["id"] for a in alerts))

        # Mark as read
        res_read = self.client.patch(f"/api/alerts/{alert['id']}/read")
        self.assertEqual(res_read.status_code, 200)
        self.assertTrue(res_read.json()["read"])

        # Delete alert
        res_del = self.client.delete(f"/api/alerts/{alert['id']}")
        self.assertEqual(res_del.status_code, 200)

    def test_06_deduplication_skips_same_scene(self):
        # Create a job that has already processed the latest scene
        job = MonitoringJobRepository.create(
            name="Dedup Monitor",
            area_name="Bengaluru",
            condition_text="Alert on change",
            condition_type="significant_general_change",
        )
        from backend.monitoring.satellite_client import SatelliteClient
        from backend.monitoring.worker import _execute_run_task
        latest = SatelliteClient.search_new_scene("Bengaluru")
        self.assertIsNotNone(latest)

        # Mark monitor as having already processed all catalog scenes
        MonitoringJobRepository.update(
            job["id"],
            {"last_scene_id": "bengaluru_2024", "last_scene_date": "2029-12-31"},
        )

        run = MonitoringRunRepository.create(
            monitor_id=job["id"],
        )
        _execute_run_task(run["id"], job["id"])

        updated_run = MonitoringRunRepository.get_by_id(run["id"])
        self.assertEqual(updated_run["status"], "skipped")
        self.assertIn("deduplication", updated_run.get("error", "").lower())

    def test_07_condition_evaluator_new_buildings(self):
        import tempfile
        from PIL import Image
        from backend.monitoring.condition_evaluator import AlertConditionEvaluator

        # Create synthetic images saved to disk
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            img1 = Image.new("RGB", (128, 128), color=(100, 100, 100))
            img2 = Image.new("RGB", (128, 128), color=(120, 120, 120))
            img1.save(f1.name)
            img2.save(f2.name)

            evaluator = AlertConditionEvaluator()
            result = evaluator.evaluate(
                condition_type="new_buildings",
                threshold=1.0,
                threshold_unit="count",
                baseline_image=f1.name,
                latest_image=f2.name,
                baseline_meta={"building_count_estimate": 10},
                latest_meta={"building_count_estimate": 15},
                area_name="Test Area",
            )
            self.assertTrue(result.satisfied)
            self.assertIn("new_buildings_count", result.evidence)
            self.assertEqual(result.evidence["new_buildings_count"], 5)

    def test_08_condition_evaluator_vegetation_decrease(self):
        import tempfile
        from PIL import Image
        from backend.monitoring.condition_evaluator import AlertConditionEvaluator

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            img1 = Image.new("RGB", (128, 128), color=(40, 180, 40))
            img2 = Image.new("RGB", (128, 128), color=(140, 100, 40))
            img1.save(f1.name)
            img2.save(f2.name)

            evaluator = AlertConditionEvaluator()
            result = evaluator.evaluate(
                condition_type="vegetation_decrease",
                threshold=5.0,
                threshold_unit="percentage_points",
                baseline_image=f1.name,
                latest_image=f2.name,
                baseline_meta={"veg_index": 0.65},
                latest_meta={"veg_index": 0.40},
                area_name="Test Forest",
            )
            self.assertTrue(result.satisfied)
            self.assertIn("Vegetation Loss", result.title)

    def test_09_agent_monitoring_intent_parsing(self):
        from backend.agent.query_parser import SemanticQueryParser
        parser = SemanticQueryParser()
        intent = parser.understand("Alert me if new buildings appear in Bengaluru East")
        self.assertTrue(intent.is_monitoring)
        self.assertEqual(intent.monitoring_condition, "new_buildings")

    def test_10_agent_monitoring_creation_flow(self):
        from backend.agent.agent import SatQueryAgent
        agent = SatQueryAgent()
        response = agent.analyze(
            query="Alert me if new buildings appear in Bengaluru",
            images=[],
            metadata={"session_id": "test_mon_session"},
        )
        self.assertIn("monitoring job", response.answer.lower())
        self.assertIsNotNone(response.monitoring_proposal)
        self.assertEqual(response.monitoring_proposal["condition_type"], "new_buildings")

        # Verify it was saved to DB
        jobs = MonitoringJobRepository.list_all(session_id="test_mon_session")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["condition_type"], "new_buildings")


if __name__ == "__main__":
    unittest.main()

