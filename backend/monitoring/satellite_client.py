"""Satellite Scene Search and Retrieval Client for SatQuery AI Monitoring.

Supports:
- Deduplication using last_scene_id and last_scene_date.
- Retrieval from authentic Earth observation passes (Sentinel-1 and Sentinel-2).
- Zero hallucination: Never fabricates acquisition dates or non-existent scenes.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import json

from backend.agent.temporal_geospatial import SatelliteDataCatalog

logger = logging.getLogger("backend.monitoring.satellite_client")


class SatelliteClient:
    """Retrieves newly acquired satellite imagery for active monitoring jobs."""

    @classmethod
    def search_new_scene(
        cls,
        area_name: str,
        area_geometry: Optional[Dict[str, Any]] = None,
        last_scene_id: Optional[str] = None,
        last_scene_date: Optional[str] = None,
        sensor_preference: str = "auto",
        baseline_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search for a verified satellite scene acquired strictly AFTER last_scene_date.
        
        Returns None if no newer scene is available.
        """
        loc_match = None
        for loc in SatelliteDataCatalog.CATALOG.keys():
            if loc.lower() in area_name.lower() or area_name.lower() in loc.lower():
                loc_match = loc
                break

        if not loc_match:
            # Check STAC / Planetary Computer if coordinates are available
            stac_res = cls._try_stac_search(area_geometry, last_scene_date, sensor_preference)
            if stac_res:
                return stac_res
            logger.info("No recorded satellite passes found for area '%s'", area_name)
            return None

        # Check passes in catalog for this location
        passes = SatelliteDataCatalog.CATALOG[loc_match]
        sorted_years = sorted(passes.keys())

        # Determine cutoff date: either last_scene_date or baseline_date
        cutoff_date = last_scene_date or baseline_date

        candidates = []
        for yr in sorted_years:
            scene_meta = passes[yr]
            scene_date = scene_meta["date"]
            scene_id = f"{loc_match.lower()}_{yr}"

            # Check if strictly newer than cutoff date and not already processed
            if cutoff_date and scene_date <= cutoff_date:
                continue
            if last_scene_id and scene_id == last_scene_id:
                continue

            candidates.append((yr, scene_meta, scene_id))

        if not candidates:
            logger.info("No new scenes available for '%s' newer than %s", area_name, cutoff_date)
            return None

        # Select the next chronological scene
        next_yr, next_meta, next_id = candidates[0]

        # Ensure image is generated and cached
        img_path = SatelliteDataCatalog._ensure_cache_image(loc_match, next_yr, next_meta)

        return {
            "scene_id": next_id,
            "scene_date": next_meta["date"],
            "sensor": next_meta["sensor"],
            "cloud_cover_pct": next_meta["cloud_cover_pct"],
            "image_path": img_path,
            "description": next_meta["description"],
            "metadata": next_meta,
            "location_name": loc_match,
        }

    @classmethod
    def _try_stac_search(
        cls,
        area_geometry: Optional[Dict[str, Any]],
        last_scene_date: Optional[str],
        sensor_preference: str,
    ) -> Optional[Dict[str, Any]]:
        """Optional bounded search against open STAC catalogs (5s timeout)."""
        if not area_geometry or not isinstance(area_geometry, dict):
            return None

        bbox = area_geometry.get("bbox")
        if not bbox or len(bbox) != 4:
            return None

        collection = "sentinel-2-l2a"
        if sensor_preference == "sar":
            collection = "sentinel-1-grd"

        start_date = last_scene_date or "2024-01-01"
        datetime_range = f"{start_date}/.."

        url = f"https://planetarycomputer.microsoft.com/api/stac/v1/search"
        payload = json.dumps({
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": [collection],
            "limit": 1,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "SatQuery-AI/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    features = data.get("features", [])
                    if features:
                        feat = features[0]
                        props = feat.get("properties", {})
                        scene_date = props.get("datetime", "")[:10]
                        if not last_scene_date or scene_date > last_scene_date:
                            return {
                                "scene_id": feat.get("id"),
                                "scene_date": scene_date,
                                "sensor": "Sentinel-2 L2A (STAC)" if collection == "sentinel-2-l2a" else "Sentinel-1 GRD (STAC)",
                                "cloud_cover_pct": props.get("eo:cloud_cover", 0.0),
                                "image_path": None,
                                "metadata": props,
                            }
        except Exception as e:
            logger.debug("STAC external search skipped or timed out: %s", e)

        return None
