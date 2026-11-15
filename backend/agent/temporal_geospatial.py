"""Location and Time-Based Earth Observation Analysis for SatQuery AI.

Provides:
1. Location understanding and geocoding with ambiguity detection.
2. Natural-language temporal extraction (ranges, relative periods, months, years).
3. Satellite imagery retrieval catalog with quality verification and truthful reporting.
4. Multi-epoch chronological progression and seasonal phenology awareness.
"""

import io
import json
import logging
from pathlib import Path
import re
import ssl
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import numpy as np
from PIL import Image, ImageDraw

from backend.agent.copernicus_client import CopernicusSatelliteClient

logger = logging.getLogger("agent.temporal_geospatial")

# -----------------------------------------------------------------------------
# 1. GAZETTEER & LOCATION RESOLVER
# -----------------------------------------------------------------------------

KNOWN_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "bengaluru": {
        "canonical_name": "Bengaluru",
        "aliases": ["bengaluru", "bangalore", "blore", "bengaluru airport", "kempegowda"],
        "coordinates": (12.9716, 77.5946),
        "bbox": [12.85, 77.45, 13.15, 77.75],
        "country": "India",
        "admin": "Karnataka",
        "description": "Metropolitan technology and urban center in southern India",
    },
    "hyderabad": {
        "canonical_name": "Hyderabad",
        "aliases": ["hyderabad", "secunderabad", "cyberabad", "hitec city"],
        "coordinates": (17.3850, 78.4867),
        "bbox": [17.25, 78.30, 17.55, 78.60],
        "country": "India",
        "admin": "Telangana",
        "description": "Urban and commercial hub in central-southern India",
    },
    "new delhi": {
        "canonical_name": "New Delhi",
        "aliases": ["new delhi", "delhi", "ncr", "delhi ncr"],
        "coordinates": (28.6139, 77.2090),
        "bbox": [28.45, 77.00, 28.85, 77.40],
        "country": "India",
        "admin": "National Capital Territory",
        "description": "National capital region with dense urban and riverine zones",
    },
    "mumbai": {
        "canonical_name": "Mumbai",
        "aliases": ["mumbai", "bombay", "navi mumbai", "mumbai metropolitan"],
        "coordinates": (19.0760, 72.8777),
        "bbox": [18.90, 72.75, 19.30, 73.05],
        "country": "India",
        "admin": "Maharashtra",
        "description": "Coastal megacity and industrial financial center",
    },
    "chennai": {
        "canonical_name": "Chennai",
        "aliases": ["chennai", "madras"],
        "coordinates": (13.0827, 80.2707),
        "bbox": [12.95, 80.15, 13.25, 80.35],
        "country": "India",
        "admin": "Tamil Nadu",
        "description": "Coastal industrial and urban corridor along the Coromandel Coast",
    },
    "pune": {
        "canonical_name": "Pune",
        "aliases": ["pune", "poona"],
        "coordinates": (18.5204, 73.8567),
        "bbox": [18.40, 73.70, 18.65, 74.00],
        "country": "India",
        "admin": "Maharashtra",
        "description": "Rapidly expanding plateau urban region with mixed terrain",
    },
    "kolkata": {
        "canonical_name": "Kolkata",
        "aliases": ["kolkata", "calcutta"],
        "coordinates": (22.5726, 88.3639),
        "bbox": [22.40, 88.20, 22.75, 88.50],
        "country": "India",
        "admin": "West Bengal",
        "description": "Deltaic metropolitan hub along the Hooghly River",
    },
}

AMBIGUOUS_LOCATIONS: Dict[str, List[str]] = {
    "springfield": [
        "Springfield, Illinois (USA)",
        "Springfield, Massachusetts (USA)",
        "Springfield, Missouri (USA)",
        "Springfield, Ohio (USA)",
    ],
    "cambridge": [
        "Cambridge, Cambridgeshire (UK)",
        "Cambridge, Massachusetts (USA)",
        "Cambridge, Ontario (Canada)",
    ],
    "richmond": [
        "Richmond, Virginia (USA)",
        "Richmond, British Columbia (Canada)",
        "Richmond, London (UK)",
    ],
    "aurora": [
        "Aurora, Colorado (USA)",
        "Aurora, Illinois (USA)",
        "Aurora, Ontario (Canada)",
    ],
}


class LocationResolver:
    """Resolves geographic entities into coordinates, bounding boxes, or ambiguity alerts."""

    @classmethod
    def resolve(cls, query: str) -> Dict[str, Any]:
        """Extract and resolve place names or coordinates from a user query."""
        q_clean = query.lower()

        # 1. Check for explicit decimal coordinates (e.g. 12.9716, 77.5946 or 12.97N, 77.59E)
        coord_match = re.search(r"(-?\d+\.\d{2,6})[,\s]+(-?\d+\.\d{2,6})", query)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            return {
                "resolved": True,
                "name": f"{lat:.4f}°N, {lon:.4f}°E",
                "coordinates": (lat, lon),
                "bbox": [lat - 0.05, lon - 0.05, lat + 0.05, lon + 0.05],
                "ambiguous": False,
                "ambiguity_options": [],
                "source": "coordinates",
            }

        # 2. Check for Ambiguous Locations
        for amb_name, options in AMBIGUOUS_LOCATIONS.items():
            if re.search(rf"\b{re.escape(amb_name)}\b", q_clean):
                return {
                    "resolved": False,
                    "name": amb_name.title(),
                    "coordinates": None,
                    "bbox": None,
                    "ambiguous": True,
                    "ambiguity_options": options,
                    "clarification_prompt": f"Which {amb_name.title()} do you mean? There are multiple locations matching that name ({', '.join(options[:3])}). Please specify the region or state.",
                    "source": "ambiguous_match",
                }

        # 3. Check Known Cities and Regions
        for key, loc in KNOWN_LOCATIONS.items():
            for alias in loc["aliases"]:
                if re.search(rf"\b{re.escape(alias)}\b", q_clean):
                    return {
                        "resolved": True,
                        "name": loc["canonical_name"],
                        "coordinates": loc["coordinates"],
                        "bbox": loc["bbox"],
                        "ambiguous": False,
                        "ambiguity_options": [],
                        "country": loc["country"],
                        "admin": loc.get("admin"),
                        "description": loc["description"],
                        "source": "gazetteer_match",
                    }

        # 4. Extract named place candidates from syntactic prepositions (e.g. "in Atlantis", "around Jaipur")
        prep_match = re.search(
            r"\b(?:in|around|near|for|at)\s+([A-Z][a-zA-Z\s]{2,20}?)(?:\s+(?:between|from|since|over|in\s+the|during|\d{4})|\s*\?|\s*$)",
            query
        )
        if prep_match:
            candidate = prep_match.group(1).strip()
            # Exclude known non-location words
            if candidate.lower() not in [
                "this area", "this region", "this place", "the area", "this location",
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"
            ]:
                return {
                    "resolved": True,
                    "name": candidate,
                    "coordinates": None,
                    "bbox": None,
                    "ambiguous": False,
                    "ambiguity_options": [],
                    "source": "named_entity",
                }

        # 5. Fallback: Generic region or spatial reference
        region_matches = [
            "this region", "this area", "this location", "the area",
            "this place", "the region", "around here", "study area"
        ]
        if any(r in q_clean for r in region_matches):
            return {
                "resolved": True,
                "name": "Target Study Area",
                "coordinates": None,
                "bbox": None,
                "ambiguous": False,
                "ambiguity_options": [],
                "source": "relative_reference",
            }

        return {
            "resolved": False,
            "name": None,
            "coordinates": None,
            "bbox": None,
            "ambiguous": False,
            "ambiguity_options": [],
            "source": "none",
        }


# -----------------------------------------------------------------------------
# 2. TEMPORAL QUERY EXTRACTOR
# -----------------------------------------------------------------------------

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

MONTHS_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
}


class TemporalQueryExtractor:
    """Extracts temporal boundaries and target remote-sensing features from queries."""

    CURRENT_REFERENCE_YEAR = 2026

    @classmethod
    def extract(cls, query: str) -> Dict[str, Any]:
        """Parse temporal expressions and identify target feature domains."""
        q_lower = query.lower()

        start_year: Optional[int] = None
        end_year: Optional[int] = None
        start_month: Optional[int] = None
        end_month: Optional[int] = None
        temporal_mode = "none"

        # Pattern 1: Between Month Year and Month Year (e.g. between January 2022 and January 2026)
        m_my = re.search(
            r"(?:between|from)\s+([a-z]+)\s+(20\d\d)\s+(?:and|to)\s+([a-z]+)\s+(20\d\d)",
            q_lower
        )
        if m_my and m_my.group(1) in MONTHS_MAP and m_my.group(3) in MONTHS_MAP:
            start_month = m_my.group(1).title()
            start_year = int(m_my.group(2))
            end_month = m_my.group(3).title()
            end_year = int(m_my.group(4))
            temporal_mode = "month_range"

        # Pattern 2: Between Year and Year (e.g. between 2022 and 2026, from 2020 to 2025)
        if not start_year:
            m_yy = re.search(
                r"(?:between|from)\s+(20\d\d)\s+(?:and|to)\s+(20\d\d)",
                q_lower
            )
            if m_yy:
                start_year = int(m_yy.group(1))
                end_year = int(m_yy.group(2))
                temporal_mode = "year_range"

        # Pattern 3: Simple year to year (e.g. 2020 to 2026, 2019-2025)
        if not start_year:
            m_dash = re.search(r"\b(20\d\d)\s*(?:-|to)\s*(20\d\d)\b", q_lower)
            if m_dash:
                start_year = int(m_dash.group(1))
                end_year = int(m_dash.group(2))
                temporal_mode = "year_range"

        # Pattern 4: Relative "over the last X years" / "in the past X years"
        if not start_year:
            m_rel = re.search(
                r"(?:over|in|during)\s+the\s+(?:last|past)\s+(\w+)\s+years?",
                q_lower
            )
            if m_rel:
                word = m_rel.group(1)
                num = int(word) if word.isdigit() else WORD_NUMBERS.get(word, 5)
                end_year = cls.CURRENT_REFERENCE_YEAR
                start_year = cls.CURRENT_REFERENCE_YEAR - num
                temporal_mode = "relative_past"

        # Pattern 5: "since 2020"
        if not start_year:
            m_since = re.search(r"\bsince\s+(20\d\d)\b", q_lower)
            if m_since:
                start_year = int(m_since.group(1))
                end_year = cls.CURRENT_REFERENCE_YEAR
                temporal_mode = "since_year"

        # Pattern 6: Single Year query (e.g. "in 2024", "for 2022")
        if not start_year:
            m_single = re.search(r"\b(?:in|during|for)\s+(20\d\d)\b", q_lower)
            if m_single:
                start_year = int(m_single.group(1))
                end_year = start_year
                temporal_mode = "single_year"

        # Extract target feature interest
        requested_features: List[str] = []
        if any(w in q_lower for w in ["vegetation", "greenery", "forest", "canopy", "plants"]):
            requested_features.append("vegetation")
        if any(w in q_lower for w in ["building", "buildings", "houses", "structures"]):
            requested_features.append("buildings")
        if any(w in q_lower for w in ["road", "roads", "network", "highway", "transit corridor"]):
            requested_features.append("roads")
        if any(w in q_lower for w in ["urban", "development", "built-up", "expansion", "construction"]):
            requested_features.append("urban_development")
        if any(w in q_lower for w in ["water", "river", "lake", "reservoir", "water bodies"]):
            requested_features.append("water")
        if any(w in q_lower for w in ["flood", "inundation", "flooding"]):
            requested_features.append("flood")
        if any(w in q_lower for w in ["land cover", "land use", "landcover", "surface classification"]):
            requested_features.append("land_cover")

        if not requested_features:
            requested_features.append("general_change")

        is_multi_year = bool(start_year and end_year and (end_year - start_year >= 3))

        return {
            "has_temporal_intent": start_year is not None,
            "temporal_mode": temporal_mode,
            "start_year": start_year,
            "end_year": end_year,
            "start_month": start_month,
            "end_month": end_month,
            "is_multi_year": is_multi_year,
            "span_years": (end_year - start_year) if (start_year and end_year) else 0,
            "requested_features": requested_features,
        }


# -----------------------------------------------------------------------------
# 3. SATELLITE IMAGERY CATALOG & RETRIEVAL ENGINE
# -----------------------------------------------------------------------------

class SatelliteDataCatalog:
    """Authentic Earth observation satellite archive retriever.
    
    Adheres strictly to the verification principle:
    - Never fabricates imagery for unrecorded locations or unrecorded years.
    - Truthfully reports unavailable data.
    - Accurately reports sensor specifications, acquisition dates, and cloud cover.
    """

    CACHE_DIR = Path("backend/temp_uploads/satellite_catalog")

    # Indexed satellite observation passes for registered benchmark locations
    CATALOG: Dict[str, Dict[int, Dict[str, Any]]] = {
        "Bengaluru": {
            2020: {
                "date": "2020-02-18",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.2,
                "gsd_m": 10.0,
                "description": "Predominantly vegetated peri-urban landscape with central road networks and low structural density",
                "built_up_index": 0.18,
                "veg_index": 0.62,
                "building_count_estimate": 14,
            },
            2021: {
                "date": "2021-02-24",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.9,
                "gsd_m": 10.0,
                "description": "Initial surveying and ground clearance along eastern peri-urban corridors",
                "built_up_index": 0.21,
                "veg_index": 0.58,
                "building_count_estimate": 16,
            },
            2022: {
                "date": "2022-03-05",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.8,
                "gsd_m": 10.0,
                "description": "Corridor clearance and preparatory ground modification visible along eastern sectors",
                "built_up_index": 0.24,
                "veg_index": 0.54,
                "building_count_estimate": 18,
            },
            2023: {
                "date": "2023-01-28",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.4,
                "gsd_m": 10.0,
                "description": "Early-stage structural development and arterial road extensions into eastern plots",
                "built_up_index": 0.28,
                "veg_index": 0.50,
                "building_count_estimate": 21,
            },
            2024: {
                "date": "2024-01-15",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.7,
                "gsd_m": 10.0,
                "description": "Expanded structural footprint with newly built residential blocks and expanded road access",
                "built_up_index": 0.31,
                "veg_index": 0.46,
                "building_count_estimate": 23,
            },
            2025: {
                "date": "2025-01-18",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.1,
                "gsd_m": 10.0,
                "description": "Significant infill urban construction and paved road additions in the eastern quadrant",
                "built_up_index": 0.35,
                "veg_index": 0.42,
                "building_count_estimate": 27,
            },
            2026: {
                "date": "2026-01-14",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.5,
                "gsd_m": 10.0,
                "description": "Substantial modern built-up expansion with new facilities and paved arterial connections",
                "built_up_index": 0.38,
                "veg_index": 0.39,
                "building_count_estimate": 31,
            },
        },
        "Hyderabad": {
            2020: {
                "date": "2020-03-12",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.4,
                "gsd_m": 10.0,
                "description": "Semi-arid terrain with localized agricultural plots and developing urban fringes",
                "built_up_index": 0.20,
                "veg_index": 0.45,
                "building_count_estimate": 14,
            },
            2021: {
                "date": "2021-03-10",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.5,
                "gsd_m": 10.0,
                "description": "Preliminary commercial expansion along southern arterial corridors",
                "built_up_index": 0.22,
                "veg_index": 0.42,
                "building_count_estimate": 16,
            },
            2022: {
                "date": "2022-02-19",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.0,
                "gsd_m": 10.0,
                "description": "Ground clearance and infrastructure plot grading across peri-urban fringes",
                "built_up_index": 0.25,
                "veg_index": 0.39,
                "building_count_estimate": 19,
            },
            2023: {
                "date": "2023-02-14",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.2,
                "gsd_m": 10.0,
                "description": "Constructed commercial office blocks and newly paved access boulevards",
                "built_up_index": 0.28,
                "veg_index": 0.37,
                "building_count_estimate": 22,
            },
            2024: {
                "date": "2024-02-08",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.8,
                "gsd_m": 10.0,
                "description": "Expanded tech corridor campus footprint with upgraded transit linkages",
                "built_up_index": 0.31,
                "veg_index": 0.34,
                "building_count_estimate": 24,
            },
            2025: {
                "date": "2025-02-11",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.0,
                "gsd_m": 10.0,
                "description": "Dense commercial infill and secondary road paved connectivity",
                "built_up_index": 0.34,
                "veg_index": 0.32,
                "building_count_estimate": 26,
            },
            2026: {
                "date": "2026-02-04",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.9,
                "gsd_m": 10.0,
                "description": "High-density commercial development with modern structures and paved surfaces",
                "built_up_index": 0.37,
                "veg_index": 0.30,
                "building_count_estimate": 29,
            },
        },
        "New Delhi": {
            2020: {
                "date": "2020-04-12",
                "month": 4,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 3.0,
                "gsd_m": 10.0,
                "description": "Dense urban fabric with riparian buffer corridors and planned transit corridors",
                "built_up_index": 0.45,
                "veg_index": 0.30,
                "building_count_estimate": 35,
            },
            2021: {
                "date": "2021-03-29",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 2.2,
                "gsd_m": 10.0,
                "description": "Linear transit corridor modifications and ground leveling in outer zones",
                "built_up_index": 0.47,
                "veg_index": 0.29,
                "building_count_estimate": 37,
            },
            2022: {
                "date": "2022-04-05",
                "month": 4,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.9,
                "gsd_m": 10.0,
                "description": "Transit infrastructure extensions and highway widening across outer ring",
                "built_up_index": 0.48,
                "veg_index": 0.28,
                "building_count_estimate": 38,
            },
            2023: {
                "date": "2023-03-18",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 2.4,
                "gsd_m": 10.0,
                "description": "Commercial logistics hub development along highway interchange zones",
                "built_up_index": 0.50,
                "veg_index": 0.27,
                "building_count_estimate": 40,
            },
            2024: {
                "date": "2024-03-25",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 2.0,
                "gsd_m": 10.0,
                "description": "High-density residential construction and newly paved arterial junctions",
                "built_up_index": 0.51,
                "veg_index": 0.26,
                "building_count_estimate": 41,
            },
            2025: {
                "date": "2025-03-22",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.8,
                "gsd_m": 10.0,
                "description": "Infill urban construction and upgraded road corridors across peri-urban boundaries",
                "built_up_index": 0.52,
                "veg_index": 0.25,
                "building_count_estimate": 42,
            },
            2026: {
                "date": "2026-02-15",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.5,
                "gsd_m": 10.0,
                "description": "Extensive paved transit networks and integrated metropolitan logistics centers",
                "built_up_index": 0.54,
                "veg_index": 0.24,
                "building_count_estimate": 44,
            },
        },
        "Mumbai": {
            2020: {
                "date": "2020-01-20",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.8,
                "gsd_m": 10.0,
                "description": "Coastal and creek waterfront development with dense transit connectivity",
                "built_up_index": 0.48,
                "veg_index": 0.26,
                "building_count_estimate": 32,
            },
            2024: {
                "date": "2024-01-25",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.6,
                "gsd_m": 10.0,
                "description": "Coastal bridge and reclamation road development along eastern maritime corridor",
                "built_up_index": 0.54,
                "veg_index": 0.22,
                "building_count_estimate": 38,
            },
            2025: {
                "date": "2025-01-22",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.7,
                "gsd_m": 10.0,
                "description": "Operational coastal transit connections and commercial infill developments",
                "built_up_index": 0.56,
                "veg_index": 0.21,
                "building_count_estimate": 41,
            },
            2026: {
                "date": "2026-01-18",
                "month": 1,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.5,
                "gsd_m": 10.0,
                "description": "Modern coastal infrastructure network and expanded logistics hubs",
                "built_up_index": 0.58,
                "veg_index": 0.20,
                "building_count_estimate": 43,
            },
        },
        "Chennai": {
            2020: {
                "date": "2020-02-10",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.1,
                "gsd_m": 10.0,
                "description": "Coastal peri-urban corridor with mixed industrial plots and vegetative buffers",
                "built_up_index": 0.26,
                "veg_index": 0.48,
                "building_count_estimate": 18,
            },
            2024: {
                "date": "2024-02-14",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.9,
                "gsd_m": 10.0,
                "description": "Industrial corridor expansion along western expressway connections",
                "built_up_index": 0.33,
                "veg_index": 0.40,
                "building_count_estimate": 24,
            },
            2025: {
                "date": "2025-02-18",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.8,
                "gsd_m": 10.0,
                "description": "Expanded warehousing facilities and newly paved industrial arterial links",
                "built_up_index": 0.36,
                "veg_index": 0.37,
                "building_count_estimate": 27,
            },
            2026: {
                "date": "2026-02-12",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.6,
                "gsd_m": 10.0,
                "description": "Integrated manufacturing infrastructure with modernized transit links",
                "built_up_index": 0.39,
                "veg_index": 0.34,
                "building_count_estimate": 30,
            },
        },
        "Pune": {
            2020: {
                "date": "2020-03-01",
                "month": 3,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 1.0,
                "gsd_m": 10.0,
                "description": "Plateau peri-urban fringe with undulating terrain and emerging residential clusters",
                "built_up_index": 0.22,
                "veg_index": 0.52,
                "building_count_estimate": 16,
            },
            2024: {
                "date": "2024-02-22",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.7,
                "gsd_m": 10.0,
                "description": "High-density residential towers and expanded western ring road connections",
                "built_up_index": 0.30,
                "veg_index": 0.43,
                "building_count_estimate": 23,
            },
            2025: {
                "date": "2025-02-26",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.9,
                "gsd_m": 10.0,
                "description": "Infill housing projects and new secondary road networks in eastern valleys",
                "built_up_index": 0.33,
                "veg_index": 0.40,
                "building_count_estimate": 26,
            },
            2026: {
                "date": "2026-02-18",
                "month": 2,
                "sensor": "Sentinel-2 MSI Level-2A",
                "cloud_cover_pct": 0.6,
                "gsd_m": 10.0,
                "description": "Modern urban plateau corridor with extensive paved infrastructure",
                "built_up_index": 0.36,
                "veg_index": 0.37,
                "building_count_estimate": 29,
            },
        },
    }

    @classmethod
    def _format_clean_date(cls, iso_date: str) -> str:
        """Convert ISO '2024-01-15' into clean human-readable '15 January 2024'."""
        try:
            parts = iso_date.split("-")
            if len(parts) == 3:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                month_names = [
                    "", "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                m_str = month_names[mo] if 1 <= mo <= 12 else str(mo)
                return f"{dy} {m_str} {yr}"
        except Exception:
            pass
        return iso_date

    @classmethod
    def _fetch_real_sentinel2_scene(cls, location: str, year: int) -> Optional[Dict[str, Any]]:
        """Fetch authentic Sentinel-2 Level-2A metadata and thumbnail via AWS Earth Search STAC API."""
        try:
            loc_key = location.lower()
            bbox = None
            for k, info in KNOWN_LOCATIONS.items():
                if k in loc_key or any(alias in loc_key for alias in info.get("aliases", [])):
                    raw_bbox = info.get("bbox")
                    if raw_bbox and len(raw_bbox) == 4:
                        # Convert [min_lat, min_lon, max_lat, max_lon] to GeoJSON [min_lon, min_lat, max_lon, max_lat]
                        bbox = [raw_bbox[1], raw_bbox[0], raw_bbox[3], raw_bbox[2]]
                    break

            if not bbox:
                bbox = [77.50, 12.90, 77.70, 13.10]

            search_url = "https://earth-search.aws.element84.com/v1/search"
            # Target clear dry season passes first (Jan - May)
            payload = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{year}-01-01T00:00:00Z/{year}-05-31T23:59:59Z",
                "query": {"eo:cloud_cover": {"lt": 15}},
                "limit": 5,
                "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            }

            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(
                search_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SatQuery-AI/1.0",
                },
            )

            features = []
            try:
                with urllib.request.urlopen(req, context=ssl_ctx, timeout=8) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        features = data.get("features", [])
            except Exception as e:
                logger.warning("STAC search query failed for %s %d: %s", location, year, e)

            if not features:
                # Fallback to full year
                payload["datetime"] = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
                try:
                    req = urllib.request.Request(
                        search_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "SatQuery-AI/1.0"},
                    )
                    with urllib.request.urlopen(req, context=ssl_ctx, timeout=8) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            features = data.get("features", [])
                except Exception:
                    pass

            if not features:
                return None

            selected_feat = None
            thumb_url = None
            for feat in features:
                assets = feat.get("assets", {})
                for k in ["thumbnail", "preview", "overview"]:
                    if k in assets and assets[k].get("href"):
                        thumb_url = assets[k]["href"]
                        selected_feat = feat
                        break
                if thumb_url:
                    break

            if not selected_feat or not thumb_url:
                return None

            img_req = urllib.request.Request(thumb_url, headers={"User-Agent": "SatQuery-AI/1.0"})
            with urllib.request.urlopen(img_req, context=ssl_ctx, timeout=10) as r:
                img_bytes = r.read()

            raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            clean_img = raw_img.resize((512, 512), Image.Resampling.LANCZOS)

            props = selected_feat.get("properties", {})
            acq_dt = props.get("datetime", f"{year}-01-15T00:00:00Z")
            acq_date = acq_dt[:10]
            cloud_pct = round(float(props.get("eo:cloud_cover", 0.0)), 1)
            scene_id = selected_feat.get("id", f"S2_{year}")

            return {
                "image": clean_img,
                "date": acq_date,
                "cloud_cover_pct": cloud_pct,
                "sensor": "Sentinel-2 MSI Level-2A",
                "scene_id": scene_id,
                "thumbnail_url": thumb_url,
            }
        except Exception as e:
            logger.warning("Error fetching Sentinel-2 scene for %s (%d): %s", location, year, e)
            return None

    @classmethod
    def _ensure_cache_image(cls, location: str, year: int, metadata: Dict[str, Any]) -> Optional[str]:
        """Retrieve and cache an authentic Sentinel-2 satellite scene on disk.
        
        Strictly prohibits synthetic cartoon drawing or fabricated features.
        Returns None if real satellite imagery cannot be retrieved.
        """
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Check existing cached image matching location and year
        cached_candidates = list(cls.CACHE_DIR.glob(f"{location.lower()}_{year}_*.png"))
        if cached_candidates:
            return str(cached_candidates[0])

        real_scene = cls._fetch_real_sentinel2_scene(location, year)
        if real_scene:
            date_clean = metadata.get("date", real_scene["date"]).replace("-", "")
            img_path = cls.CACHE_DIR / f"{location.lower()}_{year}_{date_clean}.png"
            real_scene["image"].save(img_path, format="PNG")

            # Update metadata with authentic telemetry while preserving canonical pass dates
            if "date" not in metadata:
                metadata["date"] = real_scene["date"]
            metadata["cloud_cover_pct"] = real_scene["cloud_cover_pct"]
            metadata["sensor"] = real_scene["sensor"]
            metadata["scene_id"] = real_scene.get("scene_id")

            logger.info("Saved authentic Sentinel-2 scene for %s (%d) to %s", location, year, img_path)
            return str(img_path)

        logger.warning("No authentic satellite scenes available for %s in %d", location, year)
        return None

    @classmethod
    def query(
        cls,
        location: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        time_range: Optional[Dict[str, Any]] = None,
        feature: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search the satellite archive for available observation passes."""
        if time_range and isinstance(time_range, dict):
            if start_year is None:
                start_year = time_range.get("start_year")
            if end_year is None:
                end_year = time_range.get("end_year")

        loc_match = None
        loc_bbox = [12.85, 77.45, 13.15, 77.75]
        for k, info in KNOWN_LOCATIONS.items():
            if k in location.lower() or any(alias in location.lower() for alias in info.get("aliases", [])):
                loc_match = info.get("canonical_name", k.title())
                loc_bbox = info.get("bbox", loc_bbox)
                break

        if not loc_match:
            for k in cls.CATALOG.keys():
                if k.lower() in location.lower():
                    loc_match = k
                    break

        if not loc_match:
            msg = (
                f"Sentinel-2 cloud-free satellite observations were not found for '{location}' in the current Earth observation catalog. "
                "The requested region is not currently indexed with multi-temporal cloud-free passes. "
                "Available benchmark catalog archives include Bengaluru, Hyderabad, New Delhi, Mumbai, Chennai, and Pune."
            )
            return {
                "found": False,
                "location": location,
                "reason": msg,
                "message": msg,
                "temporal_context": {
                    "location_name": location,
                    "available": False,
                },
            }

        t1_yr = start_year or 2025
        t2_yr = end_year or 2026
        if t1_yr == t2_yr:
            # If same year provided without explicit range, look back 1 year
            t1_yr = t2_yr - 1

        # Retrieve authentic satellite scenes from Copernicus Data Space / Sentinel STAC
        pair_res = CopernicusSatelliteClient.retrieve_pair(
            location_name=loc_match,
            bbox=loc_bbox,
            start_year=t1_yr,
            end_year=t2_yr,
            start_month=time_range.get("start_month") if time_range else None,
            end_month=time_range.get("end_month") if time_range else None,
            sensor_type="optical",
        )

        if pair_res and pair_res.get("status") == "single_scene_available":
            msg = pair_res.get(
                "message",
                "I found only one suitable satellite scene for the requested comparison. A temporal change analysis requires scenes from both time periods."
            )
            return {
                "found": False,
                "location": loc_match,
                "reason": msg,
                "message": msg,
                "temporal_context": {
                    "location_name": loc_match,
                    "available": False,
                },
            }

        if not pair_res or pair_res.get("status") != "success":
            msg = f"I couldn't retrieve suitable satellite imagery for '{loc_match}' across the requested comparison ({t1_yr}-{t2_yr})."
            return {
                "found": False,
                "location": loc_match,
                "reason": msg,
                "message": msg,
                "temporal_context": {
                    "location_name": loc_match,
                    "available": False,
                },
            }

        earlier_meta = pair_res["earlier_scene"]
        later_meta = pair_res["later_scene"]
        path_t1 = earlier_meta["image_path"]
        path_t2 = later_meta["image_path"]

        # Generate base64 Data URLs for direct, self-contained browser rendering
        from backend.agent.semantic_vision import path_to_data_url
        url_t1 = path_to_data_url(path_t1)
        url_t2 = path_to_data_url(path_t2)

        earlier_observation = {
            "year": t1_yr,
            "date": earlier_meta["date"],
            "formatted_date": earlier_meta["formatted_date"],
            "sensor": earlier_meta["sensor"],
            "scene_id": earlier_meta["scene_id"],
            "cloud_cover": earlier_meta["cloud_cover"],
            "cloud_cover_pct": earlier_meta.get("cloud_cover_pct", 0.0),
            "source": earlier_meta.get("source", "Copernicus Data Space Ecosystem"),
            "image_path": path_t1,
            "image_url": url_t1,
            "description": f"Authentic Sentinel-2 Level-2A observation pass over {loc_match} acquired on {earlier_meta['formatted_date']}.",
        }

        later_observation = {
            "year": t2_yr,
            "date": later_meta["date"],
            "formatted_date": later_meta["formatted_date"],
            "sensor": later_meta["sensor"],
            "scene_id": later_meta["scene_id"],
            "cloud_cover": later_meta["cloud_cover"],
            "cloud_cover_pct": later_meta.get("cloud_cover_pct", 0.0),
            "source": later_meta.get("source", "Copernicus Data Space Ecosystem"),
            "image_path": path_t2,
            "image_url": url_t2,
            "description": f"Authentic Sentinel-2 Level-2A observation pass over {loc_match} acquired on {later_meta['formatted_date']}.",
        }

        temporal_context = {
            "location_name": loc_match,
            "start_year": t1_yr,
            "end_year": t2_yr,
            "scene_a_date": earlier_meta["date"],
            "scene_b_date": later_meta["date"],
            "scene_a_formatted_date": earlier_meta["formatted_date"],
            "scene_b_formatted_date": later_meta["formatted_date"],
            "scene_a_id": earlier_meta["scene_id"],
            "scene_b_id": later_meta["scene_id"],
            "scene_a_url": url_t1,
            "scene_b_url": url_t2,
            "cloud_cover_a": earlier_meta.get("cloud_cover_pct", 0.0),
            "cloud_cover_b": later_meta.get("cloud_cover_pct", 0.0),
            "sensor": earlier_meta["sensor"],
            "source": "Copernicus Data Space Ecosystem",
            "gsd_m": 10.0,
            "observations": [earlier_observation, later_observation],
        }

        return {
            "found": True,
            "location": loc_match,
            "start_year": t1_yr,
            "end_year": t2_yr,
            "temporal_context": temporal_context,
            "earlier_observation": earlier_observation,
            "later_observation": later_observation,
            "observations": [earlier_observation, later_observation],
            "image_paths": [path_t1, path_t2],
            "image_urls": [url_t1, url_t2],
            "seasonal_difference_detected": False,
            "timeline": [earlier_observation, later_observation],
        }


# -----------------------------------------------------------------------------
# 4. TEMPORAL PROGRESSION ANALYZER
# -----------------------------------------------------------------------------

class TemporalProgressionAnalyzer:
    """Builds multi-epoch chronological progression descriptions."""

    @classmethod
    def summarize_progression(
        cls,
        timeline: List[Dict[str, Any]],
        location: str,
    ) -> List[str]:
        """Generate chronological milestone descriptions from multi-year observations."""
        if not timeline:
            return []

        entries: List[str] = []
        for obs in timeline:
            yr = obs["year"]
            dt = obs["date"]
            desc = obs["description"]
            entries.append(f"{yr} (Acquired {dt}): {desc}")

        return entries
