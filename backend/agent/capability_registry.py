"""Extensible Capability and Tool Registry for SatQuery AI.

Provides declarative capability tracking for specialist models, GIS tools, and VLMs.
Enables dynamic tool selection based on capability matching rather than hardcoded queries,
and allows future models to register capabilities without modifying planner code.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from backend.agent.schemas import CapabilityEntry

logger = logging.getLogger("agent.capability_registry")

# Default registry entries
DEFAULT_CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        name="YOLO11",
        model_type="specialist_model",
        capabilities=[
            "object_detection",
            "object_counting",
            "object_localization",
        ],
        description="xView satellite object detector identifying discrete objects (buildings, vehicles, aircraft, ships).",
        tags=["objects", "boxes", "detection", "counting"],
    ),
    CapabilityEntry(
        name="SegFormer",
        model_type="specialist_model",
        capabilities=[
            "semantic_segmentation",
            "region_extraction",
            "area_estimation",
        ],
        description="Hierarchical Transformer-based semantic segmentation model for land cover region extraction and area estimation.",
        tags=["segmentation", "regions", "masks", "vegetation", "water"],
    ),
    CapabilityEntry(
        name="BigEarthNet-ResNet50-SeCo-12B",
        model_type="specialist_model",
        capabilities=[
            "land_cover_classification",
            "multilabel_classification",
        ],
        description="12-band multispectral Sentinel-2 land-cover classifier predicting 19 BigEarthNet multi-label classes.",
        tags=["landcover", "classification", "multispectral", "sentinel-2"],
    ),
    CapabilityEntry(
        name="ChangeFormerV6",
        model_type="specialist_model",
        capabilities=[
            "change_detection",
            "bitemporal_comparison",
            "comparison_pipeline",
        ],
        description="Transformer-based bi-temporal change detector computing pixel-level change maps and statistics.",
        tags=["change", "temporal", "bitemporal", "difference"],
    ),
    CapabilityEntry(
        name="CROMA-Base",
        model_type="specialist_model",
        capabilities=[
            "cross_modal_fusion",
            "optical_sar_joint_representation",
        ],
        description="Multimodal foundation model fusing Sentinel-2 optical (12 bands) and Sentinel-1 SAR (2 bands).",
        tags=["croma", "sar", "radar", "optical", "cross_modal"],
    ),
    CapabilityEntry(
        name="GIS",
        model_type="gis_tool",
        capabilities=[
            "distance",
            "area",
            "intersection",
            "containment",
            "buffers",
            "spatial_relationships",
            "geometric_calculations",
        ],
        description="Geospatial engine for vector geometry, distance calculations, spatial buffering, and topological intersections.",
        tags=["spatial", "geometry", "distance", "buffer", "containment"],
    ),
    CapabilityEntry(
        name="RS-Qwen",
        model_type="vlm_foundation",
        capabilities=[
            "image_understanding",
            "natural_language_interpretation",
            "explanation",
            "grounded_synthesis",
        ],
        description="Remote Sensing Vision-Language Model for multimodal interpretation, scientific explanation, and grounded synthesis.",
        tags=["vlm", "reasoning", "explanation", "synthesis", "language"],
    ),
    CapabilityEntry(
        name="GeoChat",
        model_type="vlm_foundation",
        capabilities=[
            "geospatial_reasoning",
            "remote_sensing_vqa",
            "domain_explanation",
            "scene_grounding",
        ],
        description="Grounded Remote Sensing Vision-Language Model for geospatial reasoning, Earth observation VQA, and regional interpretation.",
        tags=["geochat", "vlm", "geospatial", "remote_sensing", "reasoning"],
    ),
]


class CapabilityRegistry:
    """Registry mapping tools to declared capabilities, supporting dynamic registration."""

    def __init__(self, populate_defaults: bool = True):
        self._entries: Dict[str, CapabilityEntry] = {}
        if populate_defaults:
            self.reset_defaults()

    def reset_defaults(self) -> None:
        """Reset the registry to default declared capabilities."""
        self._entries.clear()
        for entry in DEFAULT_CAPABILITIES:
            self.register(entry)

    def register(self, entry: CapabilityEntry) -> None:
        """Register a new tool or specialist model with its declared capabilities."""
        self._entries[entry.name] = entry
        logger.debug("Registered tool '%s' with capabilities: %s", entry.name, entry.capabilities)

    def get_entry(self, name: str) -> Optional[CapabilityEntry]:
        """Retrieve a registered capability entry by tool name."""
        return self._entries.get(name)

    def get_capabilities_for_tool(self, name: str) -> List[str]:
        """List all capabilities declared by a specific tool."""
        entry = self._entries.get(name)
        return entry.capabilities if entry else []

    def list_entries(self) -> Dict[str, CapabilityEntry]:
        """Return a mapping of all registered tool names to their entries."""
        return dict(self._entries)

    def list_capabilities(self) -> Dict[str, List[str]]:
        """Return a mapping of all registered tool names to their capabilities."""
        return {name: entry.capabilities for name, entry in self._entries.items()}

    def find_tools_for_capabilities(
        self,
        required_capabilities: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Find the minimal set of specialist models and tools that satisfy the required capabilities.

        Returns:
            Tuple of (selected_models, selected_tools)
        """
        req_set = set(required_capabilities)
        selected_models: List[str] = []
        selected_tools: List[str] = []

        for name, entry in self._entries.items():
            entry_caps = set(entry.capabilities)
            # If the tool satisfies at least one required capability
            if req_set & entry_caps:
                if entry.model_type == "specialist_model":
                    if name not in selected_models:
                        selected_models.append(name)
                else:
                    if name not in selected_tools:
                        selected_tools.append(name)

        return selected_models, selected_tools


# Global shared registry instance
CAPABILITY_REGISTRY = CapabilityRegistry()
