"""Tool registry for specialist deep-learning models in SatQuery AI."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("agent.registry")


class SpecialistTool:
    """Encapsulates a specialist deep-learning model."""

    def __init__(
        self,
        name: str,
        model_id: str,
        description: str,
        execute_fn: Callable[..., Dict[str, Any]],
        accepted_inputs: List[str],
    ):
        self.name = name
        self.model_id = model_id
        self.description = description
        self.execute_fn = execute_fn
        self.accepted_inputs = accepted_inputs

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute the tool and return structured evidence."""
        try:
            return self.execute_fn(*args, **kwargs)
        except Exception as e:
            logger.error("Tool execution failed [%s]: %s", self.name, e)
            return {
                "success": False,
                "model": self.model_id,
                "error": f"{self.name} execution failed: {str(e)}",
            }


# -----------------------------------------------------------------------------
# 1. Object Detection Specialist (YOLO11)
# -----------------------------------------------------------------------------
def _run_yolo(image_path: Union[str, Path], conf: float = 0.25) -> Dict[str, Any]:
    try:
        from ml.object_detection.inference import detect_objects
        return detect_objects(image_path=str(image_path), conf=conf)
    except FileNotFoundError as e:
        return {
            "success": False,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {"objects": [], "counts": {}},
            "visualization": None,
            "error": f"YOLO model weights not found: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {"objects": [], "counts": {}},
            "visualization": None,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# 2. Change Detection Specialist (ChangeFormerV6)
# -----------------------------------------------------------------------------
def _run_change_detection(
    image_a: Union[str, Path],
    image_b: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    try:
        from ml.change_detection.inference import detect_change
        result = detect_change(
            image_a=image_a,
            image_b=image_b,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
        )
        result["success"] = True
        result["error"] = None
        return result
    except Exception as e:
        return {
            "success": False,
            "model": "ChangeFormerV6",
            "image_a": str(image_a),
            "image_b": str(image_b),
            "change_map": None,
            "total_pixels": 0,
            "changed_pixels": 0,
            "unchanged_pixels": 0,
            "change_percentage": 0.0,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# 3. LandCover Specialist (12-Band Sentinel-2 SeCo-ECO ResNet-50)
# -----------------------------------------------------------------------------
def _run_landcover(
    image_path: Union[str, Path],
    threshold: float = 0.35,
    visualize: bool = False,
) -> Dict[str, Any]:
    try:
        from ml.landcover.inference import classify_landcover
        return classify_landcover(image_path=str(image_path), threshold=threshold, visualize=visualize)
    except (ImportError, AttributeError) as e:
        logger.warning("LandCover inference module is currently unavailable: %s", e)
        return {
            "success": False,
            "model": "BigEarthNet-ResNet50-SeCo-12B",
            "image": str(image_path),
            "results": {"classes": []},
            "visualization": None,
            "error": "LandCover inference module is currently unavailable or protected.",
        }
    except Exception as e:
        return {
            "success": False,
            "model": "BigEarthNet-ResNet50-SeCo-12B",
            "image": str(image_path),
            "results": {"classes": []},
            "visualization": None,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# 4. Optical + SAR Specialist (CROMA-Base)
# -----------------------------------------------------------------------------
def _run_croma(
    optical: Union[str, Path, Any],
    sar: Optional[Union[str, Path, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    allow_uninitialized: bool = True,
) -> Dict[str, Any]:
    try:
        from ml.optical_sar.inference import extract_multimodal_features
        features = extract_multimodal_features(
            optical=optical,
            sar=sar,
            checkpoint_path=checkpoint_path,
            allow_uninitialized=allow_uninitialized,
        )
        return {
            "success": True,
            "model": "CROMA-Base",
            "representations": {
                "sar_encodings_shape": list(features["SAR_encodings"].shape),
                "sar_gap_shape": list(features["SAR_GAP"].shape),
                "optical_encodings_shape": list(features["optical_encodings"].shape),
                "optical_gap_shape": list(features["optical_GAP"].shape),
                "joint_encodings_shape": list(features["joint_encodings"].shape),
                "joint_gap_shape": list(features["joint_GAP"].shape),
            },
            "summary": "Cross-modal bidirectional fused tokens and unified scene embedding extracted successfully.",
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "model": "CROMA-Base",
            "representations": {},
            "summary": None,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# Tool Registry Map
# -----------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, SpecialistTool] = {
    "yolo": SpecialistTool(
        name="yolo",
        model_id="YOLO11",
        description="xView satellite object detector identifying 60+ object classes (buildings, vehicles, aircraft, maritime vessels).",
        execute_fn=_run_yolo,
        accepted_inputs=["single_image"],
    ),
    "change_detection": SpecialistTool(
        name="change_detection",
        model_id="ChangeFormerV6",
        description="Transformer-based bi-temporal change detector computing pixel-level change maps and statistics.",
        execute_fn=_run_change_detection,
        accepted_inputs=["image_a", "image_b"],
    ),
    "landcover": SpecialistTool(
        name="landcover",
        model_id="BigEarthNet-ResNet50-SeCo-12B",
        description="12-band multispectral Sentinel-2 land-cover classifier predicting 19 BigEarthNet multi-label classes.",
        execute_fn=_run_landcover,
        accepted_inputs=["single_image_12band"],
    ),
    "croma": SpecialistTool(
        name="croma",
        model_id="CROMA-Base",
        description="Multimodal foundation model fusing Sentinel-2 optical (12 bands) and Sentinel-1 SAR (2 bands).",
        execute_fn=_run_croma,
        accepted_inputs=["optical", "sar"],
    ),
}


def get_tool(tool_name: str) -> Optional[SpecialistTool]:
    """Retrieve a specialist tool by name."""
    return TOOL_REGISTRY.get(tool_name)


def register_tool(tool: SpecialistTool) -> None:
    """Register a new tool (enables future additions such as segmentation models)."""
    TOOL_REGISTRY[tool.name] = tool

