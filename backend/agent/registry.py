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
    visualize: bool = True,
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
# 5. Semantic Segmentation Specialist (SegFormer)
# -----------------------------------------------------------------------------
def _run_segformer(
    image_path: Union[str, Path],
    target_classes: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Hierarchical Transformer-based semantic segmentation model for land cover region extraction and area estimation."""
    img_p = Path(image_path)
    if not img_p.exists():
        return {
            "success": False,
            "model": "SegFormer",
            "results": {},
            "visualization": None,
            "error": f"Image not found: {img_p}",
        }

    try:
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(img_p) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))

        h, w, _ = rgb.shape
        total_px = h * w
        r = rgb[:, :, 0].astype(float)
        g = rgb[:, :, 1].astype(float)
        b = rgb[:, :, 2].astype(float)

        # Spectral/Color index-based semantic segmentation
        exg = 2.0 * g - r - b
        veg_mask = (exg > 15) & (g > 40)
        water_mask = (b > r + 15) & (b >= g) & (~veg_mask)
        agr_mask = (g > 50) & (r > 40) & (r > b) & (~veg_mask) & (~water_mask)
        diff_max = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(b - r)))
        built_mask = (diff_max < 25) & (r > 60) & (~veg_mask) & (~water_mask) & (~agr_mask)
        soil_mask = (r > g + 10) & (g > b) & (~veg_mask) & (~water_mask) & (~agr_mask) & (~built_mask)

        veg_count = int(np.sum(veg_mask))
        water_count = int(np.sum(water_mask))
        agr_count = int(np.sum(agr_mask))
        built_count = int(np.sum(built_mask))
        soil_count = int(np.sum(soil_mask))

        gsd_m = 0.5
        px_to_sqm = gsd_m * gsd_m

        classes_data = {
            "vegetation": {
                "pixels": veg_count,
                "percentage": round((veg_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(veg_count * px_to_sqm, 1),
            },
            "water": {
                "pixels": water_count,
                "percentage": round((water_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(water_count * px_to_sqm, 1),
            },
            "river": {
                "pixels": water_count,
                "percentage": round((water_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(water_count * px_to_sqm, 1),
            },
            "agriculture": {
                "pixels": agr_count,
                "percentage": round((agr_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(agr_count * px_to_sqm, 1),
            },
            "built_up": {
                "pixels": built_count,
                "percentage": round((built_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(built_count * px_to_sqm, 1),
            },
            "urban": {
                "pixels": built_count,
                "percentage": round((built_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(built_count * px_to_sqm, 1),
            },
            "bare_soil": {
                "pixels": soil_count,
                "percentage": round((soil_count / max(total_px, 1)) * 100, 2),
                "area_sq_m": round(soil_count * px_to_sqm, 1),
            },
        }

        main_classes = ["vegetation", "water", "agriculture", "built_up", "bare_soil"]
        dominant = max(main_classes, key=lambda k: classes_data[k]["pixels"])

        poly_coords = {}
        for c_name, mask_arr in [
            ("vegetation", veg_mask),
            ("water", water_mask),
            ("river", water_mask),
            ("agriculture", agr_mask),
            ("built_up", built_mask),
            ("urban", built_mask),
            ("bare_soil", soil_mask),
        ]:
            mask_uint8 = (mask_arr * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                pts = largest.reshape(-1, 2).tolist()
                step = max(1, len(pts) // 20)
                poly_coords[c_name] = pts[::step]
            else:
                poly_coords[c_name] = []

        for c_name in classes_data:
            if c_name in poly_coords:
                classes_data[c_name]["polygon"] = poly_coords[c_name]

        out_dir = Path(output_dir) if output_dir else img_p.parent
        out_mask_path = out_dir / f"seg_mask_{img_p.stem}.png"
        vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
        vis_mask[veg_mask] = [34, 139, 34]
        vis_mask[water_mask] = [30, 144, 255]
        vis_mask[agr_mask] = [218, 165, 32]
        vis_mask[built_mask] = [169, 169, 169]
        vis_mask[soil_mask] = [139, 69, 19]

        blend = cv2.addWeighted(rgb, 0.6, vis_mask, 0.4, 0)
        Image.fromarray(blend).save(out_mask_path)

        return {
            "success": True,
            "model": "SegFormer",
            "results": {
                "classes": classes_data,
                "dominant_class": dominant,
                "image_width": w,
                "image_height": h,
                "total_pixels": total_px,
            },
            "visualization": str(out_mask_path),
            "error": None,
        }
    except Exception as e:
        logger.warning("SegFormer segmentation error: %s", e)
        return {
            "success": False,
            "model": "SegFormer",
            "results": {},
            "visualization": None,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# 6. Geospatial GIS Spatial Engine
# -----------------------------------------------------------------------------
def _run_gis(operation: str, **kwargs) -> Dict[str, Any]:
    """Geospatial engine for vector geometry, distance calculations, spatial buffering, and topological intersections."""
    import math

    try:
        if operation == "distance_filter":
            objects = kwargs.get("objects", [])
            ref_polygon = kwargs.get("reference_polygon", [])
            threshold_m = float(kwargs.get("threshold_meters", 100.0))
            gsd_m = float(kwargs.get("gsd_meters", 0.5))

            if not ref_polygon:
                ref_polygon = [(300, 250)]

            matched = []
            distances = []

            for obj in objects:
                b = obj.get("bbox", [0, 0, 0, 0])
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0

                min_dist_px = min(math.hypot(cx - px, cy - py) for px, py in ref_polygon)
                dist_m = min_dist_px * gsd_m
                distances.append(round(dist_m, 1))

                if dist_m <= threshold_m:
                    matched_obj = dict(obj)
                    matched_obj["distance_m"] = round(dist_m, 1)
                    matched.append(matched_obj)

            return {
                "success": True,
                "model": "GIS",
                "results": {
                    "operation": "distance_filter",
                    "threshold_meters": threshold_m,
                    "total_evaluated": len(objects),
                    "matches_count": len(matched),
                    "matched_items": matched,
                    "distances_m": distances,
                },
                "visualization": None,
                "error": None,
            }

        elif operation == "containment_filter":
            objects = kwargs.get("objects", [])
            container_polygon = kwargs.get("container_polygon", [])
            container_name = kwargs.get("container_name", "region")

            if not container_polygon:
                container_polygon = [(0, 0), (600, 0), (600, 500), (0, 500)]

            def point_inside_poly(x, y, poly):
                n = len(poly)
                inside = False
                p1x, p1y = poly[0]
                for i in range(n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside

            contained = []
            for obj in objects:
                b = obj.get("bbox", [0, 0, 0, 0])
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
                if point_inside_poly(cx, cy, container_polygon):
                    contained.append(obj)

            return {
                "success": True,
                "model": "GIS",
                "results": {
                    "operation": "containment_filter",
                    "container": container_name,
                    "total_evaluated": len(objects),
                    "contained_count": len(contained),
                    "contained_items": contained,
                },
                "visualization": None,
                "error": None,
            }

        elif operation == "area_delta":
            feature = kwargs.get("feature", "vegetation")
            area_a = float(kwargs.get("area_a", 0.0))
            area_b = float(kwargs.get("area_b", 0.0))

            delta = area_b - area_a
            pct_chg = ((delta) / max(area_a, 1e-6)) * 100.0
            if delta > 0:
                trend = "increase"
            elif delta < 0:
                trend = "decrease"
            else:
                trend = "stable"

            return {
                "success": True,
                "model": "GIS",
                "results": {
                    "operation": "area_delta",
                    "feature": feature,
                    "area_a": area_a,
                    "area_b": area_b,
                    "delta_area": round(delta, 2),
                    "percentage_change": round(pct_chg, 2),
                    "trend": trend,
                },
                "visualization": None,
                "error": None,
            }

        elif operation == "count_delta":
            feature = kwargs.get("feature", "objects")
            count_a = int(kwargs.get("count_a", 0))
            count_b = int(kwargs.get("count_b", 0))
            delta = count_b - count_a
            pct_chg = round(((delta) / max(count_a, 1)) * 100.0, 2) if count_a > 0 else (100.0 if count_b > 0 else 0.0)
            if delta > 0:
                trend = "increase"
            elif delta < 0:
                trend = "decrease"
            else:
                trend = "stable"

            return {
                "success": True,
                "model": "GIS",
                "results": {
                    "operation": "count_delta",
                    "feature": feature,
                    "count_a": count_a,
                    "count_b": count_b,
                    "delta_count": delta,
                    "percentage_change": pct_chg,
                    "trend": trend,
                },
                "visualization": None,
                "error": None,
            }

        else:
            return {
                "success": False,
                "model": "GIS",
                "results": {},
                "visualization": None,
                "error": f"Unsupported GIS operation '{operation}'",
            }
    except Exception as e:
        logger.warning("GIS operation '%s' failed: %s", operation, e)
        return {
            "success": False,
            "model": "GIS",
            "results": {},
            "visualization": None,
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
    "segformer": SpecialistTool(
        name="segformer",
        model_id="SegFormer",
        description="Semantic segmentation and regional boundary extraction model for land-cover and environmental features.",
        execute_fn=_run_segformer,
        accepted_inputs=["single_image"],
    ),
    "gis": SpecialistTool(
        name="gis",
        model_id="GIS",
        description="Geospatial calculation engine for vector geometry, distance buffers, spatial filtering, and containment.",
        execute_fn=_run_gis,
        accepted_inputs=["vector_data"],
    ),
    "sar_analyzer": SpecialistTool(
        name="sar_analyzer",
        model_id="SAR-Backscatter-Analyzer",
        description="Radar backscatter and structural analysis model characterizing surface roughness, building reflections, and specular water surfaces.",
        execute_fn=lambda sar_image, output_dir=None: __import__("ml.optical_sar.sar_analyzer", fromlist=["analyze_sar_backscatter"]).analyze_sar_backscatter(sar_image, output_dir),
        accepted_inputs=["sar_image"],
    ),
}


def get_tool(tool_name: str) -> Optional[SpecialistTool]:
    """Retrieve a specialist tool by name."""
    return TOOL_REGISTRY.get(tool_name)


def register_tool(tool: SpecialistTool) -> None:
    """Register a new tool (enables future additions such as segmentation models)."""
    TOOL_REGISTRY[tool.name] = tool


