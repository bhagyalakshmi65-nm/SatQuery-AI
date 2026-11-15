from pathlib import Path
from collections import Counter

from ultralytics import YOLO


# Path to the trained V1 model
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"

# xView class names
CLASS_NAMES = [
    "building",
    "fixed_wing_aircraft",
    "small_aircraft",
    "cargo_plane",
    "helicopter",
    "passenger_vehicle",
    "small_car",
    "bus",
    "pickup_truck",
    "utility_truck",
    "truck",
    "cargo_truck",
    "truck_w_box",
    "truck_tractor",
    "trailer",
    "truck_w_flatbed",
    "truck_w_liquid",
    "crane_truck",
    "railway_vehicle",
    "passenger_car",
    "cargo_car",
    "flat_car",
    "tank_car",
    "locomotive",
    "maritime_vessel",
    "motorboat",
    "sailboat",
    "tugboat",
    "barge",
    "fishing_vessel",
    "ferry",
    "yacht",
    "container_ship",
    "oil_tanker",
    "engineering_vehicle",
    "tower_crane",
    "container_crane",
    "reach_stacker",
    "straddle_carrier",
    "mobile_crane",
    "dump_truck",
    "haul_truck",
    "scraper_tractor",
    "front_loader_bulldozer",
    "excavator",
    "cement_mixer",
    "ground_grader",
    "hut_tent",
    "shed",
    "aircraft_hangar",
    "damaged_building",
    "facility",
    "construction_site",
    "vehicle_lot",
    "helipad",
    "storage_tank",
    "shipping_container_lot",
    "shipping_container",
    "pylon",
    "tower",
]


from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

# Path to configuration
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUTPUT_DIR = Path("ml/object_detection/outputs")
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

# Load trained V1 model once
_model = None


def load_inference_config() -> Dict[str, Any]:
    """Load inference settings from config.yaml if present."""
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("inference", {})
        except Exception:
            pass
    return {
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "imgsz": 640,
        "line_width": 2,
        "box_color": "#06B6D4",
    }


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def compute_object_direction(cx: float, cy: float, width: int, height: int) -> Tuple[str, str]:
    """Compute 3x3 grid spatial position and natural-language description from centroid."""
    norm_x = cx / width if width > 0 else 0.5
    norm_y = cy / height if height > 0 else 0.5

    if norm_y < 0.35:
        if norm_x < 0.35:
            return "north_west", "upper-left portion"
        elif norm_x > 0.65:
            return "north_east", "upper-right portion"
        else:
            return "north", "northern portion"
    elif norm_y > 0.65:
        if norm_x < 0.35:
            return "south_west", "lower-left portion"
        elif norm_x > 0.65:
            return "south_east", "lower-right portion"
        else:
            return "south", "southern portion"
    else:
        if norm_x < 0.35:
            return "west", "western portion"
        elif norm_x > 0.65:
            return "east", "eastern portion"
        else:
            return "central", "central portion"


def calculate_spatial_distribution(
    objects: List[Dict[str, Any]],
    width: int,
    height: int,
    target_class: str = "building",
) -> Tuple[str, str]:
    """Calculate the dominant directional concentration and natural grouping of detected objects."""
    if not objects:
        return "none", f"No {target_class}s were detected above the confidence threshold."

    quadrant_counts = Counter(obj.get("direction_label", "central portion") for obj in objects)
    total = len(objects)

    top_quads = quadrant_counts.most_common(2)
    primary_quad, primary_cnt = top_quads[0]
    primary_ratio = primary_cnt / total

    # Check for clustered secondary group
    if len(top_quads) > 1 and primary_ratio >= 0.40 and (top_quads[1][1] / total) >= 0.25:
        secondary_quad = top_quads[1][0]
        dominant_direction = f"{primary_quad} of the image"
        summary = (
            f"Most detected {target_class}s are clustered in the {primary_quad} of the scene, "
            f"with a smaller group in the {secondary_quad}."
        )
    elif primary_ratio >= 0.45:
        dominant_direction = f"{primary_quad} of the image"
        summary = f"Most of the detected {target_class}s are concentrated in the {primary_quad} of the scene."
    elif len(top_quads) >= 2 and primary_ratio >= 0.35:
        dominant_direction = f"{primary_quad} of the image"
        summary = f"Detected {target_class}s are primarily situated in the {primary_quad} and {top_quads[1][0]}."
    else:
        dominant_direction = "distributed across the scene"
        summary = f"The detected {target_class}s are distributed across the scene without a single localized cluster."

    return dominant_direction, summary


def generate_clean_detection_visualization(
    image_path: Union[str, Path],
    objects: List[Dict[str, Any]],
    counts: Dict[str, int],
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Generate a clean, high-contrast remote-sensing visualization without overlapping text labels.

    Preserves 100% of the underlying satellite imagery quality:
    - Thin (2px) crisp high-contrast cyan bounding boxes
    - Subtle 10% interior fill for instant footprint delineation
    - Intelligently spaced numbered badges [1], [2] to prevent overlapping labels
    - Top-right summary pill: '[ 8 buildings detected ]'
    - Bottom-left subtle legend: '□ Building detection'
    """
    try:
        image_path = Path(image_path)
        if not image_path.is_file():
            return None

        orig_img = Image.open(image_path).convert("RGB")
        w, h = orig_img.size

        overlay_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_rgba)

        # High-visibility colors: Cyan for buildings (#06B6D4)
        box_border_color = (6, 182, 212, 255)
        box_fill_color = (6, 182, 212, 25)  # 10% alpha

        # 1. Draw thin, clean bounding boxes
        for idx, obj in enumerate(objects):
            b = obj.get("bbox") or obj.get("box", [0, 0, 0, 0])
            x1 = max(0, min(w - 1, int(b[0])))
            y1 = max(0, min(h - 1, int(b[1])))
            x2 = max(0, min(w - 1, int(b[2])))
            y2 = max(0, min(h - 1, int(b[3])))

            if x2 <= x1 or y2 <= y1:
                continue

            # Thin, crisp 2px border
            draw.rectangle([x1, y1, x2, y2], outline=box_border_color, width=2)
            # Subtle interior fill
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=box_fill_color)

            # Intelligently place small non-overlapping numbered badge if sparse or top-left
            badge_num = str(idx + 1)
            badge_x = max(2, min(w - 22, x1))
            badge_y = max(2, min(h - 22, y1 - 18 if y1 > 20 else y1 + 2))

            # Draw small, compact 18x18 badge with index (never giant overlapping text)
            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + 18, badge_y + 16],
                radius=3,
                fill=(15, 23, 42, 235),
                outline=box_border_color,
                width=1,
            )
            draw.text((badge_x + 5, badge_y + 1), badge_num, fill=(255, 255, 255, 255))

        # 2. Top-right summary pill: '[ 8 buildings detected ]'
        total_count = len(objects)
        primary_class = list(counts.keys())[0] if counts else "object"
        count_label = f"{total_count} {primary_class}{'s' if total_count != 1 else ''} detected"
        pill_w = max(140, len(count_label) * 7 + 24)
        px, py = max(10, w - pill_w - 14), 14

        draw.rounded_rectangle(
            [px, py, px + pill_w, py + 26],
            radius=6,
            fill=(15, 23, 42, 235),
            outline=(71, 85, 105, 225),
            width=1,
        )
        draw.text((px + 12, py + 5), count_label, fill=(255, 255, 255, 255))

        # 3. Bottom-left simple legend pill: '□ Building detection'
        legend_txt = f"{primary_class.capitalize()} detection"
        leg_w = max(160, len(legend_txt) * 7 + 36)
        lx, ly = 14, max(14, h - 38)

        draw.rounded_rectangle(
            [lx, ly, lx + leg_w, ly + 24],
            radius=5,
            fill=(15, 23, 42, 230),
            outline=(51, 65, 85, 220),
            width=1,
        )
        draw.rectangle(
            [lx + 8, ly + 6, lx + 18, ly + 16],
            outline=box_border_color,
            fill=(6, 182, 212, 65),
            width=1,
        )
        draw.text((lx + 24, ly + 4), legend_txt, fill=(241, 245, 249, 255))

        # 4. Alpha composite onto clean original image
        composite = Image.alpha_composite(orig_img.convert("RGBA"), overlay_rgba).convert("RGB")

        if output_path is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = DEFAULT_OUTPUT_DIR / f"detection_{image_path.stem}_{timestamp}.png"

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        composite.save(out_p, format="PNG")
        return str(out_p.resolve())

    except Exception as e:
        logger_msg = f"Failed to generate clean detection visualization: {e}"
        print(logger_msg)
        return None


def detect_objects(
    image_path: Union[str, Path],
    conf: Optional[float] = None,
    save_visualization: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run YOLO11 object detection on a remote-sensing image.

    Returns structured results with exact coordinates, directional grounding,
    and a clean visual overlay preserving the satellite image without clutter.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return {
            "success": False,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {
                "objects": [],
                "counts": {},
                "total_count": 0,
                "average_confidence": 0.0,
                "dominant_direction": "none",
                "distribution_summary": "Image file not found.",
                "confidence_threshold": conf or DEFAULT_CONFIDENCE_THRESHOLD,
            },
            "visualization": None,
            "error": f"Image not found: {image_path}",
        }

    cfg = load_inference_config()
    resolved_conf = conf if conf is not None else cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    try:
        model = _get_model()

        # Run prediction with save=False to avoid default ugly overlapping labels
        predictions = model.predict(
            source=str(image_path),
            imgsz=cfg.get("imgsz", 640),
            conf=resolved_conf,
            max_det=cfg.get("max_det", 1089),
            save=False,
            verbose=False,
        )

        result = predictions[0]

        objects = []
        counts = Counter()

        # Get image dimensions
        orig_img = Image.open(image_path)
        img_w, img_h = orig_img.size

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()

                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = str(class_id)

                bbox_coords = [round(float(v), 2) for v in xyxy]
                cx = (bbox_coords[0] + bbox_coords[2]) / 2.0
                cy = (bbox_coords[1] + bbox_coords[3]) / 2.0

                dir_code, dir_label = compute_object_direction(cx, cy, img_w, img_h)

                objects.append(
                    {
                        "class": class_name,
                        "name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox_coords,
                        "box": bbox_coords,
                        "direction": dir_code,
                        "direction_label": dir_label,
                    }
                )
                counts[class_name] += 1

        # Calculate spatial concentration and grouped description
        primary_target = "building" if "building" in counts else (list(counts.keys())[0] if counts else "object")
        dominant_dir, dist_summary = calculate_spatial_distribution(objects, img_w, img_h, target_class=primary_target)

        avg_conf = (
            round(float(np.mean([o["confidence"] for o in objects])), 2)
            if objects
            else 0.0
        )

        # Generate clean, uncluttered visual overlay
        visualization = None
        if save_visualization:
            visualization = generate_clean_detection_visualization(
                image_path=image_path,
                objects=objects,
                counts=dict(counts),
                output_path=output_path,
            )

        return {
            "success": True,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {
                "objects": objects,
                "counts": dict(counts),
                "total_count": len(objects),
                "average_confidence": avg_conf,
                "dominant_direction": dominant_dir,
                "distribution_summary": dist_summary,
                "confidence_threshold": resolved_conf,
            },
            "visualization": visualization,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {
                "objects": [],
                "counts": {},
                "total_count": 0,
                "average_confidence": 0.0,
                "dominant_direction": "none",
                "distribution_summary": "Inference failed.",
                "confidence_threshold": resolved_conf,
            },
            "visualization": None,
            "error": str(e),
        }