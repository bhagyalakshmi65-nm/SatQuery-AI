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


# Load the trained V1 model once
_model = None


def _get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

    return _model


def detect_objects(image_path, conf=0.25, save_visualization=True):
    """
    Run object detection on a remote-sensing image.

    Returns a dictionary suitable for SATQuery AI integration.
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
            },
            "visualization": None,
            "error": f"Image not found: {image_path}",
        }

    try:
        model = _get_model()

        predictions = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=conf,
            max_det=1089,
            save=save_visualization,
            verbose=False,
        )

        result = predictions[0]

        objects = []
        counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                xyxy = box.xyxy[0].tolist()

                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = str(class_id)

                objects.append(
                    {
                        "class": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": [round(float(v), 2) for v in xyxy],
                    }
                )

                counts[class_name] += 1

        visualization = None

        if save_visualization:
            save_dir = Path(result.save_dir)
            visualization = str(
                save_dir / image_path.name
            )

        return {
            "success": True,
            "model": "YOLO11",
            "image": str(image_path),
            "results": {
                "objects": objects,
                "counts": dict(counts),
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
            },
            "visualization": None,
            "error": str(e),
        }