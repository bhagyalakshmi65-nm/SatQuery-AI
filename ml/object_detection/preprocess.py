"""
SATQuery AI - Object Detection Preprocessing Pipeline
Module: ml/object_detection/preprocess.py

Converts raw satellite GeoTIFF scenes from the official DIUx xView dataset
into sliced, normalized, multi-class YOLO11-compatible tiles and labels.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, *args, **kwargs):
        if desc:
            print(f"Starting {desc}...")
        return iterable


import yaml
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration parameters from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_and_split_scenes(
    raw_images_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    manual_train: Optional[List[str]] = None,
    manual_val: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Discover all .tif scenes from raw_images_dir and split deterministically into train and val splits.
    Filters out any non-.tif files.
    
    If manual_train and manual_val are provided, they take precedence.
    Otherwise, automatically scans raw_images_dir for all .tif scenes and performs
    a deterministic train/val scene-level partition using a fixed random seed.
    """
    if manual_train and manual_val:
        return list(manual_train), list(manual_val)

    if not os.path.isdir(raw_images_dir):
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")

    # Discover all .tif files, ignoring case and any non-.tif files
    all_tif_files = sorted([
        f for f in os.listdir(raw_images_dir)
        if f.lower().endswith(".tif")
    ])

    if not all_tif_files:
        raise ValueError(f"No .tif scene files found in {raw_images_dir}")

    # Deterministic shuffle using fixed seed
    rng = random.Random(seed)
    shuffled_scenes = list(all_tif_files)
    rng.shuffle(shuffled_scenes)

    split_idx = int(round(len(shuffled_scenes) * train_ratio))
    train_scenes = sorted(shuffled_scenes[:split_idx])
    val_scenes = sorted(shuffled_scenes[split_idx:])

    return train_scenes, val_scenes


def build_class_lookup(config: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, str], List[str]]:
    """
    Build class mappings from configuration.
    
    Returns:
        xview_to_yolo: Mapping from xView type_id -> contiguous YOLO class_id (0..nc-1)
        yolo_to_name: Mapping from YOLO class_id -> class name
        class_names_list: Ordered list of class names by YOLO class_id
    """
    classes_cfg = config.get("classes", {})
    xview_to_yolo: Dict[int, int] = {}
    yolo_to_name: Dict[int, str] = {}

    for xview_id, meta in classes_cfg.items():
        yolo_id = int(meta["yolo_id"])
        xview_id = int(xview_id)
        name = str(meta["name"])
        xview_to_yolo[xview_id] = yolo_id
        yolo_to_name[yolo_id] = name

    num_classes = len(yolo_to_name)
    class_names_list = [yolo_to_name[i] for i in range(num_classes)]

    return xview_to_yolo, yolo_to_name, class_names_list


def load_scene_annotations(
    geojson_path: str,
    target_scene_ids: Optional[Set[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Index raw xView GeoJSON annotations grouped by image_id.
    
    Args:
        geojson_path: Path to xView_train.geojson on disk.
        target_scene_ids: Optional set of scene IDs to filter; if None, indexes all.
        
    Returns:
        Dictionary mapping scene_id (e.g. '1831.tif') to list of raw annotation dicts.
    """
    print(f"Loading annotations from: {geojson_path}")
    scene_annotations: Dict[str, List[Dict[str, Any]]] = {}

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feat in tqdm(data.get("features", []), desc="Indexing GeoJSON features"):
        props = feat.get("properties", {})
        image_id = props.get("image_id")
        if not image_id:
            continue
        if target_scene_ids is not None and image_id not in target_scene_ids:
            continue

        bounds_str = props.get("bounds_imcoords")
        if not bounds_str:
            continue

        try:
            coords = [int(val.strip()) for val in bounds_str.split(",")]
            if len(coords) != 4 or coords[0] >= coords[2] or coords[1] >= coords[3]:
                continue  # Skip inverted or malformed boxes
        except Exception:
            continue

        type_id = props.get("type_id")
        if type_id is None:
            continue

        if image_id not in scene_annotations:
            scene_annotations[image_id] = []

        scene_annotations[image_id].append({
            "bounds": coords,  # [xmin, ymin, xmax, ymax]
            "type_id": int(type_id)
        })

    print(f"Indexed annotations for {len(scene_annotations)} target scenes.")
    return scene_annotations


def slice_scene(
    scene_path: str,
    annotations: List[Dict[str, Any]],
    output_img_dir: str,
    output_lbl_dir: str,
    xview_to_yolo: Dict[int, int],
    tile_size: int = 640,
    overlap: float = 0.2,
    min_area_ratio: float = 0.3,
    empty_tile_retention: float = 0.1,
    tile_format: str = "jpg",
    jpg_quality: int = 95,
    rng: Optional[random.Random] = None
) -> Dict[str, int]:
    """
    Slice one raw satellite scene GeoTIFF into overlapping tiles and generate YOLO labels.
    
    Returns:
        Statistics dictionary for this scene.
    """
    if rng is None:
        rng = random.Random(42)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    scene_name = Path(scene_path).stem  # e.g. '1831'
    img = Image.open(scene_path)
    scene_w, scene_h = img.size

    step_size = int(tile_size * (1.0 - overlap))
    step_size = max(step_size, 1)

    # Compute grid coordinates with boundary coverage
    x_starts = list(range(0, max(scene_w - tile_size + 1, 1), step_size))
    if x_starts[-1] + tile_size < scene_w:
        x_starts.append(scene_w - tile_size)

    y_starts = list(range(0, max(scene_h - tile_size + 1, 1), step_size))
    if y_starts[-1] + tile_size < scene_h:
        y_starts.append(scene_h - tile_size)

    # Pre-parse scene bounding boxes into structured numpy array for fast intersection
    raw_boxes = []
    raw_types = []
    for ann in annotations:
        tid = ann["type_id"]
        if tid in xview_to_yolo:
            raw_boxes.append(ann["bounds"])
            raw_types.append(tid)

    if raw_boxes:
        boxes_arr = np.array(raw_boxes, dtype=np.int32)
        types_arr = np.array(raw_types, dtype=np.int32)
        orig_areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    else:
        boxes_arr = np.empty((0, 4), dtype=np.int32)
        types_arr = np.empty((0,), dtype=np.int32)
        orig_areas = np.empty((0,), dtype=np.int32)

    stats = {
        "tiles_saved": 0,
        "positive_tiles": 0,
        "empty_tiles": 0,
        "total_boxes_written": 0,
        "building_boxes_written": 0
    }

    for y0 in y_starts:
        y1 = min(y0 + tile_size, scene_h)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, scene_w)

            # Find intersecting annotations
            tile_labels = []
            if len(boxes_arr) > 0:
                # Intersection coordinates
                int_x1 = np.maximum(boxes_arr[:, 0], x0)
                int_y1 = np.maximum(boxes_arr[:, 1], y0)
                int_x2 = np.minimum(boxes_arr[:, 2], x1)
                int_y2 = np.minimum(boxes_arr[:, 3], y1)

                valid_mask = (int_x2 > int_x1) & (int_y2 > int_y1)
                if np.any(valid_mask):
                    clip_areas = (int_x2[valid_mask] - int_x1[valid_mask]) * (int_y2[valid_mask] - int_y1[valid_mask])
                    target_orig_areas = orig_areas[valid_mask]
                    area_ratios = clip_areas / np.maximum(target_orig_areas, 1)

                    retained_mask = area_ratios >= min_area_ratio

                    kept_int_x1 = int_x1[valid_mask][retained_mask]
                    kept_int_y1 = int_y1[valid_mask][retained_mask]
                    kept_int_x2 = int_x2[valid_mask][retained_mask]
                    kept_int_y2 = int_y2[valid_mask][retained_mask]
                    kept_types = types_arr[valid_mask][retained_mask]

                    for k in range(len(kept_int_x1)):
                        # Translate to tile-local coordinates
                        loc_x1 = kept_int_x1[k] - x0
                        loc_y1 = kept_int_y1[k] - y0
                        loc_x2 = kept_int_x2[k] - x0
                        loc_y2 = kept_int_y2[k] - y0

                        # Normalize to [0.0, 1.0]
                        xc = (loc_x1 + loc_x2) / (2.0 * tile_size)
                        yc = (loc_y1 + loc_y2) / (2.0 * tile_size)
                        bw = (loc_x2 - loc_x1) / float(tile_size)
                        bh = (loc_y2 - loc_y1) / float(tile_size)

                        # Clamp to valid normalized ranges
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        bw = max(0.0, min(1.0, bw))
                        bh = max(0.0, min(1.0, bh))

                        if bw > 0 and bh > 0:
                            yolo_cls = xview_to_yolo[kept_types[k]]
                            tile_labels.append((yolo_cls, xc, yc, bw, bh))

            # Decide whether to save tile
            has_labels = len(tile_labels) > 0
            if not has_labels:
                if rng.random() > empty_tile_retention:
                    continue  # Drop empty tile
                stats["empty_tiles"] += 1
            else:
                stats["positive_tiles"] += 1

            # Extract tile image
            tile_img = img.crop((x0, y0, x0 + tile_size, y0 + tile_size))
            if tile_img.size != (tile_size, tile_size):
                # Pad to square if on boundary
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile_img, (0, 0))
                tile_img = padded

            tile_stem = f"{scene_name}_x{x0}_y{y0}"
            tile_img_path = os.path.join(output_img_dir, f"{tile_stem}.{tile_format}")
            tile_lbl_path = os.path.join(output_lbl_dir, f"{tile_stem}.txt")

            if tile_format.lower() in ("jpg", "jpeg"):
                tile_img.save(tile_img_path, quality=jpg_quality)
            else:
                tile_img.save(tile_img_path)

            # Write label file (empty file for negative background tile)
            with open(tile_lbl_path, "w", encoding="utf-8") as f:
                for cls_id, xc, yc, bw, bh in tile_labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    stats["total_boxes_written"] += 1
                    if cls_id == 0:  # Building is designated Class 0
                        stats["building_boxes_written"] += 1

            stats["tiles_saved"] += 1

    img.close()
    return stats


def generate_dataset_yaml(
    config: Dict[str, Any],
    class_names_list: List[str],
    output_path: str
) -> None:
    """Generate YOLO11 dataset configuration file."""
    processed_dir = Path(config["dataset"]["processed_dir"]).resolve()
    
    yaml_content = {
        "path": str(processed_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names_list),
        "names": {i: name for i, name in enumerate(class_names_list)}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"Generated YOLO dataset YAML at: {output_path}")


def run_preprocessing(
    config_path: str = "config.yaml",
    max_scenes: Optional[int] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Execute complete preprocessing pipeline according to config."""
    config = load_config(config_path)
    raw_img_dir = config["dataset"]["raw_images_dir"]
    raw_geojson = config["dataset"]["raw_geojson_path"]
    processed_dir = config["dataset"]["processed_dir"]
    dataset_yaml_path = config["dataset"]["dataset_yaml_path"]

    tiling_cfg = config.get("tiling", {})
    tile_size = tiling_cfg.get("tile_size", 640)
    overlap = tiling_cfg.get("overlap", 0.2)
    min_area_ratio = tiling_cfg.get("min_area_ratio", 0.3)
    empty_retention = tiling_cfg.get("empty_tile_retention", 0.1)
    tile_format = tiling_cfg.get("tile_format", "jpg")
    jpg_quality = tiling_cfg.get("jpg_quality", 95)

    xview_to_yolo, yolo_to_name, class_names_list = build_class_lookup(config)
    print(f"Configured {len(class_names_list)} multi-class targets. Primary target: '{class_names_list[0]}' (YOLO ID 0).")

    scenes_cfg = config.get("scenes", {})
    train_ratio = float(scenes_cfg.get("train_ratio", 0.8))
    seed = int(scenes_cfg.get("seed", 42))
    manual_train = scenes_cfg.get("train", None)
    manual_val = scenes_cfg.get("val", None)

    train_scenes, val_scenes = discover_and_split_scenes(
        raw_images_dir=raw_img_dir,
        train_ratio=train_ratio,
        seed=seed,
        manual_train=manual_train if manual_train else None,
        manual_val=manual_val if manual_val else None
    )

    if max_scenes:
        train_scenes = train_scenes[:max(1, int(max_scenes * train_ratio))]
        val_scenes = val_scenes[:max(1, int(max_scenes * (1.0 - train_ratio)))]

    all_target_scenes = set(train_scenes) | set(val_scenes)
    print(f"Discovered {len(all_target_scenes)} total .tif scenes across splits.")
    print(f"Deterministic {int(train_ratio*100)}/{int(round((1.0-train_ratio)*100))} partition (seed={seed}): {len(train_scenes)} train scenes, {len(val_scenes)} val scenes.")

    annotations = load_scene_annotations(raw_geojson, target_scene_ids=all_target_scenes)

    if dry_run:
        print("[DRY RUN] Inspection complete. Zero files written.")
        return {"status": "dry_run_success", "scenes_indexed": len(annotations)}

    rng = random.Random(42)
    overall_stats = {
        "train": Counter(),
        "val": Counter()
    }

    # Process Train Split
    train_img_dir = os.path.join(processed_dir, "train", "images")
    train_lbl_dir = os.path.join(processed_dir, "train", "labels")
    print("\n--- Slicing Training Scenes ---")
    for s_id in tqdm(train_scenes, desc="Train scenes"):
        s_path = os.path.join(raw_img_dir, s_id)
        if not os.path.exists(s_path):
            print(f"WARNING: Scene not found: {s_path}")
            continue
        scene_anns = annotations.get(s_id, [])
        s_stats = slice_scene(
            s_path, scene_anns, train_img_dir, train_lbl_dir,
            xview_to_yolo, tile_size, overlap, min_area_ratio,
            empty_retention, tile_format, jpg_quality, rng
        )
        overall_stats["train"].update(s_stats)

    # Process Val Split
    val_img_dir = os.path.join(processed_dir, "val", "images")
    val_lbl_dir = os.path.join(processed_dir, "val", "labels")
    print("\n--- Slicing Validation Scenes ---")
    for s_id in tqdm(val_scenes, desc="Val scenes"):
        s_path = os.path.join(raw_img_dir, s_id)
        if not os.path.exists(s_path):
            print(f"WARNING: Scene not found: {s_path}")
            continue
        scene_anns = annotations.get(s_id, [])
        s_stats = slice_scene(
            s_path, scene_anns, val_img_dir, val_lbl_dir,
            xview_to_yolo, tile_size, overlap, min_area_ratio,
            empty_retention, tile_format, jpg_quality, rng
        )
        overall_stats["val"].update(s_stats)

    # Generate dataset.yaml for YOLO11
    generate_dataset_yaml(config, class_names_list, dataset_yaml_path)

    print("\n================ PREPROCESSING SUMMARY ================")
    for split in ("train", "val"):
        st = overall_stats[split]
        print(f"Split [{split.upper()}]:")
        print(f"  Tiles saved: {st['tiles_saved']} (Positive: {st['positive_tiles']}, Background negatives: {st['empty_tiles']})")
        print(f"  Total bounding boxes written: {st['total_boxes_written']}")
        print(f"  Building boxes written (Class 0): {st['building_boxes_written']}")
    print("=======================================================\n")

    return overall_stats


def parse_args():
    parser = argparse.ArgumentParser(description="SATQuery AI xView to YOLO11 Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional maximum number of scenes to process (for testing).")
    parser.add_argument("--dry-run", action="store_true", help="Perform GeoJSON index and validation without writing files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(config_path=args.config, max_scenes=args.max_scenes, dry_run=args.dry_run)
