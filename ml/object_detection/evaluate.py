"""
SATQuery AI - Object Detection Evaluation Module
Module: ml/object_detection/evaluate.py

Evaluates fine-tuned YOLO11 checkpoints on the processed xView validation split.
Computes mAP@50, mAP@50-95, Precision, Recall (both overall and per-class),
and renders qualitative prediction overlays with bounding boxes, labels, and confidence scores.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import yaml


def evaluate_model(
    weights_path: str,
    data_yaml_path: str,
    device: str = "cpu",
    batch: int = 4,
    imgsz: int = 640,
    save_dir: str = "runs/detect/val_eval",
    visualize_count: int = 5,
    val_conf: Optional[float] = 0.001,
    vis_conf: float = 0.25,
    iou_thresh: float = 0.6
) -> Dict[str, Any]:
    """
    Run quantitative validation and qualitative visualization using the specified YOLO11 weights.
    """
    print("=" * 70)
    print("SATQuery AI — YOLO11 Model Evaluation")
    print("=" * 70)
    print(f"Weights path:      {weights_path}")
    print(f"Dataset YAML path: {data_yaml_path}")
    print(f"Device:            {device}")
    print(f"Image resolution:  {imgsz}x{imgsz}")
    print(f"Batch size:        {batch}")
    print(f"Val Conf Thresh:   {val_conf} (for standard mAP integration)")
    print(f"Vis Conf Thresh:   {vis_conf} (for prediction visualizations)")
    print("=" * 70)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml_path}")

    # Load model
    print("\n[1/4] Loading YOLO11 checkpoint...")
    model = YOLO(weights_path)
    num_classes = len(model.names)
    print(f"Model loaded successfully. Configured classes: {num_classes}")
    assert num_classes == 60, f"Expected 60 classes in checkpoint, found {num_classes}"
    print(f"Class 0: '{model.names.get(0)}' (PRIMARY TARGET)")

    # Run official Ultralytics validation
    print("\n[2/4] Running quantitative validation on validation split...")
    save_path = Path(save_dir)
    save_project = str(save_path.parent) if save_path.parent != Path(".") else "runs/detect"
    save_name = save_path.name

    val_kwargs = {
        "data": data_yaml_path,
        "split": "val",
        "device": device,
        "batch": batch,
        "workers": 0,
        "imgsz": imgsz,
        "iou": iou_thresh,
        "project": save_project,
        "name": save_name,
        "exist_ok": True,
        "verbose": True
    }
    if val_conf is not None:
        val_kwargs["conf"] = val_conf

    metrics = model.val(**val_kwargs)

    box_metrics = metrics.box
    overall_p = float(box_metrics.mp)
    overall_r = float(box_metrics.mr)
    overall_map50 = float(box_metrics.map50)
    overall_map50_95 = float(box_metrics.map)

    # Class-specific metrics
    per_class_results = {}
    class_indices = box_metrics.ap_class_index.tolist() if hasattr(box_metrics.ap_class_index, "tolist") else list(box_metrics.ap_class_index)

    print("\n[3/4] Extracting per-class performance...")
    for idx, cls_idx in enumerate(class_indices):
        cls_name = model.names[cls_idx]
        try:
            p, r, ap50, ap = box_metrics.class_result(idx)
        except Exception:
            p = float(box_metrics.p[idx]) if hasattr(box_metrics, "p") and len(box_metrics.p) > idx else 0.0
            r = float(box_metrics.r[idx]) if hasattr(box_metrics, "r") and len(box_metrics.r) > idx else 0.0
            ap50 = float(box_metrics.ap50[idx]) if hasattr(box_metrics, "ap50") and len(box_metrics.ap50) > idx else 0.0
            ap = float(box_metrics.ap[idx]) if hasattr(box_metrics, "ap") and len(box_metrics.ap) > idx else 0.0

        per_class_results[cls_name] = {
            "class_id": int(cls_idx),
            "precision": float(p),
            "recall": float(r),
            "map50": float(ap50),
            "map50_95": float(ap)
        }

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Class Name':<25} | {'Class ID':<8} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    print("-" * 75)
    print(f"{'ALL CLASSES (Macro)':<25} | {'-':<8} | {overall_p:<10.4f} | {overall_r:<10.4f} | {overall_map50:<10.4f} | {overall_map50_95:<10.4f}")
    print("-" * 75)

    for target_name in ("building", "small_car"):
        if target_name in per_class_results:
            c_data = per_class_results[target_name]
            print(f"{target_name:<25} | {c_data['class_id']:<8} | {c_data['precision']:<10.4f} | {c_data['recall']:<10.4f} | {c_data['map50']:<10.4f} | {c_data['map50_95']:<10.4f}")
    print("=" * 75)

    # Qualitative Visualizations
    print(f"\n[4/4] Generating qualitative prediction visualizations ({visualize_count} images)...")
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    val_img_dir = os.path.join(data_cfg["path"], data_cfg["val"])
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))

    selected_samples = []
    if val_images:
        step = max(1, len(val_images) // visualize_count)
        selected_samples = val_images[::step][:visualize_count]

    generated_vis_paths = []
    for s_path in selected_samples:
        s_stem = Path(s_path).stem
        pred_results = model.predict(source=s_path, device=device, conf=vis_conf, iou=0.45, verbose=False)
        r = pred_results[0]

        # Save annotated image plot from Ultralytics
        out_path = os.path.join(vis_dir, f"vis_{s_stem}.jpg")
        r.save(filename=out_path)
        generated_vis_paths.append(out_path)

        det_count = len(r.boxes)
        bld_count = sum(1 for b in r.boxes if int(b.cls[0]) == 0)
        print(f"  Saved: {os.path.basename(out_path)} (Total detections: {det_count}, Buildings: {bld_count})")

    report_summary = {
        "weights": weights_path,
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "map50": overall_map50,
            "map50_95": overall_map50_95
        },
        "per_class": per_class_results,
        "visualizations": generated_vis_paths
    }

    return report_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned YOLO11 model on xView validation split.")
    parser.add_argument("--weights", type=str, default="runs/detect/runs/detect/train/weights/best.pt", help="Path to checkpoint.")
    parser.add_argument("--data", type=str, default="ml/object_detection/data/processed/dataset.yaml", help="Path to dataset YAML.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on ('cpu' or '0').")
    parser.add_argument("--batch", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input resolution.")
    parser.add_argument("--save-dir", type=str, default="runs/detect/val_eval", help="Output directory for evaluations.")
    parser.add_argument("--visualize", type=int, default=5, help="Number of qualitative images to render.")
    parser.add_argument("--val-conf", type=float, default=0.001, help="Validation confidence threshold for mAP integration.")
    parser.add_argument("--vis-conf", type=float, default=0.25, help="Visualization confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        weights_path=args.weights,
        data_yaml_path=args.data,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        save_dir=args.save_dir,
        visualize_count=args.visualize,
        val_conf=args.val_conf,
        vis_conf=args.vis_conf,
        iou_thresh=args.iou
    )
