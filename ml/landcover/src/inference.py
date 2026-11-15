"""
Inference and Input Validation Pipeline for 12-Band Multispectral Land-Cover Classification.

Model: BigEarthNet-ResNet50-SeCo-12B
Input Contract: Strictly 12 Sentinel-2 Level-2A surface reflectance bands:
[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] in (12, 120, 120) grid.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tifffile
import torch

from .dataset import (
    SENTINEL2_12_BANDS,
    SECO_S2_12BAND_MEAN,
    SECO_S2_12BAND_STD,
    load_12band_sentinel2_raster,
)
from .labels import BIGEARTHNET_19_CLASSES
from .model import LandCoverClassifier
from .visualization import plot_prediction_visualization

logger = logging.getLogger("ml.landcover.inference")

_CACHED_MODEL: Optional[LandCoverClassifier] = None


def get_default_weights_path() -> str:
    """Resolve path to local trained checkpoint."""
    base_dir = Path(__file__).resolve().parent.parent
    candidate = base_dir / "weights" / "best_model.pth"
    if candidate.exists():
        return str(candidate)
    return str(candidate)


def load_classifier(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> LandCoverClassifier:
    """Loads and caches the 12-band LandCoverClassifier model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    weights_path = checkpoint_path or get_default_weights_path()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LandCoverClassifier(
        backbone="resnet50",
        num_classes=len(BIGEARTHNET_19_CLASSES),
        in_channels=12,
        pretrained=False,
    )

    if os.path.exists(weights_path):
        logger.info("Loading BigEarthNet weights from %s", weights_path)
        model.load_weights(weights_path, map_location=dev)
    else:
        logger.warning("Checkpoint not found at %s; instantiated with uninitialized weights", weights_path)

    model.to(dev)
    model.eval()
    _CACHED_MODEL = model
    return _CACHED_MODEL


def validate_landcover_input(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validates whether an input image or directory satisfies the 12-band Sentinel-2 requirement.

    Returns:
        Dict with keys:
            - valid (bool): True if genuine 12-band Sentinel-2 input, False otherwise.
            - modality (str): 'sentinel2_multispectral', 'rgb', or 'unknown'.
            - bands (int): Number of detected spectral bands.
            - message (str): Clear, user-facing validation explanation.
    """
    p = Path(image_path)
    if not p.exists():
        return {
            "valid": False,
            "modality": "unknown",
            "bands": 0,
            "message": f"Input path does not exist: {image_path}",
        }

    # Case 1: BigEarthNet patch directory containing separate band TIFFs
    if p.is_dir():
        found_bands = []
        for band_id in SENTINEL2_12_BANDS:
            matches = glob.glob(str(p / f"*{band_id}.tif")) + glob.glob(str(p / f"*{band_id}.tiff"))
            if matches:
                found_bands.append(band_id)
        if len(found_bands) == 12:
            return {
                "valid": True,
                "modality": "sentinel2_multispectral",
                "bands": 12,
                "message": "Valid BigEarthNet-S2 12-band patch directory.",
            }
        return {
            "valid": False,
            "modality": "incomplete",
            "bands": len(found_bands),
            "message": f"Patch directory contains only {len(found_bands)}/12 required Sentinel-2 bands.",
        }

    suffix = p.suffix.lower()

    # Case 2: Standard RGB optical image formats (PNG, JPG, BMP, WEBP)
    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        return {
            "valid": False,
            "modality": "rgb",
            "bands": 3,
            "message": "Multispectral land-cover classification requires 12-band Sentinel-2 surface reflectance. The provided input is 3-channel visual imagery (RGB).",
        }

    # Case 3: GeoTIFF (.tif, .tiff)
    if suffix in [".tif", ".tiff"]:
        try:
            num_bands = 0
            try:
                import rasterio
                with rasterio.open(p) as src:
                    num_bands = src.count
            except Exception:
                pass

            # If rasterio reported 1 (or failed), check if tifffile detects multi-page / 3D cube
            if num_bands != 12:
                try:
                    raw_data = tifffile.imread(str(p))
                    if raw_data.ndim == 3:
                        c = raw_data.shape[0] if raw_data.shape[0] < raw_data.shape[1] else raw_data.shape[-1]
                        if c == 12 or (num_bands <= 1 and c > 1):
                            num_bands = c
                except Exception:
                    pass

            if num_bands == 12:
                return {
                    "valid": True,
                    "modality": "sentinel2_multispectral",
                    "bands": 12,
                    "message": "Valid 12-band Sentinel-2 surface reflectance GeoTIFF.",
                }
            elif num_bands in [3, 4]:
                return {
                    "valid": False,
                    "modality": "rgb",
                    "bands": num_bands,
                    "message": f"Input raster has {num_bands} channels (visual RGB). Multispectral land-cover classification requires 12-band Sentinel-2 surface reflectance.",
                }
            else:
                return {
                    "valid": False,
                    "modality": "unsupported",
                    "bands": num_bands,
                    "message": f"Input raster has {num_bands} bands. Multispectral classification requires exactly 12 calibrated Sentinel-2 bands.",
                }
        except Exception as e:
            return {
                "valid": False,
                "modality": "unknown",
                "bands": 0,
                "message": f"Unable to parse raster structure: {e}",
            }

    # Case 4: NumPy array (.npy, .npz)
    if suffix in [".npy", ".npz"]:
        try:
            if suffix == ".npy":
                arr = np.load(str(p))
            else:
                data = np.load(str(p))
                arr = data[list(data.keys())[0]]
            
            if arr.ndim == 3:
                channels = arr.shape[0] if arr.shape[0] < arr.shape[1] else arr.shape[-1]
                if channels == 12:
                    return {
                        "valid": True,
                        "modality": "sentinel2_multispectral",
                        "bands": 12,
                        "message": "Valid 12-channel Sentinel-2 array.",
                    }
                return {
                    "valid": False,
                    "modality": "rgb" if channels in [3, 4] else "unsupported",
                    "bands": channels,
                    "message": f"Array contains {channels} channels; 12 Sentinel-2 bands required.",
                }
        except Exception as e:
            return {
                "valid": False,
                "modality": "unknown",
                "bands": 0,
                "message": f"Failed reading numpy array: {e}",
            }

    return {
        "valid": False,
        "modality": "unsupported",
        "bands": 0,
        "message": f"Unsupported file format for land-cover classification: '{suffix}'.",
    }


def classify_landcover(
    image_path: str,
    threshold: float = 0.35,
    visualize: bool = True,
    output_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes BigEarthNet 12-band land-cover classification on verified Sentinel-2 inputs.

    Strictly rejects RGB imagery and will never fabricate predictions on invalid inputs.
    """
    validation = validate_landcover_input(image_path)
    if not validation["valid"]:
        logger.info(
            "Rejecting land-cover inference for %s: %s (modality: %s, bands: %d)",
            image_path,
            validation["message"],
            validation["modality"],
            validation["bands"],
        )
        return {
            "success": False,
            "model": "BigEarthNet-ResNet50-SeCo-12B",
            "image": str(image_path),
            "modality": validation["modality"],
            "bands": validation["bands"],
            "results": {"classes": []},
            "visualization": None,
            "error": validation["message"],
        }

    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_classifier(checkpoint_path=checkpoint_path, device=dev)

        # Load 12-band cube: shape (12, 120, 120), values in [0.0, 1.0]
        cube = load_12band_sentinel2_raster(image_path, target_shape=(120, 120))

        # Normalize with SeCo statistics
        norm_cube = np.zeros_like(cube, dtype=np.float32)
        for b in range(12):
            norm_cube[b] = (cube[b] - SECO_S2_12BAND_MEAN[b]) / (SECO_S2_12BAND_STD[b] + 1e-6)

        tensor = torch.from_numpy(norm_cube).unsqueeze(0).to(dev)

        with torch.no_grad():
            probs = model.predict_proba(tensor).squeeze(0).cpu().numpy()

        # Map to class predictions
        pred_items = []
        for idx, cls_name in enumerate(BIGEARTHNET_19_CLASSES):
            prob = float(probs[idx])
            pred_items.append({"class": cls_name, "confidence": prob})

        # Sort descending by confidence
        pred_items.sort(key=lambda x: x["confidence"], reverse=True)

        # Filter by threshold
        selected_classes = [c for c in pred_items if c["confidence"] >= threshold]
        if not selected_classes and pred_items:
            # If none exceed threshold, keep top 3 informative classes
            selected_classes = pred_items[:3]

        viz_path = None
        if visualize:
            out_d = output_dir or os.path.dirname(os.path.abspath(image_path))
            base_name = Path(image_path).stem
            out_file = os.path.join(out_d, f"prediction_{base_name}.png")
            try:
                viz_path = plot_prediction_visualization(
                    image_path=image_path,
                    predictions=selected_classes,
                    output_path=out_file,
                    threshold=threshold,
                )
            except Exception as e:
                logger.warning("Visualization plotting failed: %s", e)

        return {
            "success": True,
            "model": "BigEarthNet-ResNet50-SeCo-12B",
            "image": str(image_path),
            "modality": "sentinel2_multispectral",
            "bands": 12,
            "results": {
                "classes": selected_classes,
                "all_probabilities": {c["class"]: round(c["confidence"], 4) for c in pred_items},
            },
            "visualization": viz_path,
            "error": None,
        }

    except Exception as e:
        logger.error("Error executing land-cover classifier on %s: %s", image_path, e, exc_info=True)
        return {
            "success": False,
            "model": "BigEarthNet-ResNet50-SeCo-12B",
            "image": str(image_path),
            "modality": "sentinel2_multispectral",
            "bands": 12,
            "results": {"classes": []},
            "visualization": None,
            "error": str(e),
        }
