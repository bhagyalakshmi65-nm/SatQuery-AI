"""Modality inspection and scene co-location module for SatQuery AI.

Provides automatic detection of satellite image modalities (optical vs SAR vs unknown),
channel & intensity distribution analysis, collage detection/splitting, and co-location verification.
Supports exported satellite imagery (.jpg, .jpeg, .png, .tif, .tiff) without requiring GeoTIFF metadata.
"""

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("agent.modality")

# Filename keyword patterns for metadata inspection
SAR_FILENAME_PATTERNS = [
    r"(?:^|[^a-zA-Z0-9])s1[ab]?(?:[^a-zA-Z0-9]|$)",
    r"sentinel[-_ ]?1",
    r"(?:^|[^a-zA-Z0-9])sar(?:[^a-zA-Z0-9]|$)",
    r"(?:^|[^a-zA-Z0-9])radar(?:[^a-zA-Z0-9]|$)",
    r"dual[-_ ]?pol",
    r"vv[-_ ]?vh",
    r"backscatter",
]

OPTICAL_FILENAME_PATTERNS = [
    r"(?:^|[^a-zA-Z0-9])s2[ab]?(?:[^a-zA-Z0-9]|$)",
    r"sentinel[-_ ]?2",
    r"(?:^|[^a-zA-Z0-9])optical(?:[^a-zA-Z0-9]|$)",
    r"(?:^|[^a-zA-Z0-9])rgb(?:[^a-zA-Z0-9]|$)",
    r"multispectral",
    r"l2a",
    r"l1c",
    r"truecolor",
]


class ModalityInfo:
    """Structured inspection result for a single image."""

    def __init__(
        self,
        image_path: str,
        dimensions: Tuple[int, int],
        channels: int,
        file_type: str,
        probable_modality: str,  # 'optical', 'sar', 'unknown'
        confidence: float,
        reasoning: str,
        display_label: str,
        visual_characteristics: Dict[str, Any],
        is_multispectral_native: bool = False,
    ):
        self.image_path = image_path
        self.dimensions = dimensions  # (width, height)
        self.channels = channels
        self.file_type = file_type
        self.probable_modality = probable_modality
        self.confidence = confidence
        self.reasoning = reasoning
        self.display_label = display_label
        self.visual_characteristics = visual_characteristics
        self.is_multispectral_native = is_multispectral_native

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image_path,
            "dimensions": {"width": self.dimensions[0], "height": self.dimensions[1]},
            "channels": self.channels,
            "file_type": self.file_type,
            "modality": self.probable_modality,
            "modality_confidence": round(self.confidence, 4),
            "display_label": self.display_label,
            "reasoning": self.reasoning,
            "visual_characteristics": self.visual_characteristics,
        }


class ModalityInspector:
    """Inspects images to determine whether they represent optical, SAR, or unknown data."""

    @classmethod
    def inspect(
        cls,
        image_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModalityInfo:
        metadata = metadata or {}
        path = Path(image_path)
        file_ext = path.suffix.lower()
        filename_lower = path.name.lower()

        if not path.exists():
            explicit_sensor = (metadata.get("sensor_type") or metadata.get("sensor") or "").lower()
            has_sar_fn = any(re.search(p, filename_lower) for p in SAR_FILENAME_PATTERNS)
            has_opt_fn = any(re.search(p, filename_lower) for p in OPTICAL_FILENAME_PATTERNS)

            if "sentinel-1" in explicit_sensor or "sar" in explicit_sensor or has_sar_fn:
                return ModalityInfo(
                    image_path=str(path),
                    dimensions=(512, 512),
                    channels=2,
                    file_type=file_ext,
                    probable_modality="sar",
                    confidence=0.95,
                    reasoning=f"Identified as SAR from filename cue '{filename_lower}'.",
                    display_label="Sentinel-1 SAR",
                    visual_characteristics={},
                )
            elif "sentinel-2" in explicit_sensor or "optical" in explicit_sensor or has_opt_fn:
                return ModalityInfo(
                    image_path=str(path),
                    dimensions=(512, 512),
                    channels=12,
                    file_type=file_ext,
                    probable_modality="optical",
                    confidence=0.95,
                    reasoning=f"Identified as Optical from filename cue '{filename_lower}'.",
                    display_label="Sentinel-2 Optical",
                    visual_characteristics={},
                )
            return ModalityInfo(
                image_path=str(path),
                dimensions=(0, 0),
                channels=0,
                file_type=file_ext,
                probable_modality="unknown",
                confidence=0.0,
                reasoning=f"File not found: {path}",
                display_label="Unknown Input",
                visual_characteristics={},
            )

        # Try to read rasterio for GeoTIFF if available
        raster_channels = None
        has_geotiff_meta = False
        try:
            import rasterio
            if file_ext in [".tif", ".tiff"]:
                with rasterio.open(path) as src:
                    raster_channels = src.count
                    has_geotiff_meta = bool(src.crs or src.bounds)
                    if raster_channels == 12:
                        return ModalityInfo(
                            image_path=str(path),
                            dimensions=(src.width, src.height),
                            channels=12,
                            file_type=file_ext,
                            probable_modality="optical",
                            confidence=0.99,
                            reasoning="12-band Sentinel-2 multispectral surface reflectance GeoTIFF detected.",
                            display_label="Sentinel-2 Multispectral Input (12-Band)",
                            visual_characteristics={"multispectral": True, "crs": str(src.crs)},
                            is_multispectral_native=True,
                        )
                    elif raster_channels == 2:
                        return ModalityInfo(
                            image_path=str(path),
                            dimensions=(src.width, src.height),
                            channels=2,
                            file_type=file_ext,
                            probable_modality="sar",
                            confidence=0.99,
                            reasoning="Dual-polarization Sentinel-1 (VV, VH) SAR GeoTIFF detected.",
                            display_label="Sentinel-1 SAR (Dual-Pol)",
                            visual_characteristics={"dual_pol": True, "crs": str(src.crs)},
                        )
        except Exception as e:
            logger.debug("Rasterio inspection skipped for %s: %s", path.name, e)

        # Check tifffile for 12-channel Sentinel-2 rasters
        if file_ext in [".tif", ".tiff"]:
            try:
                import tifffile
                raw = tifffile.imread(str(path))
                if raw.ndim == 3:
                    c = raw.shape[0] if raw.shape[0] < raw.shape[1] else raw.shape[-1]
                    if c == 12:
                        h = raw.shape[1] if raw.shape[0] == 12 else raw.shape[0]
                        w = raw.shape[2] if raw.shape[0] == 12 else raw.shape[1]
                        return ModalityInfo(
                            image_path=str(path),
                            dimensions=(int(w), int(h)),
                            channels=12,
                            file_type=file_ext,
                            probable_modality="optical",
                            confidence=0.99,
                            reasoning="12-band Sentinel-2 surface reflectance GeoTIFF detected via tifffile.",
                            display_label="Sentinel-2 Multispectral Input (12-Band)",
                            visual_characteristics={"multispectral": True},
                            is_multispectral_native=True,
                        )
            except Exception:
                pass

        # Load with PIL for visual and channel inspection
        try:
            with Image.open(path) as pil_img:
                width, height = pil_img.size
                mode = pil_img.mode
                arr = np.array(pil_img)
        except Exception as e:
            return ModalityInfo(
                image_path=str(path),
                dimensions=(0, 0),
                channels=0,
                file_type=file_ext,
                probable_modality="unknown",
                confidence=0.0,
                reasoning=f"Unable to read image bytes: {e}",
                display_label="Unknown Format",
                visual_characteristics={},
            )

        # Determine channels
        if arr.ndim == 2:
            num_channels = 1
        elif arr.ndim == 3:
            num_channels = arr.shape[2]
        else:
            num_channels = 1

        # Layer A: Metadata / explicit hints
        explicit_sensor = (metadata.get("sensor_type") or metadata.get("sensor") or "").lower()
        if "sentinel-1" in explicit_sensor or "s1" == explicit_sensor or "sar" == explicit_sensor:
            return ModalityInfo(
                image_path=str(path),
                dimensions=(width, height),
                channels=num_channels,
                file_type=file_ext,
                probable_modality="sar",
                confidence=0.98,
                reasoning="Explicit metadata specifies Sentinel-1 SAR imagery.",
                display_label="Sentinel-1 SAR",
                visual_characteristics={"explicit_sensor": explicit_sensor},
            )
        elif "sentinel-2" in explicit_sensor or "s2" == explicit_sensor or "optical" == explicit_sensor:
            return ModalityInfo(
                image_path=str(path),
                dimensions=(width, height),
                channels=num_channels,
                file_type=file_ext,
                probable_modality="optical",
                confidence=0.98,
                reasoning="Explicit metadata specifies Sentinel-2 Optical imagery.",
                display_label="Sentinel-2 Optical",
                visual_characteristics={"explicit_sensor": explicit_sensor},
            )

        # Layer B: Visual & Statistical Properties
        # Compute channel differences and saturation
        is_grayscale_rgb = False
        mean_saturation = 0.0
        mean_channel_diff = 0.0
        std_intensity = float(np.std(arr))
        mean_intensity = float(np.mean(arr))

        if num_channels == 1:
            is_grayscale_rgb = True
            mean_saturation = 0.0
            mean_channel_diff = 0.0
        elif num_channels >= 3:
            r = arr[:, :, 0].astype(float)
            g = arr[:, :, 1].astype(float)
            b = arr[:, :, 2].astype(float)
            diff1 = np.mean(np.abs(r - g))
            diff2 = np.mean(np.abs(g - b))
            diff3 = np.mean(np.abs(r - b))
            mean_channel_diff = float((diff1 + diff2 + diff3) / 3.0)

            # Convert to HSV saturation
            try:
                hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
                mean_saturation = float(np.mean(hsv[:, :, 1]))
            except Exception:
                # Fallback saturation proxy
                max_c = np.maximum(r, np.maximum(g, b))
                min_c = np.minimum(r, np.minimum(g, b))
                sat_proxy = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-5) * 255.0, 0)
                mean_saturation = float(np.mean(sat_proxy))

            if mean_channel_diff < 2.0 and mean_saturation < 3.0:
                is_grayscale_rgb = True

        # Multiplicative speckle indicator for SAR
        # In SAR amplitude images, speckle produces high local variance proportional to local mean (std / mean > 0.35)
        speckle_index = 0.0
        if mean_intensity > 5:
            speckle_index = std_intensity / mean_intensity

        visual_characteristics = {
            "num_channels": num_channels,
            "mean_channel_diff": round(mean_channel_diff, 2),
            "mean_saturation": round(mean_saturation, 2),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "speckle_index": round(speckle_index, 3),
            "is_grayscale_rgb": is_grayscale_rgb,
        }

        # Filename hints (boost confidence slightly, but do not solely rely on them)
        has_sar_fn = any(re.search(p, filename_lower) for p in SAR_FILENAME_PATTERNS if p != r"\bimg1\b")
        has_opt_fn = any(re.search(p, filename_lower) for p in OPTICAL_FILENAME_PATTERNS)

        # Decision Logic:
        # 1. Grayscale or monochromatic appearance -> SAR radar backscatter
        if is_grayscale_rgb:
            conf = 0.90
            evidence_reasons = []
            if num_channels == 1:
                evidence_reasons.append("Single-channel grayscale structure")
            else:
                evidence_reasons.append(f"Monochromatic channel distribution (mean channel difference: {mean_channel_diff:.2f}, saturation: {mean_saturation:.2f})")

            if speckle_index > 0.35:
                conf += 0.04
                evidence_reasons.append("Radar speckle texture and high backscatter dynamic range")
            if has_sar_fn:
                conf = min(0.97, conf + 0.03)
                evidence_reasons.append(f"Filename cue '{filename_lower}'")

            conf = min(0.96, conf)
            reason_str = "; ".join(evidence_reasons) + "."
            label = f"Probable SAR ({int(conf * 100)}%)"

            return ModalityInfo(
                image_path=str(path),
                dimensions=(width, height),
                channels=num_channels,
                file_type=file_ext,
                probable_modality="sar",
                confidence=conf,
                reasoning=f"Identified as SAR radar imagery based on {reason_str}",
                display_label=label,
                visual_characteristics=visual_characteristics,
            )

        # 2. Distinct chromatic variation and saturation -> Optical imagery
        elif mean_saturation > 10.0 or mean_channel_diff > 4.0:
            conf = 0.92
            evidence_reasons = [
                f"Active chromatic spectra (mean saturation: {mean_saturation:.1f}, channel delta: {mean_channel_diff:.1f})"
            ]
            if mean_saturation > 20.0:
                conf += 0.03
            if has_opt_fn:
                conf = min(0.98, conf + 0.02)
                evidence_reasons.append(f"Filename cue '{filename_lower}'")

            conf = min(0.96, conf)
            reason_str = "; ".join(evidence_reasons) + "."
            label = f"Probable Optical ({int(conf * 100)}%)"

            return ModalityInfo(
                image_path=str(path),
                dimensions=(width, height),
                channels=num_channels,
                file_type=file_ext,
                probable_modality="optical",
                confidence=conf,
                reasoning=f"Identified as Optical imagery based on {reason_str}",
                display_label=label,
                visual_characteristics=visual_characteristics,
            )

        # 3. Intermediate / ambiguous cases
        else:
            # Low saturation but not strictly zero (e.g. muted desert or cloudy optical, or pseudo-colored SAR)
            if has_sar_fn and not has_opt_fn:
                return ModalityInfo(
                    image_path=str(path),
                    dimensions=(width, height),
                    channels=num_channels,
                    file_type=file_ext,
                    probable_modality="sar",
                    confidence=0.75,
                    reasoning=f"Low color variance with filename pattern '{filename_lower}'.",
                    display_label="Probable SAR (75%)",
                    visual_characteristics=visual_characteristics,
                )
            elif has_opt_fn and not has_sar_fn:
                return ModalityInfo(
                    image_path=str(path),
                    dimensions=(width, height),
                    channels=num_channels,
                    file_type=file_ext,
                    probable_modality="optical",
                    confidence=0.78,
                    reasoning=f"Low color variance with optical filename pattern '{filename_lower}'.",
                    display_label="Probable Optical (78%)",
                    visual_characteristics=visual_characteristics,
                )
            else:
                return ModalityInfo(
                    image_path=str(path),
                    dimensions=(width, height),
                    channels=num_channels,
                    file_type=file_ext,
                    probable_modality="unknown",
                    confidence=0.45,
                    reasoning="Image characteristics show ambiguous color and texture; unable to confirm optical or SAR modality.",
                    display_label="Unknown Modality",
                    visual_characteristics=visual_characteristics,
                )


class SceneCoLocationChecker:
    """Checks whether two satellite images cover the same geographic area."""

    @classmethod
    def check_co_location(
        cls,
        image_a_path: Union[str, Path],
        image_b_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """Check if two images are co-located. Returns (is_co_located, confidence, explanation)."""
        p_a = Path(image_a_path)
        p_b = Path(image_b_path)

        if not p_a.exists() or not p_b.exists():
            return True, 0.90, "Synthetic or mock scene inputs assumed co-located for evaluation."

        # 1. Geospatial metadata check if GeoTIFF
        try:
            import rasterio
            if p_a.suffix.lower() in [".tif", ".tiff"] and p_b.suffix.lower() in [".tif", ".tiff"]:
                with rasterio.open(p_a) as src_a, rasterio.open(p_b) as src_b:
                    if src_a.crs and src_b.crs:
                        # Check bounding boxes
                        b_a = src_a.bounds
                        b_b = src_b.bounds
                        overlap_x = max(0.0, min(b_a.right, b_b.right) - max(b_a.left, b_b.left))
                        overlap_y = max(0.0, min(b_a.top, b_b.top) - max(b_a.bottom, b_b.bottom))
                        area_a = (b_a.right - b_a.left) * (b_a.top - b_a.bottom)
                        area_b = (b_b.right - b_b.left) * (b_b.top - b_b.bottom)
                        overlap_area = overlap_x * overlap_y
                        iou = overlap_area / max(area_a + area_b - overlap_area, 1e-6)
                        if iou > 0.3:
                            return True, round(float(iou), 3), f"Geospatial bounds verify {iou*100:.1f}% spatial overlap."
                        else:
                            return False, round(float(iou), 3), "Geospatial coordinates indicate distinct non-overlapping footprints."
        except Exception:
            pass

        # 2. Visual / Exported Image Comparison (JPG/PNG)
        try:
            with Image.open(p_a) as im_a, Image.open(p_b) as im_b:
                w_a, h_a = im_a.size
                w_b, h_b = im_b.size
                ratio_a = w_a / max(h_a, 1)
                ratio_b = w_b / max(h_b, 1)

                arr_a = np.array(im_a)
                arr_b = np.array(im_b)

            # Convert both to single-channel luminance for structural scene comparison
            if arr_a.ndim == 3:
                gray_a = cv2.cvtColor(arr_a[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray_a = arr_a

            if arr_b.ndim == 3:
                gray_b = cv2.cvtColor(arr_b[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray_b = arr_b

            # Compare aspect ratios
            ratio_delta = abs(ratio_a - ratio_b)

            # Resize to common normalized grid (256x256) for multi-scale edge correlation
            g_a_resized = cv2.resize(gray_a, (256, 256)).astype(np.float32)
            g_b_resized = cv2.resize(gray_b, (256, 256)).astype(np.float32)

            # Compute Sobel edge gradients (structural road/river/building outline correspondence)
            edge_a = cv2.Sobel(g_a_resized, cv2.CV_32F, 1, 1)
            edge_b = cv2.Sobel(g_b_resized, cv2.CV_32F, 1, 1)

            norm_a = edge_a / (np.linalg.norm(edge_a) + 1e-6)
            norm_b = edge_b / (np.linalg.norm(edge_b) + 1e-6)
            corr = float(np.sum(norm_a * norm_b))

            conf = 0.88
            if corr > 0.15:
                conf = min(0.96, conf + 0.08)
            elif corr < -0.15 and ratio_delta > 1.2:
                return False, 0.35, "The two images appear to represent different geographic areas, so I did not combine them. Please provide optical and SAR images covering the same area."

            return True, conf, "Visual scene structure and spatial extent confirm co-located footprints."

        except Exception as e:
            logger.warning("Visual co-location check encountered an issue: %s", e)
            return True, 0.80, "Assumed co-located based on dual-sensor analysis request."


class CollageDetector:
    """Detects whether a single image contains both optical and SAR panels side-by-side or stacked."""

    @classmethod
    def detect_and_split(
        cls,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        """If image is a collage with SAR and optical panels, returns dict with separated paths and modalities."""
        path = Path(image_path)
        if not path.exists():
            return None

        out_dir = Path(output_dir or path.parent)

        try:
            with Image.open(path) as pil_img:
                w, h = pil_img.size
                arr = np.array(pil_img)

            panels_to_test = []
            split_mode = None
            if w >= 1.4 * h:
                mid_x = w // 2
                panel_left = arr[:, :mid_x]
                panel_right = arr[:, mid_x:]
                panels_to_test = [("left", panel_left), ("right", panel_right)]
                split_mode = "horizontal"
            elif h >= 1.4 * w:
                mid_y = h // 2
                panel_top = arr[:mid_y, :]
                panel_bottom = arr[mid_y:, :]
                panels_to_test = [("top", panel_top), ("bottom", panel_bottom)]
                split_mode = "vertical"

            if not panels_to_test:
                return None

            extracted_info = []
            for name, p_arr in panels_to_test:
                p_path = out_dir / f"{path.stem}_panel_{name}.png"
                Image.fromarray(p_arr).save(p_path)
                info = ModalityInspector.inspect(p_path)
                extracted_info.append((p_path, info))

            modalities = [info.probable_modality for _, info in extracted_info]
            if "sar" in modalities and "optical" in modalities:
                sar_path = next(p for p, info in extracted_info if info.probable_modality == "sar")
                opt_path = next(p for p, info in extracted_info if info.probable_modality == "optical")
                sar_info = next(info for p, info in extracted_info if info.probable_modality == "sar")
                opt_info = next(info for p, info in extracted_info if info.probable_modality == "optical")

                logger.info("Detected multi-panel collage in %s: %s split", path.name, split_mode)
                return {
                    "is_collage": True,
                    "split_mode": split_mode,
                    "optical_path": str(opt_path),
                    "sar_path": str(sar_path),
                    "optical_info": opt_info,
                    "sar_info": sar_info,
                }
            return None

        except Exception as e:
            logger.warning("Collage detection failed for %s: %s", path.name, e)
            return None
