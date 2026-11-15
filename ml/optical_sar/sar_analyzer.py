"""SAR Radar Backscatter and Structural Analysis module.

Analyzes synthetic aperture radar (SAR) amplitude imagery for:
- Radar backscatter distribution & dynamic range
- High backscatter structural returns (double-bounce from buildings/urban fabric)
- Low backscatter smooth surfaces (specular reflection from calm water / smooth terrain)
- Intermediate volume scattering (vegetation canopy / rough surface texture)
- Evidence heatmap / false-color visualization
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("ml.optical_sar.analyzer")


def analyze_sar_backscatter(
    sar_image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Analyze SAR radar backscatter characteristics and generate evidence visualization."""
    path = Path(sar_image_path)
    if not path.exists():
        return {
            "success": False,
            "model": "SAR-Backscatter-Analyzer",
            "error": f"SAR image not found: {path}",
        }

    try:
        with Image.open(path) as pil_img:
            arr = np.array(pil_img)

        # Convert to single-channel amplitude
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            gray = arr[0]

        h, w = gray.shape
        total_pixels = h * w

        # Statistical distribution
        intensity = gray.astype(np.float32)
        mean_val = float(np.mean(intensity))
        std_val = float(np.std(intensity))
        min_val = float(np.min(intensity))
        max_val = float(np.max(intensity))

        # Dynamic thresholds based on intensity statistics
        # High backscatter: > mean + 0.8 * std (dense structures, corner reflectors)
        # Low backscatter: < mean - 0.7 * std or < 30 (specular reflection, smooth water/runway)
        high_thresh = min(240.0, max(120.0, mean_val + 0.75 * std_val))
        low_thresh = max(15.0, min(70.0, mean_val - 0.70 * std_val))

        high_mask = intensity >= high_thresh
        low_mask = intensity <= low_thresh
        mid_mask = (~high_mask) & (~low_mask)

        high_count = int(np.sum(high_mask))
        low_count = int(np.sum(low_mask))
        mid_count = int(np.sum(mid_mask))

        high_pct = round((high_count / max(total_pixels, 1)) * 100.0, 2)
        low_pct = round((low_count / max(total_pixels, 1)) * 100.0, 2)
        mid_pct = round((mid_count / max(total_pixels, 1)) * 100.0, 2)

        # Generate a high-contrast false-color backscatter overlay
        # Base: grayscale SAR
        vis_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Overlay high backscatter in golden amber [0, 165, 255]
        # Overlay low backscatter in deep cyan-blue [255, 120, 0]
        color_overlay = vis_bgr.copy()
        color_overlay[high_mask] = [0, 180, 255]  # High backscatter (structures)
        color_overlay[low_mask] = [230, 120, 0]   # Low backscatter (smooth/water)

        blended = cv2.addWeighted(vis_bgr, 0.55, color_overlay, 0.45, 0)

        # Draw a sleek label bar at bottom
        cv2.rectangle(blended, (0, h - 36), (w, h), (20, 24, 33), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        legend_text = f"SAR Radar Backscatter: High (built-up) {high_pct}% | Low (smooth/water) {low_pct}%"
        cv2.putText(blended, legend_text, (10, h - 12), font, 0.45, (220, 225, 235), 1, cv2.LINE_AA)

        # Save visualization
        out_dir = Path(output_dir or path.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        vis_path = out_dir / f"sar_backscatter_{path.stem}_{ts}.png"
        cv2.imwrite(str(vis_path), blended)

        findings = []
        if high_pct > 5.0:
            findings.append(
                f"Strong radar backscatter detected across {high_pct}% of the scene, consistent with double-bounce reflections from built-up structures and geometric terrain."
            )
        if low_pct > 3.0:
            findings.append(
                f"Attenuated specular backscatter observed across {low_pct}% of the scene, corresponding to smooth surfaces such as water bodies, paved areas, or flat ground."
            )
        if mid_pct > 20.0:
            findings.append(
                f"Diffuse volume scattering across {mid_pct}% of the surface, indicative of vegetation canopy and rough soil texture."
            )

        return {
            "success": True,
            "model": "SAR-Backscatter-Analyzer",
            "image": str(path),
            "statistics": {
                "mean_intensity": round(mean_val, 2),
                "std_intensity": round(std_val, 2),
                "min_intensity": round(min_val, 2),
                "max_intensity": round(max_val, 2),
                "high_backscatter_pct": high_pct,
                "low_backscatter_pct": low_pct,
                "volume_backscatter_pct": mid_pct,
            },
            "findings": findings,
            "visualization": str(vis_path),
            "error": None,
        }

    except Exception as e:
        logger.error("SAR backscatter analysis failed for %s: %s", path, e)
        return {
            "success": False,
            "model": "SAR-Backscatter-Analyzer",
            "image": str(path),
            "error": str(e),
        }
