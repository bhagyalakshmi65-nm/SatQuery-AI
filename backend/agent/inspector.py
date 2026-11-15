"""Image Understanding & Scene Inspection module for SatQuery AI.

Inspects uploaded imagery prior to specialist model routing and execution.
Determines:
- Remote sensing imagery classification & confidence
- Layout: single scene vs. side-by-side vs. vertical split vs. grid collage vs. multi-file
- True scene count (FILE COUNT != SCENE COUNT)
- Bounding boxes and separability of individual panels
- Visible dates, labels, and temporal cues (strictly descriptive; never decides routing)
- Image dimensions and metadata properties
"""

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from backend.agent.schemas import ImageSceneInspection, PanelInfo, SceneLayout

logger = logging.getLogger("agent.inspector")

# Remote sensing filename patterns & platforms
RS_FILENAME_PATTERNS = [
    r"s2[ab]?_", r"s1[ab]?_", r"sentinel", r"landsat", r"planet",
    r"levir", r"spacenet", r"xview", r"bigearthnet", r"croma",
    r"change_map", r"sat\d*", r"aerial", r"ortho", r"tile_\d+"
]

# Temporal indicators (with boundary lookaround to support underscores)
TEMPORAL_KEYWORDS = [
    r"(?<![a-zA-Z0-9])before(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])after(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])pre[-_ ]?event(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])post[-_ ]?event(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])pre(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])post(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])t1(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])t2(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])baseline(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])earlier(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])later(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])timestamp(?![a-zA-Z0-9])",
]

# Date patterns (YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD, Mon YYYY) with boundary lookaround
DATE_PATTERNS = [
    r"(?<![0-9])(19\d\d|20\d\d)[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])(?![0-9])",
    r"(?<![0-9])(19\d\d|20\d\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?![0-9])",
    r"(?<![a-zA-Z0-9])(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-_ ]+(19\d\d|20\d\d)(?![a-zA-Z0-9])",
]


class ImageInspector:
    """Pre-routing vision inspector that analyzes scene layout, panel composition, and remote sensing cues."""

    def __init__(self):
        pass

    def inspect(
        self,
        images: Optional[List[Union[str, Path]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageSceneInspection:
        """Inspect one or more image files to determine scene layout, panel count, and properties.

        CRITICAL INVARIANT: FILE COUNT != SCENE COUNT.
        A single file may contain multiple scene panels (side-by-side, grid collage).
        Multiple files may contain separate scenes.
        """
        images = images or []
        metadata = metadata or {}
        file_count = len(images)

        if file_count == 0:
            return ImageSceneInspection(
                file_count=0,
                scene_count=0,
                layout=SceneLayout.UNKNOWN,
                is_remote_sensing=False,
                remote_sensing_confidence=0.0,
                remote_sensing_rationale="No images provided for inspection.",
                panels=[],
                dimensions=[],
                visible_labels=[],
                detected_dates=[],
                has_temporal_cues=False,
                can_separate_panels=False,
                available_metadata=metadata,
                inspection_summary="No images provided for inspection.",
            )

        all_panels: List[PanelInfo] = []
        dimensions_list: List[Dict[str, Any]] = []
        all_labels: List[str] = []
        all_dates: List[str] = []
        rs_confidences: List[float] = []
        rs_rationales: List[str] = []

        # Inspect each file
        for idx, img_item in enumerate(images):
            img_path = Path(img_item)
            file_dims, pil_img, img_arr = self._load_image_and_metadata(img_path)
            dimensions_list.append(file_dims)

            # Remote sensing detection for this file
            is_rs, rs_conf, rs_rat = self._detect_remote_sensing(
                img_path=img_path,
                dims=file_dims,
                pil_img=pil_img,
                img_arr=img_arr,
                metadata=metadata,
            )
            rs_confidences.append(rs_conf)
            rs_rationales.append(rs_rat)

            # Layout and panel detection within this file
            file_layout, file_panels, file_can_sep = self._detect_layout_and_panels(
                file_idx=idx,
                img_path=img_path,
                dims=file_dims,
                img_arr=img_arr,
            )
            all_panels.extend(file_panels)

            # Text, labels, and date cues for this file
            labels, dates = self._extract_labels_and_dates(
                img_path=img_path,
                metadata=metadata,
                panels=file_panels,
            )
            all_labels.extend(labels)
            all_dates.extend(dates)

        # Determine overall layout and scene count
        scene_count = len(all_panels)
        unique_dates = sorted(list(set(all_dates)))
        unique_labels = sorted(list(set(all_labels)))

        # Assign uninvented display labels (Panel A, Panel B... or verified dates)
        for p_idx, p_info in enumerate(all_panels):
            if p_info.detected_date:
                p_info.display_name = p_info.detected_date
            elif scene_count > 1:
                label_letter = chr(65 + p_idx) if p_idx < 26 else str(p_idx + 1)
                p_info.display_name = f"Panel {label_letter}"
            else:
                p_info.display_name = "Primary Scene"

        # Determine overall layout
        if file_count > 1:
            overall_layout = SceneLayout.MULTI_FILE
            can_separate_panels = True
        elif scene_count == 1:
            overall_layout = SceneLayout.SINGLE
            can_separate_panels = False
        else:
            # Single file with multiple panels
            first_layout = file_layout  # layout determined by the single file
            overall_layout = first_layout
            can_separate_panels = True

        # Temporal cues check
        has_temporal_keywords = any(
            re.search(pat, lbl, re.IGNORECASE)
            for lbl in unique_labels
            for pat in TEMPORAL_KEYWORDS
        ) or any(
            re.search(pat, Path(img).name, re.IGNORECASE)
            for img in images
            for pat in TEMPORAL_KEYWORDS
        )
        has_multiple_dates = len(unique_dates) >= 2
        has_temporal_cues = has_temporal_keywords or has_multiple_dates

        # Remote sensing determination
        avg_rs_conf = float(np.mean(rs_confidences)) if rs_confidences else 0.0
        is_remote_sensing = avg_rs_conf >= 0.5
        rs_summary_rationale = "; ".join(rs_rationales)

        # Build concise inspection summary
        summary = self._build_inspection_summary(
            file_count=file_count,
            scene_count=scene_count,
            layout=overall_layout,
            is_remote_sensing=is_remote_sensing,
            rs_conf=avg_rs_conf,
            panels=all_panels,
            dates=unique_dates,
            has_temporal=has_temporal_cues,
        )

        return ImageSceneInspection(
            file_count=file_count,
            scene_count=scene_count,
            layout=overall_layout,
            is_remote_sensing=is_remote_sensing,
            remote_sensing_confidence=round(avg_rs_conf, 3),
            remote_sensing_rationale=rs_summary_rationale,
            panels=all_panels,
            dimensions=dimensions_list,
            visible_labels=unique_labels,
            detected_dates=unique_dates,
            has_temporal_cues=has_temporal_cues,
            can_separate_panels=can_separate_panels,
            available_metadata=metadata,
            inspection_summary=summary,
        )

    def extract_panels(
        self,
        image_path: Union[str, Path],
        inspection: Optional[ImageSceneInspection] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Extract and crop separate panel images from a multi-panel collage file.

        Args:
            image_path: Path to the input image.
            inspection: Optional precomputed ImageSceneInspection. If None, inspect() is run.
            output_dir: Optional directory to store extracted panel images.

        Returns:
            List of Path objects pointing to individual cropped panel image files.
        """
        img_p = Path(image_path)
        if not img_p.exists():
            return [img_p]

        if inspection is None:
            inspection = self.inspect([img_p])

        if not inspection.can_separate_panels or len(inspection.panels) <= 1:
            return [img_p]

        import tempfile
        out_dir = output_dir or (Path(tempfile.gettempdir()) / "satquery_panels")
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted_paths: List[Path] = []
        try:
            with Image.open(img_p) as pil_img:
                for idx, panel in enumerate(inspection.panels):
                    # bbox format: [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = panel.bbox
                    cropped = pil_img.crop((xmin, ymin, xmax, ymax))
                    panel_label = chr(65 + idx) if idx < 26 else str(idx)
                    out_name = f"{img_p.stem}_panel_{panel_label}{img_p.suffix or '.png'}"
                    out_path = out_dir / out_name
                    cropped.save(out_path)
                    extracted_paths.append(out_path)
        except Exception as e:
            logger.warning("Failed to extract panels from %s: %s", img_p, e)
            return [img_p]

        return extracted_paths

    def _load_image_and_metadata(
        self, img_path: Path
    ) -> Tuple[Dict[str, Any], Optional[Image.Image], Optional[np.ndarray]]:
        """Safely load image dimensions, PIL representation, and numpy array."""
        dims: Dict[str, Any] = {
            "path": str(img_path),
            "filename": img_path.name,
            "width": 0,
            "height": 0,
            "channels": 0,
            "format": img_path.suffix.lower(),
            "mode": "",
            "file_size_bytes": 0,
        }

        if img_path.exists():
            try:
                dims["file_size_bytes"] = img_path.stat().st_size
            except Exception:
                pass

        ext = img_path.suffix.lower()

        # Handle NumPy arrays (.npy, .npz)
        if ext in [".npy", ".npz"] and img_path.exists():
            try:
                if ext == ".npy":
                    arr = np.load(str(img_path))
                else:
                    data = np.load(str(img_path))
                    arr = data[list(data.keys())[0]]
                
                # Tensor shape handling (C, H, W) vs (H, W, C)
                if arr.ndim == 3:
                    if arr.shape[0] in [1, 2, 3, 4, 12, 14] and arr.shape[0] < arr.shape[1]:
                        # (C, H, W) -> convert to (H, W, C) for analysis
                        c, h, w = arr.shape
                        arr_trans = np.transpose(arr, (1, 2, 0))
                    else:
                        h, w, c = arr.shape
                        arr_trans = arr
                    dims.update({"width": int(w), "height": int(h), "channels": int(c), "mode": f"{c}-channel array"})
                    return dims, None, arr_trans
                elif arr.ndim == 2:
                    h, w = arr.shape
                    dims.update({"width": int(w), "height": int(h), "channels": 1, "mode": "1-channel array"})
                    return dims, None, arr[:, :, np.newaxis]
            except Exception as e:
                logger.warning("Failed reading numpy file %s: %s", img_path, e)

        # Handle GeoTIFF (.tif, .tiff)
        if ext in [".tif", ".tiff"] and img_path.exists():
            try:
                import tifffile
                raw_data = tifffile.imread(str(img_path))
                if raw_data.ndim == 3:
                    c = raw_data.shape[0] if raw_data.shape[0] < raw_data.shape[1] else raw_data.shape[-1]
                    h = raw_data.shape[1] if raw_data.shape[0] == c else raw_data.shape[0]
                    w = raw_data.shape[2] if raw_data.shape[0] == c else raw_data.shape[1]
                    dims.update({
                        "width": int(w),
                        "height": int(h),
                        "channels": int(c),
                        "mode": f"{c}-band raster",
                        "format": "TIFF",
                    })
                    if c == 12:
                        from ml.landcover.src.dataset import extract_rgb_composite
                        cube = raw_data if raw_data.shape[0] == 12 else np.transpose(raw_data, (2, 0, 1))
                        rgb_arr = extract_rgb_composite(cube)
                        rgb_pil = Image.fromarray(rgb_arr)
                        return dims, rgb_pil, rgb_arr
            except Exception as e:
                logger.debug("Tifffile inspection error for %s: %s", img_path, e)

        # Standard image reading via PIL and OpenCV
        if img_path.exists():
            try:
                with Image.open(img_path) as pil_img:
                    w, h = pil_img.size
                    mode = pil_img.mode
                    channels = len(pil_img.getbands())
                    dims.update({
                        "width": int(w),
                        "height": int(h),
                        "channels": int(channels),
                        "mode": mode,
                        "format": pil_img.format or ext.lstrip(".").upper(),
                    })
                    rgb_pil = pil_img.convert("RGB")
                    img_arr = np.array(rgb_pil)
                    return dims, rgb_pil, img_arr
            except Exception as e:
                logger.warning("PIL failed loading image %s: %s", img_path, e)

        return dims, None, None

    def _detect_remote_sensing(
        self,
        img_path: Path,
        dims: Dict[str, Any],
        pil_img: Optional[Image.Image],
        img_arr: Optional[np.ndarray],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, float, str]:
        """Classify whether the image is satellite or remote-sensing imagery."""
        reasons: List[str] = []
        score = 0.5  # Baseline prior for a satellite analysis platform

        # 1. Channels / Multispectral check
        channels = dims.get("channels", 3)
        if channels > 3:
            score = 1.0
            reasons.append(f"Multispectral / multichannel raster ({channels} channels)")
            return True, score, "; ".join(reasons)

        # 2. GeoTIFF format check
        ext = img_path.suffix.lower()
        if ext in [".tif", ".tiff"]:
            score = max(score, 0.95)
            reasons.append("GeoTIFF raster format")

        # 3. Filename indicators
        name_lower = img_path.name.lower()
        for pat in RS_FILENAME_PATTERNS:
            if re.search(pat, name_lower):
                score = max(score, 0.90)
                reasons.append(f"Filename matches remote-sensing pattern '{pat}'")
                break

        # 4. Explicit user metadata
        sensor_type = (metadata.get("sensor_type") or "").lower()
        if sensor_type in ["optical", "sar", "sentinel", "landsat", "planet", "aerial", "croma"]:
            score = max(score, 0.98)
            reasons.append(f"Verified sensor metadata ({sensor_type})")

        # 5. Image texture & frequency analysis (if array is available)
        if img_arr is not None and img_arr.size > 0:
            h, w = img_arr.shape[:2]
            aspect_ratio = w / max(h, 1)

            # Remote sensing scenes are typically square or standard 1:1 / 2:1 tiles
            if 0.75 <= aspect_ratio <= 1.33 or 1.8 <= aspect_ratio <= 2.2:
                score = min(score + 0.1, 1.0)
                reasons.append(f"Satellite aspect ratio profile ({aspect_ratio:.2f})")

            # Check edge density via Laplacian variance
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY) if img_arr.ndim == 3 else img_arr[:, :, 0]
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if laplacian_var > 80.0:
                score = min(score + 0.1, 1.0)
                reasons.append(f"High spatial texture density (Laplacian variance: {laplacian_var:.1f})")

        if not reasons:
            reasons.append("Standard visual characteristics without explicit satellite tags")

        is_rs = score >= 0.6
        return is_rs, round(score, 2), "; ".join(reasons)

    def _detect_layout_and_panels(
        self,
        file_idx: int,
        img_path: Path,
        dims: Dict[str, Any],
        img_arr: Optional[np.ndarray],
    ) -> Tuple[SceneLayout, List[PanelInfo], bool]:
        """Detect internal sub-panels, dividers, and collage structure within a single image file."""
        w = dims.get("width", 0)
        h = dims.get("height", 0)

        # Fallback if image could not be loaded
        if w == 0 or h == 0 or img_arr is None or img_arr.size == 0:
            panel = PanelInfo(
                panel_id=f"panel_{file_idx}_0",
                file_index=file_idx,
                source_file=img_path.name,
                bbox=[0, 0, h, w],
                normalized_bbox=[0.0, 0.0, 1.0, 1.0],
                width=w,
                height=h,
                aspect_ratio=w / max(h, 1) if h > 0 else 1.0,
                can_separate=False,
            )
            return SceneLayout.SINGLE, [panel], False

        aspect_ratio = w / h
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY) if img_arr.ndim == 3 else img_arr[:, :, 0]
        total_std = float(np.std(gray))

        # Blank or completely flat image -> single scene
        if total_std < 5.0:
            panel = PanelInfo(
                panel_id=f"panel_{file_idx}_0",
                file_index=file_idx,
                source_file=img_path.name,
                bbox=[0, 0, h, w],
                normalized_bbox=[0.0, 0.0, 1.0, 1.0],
                width=w,
                height=h,
                aspect_ratio=aspect_ratio,
                can_separate=False,
            )
            return SceneLayout.SINGLE, [panel], False

        # 1. Check for Vertical Divider (Sobel X along columns)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        col_grad = np.mean(np.abs(gx), axis=0)  # average along rows for each column

        # Focus on center 20% region for divider line: [0.40 * w, 0.60 * w]
        vx_start = int(w * 0.40)
        vx_end = int(w * 0.60)
        has_v_divider = False
        v_seam_x = w // 2

        if vx_end > vx_start:
            v_window = col_grad[vx_start:vx_end]
            v_peak_idx = int(np.argmax(v_window))
            v_peak_val = float(v_window[v_peak_idx])
            v_seam_x = vx_start + v_peak_idx
            bg_v_grad = float(np.mean(col_grad)) + 1e-6
            v_ratio = v_peak_val / bg_v_grad

            # Check if column is a solid uniform gutter (e.g. black/white divider line)
            col_std = float(np.std(gray[:, v_seam_x]))
            is_v_gutter = (total_std > 15.0 and col_std < 5.0)

            # Divider detected if high gradient peak ratio OR solid line gutter OR (2:1 aspect ratio with peak)
            if (v_ratio >= 3.0 and v_peak_val >= 20.0) or is_v_gutter or (1.75 <= aspect_ratio <= 2.25 and v_ratio >= 2.0 and v_peak_val >= 15.0):
                has_v_divider = True

        # 2. Check for Horizontal Divider (Sobel Y along rows)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        row_grad = np.mean(np.abs(gy), axis=1)  # average along columns for each row

        hy_start = int(h * 0.40)
        hy_end = int(h * 0.60)
        has_h_divider = False
        h_seam_y = h // 2

        if hy_end > hy_start:
            h_window = row_grad[hy_start:hy_end]
            h_peak_idx = int(np.argmax(h_window))
            h_peak_val = float(h_window[h_peak_idx])
            h_seam_y = hy_start + h_peak_idx
            bg_h_grad = float(np.mean(row_grad)) + 1e-6
            h_ratio = h_peak_val / bg_h_grad

            row_std = float(np.std(gray[h_seam_y, :]))
            is_h_gutter = (total_std > 15.0 and row_std < 5.0)

            if (h_ratio >= 3.0 and h_peak_val >= 20.0) or is_h_gutter or (0.45 <= aspect_ratio <= 0.58 and h_ratio >= 2.0 and h_peak_val >= 15.0):
                has_h_divider = True

        # 3. Determine Layout & Panels
        # Case A: Grid Collage (2x2 = 4 panels)
        if has_v_divider and has_h_divider:
            panels = [
                # Top-Left
                PanelInfo(
                    panel_id=f"panel_{file_idx}_0",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[0, 0, h_seam_y, v_seam_x],
                    normalized_bbox=[0.0, 0.0, h_seam_y / h, v_seam_x / w],
                    width=v_seam_x,
                    height=h_seam_y,
                    aspect_ratio=v_seam_x / max(h_seam_y, 1),
                    can_separate=True,
                ),
                # Top-Right
                PanelInfo(
                    panel_id=f"panel_{file_idx}_1",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[0, v_seam_x, h_seam_y, w],
                    normalized_bbox=[0.0, v_seam_x / w, h_seam_y / h, 1.0],
                    width=w - v_seam_x,
                    height=h_seam_y,
                    aspect_ratio=(w - v_seam_x) / max(h_seam_y, 1),
                    can_separate=True,
                ),
                # Bottom-Left
                PanelInfo(
                    panel_id=f"panel_{file_idx}_2",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[h_seam_y, 0, h, v_seam_x],
                    normalized_bbox=[h_seam_y / h, 0.0, 1.0, v_seam_x / w],
                    width=v_seam_x,
                    height=h - h_seam_y,
                    aspect_ratio=v_seam_x / max(h - h_seam_y, 1),
                    can_separate=True,
                ),
                # Bottom-Right
                PanelInfo(
                    panel_id=f"panel_{file_idx}_3",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[h_seam_y, v_seam_x, h, w],
                    normalized_bbox=[h_seam_y / h, v_seam_x / w, 1.0, 1.0],
                    width=w - v_seam_x,
                    height=h - h_seam_y,
                    aspect_ratio=(w - v_seam_x) / max(h - h_seam_y, 1),
                    can_separate=True,
                ),
            ]
            return SceneLayout.GRID_COLLAGE, panels, True

        # Case B: Side-by-Side (2 panels horizontally)
        if has_v_divider:
            panels = [
                # Left panel
                PanelInfo(
                    panel_id=f"panel_{file_idx}_0",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[0, 0, h, v_seam_x],
                    normalized_bbox=[0.0, 0.0, 1.0, v_seam_x / w],
                    width=v_seam_x,
                    height=h,
                    aspect_ratio=v_seam_x / max(h, 1),
                    can_separate=True,
                ),
                # Right panel
                PanelInfo(
                    panel_id=f"panel_{file_idx}_1",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[0, v_seam_x, h, w],
                    normalized_bbox=[0.0, v_seam_x / w, 1.0, 1.0],
                    width=w - v_seam_x,
                    height=h,
                    aspect_ratio=(w - v_seam_x) / max(h, 1),
                    can_separate=True,
                ),
            ]
            return SceneLayout.SIDE_BY_SIDE, panels, True

        # Case C: Vertical Split (2 panels vertically stacked)
        if has_h_divider:
            panels = [
                # Top panel
                PanelInfo(
                    panel_id=f"panel_{file_idx}_0",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[0, 0, h_seam_y, w],
                    normalized_bbox=[0.0, 0.0, h_seam_y / h, 1.0],
                    width=w,
                    height=h_seam_y,
                    aspect_ratio=w / max(h_seam_y, 1),
                    can_separate=True,
                ),
                # Bottom panel
                PanelInfo(
                    panel_id=f"panel_{file_idx}_1",
                    file_index=file_idx,
                    source_file=img_path.name,
                    bbox=[h_seam_y, 0, h, w],
                    normalized_bbox=[h_seam_y / h, 0.0, 1.0, 1.0],
                    width=w,
                    height=h - h_seam_y,
                    aspect_ratio=w / max(h - h_seam_y, 1),
                    can_separate=True,
                ),
            ]
            return SceneLayout.VERTICAL_SPLIT, panels, True

        # Case D: Single Scene
        panel = PanelInfo(
            panel_id=f"panel_{file_idx}_0",
            file_index=file_idx,
            source_file=img_path.name,
            bbox=[0, 0, h, w],
            normalized_bbox=[0.0, 0.0, 1.0, 1.0],
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            can_separate=False,
        )
        return SceneLayout.SINGLE, [panel], False

    def _extract_labels_and_dates(
        self,
        img_path: Path,
        metadata: Dict[str, Any],
        panels: List[PanelInfo],
    ) -> Tuple[List[str], List[str]]:
        """Extract visible text, dates, and temporal cues from filenames, metadata, and labels."""
        labels: List[str] = []
        dates: List[str] = []

        # 1. Search in Filename
        fname = img_path.name
        for pat in DATE_PATTERNS:
            for match in re.finditer(pat, fname, re.IGNORECASE):
                date_str = match.group(0)
                dates.append(date_str)
                labels.append(date_str)

        for pat in TEMPORAL_KEYWORDS:
            for match in re.finditer(pat, fname, re.IGNORECASE):
                labels.append(match.group(0).lower())

        # 2. Search in Metadata
        for k, v in metadata.items():
            str_v = str(v)
            for pat in DATE_PATTERNS:
                for match in re.finditer(pat, str_v, re.IGNORECASE):
                    date_str = match.group(0)
                    dates.append(date_str)
                    labels.append(date_str)
            for pat in TEMPORAL_KEYWORDS:
                for match in re.finditer(pat, str_v, re.IGNORECASE):
                    labels.append(match.group(0).lower())

        return sorted(list(set(labels))), sorted(list(set(dates)))

    def _build_inspection_summary(
        self,
        file_count: int,
        scene_count: int,
        layout: SceneLayout,
        is_remote_sensing: bool,
        rs_conf: float,
        panels: List[PanelInfo],
        dates: List[str],
        has_temporal: bool,
    ) -> str:
        """Create a human-readable visual summary of the image understanding stage."""
        rs_status = f"Remote Sensing ({rs_conf * 100:.0f}% confidence)" if is_remote_sensing else f"Non-Satellite / General Imagery ({rs_conf * 100:.0f}% confidence)"

        panel_desc = f"{scene_count} scene panel(s)"
        if layout == SceneLayout.SIDE_BY_SIDE:
            panel_desc += f" (side-by-side, split at x={panels[0].width}px)"
        elif layout == SceneLayout.VERTICAL_SPLIT:
            panel_desc += f" (vertical stack, split at y={panels[0].height}px)"
        elif layout == SceneLayout.GRID_COLLAGE:
            panel_desc += f" (2x2 grid collage)"
        elif layout == SceneLayout.MULTI_FILE:
            panel_desc += f" across {file_count} separate files"

        date_info = f", detected dates: {', '.join(dates)}" if dates else ""
        temp_info = " [temporal cues detected]" if has_temporal else ""

        return f"Inspected {file_count} file(s) -> {panel_desc}. Classification: {rs_status}{date_info}{temp_info}."

    @staticmethod
    def crop_panels(
        image_path: Union[str, Path],
        panels: List[PanelInfo],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """Crop and save individual detected panels from an image file.

        Args:
            image_path: Path to the source composite image.
            panels: List of PanelInfo objects containing bounding boxes.
            output_dir: Target directory to save cropped panels. Defaults to source parent.

        Returns:
            List of Path objects pointing to saved cropped image files.
        """
        source = Path(image_path)
        out_dir = Path(output_dir) if output_dir else source.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        cropped_paths: List[Path] = []
        with Image.open(source) as pil_img:
            rgb_img = pil_img.convert("RGB")
            for idx, panel in enumerate(panels):
                ymin, xmin, ymax, xmax = panel.bbox
                # PIL crop uses (left, top, right, bottom) = (xmin, ymin, xmax, ymax)
                cropped = rgb_img.crop((xmin, ymin, xmax, ymax))
                out_name = f"{source.stem}_panel_{idx}{source.suffix}"
                out_path = out_dir / out_name
                cropped.save(out_path)
                cropped_paths.append(out_path)

        return cropped_paths
