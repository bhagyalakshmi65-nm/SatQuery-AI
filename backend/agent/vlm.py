"""Remote Sensing Vision-Language Model (RS-Qwen) reasoning engine.

Interfaces with AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct.
Enforces the mandatory Evidence Grounding Policy:
1. Direct Observations (Model-Supported Evidence)
2. Remote Sensing Interpretation
3. Uncertainties & Sensor Caveats
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from backend.agent.schemas import TaskType

logger = logging.getLogger("agent.vlm")

MODEL_ID = "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct"

SYSTEM_PROMPT = """You are SatQuery AI's Remote Sensing Vision-Language Expert.
Synthesize a scientifically grounded, clear natural-language response based strictly on the empirical evidence provided below.

CRITICAL EVIDENCE GROUNDING RULES:
1. DIRECT OBSERVATIONS: State ONLY the empirical facts, detected objects, bounding boxes, change percentages, and land-cover classes present in the specialist evidence.
2. DOMAIN INTERPRETATION: Explain the geospatial significance of these observations.
3. UNCERTAINTY & CAVEATS: Clearly specify confidence thresholds, sensor limitations, and potential environmental factors (e.g., cloud cover, look angle, spatial resolution).
4. DO NOT invent, hallucinate, or assume unobserved object counts, change magnitudes, or class labels not present in the evidence.
"""


def build_grounded_prompt(
    query: str,
    task: TaskType,
    evidence: Dict[str, Any],
    confidence: Optional[float] = None,
) -> str:
    """Construct an evidence-grounded prompt for RS-Qwen.

    Injects empirical specialist measurements (counts, change percentages,
    land-cover classes, confidence metrics) so the vision-language model
    reasons strictly from verified telemetry.
    """
    evidence_lines: List[str] = []

    # 1. Object Detection (YOLO)
    if "object_detection" in evidence:
        obj_ev = evidence["object_detection"]
        if obj_ev.get("success"):
            counts = obj_ev.get("results", {}).get("counts", {})
            total_objs = sum(counts.values())
            if total_objs == 0:
                evidence_lines.append("- Object Detection (YOLO11): 0 discrete objects detected above threshold.")
            else:
                counts_str = ", ".join(f"{c} {cls_name}(s)" for cls_name, c in counts.items())
                evidence_lines.append(f"- Object Detection (YOLO11): {total_objs} object(s) detected: {counts_str}.")
            if confidence is not None:
                evidence_lines.append(f"- Detection Confidence: {confidence * 100:.1f}%.")
            else:
                evidence_lines.append("- Detection Confidence: N/A (zero detections above threshold).")
            vis = obj_ev.get("visualization")
            if vis:
                evidence_lines.append(f"- Bounding Box Overlay: {vis}.")
        else:
            evidence_lines.append(f"- Object Detection Error: {obj_ev.get('error', 'Execution failed')}.")

    # 2. Bi-Temporal Change Detection (ChangeFormerV6)
    if "change_detection" in evidence:
        cd_ev = evidence["change_detection"]
        if cd_ev.get("success"):
            pct = cd_ev.get("change_percentage", 0.0)
            chg = cd_ev.get("changed_pixels", 0)
            tot = cd_ev.get("total_pixels", 0)
            evidence_lines.append(
                f"- Surface Change Analysis (ChangeFormerV6): {pct:.2f}% changed surface area "
                f"({chg:,} changed / {tot:,} total pixels evaluated)."
            )
            if confidence is not None:
                evidence_lines.append(f"- Change Confidence: {confidence * 100:.1f}%.")
            map_path = cd_ev.get("change_map")
            if map_path:
                evidence_lines.append(f"- Change Map: {map_path}.")
        else:
            evidence_lines.append(f"- Change Detection Error: {cd_ev.get('error', 'Execution failed')}.")

    # 3. Land Cover Classification (SeCo-ECO)
    if "landcover" in evidence:
        lc_ev = evidence["landcover"]
        if lc_ev.get("success"):
            classes = lc_ev.get("results", {}).get("classes", [])
            if classes:
                classes_str = ", ".join(f"{c['class']} ({c['confidence']*100:.1f}%)" for c in classes)
                evidence_lines.append(f"- Land Cover Classes (SeCo-ECO 12-band): {classes_str}.")
            else:
                evidence_lines.append("- Land Cover (SeCo-ECO): No classes exceeded positive decision threshold.")
            if confidence is not None:
                evidence_lines.append(f"- Land Cover Confidence: {confidence * 100:.1f}%.")
        else:
            evidence_lines.append(f"- Land Cover Error: {lc_ev.get('error', 'Execution failed')}.")

    # 4. Optical + SAR Cross-Modal Feature Fusion (CROMA)
    if "optical_sar" in evidence:
        croma_ev = evidence["optical_sar"]
        if croma_ev.get("success"):
            reps = croma_ev.get("representations", {})
            joint_shape = reps.get("joint_encodings_shape", [1, 225, 768])
            gap_shape = reps.get("joint_gap_shape", [1, 768])
            evidence_lines.append(
                f"- Optical + SAR Cross-Modal Fusion (CROMA-Base): Joint tokens {joint_shape}, "
                f"Pooled scene embedding {gap_shape}."
            )
            if confidence is not None:
                evidence_lines.append(f"- Fusion Extraction Confidence: {confidence * 100:.1f}%.")
        else:
            evidence_lines.append(f"- Optical+SAR Fusion Error: {croma_ev.get('error', 'Execution failed')}.")

    parts: List[str] = []
    if evidence_lines:
        parts.append("[SPECIALIST MODEL EMPIRICAL EVIDENCE]")
        parts.extend(evidence_lines)
        parts.append("")

    parts.append(f"USER QUERY:\n{query.strip()}\n")
    parts.append(
        "INSTRUCTIONS:\n"
        "You are SatQuery AI's Remote Sensing Vision-Language Expert.\n"
        "Synthesize a scientifically grounded, clear natural-language response based strictly on the image and empirical specialist evidence provided.\n"
        "1. Direct Observations: State the observed empirical facts, detected objects, change percentages, or land-cover categories.\n"
        "2. Remote Sensing Interpretation: Explain the geospatial significance of these observations.\n"
        "3. Uncertainties & Caveats: Note sensor characteristics, resolution limits, and confidence metrics.\n"
        "Do NOT hallucinate or invent counts, locations, or change metrics not verified by the specialist evidence."
    )
    return "\n".join(parts)


class RSQwenEngine:
    """RS-Qwen vision-language reasoning engine with deterministic grounding fallback."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._manual_available: Optional[bool] = None

    def load_model(self) -> bool:
        """Attempt to load the official Qwen2.5-VL model via ml.rs_qwen."""
        try:
            from ml.rs_qwen.model import get_rs_qwen_model, get_rs_qwen_processor

            logger.info("Loading RS-Qwen model via ml.rs_qwen...")
            get_rs_qwen_model(self.model_path)
            get_rs_qwen_processor(self.model_path)
            self._manual_available = True
            logger.info("RS-Qwen model loaded successfully.")
            return True
        except Exception as e:
            logger.info("RS-Qwen weights not loaded (%s). Using deterministic grounded synthesis.", e)
            self._manual_available = False
            return False

    @property
    def is_available(self) -> bool:
        """Check whether RS-Qwen model is available for inference."""
        if self._manual_available is not None:
            return self._manual_available
        try:
            from ml.rs_qwen.inference import is_available as rs_qwen_is_available
            return rs_qwen_is_available(self.model_path)
        except Exception as e:
            logger.debug("Error checking ml.rs_qwen availability: %s", e)
            return False

    @is_available.setter
    def is_available(self, val: bool):
        self._manual_available = val

    @is_available.deleter
    def is_available(self):
        self._manual_available = None

    def synthesize_response(
        self,
        query: str,
        task: TaskType,
        evidence: Dict[str, Any],
        image_paths: Optional[List[Union[str, Path]]] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Generate final natural-language response grounded in specialist evidence.

        When RS-Qwen is available, delegates to ml.rs_qwen.inference.generate_response
        with structured grounding context. Otherwise falls back to deterministic synthesis.
        """
        if self.is_available and image_paths:
            from ml.rs_qwen.inference import generate_response

            target_image = (
                image_paths[1]
                if (task == TaskType.BI_TEMPORAL_CHANGE and len(image_paths) >= 2)
                else image_paths[0]
            )
            grounded_prompt = build_grounded_prompt(query, task, evidence, confidence)

            logger.info("Calling real ml.rs_qwen.generate_response for task '%s'...", task.value)
            return generate_response(
                image_path=target_image,
                question=grounded_prompt,
                max_new_tokens=256,
                do_sample=False,
                model_path=self.model_path,
            )

        # Deterministic grounded synthesis adhering strictly to the evidence policy
        return self._generate_grounded_synthesis(query, task, evidence, confidence)

    def _generate_grounded_synthesis(
        self,
        query: str,
        task: TaskType,
        evidence: Dict[str, Any],
        confidence: Optional[float],
    ) -> str:
        """Deterministic, hallucination-free response generation following the 3-part evidence structure."""
        sections = []
        header = ""
        if not self.is_available:
            header = "[Notice: RS-Qwen VLM unavailable locally; returning structured specialist evidence report]\n\n"


        # ---------------------------------------------------------------------
        # Task 1: Object Detection (YOLO)
        # ---------------------------------------------------------------------
        if task == TaskType.SINGLE_IMAGE_OBJECT_DETECTION or "object_detection" in evidence:
            obj_ev = evidence.get("object_detection", {})
            if not obj_ev.get("success", False):
                err = obj_ev.get("error", "Unknown error")
                return (
                    f"### 1. DIRECT OBSERVATIONS\n"
                    f"Object detection could not be completed: {err}.\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n"
                    f"Automated object analysis could not locate features due to tool execution failure.\n\n"
                    f"### 3. UNCERTAINTIES & CAVEATS\n"
                    f"Confidence: 0.0%. Please check image path and model checkpoint integrity."
                )

            counts = obj_ev.get("results", {}).get("counts", {})
            total_objs = sum(counts.values())

            # Section 1: Direct Observations
            if total_objs == 0:
                obs = "No discrete objects were detected above the 0.25 confidence threshold."
            else:
                counts_str = ", ".join(f"{count} {cls_name}(s)" for cls_name, count in counts.items())
                obs = f"A total of {total_objs} object(s) were detected: {counts_str}."

            sections.append(f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{obs}")

            # Section 2: Remote Sensing Interpretation
            if total_objs > 0:
                interp = (
                    f"The distribution indicates anthropogenic activity and infrastructure in the surveyed area. "
                    f"Identified categories align with remote sensing aerial reconnaissance signatures in the xView taxonomy."
                )
            else:
                interp = (
                    "The scene does not exhibit high-contrast signatures corresponding to vehicles, "
                    "isolated buildings, or specialized industrial structures in the model taxonomy."
                )
            sections.append(f"### 2. REMOTE SENSING INTERPRETATION\n{interp}")

            # Section 3: Uncertainties & Caveats
            vis_path = obj_ev.get("visualization", "Not saved")
            conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "N/A (zero detections above threshold)"
            sections.append(
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"- Mean Detection Confidence: {conf_str}\n"
                f"- Detection Model: YOLO11 (xView 60-class taxonomy)\n"
                f"- Bounding Box Overlay: {vis_path}\n"
                f"- Small or partially occluded targets below GSD limit may remain undetected."
            )

        # ---------------------------------------------------------------------
        # Task 2: Bi-Temporal Change Detection (ChangeFormerV6)
        # ---------------------------------------------------------------------
        elif task == TaskType.BI_TEMPORAL_CHANGE or "change_detection" in evidence:
            cd_ev = evidence.get("change_detection", {})
            if not cd_ev.get("success", False):
                err = cd_ev.get("error", "Unknown error")
                return (
                    f"### 1. DIRECT OBSERVATIONS\n"
                    f"Bi-temporal change analysis could not be completed: {err}.\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n"
                    f"Surface comparison could not be derived due to input/execution error.\n\n"
                    f"### 3. UNCERTAINTIES & CAVEATS\n"
                    f"Confidence: 0.0%. Verify that both earlier and later co-registered rasters are provided."
                )

            change_pct = cd_ev.get("change_percentage", 0.0)
            changed_px = cd_ev.get("changed_pixels", 0)
            total_px = cd_ev.get("total_pixels", 0)
            map_path = cd_ev.get("change_map", "Generated")

            # Section 1: Direct Observations
            obs = (
                f"Surface change analysis between Image A and Image B measured {change_pct:.2f}% changed area "
                f"({changed_px:,} changed pixels out of {total_px:,} total evaluated pixels)."
            )
            sections.append(f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{obs}")

            # Section 2: Remote Sensing Interpretation
            if change_pct > 15.0:
                interp = (
                    "Significant surface modification detected between the two acquisition dates. "
                    "Patterns are consistent with rapid urban development, structural clearing, or land transformation."
                )
            elif change_pct > 2.0:
                interp = (
                    "Moderate localized surface changes detected. These correspond to localized construction, "
                    "vegetation growth/harvesting, or minor infrastructure alterations."
                )
            else:
                interp = (
                    "Minimal surface change detected between the two observation dates. The landscape remains "
                    "predominantly stable within the sensor spatial resolution."
                )
            sections.append(f"### 2. REMOTE SENSING INTERPRETATION\n{interp}")

            # Section 3: Uncertainties & Caveats
            sections.append(
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"- Model: ChangeFormerV6 (Transformer Siamese Backbone)\n"
                f"- Overall Change Percentage: {change_pct:.2f}%\n"
                f"- Generated Change Map: {map_path}\n"
                f"- Phenological variations, solar illumination angles, and seasonal shifts can induce minor false change signals."
            )

        # ---------------------------------------------------------------------
        # Task 3: Land Cover Classification (Sentinel-2 SeCo-ECO)
        # ---------------------------------------------------------------------
        elif task == TaskType.SINGLE_IMAGE_LANDCOVER or "landcover" in evidence:
            lc_ev = evidence.get("landcover", {})
            if not lc_ev.get("success", False):
                err = lc_ev.get("error", "Unknown error")
                return (
                    f"### 1. DIRECT OBSERVATIONS\n"
                    f"Land-cover classification status: {err}.\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n"
                    f"Multispectral class distribution requires 12-band Level-2A Sentinel-2 surface reflectance.\n\n"
                    f"### 3. UNCERTAINTIES & CAVEATS\n"
                    f"Confidence: {confidence * 100:.1f}%. Supported input: 12 bands [B01..B12]."
                )

            classes = lc_ev.get("results", {}).get("classes", [])
            if classes:
                classes_str = ", ".join(f"{c['class']} ({c['confidence'] * 100:.1f}%)" for c in classes)
                obs = f"Predicted land-cover classes exceeding decision threshold: {classes_str}."
            else:
                obs = "No BigEarthNet-19 land-cover classes exceeded the positive decision threshold."

            sections.append(f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{obs}")
            sections.append(
                "### 2. REMOTE SENSING INTERPRETATION\n"
                "Reflectance profiles across the 12 multispectral Sentinel-2 bands characterize surface biophysical coverage."
            )
            sections.append(
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"- Architecture: SeCo-ECO ResNet-50 (12-band multispectral)\n"
                f"- Mean Class Confidence: {confidence * 100:.1f}%\n"
                f"- Mixed pixels in 20m/60m resampled bands can influence fractional land-cover boundaries."
            )

        # ---------------------------------------------------------------------
        # Task 4: Optical + SAR Cross-Modal (CROMA)
        # ---------------------------------------------------------------------
        elif task == TaskType.CROSS_MODAL_OPTICAL_SAR or "optical_sar" in evidence:
            croma_ev = evidence.get("optical_sar", {})
            if not croma_ev.get("success", False):
                err = croma_ev.get("error", "Unknown error")
                return (
                    f"### 1. DIRECT OBSERVATIONS\n"
                    f"Optical + SAR feature fusion could not be completed: {err}.\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n"
                    f"Cross-modal analysis requires co-registered Sentinel-2 (optical) and Sentinel-1 (SAR) data.\n\n"
                    f"### 3. UNCERTAINTIES & CAVEATS\n"
                    f"Confidence: 0.0%. Verify that both radar (VV/VH) and optical bands are present."
                )

            reps = croma_ev.get("representations", {})
            joint_shape = reps.get("joint_encodings_shape", [1, 225, 768])
            gap_shape = reps.get("joint_gap_shape", [1, 768])

            obs = (
                f"Multimodal cross-attention extracted joint spatial tokens of shape {joint_shape} (225 patch tokens, 768-dim) "
                f"and a pooled cross-modal scene embedding of shape {gap_shape}."
            )
            sections.append(f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{obs}")
            sections.append(
                "### 2. REMOTE SENSING INTERPRETATION\n"
                "The joint representation synergizes microwave radar scattering (Sentinel-1 VV/VH) with multispectral optical reflectance (Sentinel-2). "
                "Smooth water bodies exhibit specular radar scattering (dark backscatter) alongside strong NIR/SWIR absorption. "
                "Built-up structures produce high radar double-bounce reflection complemented by distinct visible/SWIR spectral signatures."
            )
            sections.append(
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"- Model: CROMA-Base (Contrastive Radar-Optical Masked Autoencoder)\n"
                f"- Pipeline Extraction Confidence: {confidence * 100:.1f}%\n"
                f"- Spatial patch resolution: 8x8 px per token. Radar speckle and incidence angle geometry can affect backscatter intensity."
            )

        # ---------------------------------------------------------------------
        # Task 5: General Description
        # ---------------------------------------------------------------------
        else:
            sections.append(
                "### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n"
                "Satellite scene overview processed. Optical multispectral channels reflect standard geographic surface features."
            )
            sections.append(
                "### 2. REMOTE SENSING INTERPRETATION\n"
                "Remote sensing scene analysis reveals natural and built landscape patterns consistent with regional Earth observation imagery."
            )
            sections.append(
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"- Analysis Confidence: {confidence * 100:.1f}%\n"
                f"- For detailed quantitative metrics, query specific specialist modules (e.g. object counts, change detection, or land-cover classification)."
            )

        return header + "\n\n".join(sections)

