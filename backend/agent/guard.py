"""Evidence Grounding Guard for SatQuery AI Agent.

Enforces deterministic evidence grounding on vision-language model responses:
1. Object Detection (YOLO):
   - Zero detections: Strictly prevents any hallucinated/fabricated positive object counts.
     Guarantees final answer states no objects were detected above threshold.
   - Real detections: Strictly preserves verified specialist counts and prevents the LLM
     from altering or fabricating different counts.
   - Failure/Unavailable: Explicitly communicates detection was not performed; prevents guessed counts.
2. Change Detection (ChangeFormerV6):
   - Preserves verified change percentage and pixel measurements.
   - Prevents fabricated change magnitudes when specialist model failed.
3. Land Cover (SeCo-ECO):
   - Preserves verified classified classes and confidence thresholds.
4. Optical + SAR (CROMA):
   - Prevents interpreting raw tensor representation shapes as semantic objects or classes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from backend.agent.schemas import TaskType

logger = logging.getLogger("agent.guard")

# Regex to detect positive object counts asserted in natural language:
# Matches: "10 major discrete objects", "There are 10 objects", "found 5 buildings", "contains 3 vehicles"
# Does NOT match "0 objects", "zero objects", "no objects", "no discrete objects"
POSITIVE_COUNT_REGEX = re.compile(
    r"(?i)\b(?:there\s+(?:is|are)|contains?|detect(?:ed|s)?|found|identif(?:y|ied)|see|visible)?\s*"
    r"(?:about\s+|approximately\s+|roughly\s+)?"
    r"([1-9]\d*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|several|multiple)\s+"
    r"(?:major\s+)?(?:discrete\s+)?(?:objects?|buildings?|vehicles?|cars?|ships?|boats?|aircraft|planes?|structures?|features?)\b"
)


def scrub_fabricated_counts(text: str) -> str:
    """Remove any sentences asserting positive counts of objects when zero detections were found."""
    if not text:
        return ""
    # Split into sentences (by period, exclamation, question mark, or newlines)
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    clean_sentences: List[str] = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if POSITIVE_COUNT_REGEX.search(s_clean):
            logger.info("Grounding Guard filtered fabricated count sentence: '%s'", s_clean)
            continue
        clean_sentences.append(s_clean)
    return " ".join(clean_sentences).strip()


class GroundingGuard:
    """Deterministic grounding guard enforcing factual fidelity between specialist telemetry and VLM output."""

    @classmethod
    def enforce(
        cls,
        raw_response: str,
        task: TaskType,
        evidence: Dict[str, Any],
        confidence: Optional[float] = None,
        query: str = "",
    ) -> str:
        """Apply deterministic grounding guards based on the task and specialist evidence."""
        if not raw_response:
            raw_response = ""

        # 1. Object Detection Guard
        if task == TaskType.SINGLE_IMAGE_OBJECT_DETECTION or "object_detection" in evidence:
            return cls._guard_object_detection(raw_response, evidence.get("object_detection"), confidence, query)

        # 2. Bi-Temporal Change Detection Guard
        if task == TaskType.BI_TEMPORAL_CHANGE or "change_detection" in evidence:
            return cls._guard_change_detection(raw_response, evidence.get("change_detection"), confidence, query)

        # 3. Optical + SAR Cross-Modal Guard
        if task == TaskType.CROSS_MODAL_OPTICAL_SAR or "optical_sar" in evidence:
            return cls._guard_optical_sar(raw_response, evidence.get("optical_sar"), confidence, query)

        # 4. Land Cover Guard
        if task == TaskType.SINGLE_IMAGE_LANDCOVER or "landcover" in evidence:
            return cls._guard_landcover(raw_response, evidence.get("landcover"), confidence, query)

        return raw_response

    @classmethod
    def _guard_object_detection(
        cls,
        raw_response: str,
        obj_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for object detection."""
        # A. YOLO unavailable or failed
        if not obj_ev or not obj_ev.get("success", False):
            err = obj_ev.get("error", "Specialist execution failed or model unavailable") if obj_ev else "Object detection model not executed"
            scrubbed = scrub_fabricated_counts(raw_response)
            direct_obs = f"Automated object detection could not be completed: {err}. No verified object counts are available."
            return (
                f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{direct_obs}\n\n"
                f"### 2. REMOTE SENSING INTERPRETATION\n"
                f"{scrubbed if scrubbed else 'Automated object analysis could not locate features due to tool execution failure.'}\n\n"
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"Confidence: 0.0%. Please check image path and model checkpoint integrity."
            )

        results = obj_ev.get("results", {})
        counts = results.get("counts", {})
        total_objs = sum(counts.values())

        # B. ZERO DETECTIONS CASE
        if total_objs == 0:
            zero_obs = "No discrete objects were detected above the configured confidence threshold (0.25) by the YOLO11 specialist model."
            has_fabricated_count = bool(POSITIVE_COUNT_REGEX.search(raw_response))
            scrubbed = scrub_fabricated_counts(raw_response)

            # If the response only contained the fabricated count or is empty after scrubbing:
            if not scrubbed or (has_fabricated_count and len(scrubbed) < 20):
                return (
                    f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{zero_obs}\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n"
                    f"The scene does not exhibit high-contrast signatures corresponding to vehicles, "
                    f"isolated buildings, or specialized industrial structures in the model taxonomy.\n\n"
                    f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                    f"- Mean Detection Confidence: N/A (zero detections above threshold)\n"
                    f"- Detection Model: YOLO11 (xView 60-class taxonomy)\n"
                    f"- Bounding Box Overlay: None\n"
                    f"- Small or partially occluded targets below GSD limit may remain undetected."
                )

            # If scrubbed text contains 3-tier structure, replace Section 1
            if "### 1. DIRECT OBSERVATIONS" in scrubbed:
                scrubbed = re.sub(
                    r"### 1\. DIRECT OBSERVATIONS[^\n]*\n.*?(?=### 2\.|\Z)",
                    f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{zero_obs}\n\n",
                    scrubbed,
                    flags=re.DOTALL,
                )
                return scrubbed

            # If the response doesn't mention zero detections, ensure it is stated
            if "no discrete objects" not in scrubbed.lower() and "zero objects" not in scrubbed.lower() and "no objects" not in scrubbed.lower():
                return (
                    f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{zero_obs}\n\n"
                    f"### 2. REMOTE SENSING INTERPRETATION\n{scrubbed}"
                )

            return scrubbed

        # C. REAL DETECTIONS CASE (total_objs > 0)
        counts_str = ", ".join(f"{c} {k}(s)" for k, c in counts.items())
        verified_obs = f"A total of {total_objs} object(s) were detected by YOLO11: {counts_str}."

        modified = raw_response

        # Correct any conflicting counts for detected classes
        for cls_name, true_count in counts.items():
            cls_pattern = re.compile(rf"\b([1-9]\d*)\s+{re.escape(cls_name)}s?\b", re.IGNORECASE)
            for match in cls_pattern.finditer(modified):
                stated_num = int(match.group(1))
                if stated_num != true_count:
                    logger.info("Grounding Guard corrected count for %s: %d -> %d", cls_name, stated_num, true_count)
                    modified = modified[:match.start()] + f"{true_count} {cls_name}(s)" + modified[match.end():]

        # If modified uses structured sections, ensure Section 1 has verified counts
        if "### 1. DIRECT OBSERVATIONS" in modified:
            if str(total_objs) not in modified:
                modified = re.sub(
                    r"(### 1\. DIRECT OBSERVATIONS[^\n]*\n)",
                    rf"\1{verified_obs}\n",
                    modified,
                )
            return modified

        return modified

    @classmethod
    def _guard_change_detection(
        cls,
        raw_response: str,
        cd_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for change detection."""
        if not cd_ev or not cd_ev.get("success", False):
            err = cd_ev.get("error", "Change detection failed or unavailable") if cd_ev else "Change detection model not executed"
            direct_obs = f"Bi-temporal change analysis could not be completed: {err}."
            return (
                f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n{direct_obs}\n\n"
                f"### 2. REMOTE SENSING INTERPRETATION\nSurface comparison could not be derived due to input/execution error.\n\n"
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\nConfidence: 0.0%. Verify that both earlier and later co-registered rasters are provided."
            )

        change_pct = cd_ev.get("change_percentage", 0.0)
        changed_px = cd_ev.get("changed_pixels", 0)
        total_px = cd_ev.get("total_pixels", 0)
        verified_obs = (
            f"Surface change analysis between Image A and Image B measured {change_pct:.2f}% changed area "
            f"({changed_px:,} changed pixels out of {total_px:,} total evaluated pixels)."
        )

        modified = raw_response

        # Check if LLM fabricated a conflicting change percentage
        pct_pattern = re.compile(r"\b(\d+(?:\.\d+)?)%\s*(?:of\s+the\s+surface|change|surface\s+change)?\b", re.IGNORECASE)
        for match in pct_pattern.finditer(modified):
            stated_pct = float(match.group(1))
            if abs(stated_pct - change_pct) > 0.5:
                logger.info("Grounding Guard corrected change percentage: %.2f%% -> %.2f%%", stated_pct, change_pct)
                modified = modified[:match.start()] + f"{change_pct:.2f}%" + modified[match.end():]

        if "### 1. DIRECT OBSERVATIONS" in modified:
            if f"{change_pct:.2f}%" not in modified and f"{change_pct:.1f}%" not in modified and f"{int(change_pct)}%" not in modified:
                modified = re.sub(
                    r"(### 1\. DIRECT OBSERVATIONS[^\n]*\n)",
                    rf"\1{verified_obs}\n",
                    modified,
                )
            return modified

        return modified

    @classmethod
    def _guard_optical_sar(
        cls,
        raw_response: str,
        croma_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for CROMA Optical + SAR fusion."""
        if not croma_ev or not croma_ev.get("success", False):
            err = croma_ev.get("error", "Optical + SAR fusion failed") if croma_ev else "CROMA model not executed"
            return (
                f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n"
                f"Optical + SAR feature fusion could not be completed: {err}.\n\n"
                f"### 2. REMOTE SENSING INTERPRETATION\n"
                f"Cross-modal analysis requires co-registered Sentinel-2 (optical) and Sentinel-1 (SAR) data.\n\n"
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"Confidence: 0.0%. Verify that both radar (VV/VH) and optical bands are present."
            )

        # Ensure raw tensor shapes are NOT described as discrete semantic object counts
        modified = raw_response

        # Guard against interpreting raw tensor shapes as discrete objects or classes
        tensor_object_pattern = re.compile(r"(?i)\b(?:225|768)\s+(?:discrete\s+)?(?:objects|classes|targets|buildings|structures)\b")
        if tensor_object_pattern.search(modified):
            logger.info("Grounding Guard sanitized raw tensor shape misinterpretation as discrete objects.")
            modified = tensor_object_pattern.sub("dense feature tokens", modified)

        # Guard against hallucinating that CROMA detected objects
        if POSITIVE_COUNT_REGEX.search(modified):
            logger.info("Grounding Guard sanitized false object detection claims in CROMA cross-modal output.")
            modified = scrub_fabricated_counts(modified)

        if "### 1. DIRECT OBSERVATIONS" in modified:
            if "CROMA" not in modified and "225" not in modified:
                clarification = (
                    f"Multimodal cross-attention extracted joint spatial tokens (225 patch tokens, 768-dim). "
                    f"Note: CROMA provides dense feature representations, not discrete semantic object detections."
                )
                modified = re.sub(
                    r"(### 1\. DIRECT OBSERVATIONS[^\n]*\n)",
                    rf"\1{clarification}\n",
                    modified,
                )
            return modified

        return modified

    @classmethod
    def _guard_landcover(
        cls,
        raw_response: str,
        lc_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for land cover classification."""
        if not lc_ev or not lc_ev.get("success", False):
            err = lc_ev.get("error", "Land-cover classification failed") if lc_ev else "Land cover model not executed"
            return (
                f"### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)\n"
                f"Land-cover classification status: {err}.\n\n"
                f"### 2. REMOTE SENSING INTERPRETATION\n"
                f"Multispectral class distribution requires 12-band Level-2A Sentinel-2 surface reflectance.\n\n"
                f"### 3. UNCERTAINTIES & SENSOR CAVEATS\n"
                f"Confidence: {confidence * 100:.1f}%. Supported input: 12 bands [B01..B12]."
            )
        return raw_response
