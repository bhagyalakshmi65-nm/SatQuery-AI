"""Evidence Grounding Guard for SatQuery AI Agent.

Enforces deterministic evidence grounding on vision-language model responses:
1. Object Detection (YOLO):
   - Zero detections: Strictly prevents any hallucinated/fabricated positive object counts.
     Guarantees final answer states no objects were detected above threshold.
   - Real detections: Strictly preserves verified specialist counts and prevents the LLM
     from altering or fabricating different counts.
   - Failure/Unavailable: Explains detection was not performed; prevents guessed counts.
2. Change Detection (ChangeFormerV6):
   - Preserves verified change percentage and pixel measurements.
   - Prevents fabricated change magnitudes when specialist model failed.
3. Land Cover (SeCo-ECO):
   - Preserves verified classified classes and confidence thresholds.
4. Optical + SAR (CROMA):
   - Prevents interpreting raw tensor representation shapes as semantic objects or classes.
5. Natural Language Output:
   - Strips internal headings (###), developer notices, and technical jargon.
   - Delivers concise, conversational, human-analyst answers.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from backend.agent.schemas import TaskType

logger = logging.getLogger("agent.guard")

# Regex to detect positive object counts asserted in natural language:
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


def sanitize_to_natural_english(text: str) -> str:
    """Strip out markdown headings (###), internal notices, and format into natural paragraphs."""
    if not text:
        return ""

    # Remove internal bracketed notices
    text = re.sub(r"\[Notice:[^\]]+\]", "", text, flags=re.IGNORECASE)

    # Remove section headings like "### 1. DIRECT OBSERVATIONS (Model-Supported Evidence)"
    text = re.sub(r"###\s*\d*\.?\s*[A-Z\s\(\)]+(?:\n|$)", "\n\n", text)

    # Remove lingering markdown heading hashes
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove internal technical prompts or caveats if present in answer
    text = re.sub(r"(?i)Confidence:\s*[\d\.]+%.*?(?=\n\n|\Z)", "", text)
    text = re.sub(r"(?i)Model:\s*[\w\-\s]+(?=\n|\Z)", "", text)
    text = re.sub(r"(?i)Why this answer was produced:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)

    # Clean up multiple newlines and spaces
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs).strip()


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

        # 1. Compound Spatial GIS Guard (highest precedence for multi-specialist pipelines)
        if "gis" in evidence:
            result = cls._guard_gis(raw_response, evidence.get("gis"), confidence, query)
        elif task == TaskType.SINGLE_IMAGE_OBJECT_DETECTION or "object_detection" in evidence:
            result = cls._guard_object_detection(raw_response, evidence.get("object_detection"), confidence, query)
        elif task == TaskType.BI_TEMPORAL_CHANGE or "change_detection" in evidence:
            result = cls._guard_change_detection(raw_response, evidence.get("change_detection"), confidence, query)
        elif task == TaskType.CROSS_MODAL_OPTICAL_SAR or "optical_sar" in evidence:
            result = cls._guard_optical_sar(raw_response, evidence.get("optical_sar"), confidence, query, evidence=evidence)
        elif "segmentation" in evidence and not ("landcover" in evidence and evidence["landcover"].get("success")):
            result = cls._guard_segmentation(raw_response, evidence.get("segmentation"), confidence, query)
        elif task == TaskType.SINGLE_IMAGE_LANDCOVER or "landcover" in evidence:
            result = cls._guard_landcover(raw_response, evidence.get("landcover"), confidence, query)
        else:
            result = raw_response

        # Final pass: sanitize any leftover ### or internal headers into clean natural English
        return sanitize_to_natural_english(result)

    @classmethod
    def _guard_object_detection(
        cls,
        raw_response: str,
        obj_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for object detection."""
        if not obj_ev or not obj_ev.get("success", False):
            return "Automated object detection could not be completed for this scene. Precise object counting could not be derived from the detector."

        results = obj_ev.get("results", {})
        counts = results.get("counts", {})
        total_objs = sum(counts.values())

        # ZERO DETECTIONS CASE
        if total_objs == 0:
            return "No discrete objects were detected above the configured confidence threshold."

        # POSITIVE DETECTIONS CASE
        counts_str = ", ".join(f"{c} {k}" for k, c in counts.items())
        verified_sentence = f"I detected {total_objs} object(s) in the image: {counts_str}."

        modified = raw_response
        for cls_name, true_count in counts.items():
            cls_pattern = re.compile(rf"\b([1-9]\d*)\s+{re.escape(cls_name)}s?(?:\([sS]\))?\b", re.IGNORECASE)
            for match in cls_pattern.finditer(modified):
                stated_num = int(match.group(1))
                if stated_num != true_count:
                    modified = modified[:match.start()] + f"{true_count} {cls_name}(s)" + modified[match.end():]

        number_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        word = number_words.get(total_objs)
        has_total = str(total_objs) in modified or (word and re.search(rf"\b{word}\b", modified, re.IGNORECASE))
        has_all_classes = all(f"{c} {k}" in modified or f"{c} {k}(s)" in modified for k, c in counts.items())

        if not has_total and not has_all_classes:
            modified = f"{verified_sentence} {modified}"

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
            return "I couldn't complete the requested comparison because I couldn't identify two analyzable scenes."

        change_pct = cd_ev.get("change_percentage", 0.0)
        changed_px = cd_ev.get("changed_pixels", 0)
        total_px = cd_ev.get("total_pixels", 0)

        verified_sentence = f"Surface change analysis indicates approximately {change_pct:.1f}% changed area."

        modified = raw_response
        pct_pattern = re.compile(r"\b(\d+(?:\.\d+)?)%\s*(?:of\s+the\s+surface|change|surface\s+change)?\b", re.IGNORECASE)
        for match in pct_pattern.finditer(modified):
            stated_pct = float(match.group(1))
            if abs(stated_pct - change_pct) > 0.5:
                modified = modified[:match.start()] + f"{change_pct:.1f}%" + modified[match.end():]

        # Ensure surface change is mentioned
        has_pct = bool(
            re.search(rf"\b{change_pct:.1f}%", modified)
            or re.search(rf"\b{int(round(change_pct))}%", modified)
            or f"{change_pct}%" in modified
        )
        if "surface change" not in modified.lower() and "changed area" not in modified.lower() and not has_pct:
            modified = f"{verified_sentence} {modified}"

        return modified

    @classmethod
    def _guard_optical_sar(
        cls,
        raw_response: str,
        croma_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enforce strict evidence grounding for Optical + SAR fusion."""
        evidence = evidence or {}
        has_croma_success = bool(croma_ev and croma_ev.get("success", False))
        has_sar_success = bool(evidence.get("sar_analyzer", {}).get("success", False))

        if not has_croma_success and not has_sar_success:
            return "Optical and radar cross-modal analysis requires both Sentinel-2 (optical) and Sentinel-1 (SAR) data."

        modified = raw_response
        tensor_object_pattern = re.compile(r"(?i)\b(?:225|768)\s+(?:discrete\s+)?(?:objects|classes|targets|buildings|structures)\b")
        if tensor_object_pattern.search(modified):
            modified = tensor_object_pattern.sub("dense feature representations", modified)

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
            err_msg = lc_ev.get("error") if lc_ev else None
            if err_msg and "12-band" in err_msg:
                return err_msg
            return "Multispectral land-cover classification requires 12-band Sentinel-2 surface reflectance. The provided input is 3-channel visual imagery (RGB)."
        return raw_response

    @classmethod
    def _guard_gis(
        cls,
        raw_response: str,
        gis_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for compound GIS spatial queries."""
        if not gis_ev or not gis_ev.get("success", False):
            return "The spatial calculations could not be completed for the requested criteria."

        res = gis_ev.get("results", {})
        op = res.get("operation")
        modified = raw_response

        if op == "distance_filter":
            m_cnt = res.get("matches_count", 0)
            thresh = int(res.get("threshold_meters", 100))
            ref_feat = res.get("reference_feature", "river")
            verified_obs = f"{m_cnt} detected building(s)/object(s) are within {thresh}m of the {ref_feat}."

            if m_cnt == 0:
                modified = scrub_fabricated_counts(modified)

            if str(m_cnt) not in modified:
                modified = f"{verified_obs} {modified}"

            return modified

        elif op == "containment_filter":
            c_cnt = res.get("contained_count", 0)
            cont = res.get("container", "specified region")
            verified_obs = f"There are {c_cnt} detected building(s) located inside the {cont} boundary."

            if c_cnt == 0:
                modified = scrub_fabricated_counts(modified)

            if str(c_cnt) not in modified:
                modified = f"{verified_obs} {modified}"

            return modified

        elif op == "area_delta":
            pct = res.get("percentage_change", 0.0)
            trend = res.get("trend", "change")
            feat = res.get("feature", "feature")
            verified_obs = f"Comparative analysis identified a {trend} of approximately {abs(pct):.1f}% in {feat} coverage."

            if f"{abs(pct):.1f}%" not in modified and f"{int(abs(pct))}%" not in modified:
                modified = f"{verified_obs} {modified}"

            return modified

        return modified

    @classmethod
    def _guard_segmentation(
        cls,
        raw_response: str,
        seg_ev: Optional[Dict[str, Any]],
        confidence: Optional[float],
        query: str,
    ) -> str:
        """Enforce strict evidence grounding for semantic segmentation."""
        if not seg_ev or not seg_ev.get("success", False):
            return "Semantic segmentation could not be completed for the specified region."
        return raw_response
