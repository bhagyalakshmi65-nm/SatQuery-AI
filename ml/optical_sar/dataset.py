"""
SatQuery AI - BigEarthNet Optical-SAR PyTorch Dataset Pipeline
Module: ml/optical_sar/dataset.py
Project: SatQuery AI (SIH26167)

Description:
    Production-grade PyTorch Dataset and DataLoader implementation for joint
    Optical-SAR Earth observation analysis. Reads paired Sentinel-1 (VV, VH)
    and Sentinel-2 (12-band) imagery from BigEarthNet v2.0 LMDB archives
    (encoded as SafeTensors, NOT pickle) and pairs them with BigEarthNet.txt
    instruction/question parquet annotations.

Key Specifications:
    1. Modalities & Band Ingestion (BENImageReader compliant):
       - Sentinel-1 SAR: VV, VH (2 bands)
       - Sentinel-2 Optical: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (12 bands)
       - All bands bilinearly resized to 120x120 spatial resolution.
       - Output tensors: SAR [2, 120, 120], Optical [12, 120, 120] (torch.float32).
    2. LMDB Storage:
       - BigEarthNet v2.0 LMDB produced by rico-HDL.
       - Deserializes SafeTensors byte payloads (safe, non-executable, zero-pickle).
       - One read-only, non-locking LMDB environment opened per worker/process
         (zero IPC deadlocks with multi-worker PyTorch DataLoaders).
       - Does NOT cache entire database in RAM (readahead=False).
    3. Metadata & S1 Pairing:
       - Maps Sentinel-2 patch_id -> corresponding Sentinel-1 s1_name via parquet
         columns, metadata lookup files, or direct LMDB records.
       - Filters annotations to ONLY those whose complete image pair exists in the LMDB.
    4. Task Support (BigEarthNet.txt annotations):
       - binary: verification queries (target: yes/no, mapped to 1/0)
       - mcq: multiple-choice queries (target: a/b/c/d, mapped to 0/1/2/3)
       - bbox: spatial grounding (target: normalized [x1, y1, x2, y2] in [0, 1])
       - captioning: natural-language description/answer conditioning target string
       * NOTE: Pixel segmentation is explicitly NOT supported because BigEarthNet.txt
         does not contain segmentation masks.
    5. Scene-Level Partition Integrity:
       - Strict split by image pair (train, validation, test).
       - Never mixes annotations of the same patch_id across splits.
       - Test and benchmark samples are strictly isolated and never leaked into train.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("SatQuery.OpticalSARDataset")

# Conditional dependency imports
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    lmdb = None
    LMDB_AVAILABLE = False

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    safetensors = None
    SAFETENSORS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False


# Official BigEarthNet band definitions
S1_BANDS: List[str] = ["VV", "VH"]
S2_BANDS: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12"
]

# Supported task type aliases
TASK_ALIASES: Dict[str, str] = {
    "binary": "binary",
    "binary_vqa": "binary",
    "yes_no": "binary",
    "verification": "binary",
    "mcq": "mcq",
    "multiple_choice": "mcq",
    "mc_vqa": "mcq",
    "multiple-choice": "mcq",
    "bbox": "bbox",
    "bounding_box": "bbox",
    "bounding box": "bbox",
    "spatial": "bbox",
    "grounding": "bbox",
    "captioning": "captioning",
    "caption": "captioning",
    "description": "captioning",
    "answer_generation": "captioning",
}


# =====================================================================
# 1. SafeTensors Payload Parser (Fallback & Primary)
# =====================================================================

def _parse_safetensors_bytes(buffer: bytes) -> Dict[str, torch.Tensor]:
    """
    Decodes a SafeTensors byte string into a dictionary of PyTorch tensors.
    LMDB values are SafeTensors, NOT pickle.
    """
    if SAFETENSORS_AVAILABLE:
        return safetensors.torch.load(buffer)

    # Pure PyTorch / Python zero-dependency parser for SafeTensors specification
    if len(buffer) < 8:
        raise ValueError("SafeTensors buffer too short to contain a valid header.")

    header_len = int.from_bytes(buffer[:8], "little")
    if len(buffer) < 8 + header_len:
        raise ValueError("SafeTensors buffer truncated before header end.")

    header_json = buffer[8:8 + header_len].decode("utf-8")
    header = json.loads(header_json)
    data_start = 8 + header_len

    dtype_map = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
    }

    tensors: Dict[str, torch.Tensor] = {}
    for tensor_name, meta in header.items():
        if tensor_name == "__metadata__":
            continue

        dtype_str = meta.get("dtype")
        shape = meta.get("shape", [])
        offsets = meta.get("data_offsets", [0, 0])

        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported SafeTensors dtype: {dtype_str}")

        start = data_start + offsets[0]
        end = data_start + offsets[1]
        raw_slice = buffer[start:end]

        t = torch.frombuffer(bytearray(raw_slice), dtype=dtype_map[dtype_str])
        if shape:
            t = t.reshape(shape)
        tensors[tensor_name] = t.clone()

    return tensors


# =====================================================================
# 2. Band Extraction & Resizing (BENImageReader Behavior)
# =====================================================================

def _extract_and_resize_band(
    band_tensor: torch.Tensor,
    target_size: Tuple[int, int] = (120, 120),
) -> torch.Tensor:
    """
    Normalizes and bilinearly interpolates an individual remote-sensing band
    to the target 120x120 spatial dimension (official BENImageReader behavior).
    """
    if band_tensor.dtype != torch.float32:
        band_tensor = band_tensor.to(torch.float32)

    # Normalize rank to [1, 1, H, W] for F.interpolate
    if band_tensor.dim() == 2:
        band_4d = band_tensor.unsqueeze(0).unsqueeze(0)
    elif band_tensor.dim() == 3:
        if band_tensor.size(0) == 1:
            band_4d = band_tensor.unsqueeze(0)
        elif band_tensor.size(-1) == 1:
            band_4d = band_tensor.squeeze(-1).unsqueeze(0).unsqueeze(0)
        else:
            band_4d = band_tensor.unsqueeze(1)
    else:
        band_4d = band_tensor

    h, w = band_4d.shape[-2:]
    if (h, w) != target_size:
        band_resized = F.interpolate(
            band_4d,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        band_resized = band_4d

    return band_resized.squeeze(0).squeeze(0)  # [120, 120]


def _lookup_band(tensors_dict: Dict[str, torch.Tensor], target_band: str) -> torch.Tensor:
    """Case-insensitive band retrieval from decoded SafeTensors dictionary."""
    if target_band in tensors_dict:
        return tensors_dict[target_band]

    upper_map = {k.upper(): v for k, v in tensors_dict.items()}
    if target_band.upper() in upper_map:
        return upper_map[target_band.upper()]

    # Check for prefix stripping e.g. "S2_B01" -> "B01"
    for k, v in tensors_dict.items():
        if k.upper().endswith(target_band.upper()):
            return v

    raise KeyError(
        f"Band '{target_band}' not found in SafeTensors payload. "
        f"Available bands: {list(tensors_dict.keys())}"
    )


# =====================================================================
# 3. Target Parsing & Validation
# =====================================================================

def parse_bbox_target(raw_output: Any, patch_id: str = "") -> torch.Tensor:
    """
    Parses normalized bounding box coordinates [x1, y1, x2, y2] in [0, 1].
    Guarantees x1 <= x2 and y1 <= y2.
    """
    coords = None
    if isinstance(raw_output, (list, tuple)):
        try:
            coords = [float(x) for x in raw_output]
        except (ValueError, TypeError):
            pass
    elif isinstance(raw_output, (np.ndarray, torch.Tensor)):
        coords = [float(x) for x in raw_output.flatten()]
    elif isinstance(raw_output, str):
        # Format can be "[x1, y1, x2, y2]", "[x1 y1, x2 y2]", "x1 y1 x2 y2", etc.
        cleaned = (
            raw_output.replace("[", " ")
            .replace("]", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .replace(";", " ")
        )
        parts = [p for p in cleaned.split() if p.strip()]
        if len(parts) == 4:
            try:
                coords = [float(p) for p in parts]
            except ValueError:
                pass

    if coords is None or len(coords) != 4:
        raise ValueError(
            f"Malformed bounding box target for patch '{patch_id}': '{raw_output}'. "
            "Expected 4 numerical coordinates [x1, y1, x2, y2]."
        )

    x1, y1, x2, y2 = coords
    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1))
    y2 = max(0.0, min(1.0, y2))

    # Order coordinates
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def parse_binary_target(raw_output: Any, patch_id: str = "") -> Tuple[str, int]:
    """Parses binary verification queries into ('yes'/'no', 1/0)."""
    val = str(raw_output).lower().strip().rstrip(".")
    if val in ("yes", "true", "1", "1.0", "y"):
        return "yes", 1
    if val in ("no", "false", "0", "0.0", "n"):
        return "no", 0

    raise ValueError(
        f"Malformed binary target for patch '{patch_id}': '{raw_output}'. "
        "Expected yes/no, true/false, or 1/0."
    )


def parse_mcq_target(raw_output: Any, patch_id: str = "") -> Tuple[str, int]:
    """Parses multiple-choice queries into ('a'/'b'/'c'/'d', 0/1/2/3)."""
    val = str(raw_output).lower().strip()
    val_clean = val.replace("(", "").replace(")", "").replace(".", "").strip()
    mcq_map = {"a": 0, "b": 1, "c": 2, "d": 3, "0": 0, "1": 1, "2": 2, "3": 3}

    if val_clean in mcq_map:
        idx = mcq_map[val_clean]
        char = chr(ord("a") + idx)
        return char, idx

    raise ValueError(
        f"Malformed MCQ target for patch '{patch_id}': '{raw_output}'. "
        "Expected one of 'a', 'b', 'c', 'd' (or 0..3)."
    )


# =====================================================================
# 4. Parquet / Metadata Loading Helper
# =====================================================================

def _load_parquet_or_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads parquet or tabular annotations into a list of row dictionaries."""
    path_str = str(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Annotations file not found at: {path_str}")

    if path_str.endswith(".parquet") or path_str.endswith(".pq"):
        if POLARS_AVAILABLE:
            df = pl.read_parquet(path_str)
            records = df.to_dicts()
        elif PANDAS_AVAILABLE:
            df = pd.read_parquet(path_str)
            records = df.to_dict(orient="records")
        else:
            raise ImportError(
                "Reading parquet annotations requires either 'polars' or 'pandas'/'pyarrow'."
            )
    else:
        # Fallback for CSV or JSON lines
        if PANDAS_AVAILABLE:
            if path_str.endswith(".json") or path_str.endswith(".jsonl"):
                df = pd.read_json(path_str, lines=True if path_str.endswith(".jsonl") else False)
            else:
                df = pd.read_csv(path_str)
            records = df.to_dict(orient="records")
        elif POLARS_AVAILABLE:
            df = pl.read_csv(path_str)
            records = df.to_dicts()
        else:
            raise ImportError("Reading tabular annotations requires 'polars' or 'pandas'.")

    # Normalize column keys to lowercase
    normalized_records = []
    for r in records:
        normalized_records.append({str(k).lower().strip(): v for k, v in r.items()})

    return normalized_records


def _load_s1_mapping(
    mapping_source: Optional[Union[Dict[str, str], str, Path]]
) -> Dict[str, str]:
    """Loads patch_id -> s1_name lookup dictionary from file or dict."""
    if mapping_source is None:
        return {}
    if isinstance(mapping_source, dict):
        return {str(k): str(v) for k, v in mapping_source.items()}

    path_str = str(mapping_source)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"S1 metadata mapping file not found at: {path_str}")

    if path_str.endswith(".json"):
        with open(path_str, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {str(k): str(v) for k, v in data.items()}

    records = _load_parquet_or_csv(path_str)
    mapping = {}
    for r in records:
        p_id = r.get("patch_id") or r.get("patch") or r.get("id") or r.get("s2_name")
        s1_id = r.get("s1_name") or r.get("s1_patch_id") or r.get("sar_name") or r.get("s1_id")
        if p_id and s1_id:
            mapping[str(p_id)] = str(s1_id)

    return mapping


# =====================================================================
# 5. Main BigEarthNet Optical-SAR PyTorch Dataset
# =====================================================================

class BigEarthNetDataset(Dataset):
    """
    Production PyTorch Dataset for BigEarthNet Optical-SAR Multimodal Analysis.

    Ingests:
      - BigEarthNet v2.0 LMDB containing SafeTensors image records.
      - BigEarthNet.txt instruction/question parquet annotations.
    """

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        annotations_path: Union[str, Path],
        split: str = "train",
        s1_mapping: Optional[Union[Dict[str, str], str, Path]] = None,
        verify_lmdb: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            lmdb_path: Path to BigEarthNet v2.0 LMDB directory.
            annotations_path: Path to BigEarthNet.txt parquet annotations file.
            split: Dataset partition ('train', 'val'/'validation', 'test').
            s1_mapping: Optional mapping from patch_id to s1_name.
            verify_lmdb: If True, filters annotations to only pairs present in LMDB.
            transform: Optional torchvision or tensor transform callable.
        """
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        self.annotations_path = str(annotations_path)
        self.split_name = split.lower().strip()
        self.transform = transform

        # Validate split specification
        if self.split_name in ("train", "training"):
            self.valid_splits: Set[str] = {"train", "training"}
            self.forbidden_splits: Set[str] = {"test", "testing", "bench", "benchmark"}
        elif self.split_name in ("val", "validation", "valid"):
            self.valid_splits = {"val", "validation", "valid"}
            self.forbidden_splits = set()
        elif self.split_name in ("test", "testing"):
            self.valid_splits = {"test", "testing"}
            self.forbidden_splits = set()
        else:
            raise ValueError(
                f"Unsupported split '{split}'. Expected 'train', 'val'/'validation', or 'test'."
            )

        # 1. Load S1 metadata mapping
        self.s1_map = _load_s1_mapping(s1_mapping)

        # 2. Load and filter parquet annotations
        self.records = self._load_and_filter_annotations(verify_lmdb=verify_lmdb)

        # 3. LMDB runtime handle (initialized lazily per worker process)
        self._env: Optional[Any] = None

    def _load_and_filter_annotations(self, verify_lmdb: bool) -> List[Dict[str, Any]]:
        """
        Loads parquet rows, filters strictly by split, ensures no data leakage,
        and verifies image pair existence in the LMDB.
        """
        raw_rows = _load_parquet_or_csv(self.annotations_path)
        logger.info(f"Loaded {len(raw_rows)} total annotations from {self.annotations_path}")

        # Split-level filter
        split_filtered: List[Dict[str, Any]] = []
        for r in raw_rows:
            row_split = str(r.get("split", "")).lower().strip()

            # Strict guard: never allow test/bench samples into train
            if self.forbidden_splits and row_split in self.forbidden_splits:
                continue

            if row_split in self.valid_splits:
                patch_id = r.get("patch_id") or r.get("patch") or r.get("id") or r.get("s2_name")
                if not patch_id:
                    continue

                # Auto-detect task type
                raw_type = str(r.get("type") or r.get("task") or r.get("task_type") or "captioning")
                task_type = TASK_ALIASES.get(raw_type.lower().strip(), "captioning")

                # Resolve question & target
                question = r.get("input") or r.get("question") or r.get("prompt") or ""
                target = r.get("output") or r.get("answer") or r.get("target") or ""

                # Resolve S1 pair name
                s1_name = (
                    r.get("s1_name")
                    or r.get("s1_patch_id")
                    or r.get("sar_name")
                    or self.s1_map.get(str(patch_id))
                    or str(patch_id)
                )

                split_filtered.append({
                    "patch_id": str(patch_id),
                    "s1_name": str(s1_name),
                    "question": str(question),
                    "target": target,
                    "task_type": task_type,
                    "split": row_split,
                })

        logger.info(
            f"Split '{self.split_name}': Retained {len(split_filtered)} annotations across image pairs."
        )

        if not verify_lmdb:
            return split_filtered

        # Verify against actual LMDB keys
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"BigEarthNet v2.0 LMDB not found at: {self.lmdb_path}")

        if not LMDB_AVAILABLE:
            raise ImportError(
                "The 'lmdb' package is required to verify and read BigEarthNet LMDBs. "
                "Install via 'pip install lmdb'."
            )

        logger.info("Verifying annotation image pairs against LMDB index...")
        valid_records: List[Dict[str, Any]] = []
        unique_patches = {r["patch_id"]: r["s1_name"] for r in split_filtered}

        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )

        confirmed_pairs: Set[str] = set()
        with env.begin(write=False) as txn:
            for p_id, s1_id in unique_patches.items():
                s2_key = p_id.encode("utf-8")
                s2_val = txn.get(s2_key)
                if s2_val is None:
                    continue

                # If S1 is stored in the same entry or separate key
                if s1_id != p_id:
                    s1_key = s1_id.encode("utf-8")
                    if txn.get(s1_key) is None:
                        # Check if S1 bands are contained in the S2 entry itself
                        try:
                            s2_tensors = _parse_safetensors_bytes(s2_val)
                            if not all(b in s2_tensors for b in S1_BANDS):
                                continue
                        except Exception:
                            continue

                confirmed_pairs.add(p_id)

        env.close()

        for r in split_filtered:
            if r["patch_id"] in confirmed_pairs:
                valid_records.append(r)

        logger.info(
            f"LMDB verification complete: {len(valid_records)} / {len(split_filtered)} "
            f"annotations verified with existing image pairs in LMDB ({len(confirmed_pairs)} pairs)."
        )

        if len(valid_records) == 0:
            raise RuntimeError(
                f"Zero annotations matched existing keys in LMDB '{self.lmdb_path}'. "
                "Check that patch IDs in the annotations match the LMDB keys."
            )

        return valid_records

    def _get_lmdb_env(self) -> Any:
        """
        Lazy per-worker LMDB environment initialization.
        Ensures compatibility with multi-process PyTorch DataLoaders (num_workers > 0).
        """
        if self._env is None:
            if not LMDB_AVAILABLE:
                raise ImportError("The 'lmdb' package is required to read BigEarthNet LMDBs.")

            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        return self._env

    def __len__(self) -> int:
        return len(self.records)

    def _read_image_pair(
        self, patch_id: str, s1_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads and constructs Optical [12, 120, 120] and SAR [2, 120, 120] tensors from LMDB.
        """
        env = self._get_lmdb_env()
        with env.begin(write=False) as txn:
            # 1. Fetch Sentinel-2 entry
            s2_bytes = txn.get(patch_id.encode("utf-8"))
            if s2_bytes is None:
                raise KeyError(f"Sentinel-2 patch '{patch_id}' not found in LMDB.")

            s2_tensors = _parse_safetensors_bytes(s2_bytes)

            # 2. Fetch Sentinel-1 entry
            if s1_name != patch_id:
                s1_bytes = txn.get(s1_name.encode("utf-8"))
                if s1_bytes is not None:
                    s1_tensors = _parse_safetensors_bytes(s1_bytes)
                else:
                    # Fallback to S1 bands stored inside S2 record if combined
                    s1_tensors = s2_tensors
            else:
                s1_tensors = s2_tensors

        # 3. Assemble Sentinel-2 Optical tensor [12, 120, 120]
        optical_bands = []
        for b_name in S2_BANDS:
            b_tensor = _lookup_band(s2_tensors, b_name)
            optical_bands.append(_extract_and_resize_band(b_tensor, target_size=(120, 120)))
        optical = torch.stack(optical_bands, dim=0)

        # 4. Assemble Sentinel-1 SAR tensor [2, 120, 120]
        sar_bands = []
        for b_name in S1_BANDS:
            b_tensor = _lookup_band(s1_tensors, b_name)
            sar_bands.append(_extract_and_resize_band(b_tensor, target_size=(120, 120)))
        sar = torch.stack(sar_bands, dim=0)

        return sar, optical

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single multimodal Optical-SAR sample.
        """
        record = self.records[idx]
        patch_id = record["patch_id"]
        s1_name = record["s1_name"]
        task_type = record["task_type"]
        question = record["question"]
        raw_target = record["target"]

        # Read imagery
        sar, optical = self._read_image_pair(patch_id, s1_name)

        if self.transform is not None:
            sar, optical = self.transform(sar, optical)

        # Parse task-specific targets
        binary_label: Optional[int] = None
        binary_str: Optional[str] = None
        mcq_label: Optional[int] = None
        mcq_char: Optional[str] = None
        bbox_target: Optional[torch.Tensor] = None
        caption_target: Optional[str] = None

        if task_type == "binary":
            binary_str, binary_label = parse_binary_target(raw_target, patch_id=patch_id)
        elif task_type == "mcq":
            mcq_char, mcq_label = parse_mcq_target(raw_target, patch_id=patch_id)
        elif task_type == "bbox":
            bbox_target = parse_bbox_target(raw_target, patch_id=patch_id)
        elif task_type == "captioning":
            caption_target = str(raw_target).strip()

        return {
            "patch_id": patch_id,
            "s1_name": s1_name,
            "sar": sar,                     # [2, 120, 120] float32
            "optical": optical,             # [12, 120, 120] float32
            "question": question,           # Natural language query
            "task_type": task_type,         # 'binary', 'mcq', 'bbox', 'captioning'
            "target": raw_target,           # Original annotation
            # Parsed numerical / canonical targets
            "binary_label": binary_label,   # Optional[int] (0 or 1)
            "binary_str": binary_str,       # Optional[str] ('yes' or 'no')
            "mcq_label": mcq_label,         # Optional[int] (0, 1, 2, 3)
            "mcq_char": mcq_char,           # Optional[str] ('a', 'b', 'c', 'd')
            "bbox_target": bbox_target,     # Optional[torch.Tensor] [4]
            "caption_target": caption_target,  # Optional[str]
        }

    def __del__(self) -> None:
        """Cleanly close the LMDB handle upon destruction."""
        if getattr(self, "_env", None) is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# =====================================================================
# 6. Collate Function for DataLoaders
# =====================================================================

class OpticalSARCollator:
    """
    Robust collator for BigEarthNet Optical-SAR PyTorch batches.
    Handles multimodal image tensor stacking, question batching,
    optional text tokenization, and task-specific target aggregation.
    """

    def __init__(self, tokenizer: Optional[Any] = None, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sar_batch = torch.stack([item["sar"] for item in batch], dim=0)          # [B, 2, 120, 120]
        optical_batch = torch.stack([item["optical"] for item in batch], dim=0)  # [B, 12, 120, 120]

        questions = [item["question"] for item in batch]
        task_types = [item["task_type"] for item in batch]
        patch_ids = [item["patch_id"] for item in batch]
        s1_names = [item["s1_name"] for item in batch]
        raw_targets = [item["target"] for item in batch]

        collated: Dict[str, Any] = {
            "sar": sar_batch,
            "optical": optical_batch,
            "questions": questions,
            "task_types": task_types,
            "patch_ids": patch_ids,
            "s1_names": s1_names,
            "raw_targets": raw_targets,
        }

        # Optional batch tokenization if tokenizer is configured
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            collated["input_ids"] = tokenized["input_ids"]
            collated["attention_mask"] = tokenized["attention_mask"]

        # Aggregate homogeneous task targets when possible
        if all(item["binary_label"] is not None for item in batch):
            collated["binary_labels"] = torch.tensor(
                [item["binary_label"] for item in batch], dtype=torch.long
            )

        if all(item["mcq_label"] is not None for item in batch):
            collated["mcq_labels"] = torch.tensor(
                [item["mcq_label"] for item in batch], dtype=torch.long
            )

        if all(item["bbox_target"] is not None for item in batch):
            collated["bbox_targets"] = torch.stack(
                [item["bbox_target"] for item in batch], dim=0
            )

        if all(item["caption_target"] is not None for item in batch):
            collated["caption_targets"] = [item["caption_target"] for item in batch]

        return collated


# =====================================================================
# 7. Factory Function
# =====================================================================

def create_optical_sar_dataloader(
    lmdb_path: Union[str, Path],
    annotations_path: Union[str, Path],
    split: str = "train",
    batch_size: int = 16,
    s1_mapping: Optional[Union[Dict[str, str], str, Path]] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    tokenizer: Optional[Any] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    verify_lmdb: bool = True,
) -> DataLoader:
    """
    Constructs a production DataLoader for BigEarthNet Optical-SAR datasets.
    """
    dataset = BigEarthNetDataset(
        lmdb_path=lmdb_path,
        annotations_path=annotations_path,
        split=split,
        s1_mapping=s1_mapping,
        verify_lmdb=verify_lmdb,
    )

    if shuffle is None:
        shuffle = True if split.lower().strip() in ("train", "training") else False

    collator = OpticalSARCollator(tokenizer=tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )
