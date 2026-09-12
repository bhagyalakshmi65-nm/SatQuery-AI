"""RS-Qwen Model Loader for Remote Sensing Vision-Language Reasoning.

Integrates AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct using
Qwen2_5_VLForConditionalGeneration and AutoProcessor.

Supports lazy loading to prevent unintentional 8 GB model downloads in test/local environments.
Weights are resolved from the RS_QWEN_MODEL_PATH environment variable or default to Hugging Face ID.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)

# Default Hugging Face repository ID validated in Google Colab
DEFAULT_MODEL_ID = "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct"
ENV_MODEL_PATH = "RS_QWEN_MODEL_PATH"

# Module-level singletons for lazy initialization
_CACHED_MODEL: Optional[Qwen2_5_VLForConditionalGeneration] = None
_CACHED_PROCESSOR: Optional[AutoProcessor] = None


def get_model_path_or_id() -> str:
    """Resolve the model path or repository ID from environment or default.

    Returns:
        Local directory path if RS_QWEN_MODEL_PATH is set and valid,
        or the default Hugging Face repository identifier.
    """
    env_path = os.environ.get(ENV_MODEL_PATH, "").strip()
    if env_path:
        return env_path
    return DEFAULT_MODEL_ID


def is_model_available(model_path: Optional[str] = None) -> bool:
    """Check whether RS-Qwen model weights or cached checkpoints are available locally.

    Does NOT trigger a remote download over the network.

    Args:
        model_path: Optional path or repo ID override.

    Returns:
        bool: True if model is already loaded in memory or weights exist locally.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return True

    target = model_path or get_model_path_or_id()

    # 1. Local filesystem path check
    local_p = Path(target)
    if local_p.is_dir():
        # Check for standard model config / weights
        has_config = (local_p / "config.json").exists()
        has_weights = (
            any(local_p.glob("*.safetensors"))
            or any(local_p.glob("*.bin"))
            or (local_p / "model.safetensors.index.json").exists()
        )
        if has_config and (has_weights or any(local_p.iterdir())):
            return True
        return False
    elif local_p.is_file():
        return True

    # 2. Check Hugging Face local cache without downloading
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_file = try_to_load_from_cache(target, "config.json")
        if isinstance(cached_file, str) and os.path.exists(cached_file):
            return True
    except Exception as e:
        logger.debug("Hugging Face cache inspection failed: %s", e)

    # 3. Explicit override for environments configured to download
    if os.environ.get("RS_QWEN_FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True

    return False


def get_rs_qwen_processor(model_path: Optional[str] = None) -> AutoProcessor:
    """Lazily load and return the AutoProcessor for RS-Qwen.

    Args:
        model_path: Optional model directory or HF repo ID.

    Returns:
        AutoProcessor instance.

    Raises:
        RuntimeError: If model files are not locally available and downloads are not enabled.
    """
    global _CACHED_PROCESSOR
    if _CACHED_PROCESSOR is not None:
        return _CACHED_PROCESSOR

    target = model_path or get_model_path_or_id()

    if not is_model_available(target):
        raise RuntimeError(
            f"RS-Qwen model is not available at '{target}'. "
            "Please ensure weights are cached or set RS_QWEN_MODEL_PATH to the checkpoint directory."
        )

    logger.info("Loading AutoProcessor for RS-Qwen from: %s", target)
    _CACHED_PROCESSOR = AutoProcessor.from_pretrained(target)
    return _CACHED_PROCESSOR


def get_rs_qwen_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Qwen2_5_VLForConditionalGeneration:
    """Lazily load and return the Qwen2_5_VLForConditionalGeneration model.

    In GPU environments, loads with float16 / bfloat16 and device_map='auto'.
    In CPU environments, loads with float32 and low_cpu_mem_usage=True.

    Args:
        model_path: Optional model directory or HF repo ID.
        device: Target device ('cuda', 'cpu', etc.). Defaults to 'cuda' if available.
        torch_dtype: Optional torch data type. Defaults to float16 on CUDA, float32 on CPU.

    Returns:
        Qwen2_5_VLForConditionalGeneration instance in eval mode.

    Raises:
        RuntimeError: If model files are not locally available.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    target = model_path or get_model_path_or_id()

    if not is_model_available(target):
        raise RuntimeError(
            f"RS-Qwen model is not available at '{target}'. "
            "To use RS-Qwen, provide the checkpoint path via the RS_QWEN_MODEL_PATH "
            "environment variable or run in a GPU environment with model weights downloaded."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(
        "Loading RS-Qwen model from '%s' on %s (%s)...",
        target,
        device,
        torch_dtype,
    )

    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    model.eval()
    _CACHED_MODEL = model
    return _CACHED_MODEL


def reset_model_cache() -> None:
    """Clear cached model and processor instances.

    Useful for testing or memory reclamation.
    """
    global _CACHED_MODEL, _CACHED_PROCESSOR
    _CACHED_MODEL = None
    _CACHED_PROCESSOR = None

