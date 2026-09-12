# This file is part of the SatQuery AI Optical + SAR module.
"""Model definition and checkpoint loader for CROMA Optical + SAR fusion."""

import logging
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

# Ensure directory is on sys.path so 'from use_croma import PretrainedCROMA' resolves cleanly
_MODULE_DIR = str(Path(__file__).parent.resolve())
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from use_croma import PretrainedCROMA  # noqa: E402

logger = logging.getLogger(__name__)

# Default checkpoint path and environment variable name
ENV_CHECKPOINT_KEY = "CROMA_CHECKPOINT_PATH"
DEFAULT_CHECKPOINT_PATH = Path("ml/optical_sar/weights/CROMA_base.pt")
DOWNLOAD_URL = "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt"

# Global model cache to avoid re-instantiation across inference calls
_MODEL_CACHE: Dict[str, PretrainedCROMA] = {}


def resolve_checkpoint_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the CROMA checkpoint path with priority: argument > env var > default."""
    if custom_path:
        return Path(custom_path)
    env_path = os.environ.get(ENV_CHECKPOINT_KEY)
    if env_path:
        return Path(env_path)
    return DEFAULT_CHECKPOINT_PATH


def get_croma_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    size: str = "base",
    modality: str = "both",
    image_resolution: int = 120,
    allow_uninitialized: bool = False,
) -> PretrainedCROMA:
    """Instantiate and load the official CROMA model for Optical + SAR multimodal analysis.

    Args:
        checkpoint_path: Optional explicit path to pretrained weights (.pt).
        device: Target execution device ('cuda', 'cpu', or auto-detected).
        size: CROMA model scale ('base' or 'large').
        modality: Modality ('both', 'SAR', or 'optical').
        image_resolution: Input patch height/width in pixels (default 120).
        allow_uninitialized: If True, allows instantiating the model without weights for testing.

    Returns:
        PretrainedCROMA model in eval mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint file cannot be found and allow_uninitialized is False.
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    resolved_path = resolve_checkpoint_path(checkpoint_path)
    cache_key = f"{resolved_path}_{size}_{modality}_{image_resolution}_{target_device}_{allow_uninitialized}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    actual_ckpt_str = None
    if resolved_path.is_file():
        actual_ckpt_str = str(resolved_path.resolve())
        logger.info("Loading official CROMA weights from %s", actual_ckpt_str)
    else:
        if not allow_uninitialized:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"[ERROR] CROMA pretrained checkpoint not found at: '{resolved_path}'\n\n"
                f"Please download the official 'CROMA_base.pt' (~741MB) from:\n"
                f"  {DOWNLOAD_URL}\n\n"
                f"and place it at:\n"
                f"  {DEFAULT_CHECKPOINT_PATH}\n\n"
                f"Or specify the location by setting the environment variable:\n"
                f"  set {ENV_CHECKPOINT_KEY}=/path/to/CROMA_base.pt\n"
                f"{'='*70}\n"
            )
        logger.warning(
            "CROMA checkpoint not found at %s. Initialized with uninitialized weights (allow_uninitialized=True).",
            resolved_path,
        )

    model = PretrainedCROMA(
        pretrained_path=actual_ckpt_str,
        size=size,
        modality=modality,
        image_resolution=image_resolution,
    )
    model.to(target_device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model

