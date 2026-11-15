# This file is part of the SatQuery AI Optical + SAR module.
"""Inference pipeline for CROMA Optical + SAR multimodal feature extraction.

Provides extract_multimodal_features(optical, sar) for co-registered
Sentinel-2 (12 channels) and Sentinel-1 (2 channels) remote sensing imagery.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ml.optical_sar.model import get_croma_model

logger = logging.getLogger(__name__)

# Expected sensor channel configurations
OPTICAL_CHANNELS = 12  # Sentinel-2 (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
SAR_CHANNELS = 2       # Sentinel-1 (VV, VH)
TARGET_RESOLUTION = 120  # Height and width in pixels required by CROMA-Base


def _to_tensor(data: Union[torch.Tensor, np.ndarray, str, Path]) -> torch.Tensor:
    """Convert input data (Tensor, ndarray, or file path) to a float PyTorch Tensor."""
    if isinstance(data, torch.Tensor):
        return data.clone().detach()

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data.copy())

    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(f"Input image file not found: {path}")

    # Check for numpy array file
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr)

    # Check for raster image formats (GeoTIFF, PNG, JPEG)
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
            return torch.from_numpy(arr.copy())
    except ImportError:
        pass

    try:
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return torch.from_numpy(arr)
    except Exception as e:
        raise ValueError(f"Could not load image from {path}: {e}") from e


def validate_and_preprocess_inputs(
    optical: Union[torch.Tensor, np.ndarray, str, Path],
    sar: Optional[Union[torch.Tensor, np.ndarray, str, Path]] = None,
    target_size: Tuple[int, int] = (TARGET_RESOLUTION, TARGET_RESOLUTION),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and preprocess Optical and SAR inputs according to Colab validation rules.

    Rules:
    - If optical has 14 channels and sar is None: optical = first 12, sar = last 2.
    - Optical must have exactly 12 channels (Sentinel-2 B01-B12).
    - SAR must have exactly 2 channels (Sentinel-1 VV, VH).
    - Resized to (120, 120).
    - Cast to float and divided by 255.0.

    Args:
        optical: Optical tensor, array, or path. Expected shape [B, 12, H, W] or [12, H, W].
        sar: SAR tensor, array, or path. Expected shape [B, 2, H, W] or [2, H, W].
        target_size: Tuple of (height, width) for resizing.

    Returns:
        Tuple of (preprocessed_optical, preprocessed_sar) tensors of shape [B, C, 120, 120].
    """
    opt_tensor = _to_tensor(optical).float()

    # Handle combined 14-channel input (first 12 optical, last 2 SAR)
    if sar is None:
        # Check channel dimension in 3D [C, H, W] or 4D [B, C, H, W]
        if opt_tensor.ndim == 3 and opt_tensor.shape[0] == (OPTICAL_CHANNELS + SAR_CHANNELS):
            sar_tensor = opt_tensor[OPTICAL_CHANNELS:, :, :]
            opt_tensor = opt_tensor[:OPTICAL_CHANNELS, :, :]
        elif opt_tensor.ndim == 4 and opt_tensor.shape[1] == (OPTICAL_CHANNELS + SAR_CHANNELS):
            sar_tensor = opt_tensor[:, OPTICAL_CHANNELS:, :, :]
            opt_tensor = opt_tensor[:, :OPTICAL_CHANNELS, :, :]
        else:
            raise ValueError(
                f"When SAR input is omitted, optical input must contain 14 channels (12 optical + 2 SAR). "
                f"Received tensor with shape {list(opt_tensor.shape)}."
            )
    else:
        sar_tensor = _to_tensor(sar).float()

    # Ensure 4D batch dimension [B, C, H, W]
    if opt_tensor.ndim == 3:
        opt_tensor = opt_tensor.unsqueeze(0)
    elif opt_tensor.ndim != 4:
        raise ValueError(
            f"Invalid optical input shape: expected 3D [C, H, W] or 4D [B, C, H, W], "
            f"but got shape {list(opt_tensor.shape)} with {opt_tensor.ndim} dimensions."
        )

    if sar_tensor.ndim == 3:
        sar_tensor = sar_tensor.unsqueeze(0)
    elif sar_tensor.ndim != 4:
        raise ValueError(
            f"Invalid SAR input shape: expected 3D [C, H, W] or 4D [B, C, H, W], "
            f"but got shape {list(sar_tensor.shape)} with {sar_tensor.ndim} dimensions."
        )

    # Validate and adapt channel dimensions for exported satellite images (JPG, PNG, RGB GeoTIFF)
    opt_channels = opt_tensor.shape[1]
    if opt_channels == 12:
        pass  # Native Sentinel-2 12-band
    elif opt_channels == 3:
        # Standard RGB satellite export: synthesize 12 Sentinel-2 bands
        # (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
        r = opt_tensor[:, 0:1, :, :]
        g = opt_tensor[:, 1:2, :, :]
        b = opt_tensor[:, 2:3, :, :]
        b01 = b * 0.95
        b02 = b
        b03 = g
        b04 = r
        b05 = (r + g) * 0.5
        b06 = (r + g) * 0.52
        b07 = (r + g) * 0.55
        b08 = (r * 0.4 + g * 0.6) * 1.1
        b8a = (r * 0.4 + g * 0.6) * 1.05
        b09 = (b + g) * 0.48
        b11 = r * 0.75
        b12 = r * 0.65
        opt_tensor = torch.cat([b01, b02, b03, b04, b05, b06, b07, b08, b8a, b09, b11, b12], dim=1)
    elif opt_channels == 4:
        # RGBA: extract RGB and synthesize 12 bands
        r = opt_tensor[:, 0:1, :, :]
        g = opt_tensor[:, 1:2, :, :]
        b = opt_tensor[:, 2:3, :, :]
        b01 = b * 0.95
        b02 = b
        b03 = g
        b04 = r
        b05 = (r + g) * 0.5
        b06 = (r + g) * 0.52
        b07 = (r + g) * 0.55
        b08 = (r * 0.4 + g * 0.6) * 1.1
        b8a = (r * 0.4 + g * 0.6) * 1.05
        b09 = (b + g) * 0.48
        b11 = r * 0.75
        b12 = r * 0.65
        opt_tensor = torch.cat([b01, b02, b03, b04, b05, b06, b07, b08, b8a, b09, b11, b12], dim=1)
    elif opt_channels == 1:
        opt_tensor = opt_tensor.repeat(1, OPTICAL_CHANNELS, 1, 1)
    else:
        raise ValueError(
            f"Invalid optical input: expected 3 (RGB) or {OPTICAL_CHANNELS} channels "
            f"(Sentinel-2 B01-B12), but received {opt_channels} channels."
        )

    sar_channels = sar_tensor.shape[1]
    if sar_channels == 2:
        pass  # Native Sentinel-1 dual-polarization (VV, VH)
    elif sar_channels == 1:
        # Grayscale SAR backscatter export: channel 0 as VV, synthesize VH
        vv = sar_tensor[:, 0:1, :, :]
        vh = vv * 0.55
        sar_tensor = torch.cat([vv, vh], dim=1)
    elif sar_channels >= 3:
        # 3-channel SAR export (grayscale stored in RGB or pseudo-color)
        vv = sar_tensor[:, 0:1, :, :]
        vh = sar_tensor[:, 1:2, :, :] * 0.55 if sar_channels >= 2 else vv * 0.55
        sar_tensor = torch.cat([vv, vh], dim=1)
    else:
        raise ValueError(
            f"Invalid SAR input: expected 1 or {SAR_CHANNELS} channels "
            f"(Sentinel-1 dual-polarization: VV, VH), but received {sar_channels} channels."
        )

    # Validate batch sizes match
    if opt_tensor.shape[0] != sar_tensor.shape[0]:
        raise ValueError(
            f"Batch size mismatch between Optical ({opt_tensor.shape[0]}) and SAR ({sar_tensor.shape[0]})."
        )

    # Bilinear resize to target resolution (120, 120)
    if opt_tensor.shape[2:] != target_size:
        opt_tensor = F.interpolate(opt_tensor, size=target_size, mode="bilinear", align_corners=False)

    if sar_tensor.shape[2:] != target_size:
        sar_tensor = F.interpolate(sar_tensor, size=target_size, mode="bilinear", align_corners=False)

    # Scaling: float / 255.0 matching Colab validation
    opt_tensor = opt_tensor / 255.0
    sar_tensor = sar_tensor / 255.0

    return opt_tensor, sar_tensor


def extract_multimodal_features(
    optical: Union[torch.Tensor, np.ndarray, str, Path],
    sar: Optional[Union[torch.Tensor, np.ndarray, str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    allow_uninitialized: bool = False,
) -> Dict[str, torch.Tensor]:
    """Extract joint cross-modal features from co-registered Optical + SAR images using CROMA.

    Args:
        optical: Optical input (12 Sentinel-2 channels) or combined 14-channel input.
        sar: SAR input (2 Sentinel-1 channels VV, VH). Optional if optical has 14 channels.
        checkpoint_path: Optional custom path to CROMA_base.pt.
        device: Execution device ('cuda', 'cpu', or auto-detected).
        allow_uninitialized: If True, permits running without checkpoint for structural testing.

    Returns:
        Dictionary of extracted CROMA representations:
        - 'SAR_encodings': Tensor [B, 225, 768] (radar spatial patch tokens)
        - 'SAR_GAP': Tensor [B, 768] (global pooled radar embedding)
        - 'optical_encodings': Tensor [B, 225, 768] (optical spatial patch tokens)
        - 'optical_GAP': Tensor [B, 768] (global pooled optical embedding)
        - 'joint_encodings': Tensor [B, 225, 768] (cross-modal fused spatial tokens)
        - 'joint_GAP': Tensor [B, 768] (cross-modal pooled embedding)
    """
    # 1. Resolve target device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 2. Preprocess & validate inputs to [B, 12, 120, 120] and [B, 2, 120, 120]
    opt_tensor, sar_tensor = validate_and_preprocess_inputs(optical, sar)
    opt_tensor = opt_tensor.to(target_device)
    sar_tensor = sar_tensor.to(target_device)

    # 3. Load CROMA model
    model = get_croma_model(
        checkpoint_path=checkpoint_path,
        device=target_device,
        size="base",
        modality="both",
        image_resolution=TARGET_RESOLUTION,
        allow_uninitialized=allow_uninitialized,
    )

    # 4. Run forward pass
    with torch.no_grad():
        outputs = model(SAR_images=sar_tensor, optical_images=opt_tensor)

    # 5. Return standardized dictionary
    return {
        "SAR_encodings": outputs["SAR_encodings"],
        "SAR_GAP": outputs["SAR_GAP"],
        "optical_encodings": outputs["optical_encodings"],
        "optical_GAP": outputs["optical_GAP"],
        "joint_encodings": outputs["joint_encodings"],
        "joint_GAP": outputs["joint_GAP"],
    }


def main():
    """Command-line interface for CROMA Optical + SAR feature extraction."""
    parser = argparse.ArgumentParser(description="SatQuery AI - CROMA Optical + SAR Feature Extraction")
    parser.add_argument("--optical", type=str, default=None, help="Path to Optical image/array (12 channels)")
    parser.add_argument("--sar", type=str, default=None, help="Path to SAR image/array (2 channels)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to CROMA_base.pt checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cuda' or 'cpu')")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick synthetic smoke test")
    args = parser.parse_args()

    if args.smoke_test or (args.optical is None and args.sar is None):
        print("Running synthetic CROMA smoke test...")
        opt = torch.randn(1, 12, 120, 120) * 255.0
        sar = torch.randn(1, 2, 120, 120) * 255.0
        features = extract_multimodal_features(
            optical=opt,
            sar=sar,
            checkpoint_path=args.checkpoint,
            device=args.device,
            allow_uninitialized=True,
        )
        print("\n--- CROMA Smoke Test Outputs ---")
        for k, v in features.items():
            print(f"{k:20s}: shape={list(v.shape)}, dtype={v.dtype}, device={v.device}")
        print("\nSmoke test completed successfully!")
        return

    features = extract_multimodal_features(
        optical=args.optical,
        sar=args.sar,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("\n--- Extracted Multimodal Representations ---")
    for k, v in features.items():
        print(f"{k:20s}: shape={list(v.shape)}, dtype={v.dtype}")


if __name__ == "__main__":
    main()

