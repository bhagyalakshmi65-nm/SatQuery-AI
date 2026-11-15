# This file is part of the SatQuery AI Optical + SAR module.
"""Real-data validation script for CROMA Optical + SAR multimodal fusion.

Reproduces and preserves the Colab validation pipeline using a paired
DFC2020 sample (12 optical bands + 2 SAR bands = 14 channels).

Pipeline Steps:
1. Load real DFC2020 sample containing 14 channels.
2. Split:
   - sample[:12] -> Optical / Sentinel-2
   - sample[12:] -> SAR / Sentinel-1
3. Verify shapes:
   - Optical = [12, 96, 96]
   - SAR     = [2, 96, 96]
4. Add batch dimension:
   - Optical = [1, 12, 96, 96]
   - SAR     = [1, 2, 96, 96]
5. Bilinear interpolation resize to 120x120:
   - Optical = [1, 12, 120, 120]
   - SAR     = [1, 2, 120, 120]
6. Convert to float32.
7. Divide by 255.0.
8. Execute official PretrainedCROMA forward pass:
   model(SAR_images=sar_input, optical_images=optical_input)
9. Print and verify all 6 CROMA outputs and shapes:
   - SAR_encodings:     [1, 225, 768]
   - SAR_GAP:           [1, 768]
   - optical_encodings: [1, 225, 768]
   - optical_GAP:       [1, 768]
   - joint_encodings:   [1, 225, 768]
   - joint_GAP:         [1, 768]

DISCLAIMER:
This script validates data pipeline integrity, tensor transformation fidelity,
and multimodal feature extraction throughput matching Google Colab experiments.
It does NOT claim semantic, segmentation, water, or building detection accuracy,
which requires downstream task-specific classification/segmentation heads.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Ensure local module directory is in sys.path
_MODULE_DIR = str(Path(__file__).parent.resolve())
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
for p in [_MODULE_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.optical_sar.model import get_croma_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("validate_real_data")

# Constants matching DFC2020 and CROMA specifications
EXPECTED_TOTAL_CHANNELS = 14
OPTICAL_CHANNELS = 12
SAR_CHANNELS = 2
RAW_DFC_RESOLUTION = (96, 96)
CROMA_TARGET_RESOLUTION = (120, 120)

EXPECTED_OUTPUT_SHAPES = {
    "SAR_encodings": (1, 225, 768),
    "SAR_GAP": (1, 768),
    "optical_encodings": (1, 225, 768),
    "optical_GAP": (1, 768),
    "joint_encodings": (1, 225, 768),
    "joint_GAP": (1, 768),
}


def load_raw_dfc_sample(
    data_path: Union[str, Path],
    sample_index: int = 0,
) -> torch.Tensor:
    """Load a 14-channel DFC2020 sample from disk (.pt or .npy).

    Args:
        data_path: Path to the data file.
        sample_index: Index to extract if the file contains a batch/dataset.

    Returns:
        PyTorch tensor of shape [14, 96, 96].
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sample data file not found at: {path.resolve()}")

    logger.info("Loading DFC2020 sample from: %s", path)

    if path.suffix.lower() == ".npy":
        raw_data = np.load(path)
        tensor = torch.from_numpy(raw_data)
    else:
        # Load PyTorch object (.pt or .pth)
        raw_data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw_data, dict):
            # Try common keys
            for key in ["sample", "patch", "image", "data", "dfc"]:
                if key in raw_data and isinstance(raw_data[key], (torch.Tensor, np.ndarray)):
                    raw_data = raw_data[key]
                    break
            else:
                # Use first tensor value found
                for val in raw_data.values():
                    if isinstance(val, (torch.Tensor, np.ndarray)):
                        raw_data = val
                        break

        if isinstance(raw_data, (list, tuple)):
            raw_data = raw_data[sample_index]

        if isinstance(raw_data, np.ndarray):
            tensor = torch.from_numpy(raw_data)
        elif isinstance(raw_data, torch.Tensor):
            tensor = raw_data.clone().detach()
        else:
            raise TypeError(f"Unsupported data container type: {type(raw_data)}")

    # Normalize dimensions to [14, 96, 96]
    if tensor.ndim == 4:
        # [N, 14, H, W] -> select sample_index
        if tensor.shape[0] > sample_index:
            tensor = tensor[sample_index]
        else:
            raise IndexError(
                f"Requested sample_index={sample_index}, but dataset contains only {tensor.shape[0]} samples."
            )
    elif tensor.ndim == 3 and tensor.shape[0] != EXPECTED_TOTAL_CHANNELS and tensor.shape[-1] == EXPECTED_TOTAL_CHANNELS:
        # Transpose channel-last [H, W, 14] -> [14, H, W]
        tensor = tensor.permute(2, 0, 1)

    return tensor


def create_synthetic_dfc_sample(
    save_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Generate a synthetic 14-channel DFC2020 sample [14, 96, 96] for pipeline verification."""
    # First 12 optical channels in realistic uint8 range [0, 255]
    optical_sim = torch.randint(10, 240, (12, 96, 96), dtype=torch.uint8)
    # Last 2 SAR channels in realistic uint8 range [0, 255]
    sar_sim = torch.randint(5, 250, (2, 96, 96), dtype=torch.uint8)
    sample = torch.cat([optical_sim, sar_sim], dim=0)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, p)
        logger.info("Saved synthetic DFC2020 sample to: %s", p.resolve())

    return sample


def validate_croma_real_data(
    sample_tensor: torch.Tensor,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    allow_uninitialized: bool = False,
) -> Dict[str, Any]:
    """Execute the exact Colab validation pipeline on a 14-channel Optical + SAR sample.

    Args:
        sample_tensor: Raw 14-channel tensor of shape [14, 96, 96].
        checkpoint_path: Path to CROMA_base.pt.
        device: Target execution device ('cuda', 'cpu', or auto).
        allow_uninitialized: Whether to permit uninitialized weights for smoke testing.

    Returns:
        Dictionary containing verified outputs and pipeline telemetry.
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print("\n" + "=" * 78)
    print("  SatQuery AI - CROMA Real-Data Optical + SAR Validation Pipeline")
    print("=" * 78)
    print(f"Target Execution Device: {target_device}")

    # -------------------------------------------------------------------------
    # Step 1: Input Sample Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Inspecting Input Sample:")
    print(f"  Input Tensor Shape : {list(sample_tensor.shape)}")
    print(f"  Input Tensor Dtype : {sample_tensor.dtype}")
    if sample_tensor.ndim != 3 or sample_tensor.shape[0] != EXPECTED_TOTAL_CHANNELS:
        raise ValueError(
            f"Expected a 14-channel tensor of shape [14, 96, 96], but received shape {list(sample_tensor.shape)}."
        )

    # -------------------------------------------------------------------------
    # Step 2: Split into Optical and SAR Modalities
    # -------------------------------------------------------------------------
    print("\n[Step 2] Splitting into Optical (Sentinel-2) and SAR (Sentinel-1):")
    optical_raw = sample_tensor[:OPTICAL_CHANNELS]   # Channels 0..11
    sar_raw = sample_tensor[OPTICAL_CHANNELS:]       # Channels 12..13
    print(f"  Optical slice (sample[:12]) shape : {list(optical_raw.shape)}")
    print(f"  SAR slice (sample[12:]) shape     : {list(sar_raw.shape)}")

    # -------------------------------------------------------------------------
    # Step 3: Verify Raw Shapes
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Raw Modality Shapes:")
    assert optical_raw.shape == torch.Size([OPTICAL_CHANNELS, *RAW_DFC_RESOLUTION]), (
        f"Optical shape mismatch: expected {[OPTICAL_CHANNELS, *RAW_DFC_RESOLUTION]}, got {list(optical_raw.shape)}"
    )
    assert sar_raw.shape == torch.Size([SAR_CHANNELS, *RAW_DFC_RESOLUTION]), (
        f"SAR shape mismatch: expected {[SAR_CHANNELS, *RAW_DFC_RESOLUTION]}, got {list(sar_raw.shape)}"
    )
    print("  [PASS] Optical raw shape verified : [12, 96, 96]")
    print("  [PASS] SAR raw shape verified     : [2, 96, 96]")

    # -------------------------------------------------------------------------
    # Step 4: Add Batch Dimension
    # -------------------------------------------------------------------------
    print("\n[Step 4] Adding Batch Dimension:")
    optical_batch = optical_raw.unsqueeze(0)  # [1, 12, 96, 96]
    sar_batch = sar_raw.unsqueeze(0)          # [1, 2, 96, 96]
    print(f"  Optical with batch dim : {list(optical_batch.shape)}")
    print(f"  SAR with batch dim     : {list(sar_batch.shape)}")

    # -------------------------------------------------------------------------
    # Step 5: Bilinear Interpolation Resize from 96x96 to 120x120
    # -------------------------------------------------------------------------
    print("\n[Step 5] Resizing to CROMA Native Resolution (96x96 -> 120x120 via Bilinear Interpolation):")
    # Cast to float for interpolation
    optical_resized = F.interpolate(
        optical_batch.float(),
        size=CROMA_TARGET_RESOLUTION,
        mode="bilinear",
        align_corners=False,
    )
    sar_resized = F.interpolate(
        sar_batch.float(),
        size=CROMA_TARGET_RESOLUTION,
        mode="bilinear",
        align_corners=False,
    )
    assert optical_resized.shape == torch.Size([1, OPTICAL_CHANNELS, *CROMA_TARGET_RESOLUTION])
    assert sar_resized.shape == torch.Size([1, SAR_CHANNELS, *CROMA_TARGET_RESOLUTION])
    print(f"  [PASS] Resized Optical shape : {list(optical_resized.shape)}")
    print(f"  [PASS] Resized SAR shape     : {list(sar_resized.shape)}")

    # -------------------------------------------------------------------------
    # Step 6 & 7: Convert to float32 and Scale (x / 255.0)
    # -------------------------------------------------------------------------
    print("\n[Step 6 & 7] Casting to float32 and Normalizing by 255.0:")
    optical_input = optical_resized.float() / 255.0
    sar_input = sar_resized.float() / 255.0

    print(f"  Optical input dtype : {optical_input.dtype}")
    print(f"  Optical min / max   : {optical_input.min().item():.4f} / {optical_input.max().item():.4f}")
    print(f"  SAR input dtype     : {sar_input.dtype}")
    print(f"  SAR min / max       : {sar_input.min().item():.4f} / {sar_input.max().item():.4f}")

    # Transfer to target device
    optical_input = optical_input.to(target_device)
    sar_input = sar_input.to(target_device)

    # -------------------------------------------------------------------------
    # Step 8: Load Official PretrainedCROMA & Execute Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 8] Instantiating Official PretrainedCROMA Model:")
    model = get_croma_model(
        checkpoint_path=checkpoint_path,
        device=target_device,
        size="base",
        modality="both",
        image_resolution=CROMA_TARGET_RESOLUTION[0],
        allow_uninitialized=allow_uninitialized,
    )

    print("\nExecuting forward pass: model(SAR_images=sar_input, optical_images=optical_input)...")
    with torch.no_grad():
        outputs = model(SAR_images=sar_input, optical_images=optical_input)

    # -------------------------------------------------------------------------
    # Step 9: Verify Outputs and Print Verification Matrix
    # -------------------------------------------------------------------------
    print("\n[Step 9] Verifying All 6 CROMA Output Representations:")
    print("-" * 78)
    print(f"{'Output Key':<20} | {'Actual Shape':<18} | {'Expected Shape':<18} | {'Status'}")
    print("-" * 78)

    verification_passed = True
    for key, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        if key not in outputs:
            print(f"{key:<20} | {'MISSING':<18} | {str(list(expected_shape)):<18} | [FAIL]")
            verification_passed = False
            continue

        actual_shape = tuple(outputs[key].shape)
        status = "[PASS]" if actual_shape == expected_shape else "[FAIL]"
        if actual_shape != expected_shape:
            verification_passed = False

        print(f"{key:<20} | {str(list(actual_shape)):<18} | {str(list(expected_shape)):<18} | {status}")

    print("-" * 78)
    if not verification_passed:
        raise AssertionError("CROMA output shape verification failed! One or more outputs mismatched.")

    print("\nAll 6 CROMA output representations successfully verified!")

    # -------------------------------------------------------------------------
    # Step 10: Modality and Representation Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Multimodal Representation Summary")
    print("=" * 78)
    print(f"  * Radar Patch Tokens (SAR_encodings)   : {list(outputs['SAR_encodings'].shape)} (15x15 = 225 patches, dim 768)")
    print(f"  * Radar Scene Embedding (SAR_GAP)      : {list(outputs['SAR_GAP'].shape)} (global average pooled)")
    print(f"  * Optical Patch Tokens (optical_enc.)  : {list(outputs['optical_encodings'].shape)} (15x15 = 225 patches, dim 768)")
    print(f"  * Optical Scene Embedding (opt_GAP)    : {list(outputs['optical_GAP'].shape)} (global average pooled)")
    print(f"  * Cross-Modal Tokens (joint_encodings) : {list(outputs['joint_encodings'].shape)} (bidirectional fused patches)")
    print(f"  * Multimodal Embedding (joint_GAP)     : {list(outputs['joint_GAP'].shape)} (unified scene representation)")
    print("=" * 78)

    print("\n[DISCLAIMER]")
    print("This real-data validation script validates data pipeline compatibility,")
    print("tensor transformation fidelity, and multimodal representation extraction")
    print("throughput matching Google Colab experiments.")
    print("It does NOT assert downstream semantic, land-cover segmentation, water-detection,")
    print("or building-detection accuracy (which requires task-specific classification heads).")
    print("=" * 78 + "\n")

    return {
        "outputs": outputs,
        "optical_shape": list(optical_input.shape),
        "sar_shape": list(sar_input.shape),
        "verification_passed": verification_passed,
    }


def main():
    """Command-line interface for real-data CROMA validation."""
    parser = argparse.ArgumentParser(
        description="SatQuery AI - Validate Real Optical + SAR DFC2020 Data with CROMA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to real DFC2020 sample (.pt or .npy) containing 14 channels.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of sample to extract if the data file contains multiple patches.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to official CROMA_base.pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('cuda', 'cpu', or auto-detect).",
    )
    parser.add_argument(
        "--allow_uninitialized",
        action="store_true",
        help="Allow running with uninitialized weights for pipeline smoke testing when checkpoint is not present.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run an automated end-to-end smoke test using a synthetic 14-channel DFC sample [14, 96, 96].",
    )
    parser.add_argument(
        "--create_synthetic_sample",
        type=str,
        default=None,
        help="Optional path to save a generated synthetic 14-channel DFC sample (.pt).",
    )
    args = parser.parse_args()

    # If requested, generate and save synthetic sample
    if args.create_synthetic_sample:
        create_synthetic_dfc_sample(args.create_synthetic_sample)
        print(f"Synthetic DFC sample successfully created at: {args.create_synthetic_sample}")
        if not args.smoke_test and args.data_path is None:
            return

    # Determine input sample
    if args.smoke_test or args.data_path is None:
        print("[INFO] Operating in smoke test mode with synthetic DFC2020 sample [14, 96, 96].")
        sample_tensor = create_synthetic_dfc_sample()
        # Default allow_uninitialized to True in smoke test mode if checkpoint is absent
        allow_uninit = True if args.checkpoint is None else args.allow_uninitialized
    else:
        sample_tensor = load_raw_dfc_sample(args.data_path, sample_index=args.sample_index)
        allow_uninit = args.allow_uninitialized

    # Execute validation
    validate_croma_real_data(
        sample_tensor=sample_tensor,
        checkpoint_path=args.checkpoint,
        device=args.device,
        allow_uninitialized=allow_uninit,
    )


if __name__ == "__main__":
    main()

