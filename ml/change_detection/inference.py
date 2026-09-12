# This file is part of the SatQuery AI Change Detection module.
"""Inference pipeline for ChangeFormerV6 remote sensing change detection.

Provides detect_change(image_a, image_b) for bi-temporal image pairs.
"""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from ml.change_detection.model import ChangeFormerV6, load_changeformer_weights

logger = logging.getLogger(__name__)

# Default weights location
DEFAULT_WEIGHTS_PATH = Path("ml/change_detection/weights/ChangeFormerV6.pth")
DEFAULT_OUTPUT_DIR = Path("ml/change_detection/outputs")

# Global cache for loaded model instances
_MODEL_CACHE: Dict[str, ChangeFormerV6] = {}

# Preprocessing transform: Resize -> Tensor [0, 1] -> Normalize to [-1, 1]
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def preprocess_image(
    image_path: Union[str, Path],
    target_size: tuple = (256, 256),
) -> torch.Tensor:
    """Load an image from disk, convert to RGB, and preprocess for ChangeFormerV6.

    Args:
        image_path: Path to the image file.
        target_size: Tuple (height, width) for resizing.

    Returns:
        Preprocessed tensor of shape (1, 3, 256, 256) normalized to [-1, 1].
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image as RGB from {image_path}: {e}") from e

    # Apply transform: (3, H, W) in range [-1, 1]
    tensor = PREPROCESS_TRANSFORM(pil_img)
    # Add batch dimension: (1, 3, H, W)
    return tensor.unsqueeze(0)


def get_changeformer_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> ChangeFormerV6:
    """Retrieve or initialize a ChangeFormerV6 model, caching across invocations.

    Args:
        checkpoint_path: Optional path to pretrained weights (.pth / .pt).
        device: Device to place the model on ('cuda', 'cpu', or auto-detected).

    Returns:
        ChangeFormerV6 model ready for inference in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt_str = str(checkpoint_path) if checkpoint_path else str(DEFAULT_WEIGHTS_PATH)
    cache_key = f"{ckpt_str}_{device}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.info("Initializing ChangeFormerV6 architecture (input_nc=3, output_nc=2, embed_dim=256)...")
    model = ChangeFormerV6(input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256)

    resolved_ckpt = Path(ckpt_str)
    if resolved_ckpt.is_file():
        logger.info("Loading pretrained weights from %s...", resolved_ckpt)
        load_changeformer_weights(model, resolved_ckpt, device=device)
    else:
        logger.warning(
            "Pretrained weights not found at %s. Running with initialized weights. "
            "To use pretrained weights, place ChangeFormerV6.pth in ml/change_detection/weights/ "
            "or provide --checkpoint_path.",
            resolved_ckpt,
        )

    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def save_change_map(
    change_mask: np.ndarray,
    output_path: Union[str, Path],
) -> str:
    """Save a binary change mask (0: unchanged, 1: changed) as a visual image.

    Args:
        change_mask: 2D numpy array of shape (256, 256) with 0 and 1 values.
        output_path: Destination path for saved image.

    Returns:
        Absolute string path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert binary mask: 0 -> 0 (black), 1 -> 255 (white)
    mask_uint8 = (change_mask.astype(np.uint8)) * 255
    mask_img = Image.fromarray(mask_uint8, mode="L")
    mask_img.save(output_path)
    return str(output_path.resolve())


def detect_change(
    image_a: Union[str, Path],
    image_b: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Perform bi-temporal change detection on two remote sensing images.

    Args:
        image_a: Path to first / earlier image (time 1).
        image_b: Path to second / later image (time 2).
        checkpoint_path: Optional path to ChangeFormerV6 weights (.pth).
        output_path: Optional path to save the generated change map image.
        device: Device to use ('cuda', 'cpu', or auto-detected).

    Returns:
        Structured dictionary:
        {
            "model": "ChangeFormerV6",
            "image_a": str,
            "image_b": str,
            "change_map": str,
            "total_pixels": int,
            "changed_pixels": int,
            "unchanged_pixels": int,
            "change_percentage": float,
        }
    """
    path_a = Path(image_a)
    path_b = Path(image_b)

    # 1. Resolve device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 2. Preprocess input images to (1, 3, 256, 256)
    x1 = preprocess_image(path_a).to(target_device)
    x2 = preprocess_image(path_b).to(target_device)

    # 3. Load model
    model = get_changeformer_model(checkpoint_path=checkpoint_path, device=target_device)

    # 4. Inference
    with torch.no_grad():
        outputs = model(x1, x2)

    # Extract final full-resolution output [1, 2, 256, 256]
    final_pred = outputs[-1]

    # 5. Generate binary change map: argmax over class dim (0: unchanged, 1: changed)
    pred_mask = torch.argmax(final_pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 6. Compute statistics
    total_pixels = int(pred_mask.size)
    changed_pixels = int(np.sum(pred_mask == 1))
    unchanged_pixels = int(np.sum(pred_mask == 0))
    change_percentage = round((changed_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    # 7. Determine output path and save change map
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem_a = path_a.stem
        stem_b = path_b.stem
        output_path = DEFAULT_OUTPUT_DIR / f"change_map_{stem_a}_{stem_b}_{timestamp}.png"

    saved_path = save_change_map(pred_mask, output_path)

    # 8. Return standardized result dictionary
    return {
        "model": "ChangeFormerV6",
        "image_a": str(path_a),
        "image_b": str(path_b),
        "change_map": saved_path,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "change_percentage": change_percentage,
    }


def main():
    """Command-line interface for Change Detection inference."""
    parser = argparse.ArgumentParser(description="SatQuery AI - ChangeFormerV6 Change Detection")
    parser.add_argument("--image_a", type=str, required=True, help="Path to first / earlier image (time 1)")
    parser.add_argument("--image_b", type=str, required=True, help="Path to second / later image (time 2)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained ChangeFormerV6 weights")
    parser.add_argument("--output", type=str, default=None, help="Output path for the generated change map image")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    result = detect_change(
        image_a=args.image_a,
        image_b=args.image_b,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
    )

    print("\n--- Change Detection Results ---")
    print(f"Model:             {result['model']}")
    print(f"Image A (T1):      {result['image_a']}")
    print(f"Image B (T2):      {result['image_b']}")
    print(f"Change Map Saved:  {result['change_map']}")
    print(f"Total Pixels:      {result['total_pixels']}")
    print(f"Changed Pixels:    {result['changed_pixels']}")
    print(f"Unchanged Pixels:  {result['unchanged_pixels']}")
    print(f"Change Percentage: {result['change_percentage']} %")


if __name__ == "__main__":
    main()
