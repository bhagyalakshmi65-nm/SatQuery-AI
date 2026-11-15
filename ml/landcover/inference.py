"""
Public Inference API for SATQuery AI Land-Cover Classification.

This module exposes classify_landcover() for downstream integration
with the central SATQuery AI agent.

The actual 12-band multispectral inference implementation is handled
inside src/inference.py.
"""

from typing import Optional, Dict, Any

from .src.inference import (
    classify_landcover as _classify_landcover,
    validate_landcover_input as _validate_landcover_input,
)


def validate_landcover_input(image_path: str) -> Dict[str, Any]:
    """Validates whether an input image or directory satisfies the 12-band Sentinel-2 requirement."""
    return _validate_landcover_input(image_path=image_path)


def classify_landcover(
    image_path: str,
    threshold: float = 0.5,
    visualize: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public SATQuery AI land-cover classification API.

    Args:
        image_path: Path to a Sentinel-2 / BigEarthNet-compatible
            12-band multispectral image or patch.
        threshold: Confidence threshold used for multi-label predictions.
        visualize: Whether to generate a prediction visualization.
        output_dir: Directory for the visualization output.

    Returns:
        Dictionary containing classification results in the
        SATQuery AI tool interface format.
    """
    return _classify_landcover(
        image_path=image_path,
        threshold=threshold,
        visualize=visualize,
        output_dir=output_dir,
    )


__all__ = ["classify_landcover", "validate_landcover_input"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = classify_landcover(sys.argv[1])
        print(result)
    else:
        print("Usage: python -m ml.landcover.inference <image_path>")