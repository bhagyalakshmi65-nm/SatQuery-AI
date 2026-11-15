"""
Visualization generator for 12-Band Multispectral Land-Cover Predictions.
Extracts True-Color RGB Composite (B04, B03, B02) from the 12-band cube.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .dataset import load_12band_sentinel2_raster, extract_rgb_composite


def plot_prediction_visualization(
    image_path: str,
    predictions: list[dict],
    output_path: str = None,
    threshold: float = 0.5,
) -> str:
    """
    Renders an informative side-by-side plot displaying the true-color RGB composite
    (B04, B03, B02) extracted from the 12-band Sentinel-2 raster alongside predicted
    multi-label land-cover classes and confidence scores.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(image_path))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_prediction.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load 12-band cube and extract RGB composite
    try:
        cube = load_12band_sentinel2_raster(image_path)
        rgb_img = extract_rgb_composite(cube)
    except Exception:
        # Fallback if standard image
        with Image.open(image_path) as img:
            rgb_img = np.array(img.convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    # Left: Sentinel-2 True-Color Composite (B04-B03-B02)
    axes[0].imshow(rgb_img)
    patch_name = os.path.basename(image_path)
    axes[0].set_title(f"Sentinel-2 Patch (B04-B03-B02)\n{patch_name}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Right: Multi-label predictions horizontal bar chart
    if predictions:
        top_preds = predictions[:7]
        classes = [p["class"] for p in reversed(top_preds)]
        confs = [p["confidence"] for p in reversed(top_preds)]

        colors = ["#27ae60" if c >= threshold else "#e74c3c" for c in confs]
        bars = axes[1].barh(classes, confs, color=colors, height=0.55, edgecolor="black", linewidth=0.7)

        for bar, conf in zip(bars, confs):
            axes[1].text(
                conf + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{conf * 100:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#2c3e50",
            )

        axes[1].axvline(threshold, color="#e67e22", linestyle="--", linewidth=1.5, label=f"Cutoff ({threshold:.2f})")
        axes[1].set_xlim(0, 1.15)
        axes[1].set_xlabel("Confidence Score", fontsize=10, fontweight="bold")
        axes[1].set_title("Predicted Land-Cover Classes (BigEarthNet-19)", fontsize=11, fontweight="bold")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    else:
        axes[1].text(
            0.5, 0.5, "No land-cover classes\nexceeded the confidence threshold",
            ha="center", va="center", fontsize=10, color="gray"
        )
        axes[1].set_title("Multi-Label Predictions (Empty)", fontsize=11)
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
