"""
SATQuery AI Land-Cover Classification Module.
12-Band Multispectral Sentinel-2 SeCo-Compatible Pipeline.
"""
from .labels import (
    BIGEARTHNET_19_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLC43_TO_BEN19,
    SENTINEL2_12_BANDS,
    BAND_NAMES_12,
    RGB_BAND_INDICES,
)
from .model import LandCoverClassifier
from .dataset import (
    BigEarthNetDataset,
    load_12band_sentinel2_raster,
    extract_rgb_composite,
    resample_band_to_grid,
    create_synthetic_bigearthnet_samples,
)
from .evaluate import (
    compute_tier1_frequency_baseline,
    compute_multilabel_metrics,
    measure_inference_latency,
)
from .visualization import plot_prediction_visualization
from .train import train_tier2_linear_probe, train_tier3_finetune

__all__ = [
    "BIGEARTHNET_19_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "CLC43_TO_BEN19",
    "SENTINEL2_12_BANDS",
    "BAND_NAMES_12",
    "RGB_BAND_INDICES",
    "LandCoverClassifier",
    "BigEarthNetDataset",
    "load_12band_sentinel2_raster",
    "extract_rgb_composite",
    "resample_band_to_grid",
    "create_synthetic_bigearthnet_samples",
    "compute_tier1_frequency_baseline",
    "compute_multilabel_metrics",
    "measure_inference_latency",
    "plot_prediction_visualization",
    "train_tier2_linear_probe",
    "train_tier3_finetune",
]
