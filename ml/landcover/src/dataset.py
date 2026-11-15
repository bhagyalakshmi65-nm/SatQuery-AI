"""
BigEarthNet-S2 12-Band Multispectral Dataset and Preprocessing Pipeline.

Configured exclusively for the Sentinel-2 SeCo-ECO 12-channel ResNet-50 architecture.
Bands: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
Common aligned spatial grid: (12, 120, 120)
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tifffile

from .labels import (
    BIGEARTHNET_19_CLASSES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    labels_to_binary_vector,
)

# 12 Sentinel-2 Level-2A surface reflectance bands in strict SeCo order
SENTINEL2_12_BANDS = [
    "B01",  # 60m Coastal aerosol
    "B02",  # 10m Blue
    "B03",  # 10m Green
    "B04",  # 10m Red
    "B05",  # 20m Vegetation Red Edge 1
    "B06",  # 20m Vegetation Red Edge 2
    "B07",  # 20m Vegetation Red Edge 3
    "B08",  # 10m Broad NIR
    "B8A",  # 20m Narrow NIR
    "B09",  # 60m Water vapour
    "B11",  # 20m SWIR 1
    "B12",  # 20m SWIR 2
]

# Native Ground Sample Distance (GSD in meters) and native grid dimensions in BigEarthNet-S2
BAND_METADATA_S2 = {
    "B01": {"gsd": 60, "native_shape": (20, 20)},
    "B02": {"gsd": 10, "native_shape": (120, 120)},
    "B03": {"gsd": 10, "native_shape": (120, 120)},
    "B04": {"gsd": 10, "native_shape": (120, 120)},
    "B05": {"gsd": 20, "native_shape": (60, 60)},
    "B06": {"gsd": 20, "native_shape": (60, 60)},
    "B07": {"gsd": 20, "native_shape": (60, 60)},
    "B08": {"gsd": 10, "native_shape": (120, 120)},
    "B8A": {"gsd": 20, "native_shape": (60, 60)},
    "B09": {"gsd": 60, "native_shape": (20, 20)},
    "B11": {"gsd": 20, "native_shape": (60, 60)},
    "B12": {"gsd": 20, "native_shape": (60, 60)},
}

# SeCo / TorchGeo BigEarthNet-S2 12-Band Normalization Parameters
# Sources:
# 1. SeCo (Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data, Manas et al., ICCV 2021)
# 2. TorchGeo BigEarthNet Sentinel-2 benchmark statistics (Sumbul et al.)
# Computed on Level-2A Bottom-of-Atmosphere (BOA) reflectance scaled by 1/10000.0 (reflectance range [0.0, 1.0])
SECO_S2_12BAND_MEAN = np.array([
    0.034077,  # B01 (Coastal aerosol)
    0.042991,  # B02 (Blue)
    0.061422,  # B03 (Green)
    0.059024,  # B04 (Red)
    0.095068,  # B05 (RedEdge 1)
    0.179246,  # B06 (RedEdge 2)
    0.207546,  # B07 (RedEdge 3)
    0.221803,  # B08 (Broad NIR)
    0.226646,  # B8A (Narrow NIR)
    0.224606,  # B09 (Water vapour)
    0.159442,  # B11 (SWIR 1)
    0.100923,  # B12 (SWIR 2)
], dtype=np.float32)

SECO_S2_12BAND_STD = np.array([
    0.055481,  # B01
    0.057250,  # B02
    0.058288,  # B03
    0.067589,  # B04
    0.072990,  # B05
    0.109601,  # B06
    0.127497,  # B07
    0.136545,  # B08
    0.137942,  # B8A
    0.135614,  # B09
    0.107996,  # B11
    0.082111,  # B12
], dtype=np.float32)

# Aliases for convenience
S2_12BAND_MEAN = SECO_S2_12BAND_MEAN
S2_12BAND_STD = SECO_S2_12BAND_STD


def resample_band_to_grid(
    band_array: np.ndarray,
    target_shape: tuple[int, int] = (120, 120),
) -> np.ndarray:
    """
    Spatially aligns a single 2D Sentinel-2 band raster to the standard BigEarthNet 120x120 grid.

    - 10m bands (120x120): Unchanged (native resolution).
    - 20m bands (60x60): Bilinearly upsampled by 2x to (120, 120).
    - 60m bands (20x20): Bilinearly upsampled by 6x to (120, 120).

    Args:
        band_array: 2D numpy array of shape (H, W).
        target_shape: Destination spatial dimensions (120, 120).

    Returns:
        2D numpy array of shape target_shape, dtype np.float32.
    """
    if band_array.shape == target_shape:
        return band_array.astype(np.float32)

    tensor = torch.from_numpy(band_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(
        tensor,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resampled.squeeze(0).squeeze(0).numpy()


def load_12band_sentinel2_raster(
    patch_path: str,
    target_shape: tuple[int, int] = (120, 120),
) -> np.ndarray:
    """
    Loads a 12-band Sentinel-2 patch from disk and aligns all 12 bands to the common (12, 120, 120) grid.
    Strictly enforces 12-band multispectral input; rejects standard RGB images.

    Supported structures:
    1. Official BigEarthNet Patch Directory:
       Folder containing 12 separate GeoTIFFs matching '*_B01.tif' ... '*_B12.tif'.
    2. Single 12-Band GeoTIFF (.tif, .tiff):
       Multi-band raster with exactly 12 channels.
    3. NumPy array (.npy, .npz):
       Array of shape (12, H, W) or (H, W, 12).

    Returns:
        np.ndarray of shape (12, 120, 120), dtype np.float32, scaled to physical BOA reflectance in [0.0, 1.0].
    """
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Sentinel-2 patch path not found: {patch_path}")

    # Explicit rejection of RGB image formats (PNG, JPG, BMP)
    if patch_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise ValueError(
            f"Invalid image format: '{patch_path}'. RGB images cannot be used in the "
            f"12-band BigEarthNet-S2 pipeline. The pipeline requires all 12 Sentinel-2 "
            f"multispectral bands (B01-B12) in multi-band GeoTIFF or BigEarthNet patch folder format."
        )

    # Case 1: BigEarthNet patch directory containing separate band GeoTIFF files
    if os.path.isdir(patch_path):
        band_arrays = []
        for band_id in SENTINEL2_12_BANDS:
            # Match standard BigEarthNet naming: <patch_id>_<band_id>.tif or .tiff
            matches = glob.glob(os.path.join(patch_path, f"*{band_id}.tif"))
            if not matches:
                matches = glob.glob(os.path.join(patch_path, f"*{band_id}.tiff"))

            if not matches:
                raise FileNotFoundError(
                    f"Required Sentinel-2 band '{band_id}' missing in patch directory: {patch_path}"
                )

            band_file = matches[0]
            raw_band = tifffile.imread(band_file)
            aligned_band = resample_band_to_grid(raw_band, target_shape)
            band_arrays.append(aligned_band)

        cube = np.stack(band_arrays, axis=0)  # Shape: (12, 120, 120)

    # Case 2: NumPy container
    elif patch_path.lower().endswith((".npy", ".npz")):
        if patch_path.lower().endswith(".npz"):
            with np.load(patch_path) as data:
                cube = data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
        else:
            cube = np.load(patch_path)

        if cube.ndim == 3 and cube.shape[-1] == 12:
            cube = np.transpose(cube, (2, 0, 1))

        if cube.ndim != 3 or cube.shape[0] != 12:
            raise ValueError(
                f"Array at '{patch_path}' has shape {cube.shape}. Expected 12 channels for Sentinel-2."
            )

        aligned_bands = [resample_band_to_grid(cube[i], target_shape) for i in range(12)]
        cube = np.stack(aligned_bands, axis=0)

    # Case 3: Single multi-band GeoTIFF
    elif patch_path.lower().endswith((".tif", ".tiff")):
        raw_data = tifffile.imread(patch_path)
        if raw_data.ndim == 3 and raw_data.shape[-1] == 12:
            raw_data = np.transpose(raw_data, (2, 0, 1))

        if raw_data.ndim != 3 or raw_data.shape[0] != 12:
            raise ValueError(
                f"Multi-band raster at '{patch_path}' has shape {raw_data.shape}. "
                f"Expected exactly 12 bands: {SENTINEL2_12_BANDS}. "
                f"RGB or single-band rasters cannot be used in the 12-band pipeline."
            )

        aligned_bands = [resample_band_to_grid(raw_data[i], target_shape) for i in range(12)]
        cube = np.stack(aligned_bands, axis=0)

    else:
        raise ValueError(f"Unsupported file format for 12-band Sentinel-2 data: {patch_path}")

    # Scale raw Sentinel-2 Level-2A digital numbers to physical BOA reflectance in [0.0, 1.0]
    # Sentinel-2 L2A BOA reflectance values are stored as integers with a scale factor of 10000
    if cube.dtype in (np.uint16, np.int16, np.int32) or np.max(cube) > 10.0:
        cube = np.clip(cube.astype(np.float32) / 10000.0, 0.0, 1.0)
    else:
        cube = np.clip(cube.astype(np.float32), 0.0, 1.0)

    return cube


def extract_rgb_composite(cube: np.ndarray) -> np.ndarray:
    """
    Extracts a visually normalized true-color RGB composite (B04, B03, B02)
    from a 12-band Sentinel-2 cube of shape (12, H, W).
    Returns uint8 array of shape (H, W, 3).
    """
    # In SeCo-ECO band order: index 3 is B04 (Red), index 2 is B03 (Green), index 1 is B02 (Blue)
    rgb = cube[[3, 2, 1], :, :].transpose(1, 2, 0).astype(np.float32)
    
    # Robust 2% - 98% percentile contrast stretch
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


class BigEarthNetDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet-S2 12-band multispectral classification.
    
    Each item returns:
        - 12-band normalized tensor of shape (12, 120, 120), dtype torch.float32
        - 19-class binary target vector of shape (19,), dtype torch.float32
        - sample_id: string identifier of the patch
    """

    def __init__(
        self,
        samples: list[dict],
        split: str = "train",
        normalize: bool = True,
    ):
        """
        Args:
            samples: List of dicts, each containing:
                - 'image_path': Path to 12-band GeoTIFF, patch folder, or .npy file
                - 'labels': List of class names (e.g. ['Arable land', 'Pastures'])
                - 'sample_id': (Optional) unique identifier
            split: 'train', 'val', or 'test'
            normalize: Whether to apply SeCo 12-band channel standardization
        """
        self.samples = samples
        self.split = split
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        patch_path = sample["image_path"]
        labels = sample.get("labels", [])
        sample_id = sample.get("sample_id", os.path.basename(os.path.normpath(patch_path)))

        # Load and align all 12 bands to (12, 120, 120)
        cube = load_12band_sentinel2_raster(patch_path, target_shape=(120, 120))

        # Geometric data augmentations for training split (preserving rotational invariance)
        if self.split == "train":
            # Random horizontal flip
            if np.random.rand() > 0.5:
                cube = np.flip(cube, axis=2).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                cube = np.flip(cube, axis=1).copy()
            # Random 90-degree rotations
            k_rot = np.random.randint(0, 4)
            if k_rot > 0:
                cube = np.rot90(cube, k=k_rot, axes=(1, 2)).copy()

        tensor = torch.from_numpy(cube)  # Shape: (12, 120, 120), dtype float32

        # Standardize using SeCo / TorchGeo Sentinel-2 12-band statistics
        if self.normalize:
            mean = torch.from_numpy(SECO_S2_12BAND_MEAN).view(12, 1, 1)
            std = torch.from_numpy(SECO_S2_12BAND_STD).view(12, 1, 1)
            tensor = (tensor - mean) / std

        # 19-class multi-label binary target vector
        target_vector = torch.tensor(
            labels_to_binary_vector(labels),
            dtype=torch.float32,
        )

        return tensor, target_vector, sample_id


def extract_rgb_composite(cube: np.ndarray) -> np.ndarray:
    """
    Extract a true-color RGB composite from a 12-band Sentinel-2 cube [C=12, H, W].
    Uses Band 4 (Red, index 3), Band 3 (Green, index 2), and Band 2 (Blue, index 1).
    Used solely for visualization and human inspection, not for model training.
    """
    if cube.ndim != 3 or cube.shape[0] != 12:
        raise ValueError(f"Expected 12-band cube with shape (12, H, W), got {cube.shape}")

    r = cube[3]  # B04 (Red, 665nm)
    g = cube[2]  # B03 (Green, 560nm)
    b = cube[1]  # B02 (Blue, 490nm)

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # 2% to 98% percentile linear stretch for natural color rendering
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255).astype(np.uint8)


def create_synthetic_bigearthnet_samples(
    output_dir: str,
    num_samples: int = 16,
    target_shape: tuple[int, int] = (120, 120),
) -> list[dict]:
    """
    Creates synthetic 12-band Sentinel-2 TIFF patches strictly for development,
    unit testing, and CI verification when the full archive is not downloaded.

    Generates 12-channel GeoTIFF files (.tif) with realistic spectral signatures.
    RGB PNGs are strictly NOT generated to prevent accidental usage in training.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    archetypes = [
        ["Urban fabric", "Industrial or commercial units"],
        ["Arable land", "Complex cultivation patterns"],
        ["Pastures", "Land principally occupied by agriculture, with significant areas of natural vegetation"],
        ["Broad-leaved forest", "Mixed forest"],
        ["Coniferous forest", "Moors, heathland and sclerophyllous vegetation"],
        ["Inland waters", "Inland wetlands", "Pastures"],
        ["Marine waters", "Coastal wetlands", "Beaches, dunes, sands"],
        ["Arable land", "Pastures", "Agro-forestry areas"],
    ]

    for i in range(num_samples):
        sample_id = f"S2A_MSIL2A_20170613T101031_N0205_R022_T32UNE_{i:04d}"
        patch_path = os.path.join(output_dir, f"{sample_id}.tif")

        archetype = archetypes[i % len(archetypes)]
        np.random.seed(200 + i)

        # Spectral reflectance templates for different terrain archetypes
        if "Inland waters" in archetype or "Marine waters" in archetype:
            base_spectrum = np.array([0.05, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.003], dtype=np.float32)
        elif any("forest" in a.lower() for a in archetype):
            base_spectrum = np.array([0.02, 0.03, 0.04, 0.03, 0.08, 0.22, 0.28, 0.30, 0.32, 0.03, 0.14, 0.07], dtype=np.float32)
        elif "Urban fabric" in archetype:
            base_spectrum = np.array([0.08, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.06, 0.20, 0.18], dtype=np.float32)
        else:
            base_spectrum = np.array([0.03, 0.04, 0.06, 0.06, 0.10, 0.18, 0.22, 0.24, 0.25, 0.04, 0.18, 0.12], dtype=np.float32)

        base_dn = (base_spectrum * 10000.0).astype(np.float32)

        # Build 12-channel raster with native resolution structures resampled to (12, 120, 120)
        cube = np.zeros((12, target_shape[0], target_shape[1]), dtype=np.uint16)
        for b_idx, band_id in enumerate(SENTINEL2_12_BANDS):
            native_h, native_w = BAND_METADATA_S2[band_id]["native_shape"]
            native_noise = np.random.normal(0, base_dn[b_idx] * 0.10 + 10, size=(native_h, native_w))
            raw_native = np.clip(base_dn[b_idx] + native_noise, 0, 10000).astype(np.float32)
            resampled = resample_band_to_grid(raw_native, target_shape)
            cube[b_idx] = np.clip(resampled, 0, 10000).astype(np.uint16)

        # Write strictly 12-band GeoTIFF
        tifffile.imwrite(patch_path, cube)

        samples.append({
            "sample_id": sample_id,
            "image_path": patch_path,
            "labels": archetype,
        })

    return samples
