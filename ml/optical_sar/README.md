# SatQuery AI - Optical + SAR Module (CROMA)

This module provides cross-modal representation learning and feature fusion for co-registered Optical and SAR remote sensing imagery using the official **CROMA** (*Contrastive Radar-Optical Masked Autoencoders*, NeurIPS 2023) architecture.

---

## 1. What CROMA Does in SatQuery AI

CROMA serves as the multimodal foundation model in SatQuery AI for joint optical and radar analysis. It processes co-registered multi-spectral and synthetic aperture radar data through dedicated sensor encoders and fuses them via cross-attention to produce rich joint representations for downstream tasks (such as identifying built-up areas and water-covered regions).

### Modalities & Sensor Mapping
- **Optical (Sentinel-2)**: 12 multispectral bands (`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`).
- **SAR (Sentinel-1)**: 2 dual-polarization channels (`VV, VH` backscatter in dB scale).

### Why Both Are Used
1. **Water-Covered Regions**:
   - **SAR**: Smooth water acts as a specular reflector, bouncing radar energy away from the sensor and appearing extremely dark (very low VV/VH backscatter).
   - **Optical**: Water absorbs infrared wavelengths (NIR/SWIR). Combining both avoids false positives from asphalt, cloud shadows, or flat terrain.
2. **Built-Up / Urban Regions**:
   - **SAR**: Buildings, vertical walls, and infrastructure act as corner reflectors producing bright double-bounce scattering (very high backscatter).
   - **Optical**: Urban features exhibit distinctive reflectance in visible/SWIR bands. Combining both separates buildings from dry bare soils and rocky outcrops.
3. **All-Weather Capability**:
   - SAR microwaves penetrate cloud cover, smoke, and atmospheric haze, anchoring optical observations during overcast conditions.

---

## 2. Input Specifications & Preprocessing

### Expected Input Shapes
- **Optical Input**: `[B, 12, H, W]` or `[12, H, W]` (Sentinel-2 12-band reflectance).
- **SAR Input**: `[B, 2, H, W]` or `[2, H, W]` (Sentinel-1 VV, VH backscatter).
- **Combined Input**: Alternatively, a single 14-channel input `[B, 14, H, W]` where channels 0–11 are Optical and 12–13 are SAR.

### Preprocessing Pipeline (Matching Colab Validation)
1. **Validation**: Enforces exactly 12 optical channels and 2 SAR channels; raises informative `ValueError` on invalid dimensions.
2. **Spatial Resizing**: Resizes both modalities to $120 \times 120$ pixels via bilinear interpolation (`TARGET_RESOLUTION = 120`).
3. **Type Casting & Normalization**: Converts to `torch.float32` and divides by `255.0`.
4. **Device Dispatch**: Automatically dispatches tensors to CUDA if available, otherwise CPU.

---

## 3. Output Representations

CROMA produces 6 multimodal feature representations:

| Output Key | Shape | Description |
| :--- | :--- | :--- |
| `SAR_encodings` | `[B, 225, 768]` | Spatial patch tokens from the dedicated Sentinel-1 radar ViT encoder. |
| `SAR_GAP` | `[B, 768]` | Global Average Pooled radar embedding after FFN projection. |
| `optical_encodings` | `[B, 225, 768]` | Spatial patch tokens from the dedicated Sentinel-2 optical ViT encoder. |
| `optical_GAP` | `[B, 768]` | Global Average Pooled optical embedding after FFN projection. |
| `joint_encodings` | `[B, 225, 768]` | Cross-modal fused spatial tokens produced by bidirectional cross-attention. |
| `joint_GAP` | `[B, 768]` | Unified multimodal scene embedding (global average pooled over joint tokens). |

*Note: With an input size of $120 \times 120$ and patch size of 8, the token sequence length is $(120/8)^2 = 15 \times 15 = 225$ patches.*

---

## 4. Pretrained Weights Setup

Pretrained weights (`CROMA_base.pt`, ~741MB) are not committed to Git. Download the official checkpoint:

```bash
# Place in weights directory:
wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt -O ml/optical_sar/weights/CROMA_base.pt
```

### Configurable Checkpoint Path
You can specify the weights location through:
1. Direct function parameter: `extract_multimodal_features(..., checkpoint_path="/path/to/CROMA_base.pt")`
2. Environment variable: `set CROMA_CHECKPOINT_PATH=/path/to/CROMA_base.pt`
3. Default path: `ml/optical_sar/weights/CROMA_base.pt`

---

## 5. Usage

### Python API

```python
import torch
from ml.optical_sar.inference import extract_multimodal_features

# Create or load optical [B, 12, H, W] and SAR [B, 2, H, W]
optical = torch.randn(1, 12, 120, 120) * 255.0
sar = torch.randn(1, 2, 120, 120) * 255.0

# Extract cross-modal representations
features = extract_multimodal_features(optical, sar)

print("Joint spatial tokens:", features["joint_encodings"].shape) # [1, 225, 768]
print("Joint scene embedding:", features["joint_GAP"].shape)       # [1, 768]
```

### CLI Feature Extraction & Smoke Test

```bash
# Run synthetic verification smoke test:
python -m ml.optical_sar.inference --smoke_test

# Run inference on data files:
python -m ml.optical_sar.inference --optical path/to/opt.tif --sar path/to/sar.tif
```

---

## 6. Real-Data Validation (DFC2020 Paired Optical + SAR)

To reproduce and preserve the Colab experiment validating real co-registered Optical + SAR remote sensing data, `validate_real_data.py` executes the exact preprocessing and verification pipeline on a paired 14-channel sample.

### Channel Layout
A DFC2020 co-registered sample contains 14 channels:
- `sample[:12]`: 12-channel Sentinel-2 optical multispectral bands (`[12, 96, 96]`)
- `sample[12:]`: 2-channel Sentinel-1 SAR dual-pol bands (`[2, 96, 96]`)

### Preprocessing & Verification Pipeline
1. **Load Sample**: Load 14-channel patch from `.pt` or `.npy`.
2. **Modal Split**: Slice into `optical = sample[:12]` and `sar = sample[12:]`.
3. **Shape Check**: Verify raw dimensions `[12, 96, 96]` and `[2, 96, 96]`.
4. **Batch Dimension**: Unsqueeze to `[1, 12, 96, 96]` and `[1, 2, 96, 96]`.
5. **Spatial Resampling**: Bilinear interpolation from $96 \times 96$ to native CROMA $120 \times 120$.
6. **Data Type**: Convert to `torch.float32`.
7. **Scale Normalization**: Divide by `255.0` (`x / 255.0`).
8. **Forward Execution**: Run official `PretrainedCROMA`:
   ```python
   outputs = model(SAR_images=sar_input, optical_images=optical_input)
   ```
9. **Representation Matrix Verification**: Check that all 6 output tensors match expected shapes:
   - `SAR_encodings`: `[1, 225, 768]`
   - `SAR_GAP`: `[1, 768]`
   - `optical_encodings`: `[1, 225, 768]`
   - `optical_GAP`: `[1, 768]`
   - `joint_encodings`: `[1, 225, 768]`
   - `joint_GAP`: `[1, 768]`

### Validation CLI Commands

```bash
# 1. Run real-data validation on a preprocessed DFC2020 sample:
python -m ml.optical_sar.validate_real_data --data_path path/to/DFC_preprocessed.pt

# 2. Run with custom checkpoint and specific sample index:
python -m ml.optical_sar.validate_real_data --data_path path/to/DFC_preprocessed.pt --checkpoint ml/optical_sar/weights/CROMA_base.pt --sample_index 0

# 3. Run pipeline smoke test (synthetic 14-channel [14, 96, 96] verification):
python -m ml.optical_sar.validate_real_data --smoke_test
```

> [!NOTE]
> **Validation Scope**: This script validates end-to-end data pipeline integrity, tensor transformation fidelity, and multimodal representation extraction throughput matching Google Colab experiments. It does **not** assert downstream semantic, land-cover segmentation, water, or building detection accuracy (which requires task-specific classification heads).


