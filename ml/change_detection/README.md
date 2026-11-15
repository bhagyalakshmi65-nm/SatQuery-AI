# SatQuery AI - Change Detection Module

This module provides bi-temporal remote sensing change detection using the official **ChangeFormerV6** architecture (*"A Transformer-Based Siamese Network for Change Detection"*, Bandara & Patel, IGARSS 2022).

---

## Architecture Overview

ChangeFormerV6 is an end-to-end Siamese Transformer model designed for high-resolution bi-temporal change detection:

1. **Siamese Hierarchical Transformer Encoder (`EncoderTransformer_v3`)**:
   - 4-stage pyramid architecture extracting multi-scale contextual features from bi-temporal pairs `(x1, x2)`.
   - Overlapping Patch Embeddings (`OverlapPatchEmbed`) with stage dimensions: `[64, 128, 320, 512]` and depths: `[3, 3, 4, 3]`.
   - Spatial Reduction Self-Attention (`Attention`) with reduction ratios: `[8, 4, 2, 1]`.
   - Depthwise Convolution Mix-FFN (`DWConv` + `Mlp`).

2. **Multi-Scale Difference MLP Decoder (`DecoderTransformer_v3`)**:
   - Convolutional Difference Modules (`conv_diff`) computing feature differences across all 4 scales.
   - Intermediate prediction heads (`make_prediction`) providing deep supervision.
   - Linear multi-scale fusion (`linear_fuse`) combined with progressive residual upsampling blocks (`UpsampleConvLayer` + `ResidualBlock`).
   - Returns 5 multi-scale predictions `[p_c4, p_c3, p_c2, p_c1, cp]`, where the final output `cp` (`outputs[-1]`) has shape `(B, 2, 256, 256)`.

---

## Pretrained Weights

Pretrained ChangeFormerV6 weights (e.g., trained on LEVIR-CD or DSIFN-CD) should be placed at:

```
ml/change_detection/weights/ChangeFormerV6.pth
```

If the checkpoint file is not present, the model will initialize cleanly with a warning for testing and verification purposes.

---

## Python API: `detect_change(image_a, image_b)`

```python
from ml.change_detection.inference import detect_change

result = detect_change(
    image_a="path/to/time1_image.png",
    image_b="path/to/time2_image.png",
    checkpoint_path="ml/change_detection/weights/ChangeFormerV6.pth", # Optional
    output_path="path/to/output_change_map.png",                     # Optional
)

print(result)
```

### Standardized Output Structure

```python
{
    "model": "ChangeFormerV6",
    "image_a": "path/to/time1_image.png",
    "image_b": "path/to/time2_image.png",
    "change_map": "path/to/output_change_map.png",
    "total_pixels": 65536,
    "changed_pixels": 4520,
    "unchanged_pixels": 61016,
    "change_percentage": 6.90
}
```

---

## CLI Usage

Run inference directly from the terminal:

```bash
python -m ml.change_detection.inference \
    --image_a path/to/t1.png \
    --image_b path/to/t2.png \
    --checkpoint ml/change_detection/weights/ChangeFormerV6.pth \
    --output outputs/change_map.png
```
