# SATQuery AI - Land-Cover Classification Module (`ml/landcover/`)

Welcome to the **12-Band Multispectral Land-Cover Classification** component of the **SATQuery AI** system. This module analyzes Sentinel-2 Level-2A satellite imagery using a **SeCo-compatible 12-Band ResNet-50** multi-label deep learning model.

---

## 🛰️ 1. Task & Dataset Formulation: BigEarthNet-S2

* **Primary Benchmark**: **BigEarthNet-S2**
* **Multi-Label Formulation**:
  Satellite patches ($1.2\text{ km} \times 1.2\text{ km}$) naturally contain **multiple co-occurring land covers** (such as *Arable land*, *Pastures*, and *Mixed forest*).
  * **Target Vector**: $\mathbf{y} \in \{0, 1\}^{19}$ where each element is an independent binary indicator.
  * **Loss Objective**: Multi-label Binary Cross-Entropy with Logits (`torch.nn.BCEWithLogitsLoss`).
* **Important Task Clarification**:
  BigEarthNet-S2 provides **patch-level multi-label land-cover annotations**. It does **not** provide bounding boxes (object detection) or pixel-level segmentation masks.

---

## 🌈 2. 12-Band Sentinel-2 Configuration & Spatial Grid Alignment

To maintain strict input compatibility with the **Sentinel-2 SeCo-ECO ResNet-50** pretrained checkpoint, the pipeline processes all 12 Level-2A surface reflectance bands in standard spectral sequence:

$$\mathcal{B}_{\text{input}} = [\text{B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12}]$$

| Index | Band ID | Description | Native GSD | Native Resolution | Aligned Grid |
|:---:|:---:|:---|:---:|:---:|:---:|
| 0 | **B01** | Coastal aerosol | 60 m | $20 \times 20$ px | $120 \times 120$ px (Bilinear $\times 6$) |
| 1 | **B02** | Blue | 10 m | $120 \times 120$ px | $120 \times 120$ px (Native) |
| 2 | **B03** | Green | 10 m | $120 \times 120$ px | $120 \times 120$ px (Native) |
| 3 | **B04** | Red | 10 m | $120 \times 120$ px | $120 \times 120$ px (Native) |
| 4 | **B05** | Vegetation Red Edge 1 | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |
| 5 | **B06** | Vegetation Red Edge 2 | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |
| 6 | **B07** | Vegetation Red Edge 3 | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |
| 7 | **B08** | Broad NIR | 10 m | $120 \times 120$ px | $120 \times 120$ px (Native) |
| 8 | **B8A** | Narrow NIR | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |
| 9 | **B09** | Water vapour | 60 m | $20 \times 20$ px | $120 \times 120$ px (Bilinear $\times 6$) |
| 10 | **B11** | SWIR 1 | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |
| 11 | **B12** | SWIR 2 | 20 m | $60 \times 60$ px | $120 \times 120$ px (Bilinear $\times 2$) |

* **Common Spatial Grid**: All bands are resampled to a uniform $120 \times 120$ pixel grid ($1.2\text{ km} \times 1.2\text{ km}$ at 10m GSD).
* **Radiometric Preprocessing**: Reflectance values are scaled by $\frac{1}{10000.0}$ and standardized per-channel.

---

## 🏷️ 3. Label Space (BigEarthNet-19)

The 19 aggregated classes defined by Sumbul et al. from the Corine Land Cover (CLC-43) taxonomy:

1. **Urban fabric**
2. **Industrial or commercial units**
3. **Arable land**
4. **Permanent crops**
5. **Pastures**
6. **Complex cultivation patterns**
7. **Land principally occupied by agriculture, with significant areas of natural vegetation**
8. **Agro-forestry areas**
9. **Broad-leaved forest**
10. **Coniferous forest**
11. **Mixed forest**
12. **Natural grassland and sparsely vegetated areas**
13. **Moors, heathland and sclerophyllous vegetation**
14. **Transitional woodland, shrub**
15. **Beaches, dunes, sands**
16. **Inland wetlands**
17. **Coastal wetlands**
18. **Inland waters**
19. **Marine waters**

---

## 🔬 4. Three-Tier Baseline Strategy

We establish a rigorous three-tier experimental progression. All metrics and timings are treated as reference targets and measured empirically during experimentation:

1. **Tier 1 (Class-Frequency Baseline)**:
   Predicts positive classes based purely on marginal class priors in the training split. Measures the statistical performance floor.
2. **Tier 2 (Frozen Pretrained Backbone + Linear Probe)**:
   Freezes the 12-band SeCo ResNet-50 feature extractor and trains exclusively the `Linear(2048, 19)` head with `BCEWithLogitsLoss`. Evaluates representation quality prior to full fine-tuning.
3. **Tier 3 (End-to-End Fine-Tuning)**:
   Fine-tunes all layers with differential learning rates ($10^{-5}$ for backbone, $3 \times 10^{-4}$ for stem & head).

---

## 📁 5. Directory Structure

```text
ml/landcover/
├── configs/
│   └── config.yaml                     # 12-band Sentinel-2 configuration, thresholds, paths
├── notebooks/
│   ├── 01_data_exploration.ipynb       # 12-band spectra, spatial alignment (10m/20m/60m), 19-class co-occurrence
│   ├── 02_baseline.ipynb               # Tier 1 (Class frequency) & Tier 2 (Frozen SeCo linear probe)
│   ├── 03_training.ipynb               # Tier 3 (End-to-end 12-band fine-tuning with BCEWithLogitsLoss)
│   └── 04_evaluation.ipynb             # Holdout test metrics (Micro/Macro F1, mAP, per-class breakdown)
├── src/
│   ├── __init__.py                     # Package exports
│   ├── labels.py                       # 19-class definitions, indices, and CLC-43 translation
│   ├── dataset.py                      # 12-band raster loader, bilinear spatial alignment, BOA scaling
│   ├── model.py                        # 12-channel SeCo-compatible ResNet-50 with multi-label head
│   ├── train.py                        # Tier 2 & Tier 3 training loops with early stopping
│   ├── evaluate.py                     # Multi-label metrics (Micro/Macro F1, precision, recall, mAP)
│   └── visualization.py                # True-color RGB composite (B04-B03-B02) & prediction plot
├── inference.py                        # SATQuery AI exportable API: classify_landcover() for 12-band S2
├── weights/
│   └── best_model.pth                  # Saved checkpoint weights
├── outputs/
│   └── ..._landcover_vis.png           # Annotated visualization outputs
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation and usage guide
```

---

## 🚀 6. SATQuery AI Integration

Another developer or the central SATQuery AI agent can import and execute `classify_landcover`:

```python
from ml.landcover.inference import classify_landcover

# Query a 12-band Sentinel-2 patch (.tif, .npy, or band folder)
result = classify_landcover("path/to/sentinel2_12band_patch.tif", threshold=0.3, visualize=True)
print(result)
```

### Standard Output Format

```json
{
  "success": true,
  "model": "BigEarthNet-ResNet50-SeCo-12B",
  "image": "path/to/sentinel2_12band_patch.tif",
  "results": {
    "classes": [
      {
        "class": "Arable land",
        "confidence": 0.4442
      },
      {
        "class": "Pastures",
        "confidence": 0.4338
      },
      {
        "class": "Coastal wetlands",
        "confidence": 0.3653
      }
    ]
  },
  "visualization": "ml/landcover/outputs/..._landcover_vis.png",
  "error": null
}
```
