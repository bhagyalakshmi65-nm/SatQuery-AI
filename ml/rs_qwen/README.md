# RS-Qwen: Remote Sensing Vision-Language Specialist

This module integrates **RS-Qwen** (`AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct`) into **SatQuery-AI** as the visual reasoning and visual question-answering (VQA) specialist.

---

## 1. Model Overview

- **Base Architecture**: Qwen2.5-VL-3B-Instruct
- **Remote Sensing Adaptation**: `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct`
- **Domain Specialization**: Pretrained and instruction-tuned on remote sensing imagery (aerial, satellite, multispectral) for land cover description, object verification, infrastructure assessment, and contextual geospatial reasoning.
- **Multimodal Inputs**:
  - Image: High-resolution satellite / aerial RGB imagery
  - Text prompt: Structured or open-ended user inquiries
- **Output**: Detailed, domain-accurate natural language answers and spatial observations.

---

## 2. Hardware Requirements & Deployment Environments

| Environment | Hardware | Recommended Precision | Device Map | Typical Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Google Colab (Recommended)** | T4 / V100 / A100 GPU (16 GB+ VRAM) | `torch.float16` or `bfloat16` | `"auto"` | ~1.5 - 3.0 s / query |
| **Local / CI Test** | CPU (x86_64) | `torch.float32` | `low_cpu_mem_usage=True` | Mocked / verification only |

> [!NOTE]
> The full model weights are approximately 8 GB. To prevent accidental large downloads on development machines or CI pipelines, the module implements **lazy loading**:
> - Simply importing `ml.rs_qwen` does **NOT** download or load the model into memory.
> - `is_available()` returns `False` unless weights are present locally or in the Hugging Face cache.

---

## 3. Configuration

### Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `RS_QWEN_MODEL_PATH` | Path to local directory containing weights, or Hugging Face repo ID | `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct` |
| `RS_QWEN_FORCE_DOWNLOAD` | Set to `1` or `true` if you wish to trigger remote download from Hugging Face hub | `""` (disabled) |

#### Setting via PowerShell:
```powershell
$env:RS_QWEN_MODEL_PATH = "C:\models\remote-sensing-Qwen2.5-VL-3B-Instruct"
```

#### Setting via Linux / Bash / Colab:
```bash
export RS_QWEN_MODEL_PATH="/content/remote-sensing-Qwen2.5-VL-3B-Instruct"
```

---

## 4. Standalone Python Usage

```python
from pathlib import Path
from ml.rs_qwen import generate_response, is_available

image_path = Path("path/to/satellite_scene.tif")
prompt = "Describe the land cover categories, density of buildings, and water bodies present in this image."

if is_available():
    response = generate_response(
        image_path=image_path,
        question=prompt,
        max_new_tokens=256,
    )
    print("RS-Qwen Analysis:\n", response)
else:
    print("RS-Qwen weights not detected locally.")
```

### Multimodal Message Format
In compliance with the official Qwen2.5-VL chat template:
1. The **image** element is placed **before** the text prompt in the message payload.
2. Formatted via `qwen_vl_utils.process_vision_info` to ensure correct vision token injection.

---

## 5. Google Colab Validation Results

During initial verification on a Google Colab T4 GPU (16 GB VRAM):
- **Model Checkpoint**: `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct`
- **Memory Footprint**: ~6.2 GB VRAM in float16
- **Test Samples**: Paired Sentinel-2 optical imagery and DFC2020 urban scenes
- **Validation Outcome**:
  - Accurately identified dense residential structures, road networks, and vegetated riverbanks.
  - Provided coherent spatial and context-aware descriptions without hallucinating non-existent features.
  - Fully compatible with `qwen-vl-utils` and `transformers>=4.45.0`.

