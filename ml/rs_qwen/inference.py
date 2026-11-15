"""RS-Qwen Inference Pipeline for Remote Sensing Vision-Language Question Answering.

Exposes:
- generate_response(image_path, question, max_new_tokens=256)
- is_available()
- build_multimodal_message(image_path, question)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import torch

from ml.rs_qwen.model import (
    get_model_path_or_id,
    get_rs_qwen_model,
    get_rs_qwen_processor,
    is_model_available,
)

logger = logging.getLogger(__name__)


def is_available(model_path: Optional[str] = None) -> bool:
    """Check if the RS-Qwen model is available for inference.

    Returns:
        bool: True if model is loaded or accessible locally without downloading.
    """
    return is_model_available(model_path)


def validate_image(image_input: Union[str, Path, Image.Image]) -> Union[Path, Image.Image]:
    """Validate that the image exists and is readable.

    Args:
        image_input: Path to an image file or a PIL Image instance.

    Returns:
        Validated Path or PIL.Image instance.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the file is not a readable or valid image format.
        TypeError: If the input type is unsupported.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        if not p.is_file():
            raise ValueError(f"Image path is not a file: {image_input}")
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"File at '{image_input}' is not a valid image: {e}") from e
        return p
    elif isinstance(image_input, Image.Image):
        return image_input
    else:
        raise TypeError(
            f"Expected image_path to be str, Path, or PIL.Image, got {type(image_input)}"
        )


def build_multimodal_message(
    image_input: Union[str, Path, Image.Image],
    question: str,
) -> List[Dict[str, Any]]:
    """Construct the official Qwen2.5-VL multimodal message format.

    Ensures the image is placed strictly before the question text prompt.

    Args:
        image_input: Path to the image or PIL Image object.
        question: Text prompt / question for the model.

    Returns:
        List of message dictionaries matching Qwen2.5-VL chat template.
    """
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    image_ref: Union[str, Image.Image]
    if isinstance(image_input, Path):
        image_ref = str(image_input.resolve())
    elif isinstance(image_input, str):
        image_ref = str(Path(image_input).resolve())
    else:
        image_ref = image_input

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_ref,
                },
                {
                    "type": "text",
                    "text": str(question).strip(),
                },
            ],
        }
    ]


def generate_response(
    image_path: Union[str, Path, Image.Image],
    question: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    model_path: Optional[str] = None,
) -> str:
    """Generate a descriptive answer for a remote-sensing image using RS-Qwen.

    Validates inputs, formats the Qwen2.5-VL multimodal prompt with image preceding text,
    runs inference, and decodes the final response cleanly matching the proven Qwen.ipynb pipeline.

    Args:
        image_path: Path to the satellite/aerial image, or PIL Image.
        question: User query or prompt about the image.
        max_new_tokens: Generation budget for response tokens (default: 256).
        do_sample: Whether to use sampling during generation (default: False, matching Qwen.ipynb).
        model_path: Optional override for model checkpoint directory or ID.

    Returns:
        str: Generated textual response.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image file is corrupt or question is empty.
        RuntimeError: If RS-Qwen model is not available.
    """
    # 1. Validate inputs
    validated_image = validate_image(image_path)
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    # 2. Check model availability
    if not is_available(model_path):
        target = model_path or get_model_path_or_id()
        raise RuntimeError(
            f"RS-Qwen model is not available at '{target}'. "
            "Please provide the checkpoint via the RS_QWEN_MODEL_PATH environment variable "
            "or run in a GPU environment where weights are downloaded."
        )

    # 3. Load processor and model lazily
    processor = get_rs_qwen_processor(model_path=model_path)
    model = get_rs_qwen_model(model_path=model_path)

    # 4. Construct multimodal message (image before text, matching Qwen.ipynb)
    messages = build_multimodal_message(validated_image, question)

    # 5. Process vision and text inputs
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs, video_inputs = None, None

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 6. Move inputs to model device (matching Qwen.ipynb: .to(model.device))
    if hasattr(model, "device") and isinstance(model.device, (str, torch.device)):
        device = model.device
    elif hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # 7. Generate tokens using torch.inference_mode() matching Qwen.ipynb
    inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_ctx():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    # 8. Trim prompt tokens and decode (clean_up_tokenization_spaces=False matching Qwen.ipynb)
    input_ids = getattr(inputs, "input_ids", inputs.get("input_ids") if isinstance(inputs, dict) else None)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()

