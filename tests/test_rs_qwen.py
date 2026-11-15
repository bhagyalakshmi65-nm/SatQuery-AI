"""Unit tests for ml/rs_qwen specialist module.

Verifies:
- Clean module imports
- Model configuration and environment variable resolution
- Strict image path and validity validation
- Clear error reporting for model unavailability
- Correct multimodal message layout (image before text)
- Mocked generation pipeline execution
- Lazy loading guarantees (no unintended 8 GB downloads)
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

import ml.rs_qwen
from ml.rs_qwen import (
    DEFAULT_MODEL_ID,
    ENV_MODEL_PATH,
    build_multimodal_message,
    generate_response,
    get_model_path_or_id,
    get_rs_qwen_model,
    get_rs_qwen_processor,
    is_available,
    is_model_available,
    reset_model_cache,
    validate_image,
)


class TestRSQwenModule(unittest.TestCase):
    """Test suite for RS-Qwen Vision-Language Specialist."""

    def setUp(self):
        """Prepare temporary test images and reset cached instances."""
        reset_model_cache()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        # Create a valid test image
        self.valid_img_path = self.temp_dir_path / "test_satellite.png"
        img = Image.new("RGB", (64, 64), color="forestgreen")
        img.save(self.valid_img_path)

        # Create a corrupt non-image file
        self.corrupt_file_path = self.temp_dir_path / "corrupt.png"
        with open(self.corrupt_file_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_CONTENT")

    def tearDown(self):
        """Clean up temporary directory and caches."""
        reset_model_cache()
        self.temp_dir.cleanup()

    def test_imports_cleanly(self):
        """Verify that all public APIs are exposed and callable."""
        self.assertTrue(callable(generate_response))
        self.assertTrue(callable(is_available))
        self.assertTrue(callable(build_multimodal_message))
        self.assertTrue(callable(get_model_path_or_id))
        self.assertTrue(callable(get_rs_qwen_model))
        self.assertTrue(callable(get_rs_qwen_processor))

    def test_default_model_id_resolution(self):
        """Verify default Hugging Face model repository ID."""
        with patch.dict(os.environ, {}, clear=True):
            resolved_id = get_model_path_or_id()
            self.assertEqual(resolved_id, DEFAULT_MODEL_ID)
            self.assertEqual(resolved_id, "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct")

    def test_custom_model_path_resolution(self):
        """Verify RS_QWEN_MODEL_PATH environment variable override."""
        custom_path = "C:/models/rs_qwen_custom"
        with patch.dict(os.environ, {ENV_MODEL_PATH: custom_path}):
            resolved_id = get_model_path_or_id()
            self.assertEqual(resolved_id, custom_path)

    def test_validate_image_valid_path(self):
        """Verify that valid image path passes validation."""
        validated = validate_image(self.valid_img_path)
        self.assertEqual(validated, self.valid_img_path)

    def test_validate_image_pil_object(self):
        """Verify that PIL Image instances pass validation."""
        img = Image.new("RGB", (32, 32))
        validated = validate_image(img)
        self.assertEqual(validated, img)

    def test_validate_image_missing_file_raises_not_found(self):
        """Verify FileNotFoundError when image path does not exist."""
        non_existent = self.temp_dir_path / "does_not_exist.jpg"
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_image(non_existent)
        self.assertIn("does not exist", str(ctx.exception))

    def test_validate_image_corrupt_file_raises_value_error(self):
        """Verify ValueError when file is not a valid image format."""
        with self.assertRaises(ValueError) as ctx:
            validate_image(self.corrupt_file_path)
        self.assertIn("not a valid image", str(ctx.exception))

    def test_validate_image_unsupported_type_raises_type_error(self):
        """Verify TypeError when input type is neither path nor PIL Image."""
        with self.assertRaises(TypeError):
            validate_image(12345)

    def test_empty_question_rejected(self):
        """Verify ValueError when question text is empty or only whitespace."""
        with self.assertRaises(ValueError) as ctx:
            generate_response(self.valid_img_path, "")
        self.assertIn("Question cannot be empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate_response(self.valid_img_path, "   \t\n  ")
        self.assertIn("Question cannot be empty", str(ctx.exception))

    def test_multimodal_message_construction_image_before_text(self):
        """Verify Qwen2.5-VL multimodal message format placing image strictly before text."""
        question = "Identify built-up areas and road structures."
        messages = build_multimodal_message(self.valid_img_path, question)

        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 1)

        user_message = messages[0]
        self.assertEqual(user_message.get("role"), "user")
        content = user_message.get("content")
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        # First item MUST be image
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["image"], str(self.valid_img_path.resolve()))

        # Second item MUST be text
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], question)

    def test_model_unavailability_reported_without_download(self):
        """Verify clear error reporting when weights are not downloaded locally."""
        # By default in this test environment, weights are not downloaded
        self.assertFalse(is_available())

        with self.assertRaises(RuntimeError) as ctx:
            generate_response(
                image_path=self.valid_img_path,
                question="What is shown in this scene?",
            )
        err_msg = str(ctx.exception)
        self.assertIn("RS-Qwen model is not available", err_msg)
        self.assertIn("RS_QWEN_MODEL_PATH", err_msg)

    def test_lazy_loading_prevents_eager_model_instantiation(self):
        """Verify that importing or querying availability does NOT instantiate heavy models."""
        self.assertIsNone(ml.rs_qwen.model._CACHED_MODEL)
        self.assertIsNone(ml.rs_qwen.model._CACHED_PROCESSOR)

        # Query availability
        avail = is_available()
        self.assertFalse(avail)

        # Caches should remain None
        self.assertIsNone(ml.rs_qwen.model._CACHED_MODEL)
        self.assertIsNone(ml.rs_qwen.model._CACHED_PROCESSOR)

    @patch("ml.rs_qwen.inference.is_available", return_value=True)
    @patch("ml.rs_qwen.inference.get_rs_qwen_processor")
    @patch("ml.rs_qwen.inference.get_rs_qwen_model")
    def test_mocked_generation_pipeline(
        self,
        mock_get_model,
        mock_get_processor,
        mock_is_available,
    ):
        """Verify complete inference pipeline with mocked model and processor."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor

        # Dummy model parameter device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Mock processor chat template and tokenization
        mock_processor.apply_chat_template.return_value = "<|image_pad|> Describe this."
        dummy_inputs = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_processor.return_value = dummy_inputs

        # Mock generation IDs (prompt + 2 new tokens)
        mock_model.generate.return_value = torch.tensor([[101, 102, 201, 202]])

        # Mock batch decode
        mock_processor.batch_decode.return_value = ["Dense urban structures and water body."]

        result = generate_response(
            image_path=self.valid_img_path,
            question="Describe this scene.",
            max_new_tokens=128,
        )

        self.assertEqual(result, "Dense urban structures and water body.")
        mock_processor.apply_chat_template.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
