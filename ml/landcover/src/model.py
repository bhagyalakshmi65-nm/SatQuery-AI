"""
12-Band Multispectral ResNet-50 for Sentinel-2 Land-Cover Classification.

Architectural Design:
    - Backbone: ResNet-50 initialized with TorchGeo's Sentinel-2 SeCo-ECO
      (Seasonal Contrast for Ecology) pretrained weights.
    - Input: Strictly 12 Sentinel-2 Level-2A surface reflectance bands in SeCo order:
      [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
      Nominal input shape: (B, 12, 120, 120).
    - Head: 19-class multi-label classification head (Dropout + Linear(2048, 19)).
    - Loss Compatibility: forward() returns raw unnormalized logits for nn.BCEWithLogitsLoss.
    - Probabilities: predict_proba() strictly applies torch.sigmoid().
    - Zero ImageNet RGB weight duplication: Uses genuine Sentinel-2 multispectral
      pretraining directly from TorchGeo (ResNet50_Weights.SENTINEL2_ALL_SECO_ECO).
    - No RGB/3-channel fallback: Rejects any input that does not match the 12-channel Sentinel-2 format.
"""

import os
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torchgeo.models import resnet50, ResNet50_Weights


class LandCoverClassifier(nn.Module):
    """
    12-Band Multispectral Multi-Label Land-Cover Classifier for Sentinel-2.

    Uses a Sentinel-2 pretrained ResNet-50 backbone initialized with TorchGeo's
    SeCo-ECO (Seasonal Contrast for Ecology) weights.

    Input Specification:
        - Exactly 12 channels corresponding to Sentinel-2 Level-2A BOA bands:
          [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
        - Input shape: (B, 12, H, W), typically (B, 12, 120, 120).

    Output Specification:
        - Multi-label classification producing exactly 19 unnormalized raw logits.
        - Direct numerical compatibility with torch.nn.BCEWithLogitsLoss.
        - Sigmoid activation applied strictly inside predict_proba().
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 19,
        in_channels: int = 12,
        pretrained: bool = True,
        weights: Optional[Union[str, ResNet50_Weights]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Strict validation: input MUST be exactly 12 channels (no RGB or 3-channel fallback)
        if in_channels != 12:
            raise ValueError(
                f"LandCoverClassifier strictly requires in_channels=12 for the 12 Sentinel-2 bands "
                f"[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]. "
                f"Received in_channels={in_channels}. RGB or 3-channel models are NOT supported."
            )

        if backbone != "resnet50":
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. LandCoverClassifier requires 'resnet50' "
                f"for Sentinel-2 SeCo-ECO pretraining."
            )

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained

        # Resolve TorchGeo SeCo-ECO weights
        selected_weights: Optional[ResNet50_Weights] = None
        local_checkpoint_path: Optional[str] = None

        if isinstance(weights, ResNet50_Weights):
            selected_weights = weights
        elif isinstance(weights, str):
            if weights == "SENTINEL2_ALL_SECO_ECO" or weights.lower() in ("seco-eco", "seco_eco"):
                selected_weights = ResNet50_Weights.SENTINEL2_ALL_SECO_ECO
            elif os.path.exists(weights):
                local_checkpoint_path = weights
            else:
                try:
                    selected_weights = getattr(ResNet50_Weights, weights)
                except AttributeError:
                    raise ValueError(f"Unknown weight name or checkpoint path: {weights}")
        elif pretrained:
            # Default pretrained model is the actual Sentinel-2 SeCo-ECO ResNet-50 from TorchGeo
            selected_weights = ResNet50_Weights.SENTINEL2_ALL_SECO_ECO

        # Instantiate 12-channel ResNet-50 feature extractor using TorchGeo
        # num_classes=0 returns the 2048-dimensional pooled feature representation
        if selected_weights is not None:
            self.backbone = resnet50(weights=selected_weights, num_classes=0)
        else:
            self.backbone = resnet50(weights=None, in_chans=12, num_classes=0)

        # 19-class multi-label classification head
        in_features = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # If a local checkpoint file was supplied, load it
        if local_checkpoint_path is not None:
            self.load_weights(local_checkpoint_path)

    def freeze_backbone(self):
        """Freeze feature extractor parameters for Tier 2 linear probe training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all parameters for Tier 3 end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized raw logits for BCEWithLogitsLoss.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, 19).
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor of shape (B, 12, H, W), but received {x.ndim}D tensor."
            )
        if x.shape[1] != 12:
            raise ValueError(
                f"Input tensor must have exactly 12 channels corresponding to Sentinel-2 bands "
                f"[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]. "
                f"Received tensor with shape {tuple(x.shape)}. RGB/3-channel fallback is disabled."
            )

        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates independent sigmoid probabilities for each of the 19 land-cover classes.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W).

        Returns:
            torch.Tensor: Independent class probabilities in [0.0, 1.0] of shape (B, 19).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def save_weights(self, path: str, extra_info: Optional[Dict[str, Any]] = None):
        """Save checkpoint dictionary to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "state_dict": self.state_dict(),
        }
        if extra_info:
            payload.update(extra_info)
        torch.save(payload, path)

    def load_weights(self, path: str, map_location=None) -> Dict[str, Any]:
        """Load trained SATQuery checkpoint from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        map_location = map_location or torch.device("cpu")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("module.", "")
            clean_state_dict[clean_key] = v

        missing, unexpected = self.load_state_dict(clean_state_dict, strict=False)
        return {"missing_keys": missing, "unexpected_keys": unexpected}


