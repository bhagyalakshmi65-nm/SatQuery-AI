# This file is part of the SatQuery AI Change Detection module.
"""ChangeFormerV6 Model Architecture for Remote Sensing Change Detection.

Based on the official ChangeFormer implementation (Bandara & Patel):
"A Transformer-Based Siamese Network for Change Detection" (IGARSS 2022).

Architecture:
- Hierarchical Siamese Transformer Encoder (EncoderTransformer_v3) with 4 stages.
- Multi-Scale Difference MLP Decoder (DecoderTransformer_v3) with progressive feature fusion.
- Output: 5 multi-scale predictions where output[-1] is full-resolution (B, 2, H, W).
"""

from collections.abc import Iterable
from functools import partial
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def to_2tuple(x: Any) -> Tuple[Any, Any]:
    """Convert input to a 2-tuple if it is not already an iterable."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return tuple(x)
    return (x, x)


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _init_weights(m: nn.Module) -> None:
    """Initialize weights using truncated normal for linear layers and normal for conv."""
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv(nn.Module):
    """Depthwise Convolution for spatial feature mixing in MLP."""

    def __init__(self, dim: int = 768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """Feed-Forward Network with Depthwise Convolution."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Efficient Multi-Head Self-Attention with Spatial Reduction."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Encoder Block with Attention and Mix-FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Any = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class EncoderTransformer_v3(nn.Module):
    """Hierarchical Siamese Transformer Encoder for ChangeFormerV6."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 7,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
        depths: List[int] = [3, 3, 4, 3],
        sr_ratios: List[int] = [8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # Overlapping patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stage-1 (scale 1/4)
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (scale 1/8)
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (scale 1/16)
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (scale 1/32)
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(_init_weights)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        # Stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # Stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for blk in self.block2:
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # Stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for blk in self.block3:
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # Stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for blk in self.block4:
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        return outs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x)


class MLP(nn.Module):
    """Linear channel embedding projector."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def conv_diff(in_channels: int, out_channels: int) -> nn.Sequential:
    """Convolutional difference module."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


def make_prediction(in_channels: int, out_channels: int) -> nn.Sequential:
    """Intermediate prediction module."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        return torch.add(out, residual)


class DecoderTransformer_v3(nn.Module):
    """Transformer Decoder with multi-scale feature difference and progressive fusion."""

    def __init__(
        self,
        input_transform: str = "multiple_select",
        in_index: List[int] = [0, 1, 2, 3],
        align_corners: bool = False,
        in_channels: List[int] = [64, 128, 320, 512],
        embedding_dim: int = 256,
        output_nc: int = 2,
        decoder_softmax: bool = False,
        feature_strides: List[int] = [2, 4, 8, 16],
    ):
        super(DecoderTransformer_v3, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # Convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # Intermediate predictions from encoder scales
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
        )

        # Final prediction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            return torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            return [inputs[i] for i in self.in_index]
        else:
            return inputs[self.in_index]

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        x_1 = self._transform_inputs(inputs1)
        x_2 = self._transform_inputs(inputs2)

        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        n, _, h, w = c4_1.shape
        outputs = []

        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = F.interpolate(_c4, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(
            _c4, scale_factor=2, mode="bilinear", align_corners=False
        )
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = F.interpolate(_c3, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(
            _c3, scale_factor=2, mode="bilinear", align_corners=False
        )
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = F.interpolate(_c2, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(
            _c2, scale_factor=2, mode="bilinear", align_corners=False
        )
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear fusion of difference maps from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # Progressive upsampling to x1 scale
        x = self.convd2x(_c)
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x)

        # Final prediction logits (B, output_nc, H, W)
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            outputs = [self.active(pred) for pred in outputs]

        return outputs


class ChangeFormerV6(nn.Module):
    """ChangeFormerV6: Transformer-Based Siamese Network for Change Detection.

    Accepts two RGB images `model(x1, x2)` and returns 5 multi-scale outputs.
    The final output `outputs[-1]` has shape (B, output_nc, 256, 256).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        decoder_softmax: bool = False,
        embed_dim: int = 256,
    ):
        super(ChangeFormerV6, self).__init__()
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        # Transformer Encoder
        self.Tenc_x2 = EncoderTransformer_v3(
            img_size=256,
            patch_size=7,
            in_chans=input_nc,
            num_classes=output_nc,
            embed_dims=self.embed_dims,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop,
            drop_path_rate=self.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=self.depths,
            sr_ratios=[8, 4, 2, 1],
        )

        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer_v3(
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3],
            align_corners=False,
            in_channels=self.embed_dims,
            embedding_dim=self.embedding_dim,
            output_nc=output_nc,
            decoder_softmax=decoder_softmax,
            feature_strides=[2, 4, 8, 16],
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass for bi-temporal image pair.

        Args:
            x1: First / earlier temporal image tensor of shape (B, 3, 256, 256).
            x2: Second / later temporal image tensor of shape (B, 3, 256, 256).

        Returns:
            List of 5 multi-scale predictions [p_c4, p_c3, p_c2, p_c1, cp]
            where cp is output[-1] with shape (B, output_nc, 256, 256).
        """
        fx1 = self.Tenc_x2(x1)
        fx2 = self.Tenc_x2(x2)
        cp = self.TDec_x2(fx1, fx2)
        return cp


def load_changeformer_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> nn.Module:
    """Load pretrained weights into ChangeFormerV6 model with prefix and format handling.

    Args:
        model: Initialized ChangeFormerV6 model instance.
        checkpoint_path: Path to checkpoint file (.pth or .pt).
        device: Target device for loading.
        strict: Whether to enforce strict key matching.

    Returns:
        The model with loaded weights.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Extract state dict from dictionary wrappers
    if isinstance(checkpoint, dict):
        if "model_G_state_dict" in checkpoint:
            state_dict = checkpoint["model_G_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "net_G" in checkpoint:
            state_dict = checkpoint["net_G"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip common prefixes from distributed or wrapped training
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("net_G."):
            key = key[6:]
        new_state_dict[key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected[:5])

    logger.info("Successfully loaded ChangeFormerV6 weights from %s", checkpoint_path)
    return model
