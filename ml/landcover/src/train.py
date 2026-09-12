"""
Training routines implementing Tier 2 (Linear Probe) and Tier 3 (End-to-End Fine-Tuning).
Uses multi-label BCEWithLogitsLoss on 12-band Sentinel-2 inputs.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import LandCoverClassifier
from .evaluate import compute_multilabel_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0

    for tensors, targets, _ in dataloader:
        tensors = tensors.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tensors.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict]:
    """Evaluate model and compute multi-label metrics."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for tensors, targets, _ in dataloader:
        tensors = tensors.to(device)
        targets = targets.to(device)

        logits = model(tensors)
        loss = criterion(logits, targets)
        running_loss += loss.item() * tensors.size(0)

        probs = torch.sigmoid(logits)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    import numpy as np
    y_true = np.vstack(all_targets)
    y_probs = np.vstack(all_probs)

    metrics = compute_multilabel_metrics(y_true, y_probs, threshold=threshold)
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, metrics


def train_tier2_linear_probe(
    model: LandCoverClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    device: torch.device = None,
) -> dict:
    """
    Tier 2 Baseline: Trains ONLY the linear head while keeping the 12-band backbone frozen.
    Evaluates the linear separability of pretrained features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.freeze_backbone()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
    print(f"[Tier 2: Linear Probe] Training classification head ({num_epochs} epochs) on {device}...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        dur = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"  Epoch {epoch:02d}/{num_epochs:02d} [{dur:.1f}s] - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
            f"Micro F1: {val_metrics['micro_f1']:.4f} - Macro F1: {val_metrics['macro_f1']:.4f}"
        )

    return history


def train_tier3_finetune(
    model: LandCoverClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    backbone_lr: float = 1e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 0.0001,
    save_path: str = "ml/landcover/weights/best_model.pth",
    device: torch.device = None,
) -> dict:
    """
    Tier 3: End-to-End Fine-Tuning with differential learning rates across backbone and head.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.unfreeze_backbone()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
    best_macro_f1 = 0.0
    print(f"[Tier 3: Fine-Tuning] Training all layers ({num_epochs} epochs) on {device}...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        dur = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"  Epoch {epoch:02d}/{num_epochs:02d} [{dur:.1f}s] - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
            f"Micro F1: {val_metrics['micro_f1']:.4f} - Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            if save_path:
                model.save_weights(save_path, extra_info={"epoch": epoch, "metrics": val_metrics})
                print(f"    --> Saved best checkpoint to {save_path} (Macro F1: {best_macro_f1:.4f})")

    return history
