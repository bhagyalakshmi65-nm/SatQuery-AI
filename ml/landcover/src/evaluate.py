"""
Multi-Label Evaluation Metrics, Tier 1 Frequency Baseline, and Latency Profiling.
"""

import time
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from .labels import BIGEARTHNET_19_CLASSES


def compute_tier1_frequency_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Tier 1 Baseline: Predicts classes purely based on marginal class frequency in training set.
    Establishes the empirical statistical performance floor.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Marginal class frequencies: P(y_k = 1)
    class_priors = np.mean(y_train, axis=0)  # Shape: (19,)

    # Predict test samples using class priors
    y_pred_probs = np.tile(class_priors, (len(y_test), 1))
    metrics = compute_multilabel_metrics(y_test, y_pred_probs, threshold=threshold)
    metrics["class_priors"] = {
        cls_name: round(float(p), 4)
        for cls_name, p in zip(BIGEARTHNET_19_CLASSES, class_priors)
    }
    return metrics


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Computes standard multi-label metrics on 19 land-cover classes.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)
    y_pred_bin = (y_pred_probs >= threshold).astype(int)

    micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)

    micro_precision = precision_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)

    micro_recall = recall_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)

    try:
        map_score = average_precision_score(y_true, y_pred_probs, average="macro")
    except Exception:
        map_score = 0.0

    per_class_f1 = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    per_class_metrics = {}
    for idx, class_name in enumerate(BIGEARTHNET_19_CLASSES[: y_true.shape[1]]):
        per_class_metrics[class_name] = {
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(np.sum(y_true[:, idx])),
        }

    return {
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "micro_precision": round(float(micro_precision), 4),
        "macro_precision": round(float(macro_precision), 4),
        "micro_recall": round(float(micro_recall), 4),
        "macro_recall": round(float(macro_recall), 4),
        "mAP": round(float(map_score), 4),
        "threshold": threshold,
        "per_class": per_class_metrics,
    }


def measure_inference_latency(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup: int = 5,
    runs: int = 30,
) -> dict:
    """
    Empirically measure inference latency per patch.
    """
    model.eval()
    sample_input = sample_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "device": str(device),
        "num_runs": runs,
    }
