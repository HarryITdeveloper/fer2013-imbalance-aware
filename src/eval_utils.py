import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    macro_f1 = per_class_f1.mean().item()
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }
    for idx, name in enumerate(class_names):
        metrics[f"f1_{name}"] = per_class_f1[idx].item()
    return metrics


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 1.5 if cm.max() != 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value:.2f}" if normalize else f"{value:.0f}", ha="center", va="center", color=text_color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metrics_path: Path,
    cm_path: Path,
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred, class_names)
    save_metrics(metrics, metrics_path)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=cm_path)
    return metrics
