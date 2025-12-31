import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def load_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_f1_list(metrics: dict, class_names):
    if "per_class_f1" in metrics:
        return metrics["per_class_f1"]
    f1s = []
    for name in class_names:
        key = f"f1_{name}"
        f1s.append(metrics.get(key, 0.0))
    return f1s


def plot_confusions(e0_path: Path, e5_path: Path, save_path: Path) -> None:
    img_e0 = Image.open(e0_path)
    img_e5 = Image.open(e5_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_e0)
    axes[0].set_title("E0_match Confusion")
    axes[0].axis("off")
    axes[1].imshow(img_e5)
    axes[1].set_title("E5_balanced Confusion")
    axes[1].axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(e0_metrics, e5_metrics, class_names, save_path: Path) -> None:
    f1_e0 = extract_f1_list(e0_metrics, class_names)
    f1_e5 = extract_f1_list(e5_metrics, class_names)
    x = range(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], f1_e0, width, label="E0_match")
    ax.bar([i + width / 2 for i in x], f1_e5, width, label="E5_balanced")
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1: E0 vs E5")
    ax.set_ylim(0, 1.0)
    # Highlight minority classes
    for idx in [1, 2]:  # disgust, fear
        ax.get_xticklabels()[idx].set_color("red")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    runs_dir = Path("runs")
    e0_dir = runs_dir / "E0_match"
    e5_dir = runs_dir / "E5_softmax_balanced_30ep"

    e0_metrics = load_metrics(e0_dir / "metrics.json")
    e5_metrics = load_metrics(e5_dir / "metrics.json")

    class_names = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    # Confusion comparison
    plot_confusions(
        e0_dir / "confusion_matrix.png",
        e5_dir / "confusion_matrix.png",
        Path("figures/confusion_E0_vs_E5.png"),
    )

    # Per-class F1 bars
    plot_per_class_f1(
        e0_metrics,
        e5_metrics,
        class_names,
        Path("figures/per_class_f1_E0_vs_E5.png"),
    )
    print("Figures written to figures/confusion_E0_vs_E5.png and figures/per_class_f1_E0_vs_E5.png")


if __name__ == "__main__":
    main()
