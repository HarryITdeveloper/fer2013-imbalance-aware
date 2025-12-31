import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

SPLIT_MAP = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def save_image(pixels: str, save_path: Path) -> None:
    """Convert space-separated pixel string to 48x48 grayscale PNG."""
    arr = np.fromstring(pixels, dtype=np.uint8, sep=" ").reshape(48, 48)
    img = Image.fromarray(arr, mode="L")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def convert_csv(csv_path: Path, output_root: Path) -> None:
    df = pd.read_csv(csv_path)
    if not {"emotion", "pixels", "Usage"}.issubset(df.columns):
        raise ValueError("CSV missing required columns: emotion, pixels, Usage")

    for usage, split in SPLIT_MAP.items():
        split_df = df[df["Usage"] == usage]
        print(f"{usage}: {len(split_df)} samples -> {split}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        usage = row["Usage"]
        split = SPLIT_MAP.get(usage)
        if split is None:
            continue
        label_idx = int(row["emotion"])
        class_name = CLASS_NAMES[label_idx]
        save_dir = output_root / split / class_name
        save_path = save_dir / f"{idx}.png"
        save_image(row["pixels"], save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert fer2013.csv to ImageFolder layout."
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("data/raw/fer2013.csv"),
        help="Path to fer2013.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/fer2013"),
        help="Output root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    convert_csv(args.csv_path, args.output_root)
    print(f"Done. Data written to {args.output_root}")


if __name__ == "__main__":
    main()
