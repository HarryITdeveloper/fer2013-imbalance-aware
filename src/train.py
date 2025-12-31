import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from . import eval_utils
from .losses import compute_class_weights, get_loss_fn
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FER-2013 training script")
    parser.add_argument("--data_root", type=Path, default=Path("data/fer2013"))
    parser.add_argument("--exp", type=str, default="E0")
    parser.add_argument("--head", type=str, choices=["softmax", "arcface"], default="softmax")
    parser.add_argument("--loss", type=str, choices=["ce", "wce", "focal"], default="ce")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.5)
    parser.add_argument("--sampler", type=str, choices=["default", "balanced"], default="default")
    parser.add_argument("--weight_mode", type=str, choices=["inv", "inv_sqrt", "effective"], default="inv_sqrt")
    parser.add_argument("--beta", type=float, default=0.999, help="beta for effective class weights")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint (backbone/head) to resume/eval")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomCrop(44, padding=4),
            transforms.Resize(48),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(48),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return train_tf, test_tf


def prepare_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int = 4,
    sampler_mode: str = "default",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_tf, test_tf = build_transforms()
    train_set = datasets.ImageFolder(root=data_root / "train", transform=train_tf)
    val_set = datasets.ImageFolder(root=data_root / "val", transform=test_tf)
    test_set = datasets.ImageFolder(root=data_root / "test", transform=test_tf)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if sampler_mode == "balanced":
        targets = torch.tensor(train_set.targets)
        class_counts = torch.bincount(targets, minlength=len(train_set.classes)).float().clamp_min(1.0)
        # gentler weighting: 1/sqrt(freq)
        class_weights = 1.0 / torch.sqrt(class_counts)
        class_weights = class_weights / class_weights.mean()
        sample_weights = class_weights[targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_set, shuffle=False, sampler=sampler, **kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)

    kwargs_eval = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_set, **kwargs_eval)
    test_loader = DataLoader(test_set, **kwargs_eval)
    return train_loader, val_loader, test_loader, train_set.classes


def compute_logits(head_type: str, head, feats, labels=None):
    if head_type == "arcface":
        return head(feats, labels)
    return head(feats)


def train_one_epoch(
    loader: DataLoader,
    backbone,
    head,
    head_type: str,
    loss_fn,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    backbone.train()
    head.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            feats = backbone(images)
            logits = compute_logits(head_type, head, feats, labels)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            print("Non-finite loss encountered. Stopping training early.")
            raise RuntimeError("Loss became non-finite.")

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 5.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return {"loss": avg_loss, "acc": acc}


def run_inference(
    loader: DataLoader,
    backbone,
    head,
    head_type: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    head.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            feats = backbone(images)
            logits = compute_logits(head_type, head, feats, labels=None)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return y_true, y_pred


def save_checkpoint(
    path: Path,
    epoch: int,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Disable AMP for ArcFace if requested to avoid instability
    amp_enabled = args.amp
    if args.head == "arcface" and args.amp:
        print("AMP disabled for ArcFace for stability.")
        amp_enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_root = Path("runs")
    run_dir = runs_root / args.exp
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        args.data_root, args.batch_size, sampler_mode=args.sampler
    )

    num_classes = len(class_names)
    backbone, head = build_model(num_classes, args.head, arc_s=args.arc_s, arc_m=args.arc_m)
    backbone.to(device)
    head.to(device)

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone"])
        head.load_state_dict(checkpoint["head"])
        print(f"Loaded checkpoint from {ckpt_path}")

    train_targets = torch.tensor(train_loader.dataset.targets)
    class_weights = None
    if args.loss in {"wce", "focal"}:
        class_weights = compute_class_weights(
            train_targets, num_classes, mode=args.weight_mode, beta=args.beta
        ).to(device)
    loss_fn = get_loss_fn(args.loss, class_weights=class_weights)

    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=run_dir)
    best_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            train_loader, backbone, head, args.head, loss_fn, optimizer, scaler, device, amp_enabled
        )
        scheduler.step()

        y_true_val, y_pred_val = run_inference(val_loader, backbone, head, args.head, device)
        val_metrics = eval_utils.compute_metrics(y_true_val, y_pred_val, class_names)

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/acc", train_stats["acc"], epoch)
        writer.add_scalar("val/acc", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        writer.flush()

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(best_ckpt_path, epoch, backbone, head, optimizer, scheduler, best_f1)
            print(f"Epoch {epoch}: new best macro-F1 {best_f1:.4f}")

    print(f"Training done. Best epoch {best_epoch} with macro-F1 {best_f1:.4f}")

    # Load best checkpoint for evaluation
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone"])
        head.load_state_dict(checkpoint["head"])

    y_true_test, y_pred_test = run_inference(test_loader, backbone, head, args.head, device)
    metrics_path = run_dir / "metrics.json"
    cm_path = run_dir / "confusion_matrix.png"
    metrics = eval_utils.evaluate_predictions(
        y_true_test, y_pred_test, class_names, metrics_path=metrics_path, cm_path=cm_path
    )
    print(json.dumps(metrics, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
