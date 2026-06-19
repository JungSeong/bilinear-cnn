from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AICVisionOffsetMultiView
from models import build_model
from reproducibility import make_generator, seed_everything, seed_worker


def make_transform(image_size: int, train: bool):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(steps)


def make_dataset(args, split: str):
    transform = make_transform(args.image_size, train=(split == "train"))
    connectors = tuple(item.strip().upper() for item in args.connectors.split(",") if item.strip())
    return AICVisionOffsetMultiView(
        args.dataset_root,
        split=split,
        connectors=connectors,
        transform=transform,
        require_all_views=True,
        hf_repo_id=args.dataset_hf_repo_id,
        hf_revision=args.dataset_hf_revision,
    )


def to_device(batch, device):
    images, labels, _meta = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    total_loss = 0.0
    total_count = 0
    abs_error_sum = torch.zeros(6, device=device)
    with torch.no_grad():
        for batch in loader:
            images, labels = to_device(batch, device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            count = labels.size(0)
            total_loss += float(loss.item()) * count
            total_count += count
            abs_error_sum += torch.abs(pred - labels).sum(dim=0)
    mae = abs_error_sum / max(1, total_count)
    return {
        "loss": total_loss / max(1, total_count),
        "xyz_mae_mm": (mae[:3] * 1000.0).detach().cpu().tolist(),
        "rpy_mae_deg": torch.rad2deg(mae[3:]).detach().cpu().tolist(),
    }


def write_model_card(args, output_dir: Path, best_loss: float, best_metrics: dict) -> None:
    card_path = output_dir / "README.md"
    card_path.write_text(
        "\n".join(
            [
                "---",
                "library_name: pytorch",
                "tags:",
                "- computer-vision",
                "- robotics",
                "- regression",
                "- aic-sejong",
                "---",
                "",
                "# AIC Vision Offset Regression Model",
                "",
                "This model predicts a 6D base_link correction vector for FinalPolicy ALIGN:",
                "",
                "```text",
                "[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]",
                "```",
                "",
                "## Training Configuration",
                "",
                f"- model: `{args.model}`",
                f"- backbone: `{args.backbone_name}`",
                f"- pretrained backbone: `{args.pretrained}`",
                f"- feature_dim: `{args.feature_dim}`",
                f"- image_size: `{args.image_size}`",
                f"- connectors: `{args.connectors}`",
                f"- dataset repo: `{args.dataset_hf_repo_id}`",
                f"- dataset revision: `{args.dataset_hf_revision}`",
                f"- seed: `{args.seed}`",
                "",
                "## Best Validation Metrics",
                "",
                f"- best val loss: `{best_loss:.8f}`",
                f"- xyz MAE mm: `{best_metrics.get('xyz_mae_mm', [])}`",
                f"- rpy MAE deg: `{best_metrics.get('rpy_mae_deg', [])}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def push_output_to_hub(args, output_dir: Path) -> None:
    if not args.push_to_hub:
        return
    if not args.hub_repo_id:
        raise ValueError("--hub-repo-id is required when --push-to-hub is set")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --push-to-hub. "
            "Install requirements.txt first."
        ) from exc

    api = HfApi(token=args.hub_token)
    api.whoami()
    api.create_repo(
        repo_id=args.hub_repo_id,
        repo_type="model",
        private=args.hub_private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.hub_repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        path_in_repo=args.hub_path_in_repo,
        revision=args.hub_revision,
        commit_message=args.hub_commit_message,
    )
    print(f"Uploaded model artifacts to https://huggingface.co/{args.hub_repo_id}")


def train(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator = make_generator(args.seed)

    train_data = make_dataset(args, "train")
    val_data = make_dataset(args, "val")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1),
    )

    model = build_model(
        args.model,
        feature_dim=args.feature_dim,
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    loss_fn = nn.SmoothL1Loss()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_metrics = {}
    best_checkpoint_path = output_dir / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            images, labels = to_device(batch, device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = labels.size(0)
            running_loss += float(loss.item()) * count
            running_count += count

        train_loss = running_loss / max(1, running_count)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["loss"])
        is_best = metrics["loss"] < best_loss
        if is_best:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "backbone_name": args.backbone_name,
                    "pretrained": args.pretrained,
                    "feature_dim": args.feature_dim,
                    "image_size": args.image_size,
                    "label_order": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
                    "dataset_root": str(Path(args.dataset_root).expanduser()),
                },
                best_checkpoint_path,
            )
            best_metrics = metrics
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={metrics['loss']:.6f} "
            f"xyz_mae_mm={metrics['xyz_mae_mm']} "
            f"rpy_mae_deg={metrics['rpy_mae_deg']} "
            f"{'*' if is_best else ''}"
        )

    summary = {
        "model": args.model,
        "backbone_name": args.backbone_name,
        "pretrained": args.pretrained,
        "feature_dim": args.feature_dim,
        "image_size": args.image_size,
        "label_order": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "dataset_hf_repo_id": args.dataset_hf_repo_id,
        "dataset_hf_revision": args.dataset_hf_revision,
        "connectors": args.connectors,
        "seed": args.seed,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "best_checkpoint": str(best_checkpoint_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_model_card(args, output_dir, best_loss, best_metrics)
    push_output_to_hub(args, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIC vision-offset CNN models.")
    parser.add_argument("--dataset-root", default="/home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset")
    parser.add_argument("--dataset-hf-repo-id", default="aic-sejong-team/aic-vision-offset-dataset")
    parser.add_argument("--dataset-hf-revision", default=None)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--model", choices=["simple_cnn", "shared_bilinear", "multiview_bilinear"], required=True)
    parser.add_argument("--connectors", default="SFP,SC")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-name", default="efficientnetv2_rw_s")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default=None)
    parser.add_argument("--hub-revision", default=None)
    parser.add_argument("--hub-path-in-repo", default=".")
    parser.add_argument("--hub-commit-message", default="Upload AIC vision-offset model")
    parser.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hub-token", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
