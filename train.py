from __future__ import annotations

import argparse
import csv
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


MODEL_NAMES = ("simple_cnn", "shared_bilinear", "multiview_bilinear")


def model_names_from_args(args) -> list[str]:
    if args.model == "all":
        return list(MODEL_NAMES)
    return [args.model]


def write_model_card(args, model_name: str, output_dir: Path, best_loss: float, best_metrics: dict) -> None:
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
                f"- model: `{model_name}`",
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


def flatten_summary_for_csv(summary: dict) -> dict:
    metrics = summary.get("best_metrics", {})
    xyz = metrics.get("xyz_mae_mm", [None, None, None])
    rpy = metrics.get("rpy_mae_deg", [None, None, None])
    xyz_values = [float(value) for value in xyz if value is not None]
    rpy_values = [float(value) for value in rpy if value is not None]
    mean_xyz_mae_mm = sum(xyz_values) / len(xyz_values) if xyz_values else None
    mean_rpy_mae_deg = sum(rpy_values) / len(rpy_values) if rpy_values else None
    rpy_score_weight = float(summary.get("rpy_score_weight", 1.0))
    selection_score = None
    if mean_xyz_mae_mm is not None and mean_rpy_mae_deg is not None:
        selection_score = mean_xyz_mae_mm + rpy_score_weight * mean_rpy_mae_deg
    return {
        "model": summary.get("model"),
        "selection_score": selection_score,
        "mean_xyz_mae_mm": mean_xyz_mae_mm,
        "mean_rpy_mae_deg": mean_rpy_mae_deg,
        "best_loss": summary.get("best_loss"),
        "x_mae_mm": xyz[0] if len(xyz) > 0 else None,
        "y_mae_mm": xyz[1] if len(xyz) > 1 else None,
        "z_mae_mm": xyz[2] if len(xyz) > 2 else None,
        "roll_mae_deg": rpy[0] if len(rpy) > 0 else None,
        "pitch_mae_deg": rpy[1] if len(rpy) > 1 else None,
        "yaw_mae_deg": rpy[2] if len(rpy) > 2 else None,
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def write_comparison_files(output_dir: Path, summaries: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise ValueError("No model summaries were produced.")
    rows = [flatten_summary_for_csv(summary) for summary in summaries]
    rows = sorted(
        rows,
        key=lambda item: (
            float(item["selection_score"]) if item["selection_score"] is not None else float("inf"),
            float(item["best_loss"]) if item["best_loss"] is not None else float("inf"),
        ),
    )
    sorted_summaries = []
    for row in rows:
        summary = next(summary for summary in summaries if summary.get("model") == row["model"])
        summary = dict(summary)
        summary["selection_score"] = row["selection_score"]
        summary["mean_xyz_mae_mm"] = row["mean_xyz_mae_mm"]
        summary["mean_rpy_mae_deg"] = row["mean_rpy_mae_deg"]
        sorted_summaries.append(summary)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(sorted_summaries, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Model Comparison (sorted by selection_score) ===")
    for row in rows:
        score_text = f"{float(row['selection_score']):.6f}" if row["selection_score"] is not None else "nan"
        loss_text = f"{float(row['best_loss']):.6f}" if row["best_loss"] is not None else "nan"
        print(
            f"{row['model']}: score={score_text}, "
            f"mean_xyz_mae_mm={row['mean_xyz_mae_mm']}, "
            f"mean_rpy_mae_deg={row['mean_rpy_mae_deg']}, "
            f"loss={loss_text}, "
            f"xyz_mae_mm=({row['x_mae_mm']}, {row['y_mae_mm']}, {row['z_mae_mm']}), "
            f"rpy_mae_deg=({row['roll_mae_deg']}, {row['pitch_mae_deg']}, {row['yaw_mae_deg']})"
        )
    print(f"Saved comparison: {csv_path}")


def train_one_model(args, model_name: str, output_dir: Path) -> dict:
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
        model_name,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_metrics = {}
    best_epoch = 0
    stopped_epoch = None
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / f"{model_name}_best.pt"
    summary_path = output_dir / "training_summary.json"

    if args.skip_existing and best_checkpoint_path.exists() and summary_path.exists():
        print(f"[Skip] Existing model found: {best_checkpoint_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

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
        is_best = metrics["loss"] < (best_loss - args.early_stopping_min_delta)
        if is_best:
            best_loss = metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
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
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={metrics['loss']:.6f} "
            f"xyz_mae_mm={metrics['xyz_mae_mm']} "
            f"rpy_mae_deg={metrics['rpy_mae_deg']} "
            f"no_improve={epochs_without_improvement}/{args.early_stopping_patience} "
            f"{'*' if is_best else ''}"
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_epoch = epoch
            print(
                f"Early stopping {model_name} at epoch {epoch}. "
                f"best_epoch={best_epoch}, best_loss={best_loss:.6f}"
            )
            break

    summary = {
        "model": model_name,
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
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "rpy_score_weight": args.rpy_score_weight,
        "best_checkpoint": str(best_checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_model_card(args, model_name, output_dir, best_loss, best_metrics)
    return summary


def train(args):
    root_output_dir = Path(args.output_dir).expanduser()
    selected_models = model_names_from_args(args)
    summaries = []

    for model_name in selected_models:
        model_output_dir = root_output_dir / model_name if len(selected_models) > 1 else root_output_dir
        print(f"\n=== Training {model_name} ===")
        summaries.append(train_one_model(args, model_name, model_output_dir))

    write_comparison_files(root_output_dir, summaries)
    push_output_to_hub(args, root_output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIC vision-offset CNN models.")
    parser.add_argument("--dataset-root", default="/home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset")
    parser.add_argument("--dataset-hf-repo-id", default="aic-sejong-team/aic-vision-offset-dataset")
    parser.add_argument("--dataset-hf-revision", default=None)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--model", choices=["simple_cnn", "shared_bilinear", "multiview_bilinear", "all"], required=True)
    parser.add_argument("--connectors", default="SFP,SC")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-name", default="efficientnetv2_rw_s")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rpy-score-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default="aic-sejong-team/aic-vision-offset-models")
    parser.add_argument("--hub-revision", default=None)
    parser.add_argument("--hub-path-in-repo", default=".")
    parser.add_argument("--hub-commit-message", default="Upload AIC vision-offset model")
    parser.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hub-token", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
