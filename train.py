from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AICVisionOffsetMultiView, DEFAULT_CONNECTORS
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


def connector_names_from_args(args) -> list[str]:
    connectors = tuple(item.strip().upper() for item in args.connectors.split(",") if item.strip())
    allowed = ", ".join(("all", *DEFAULT_CONNECTORS))
    if len(connectors) != 1:
        raise ValueError(f"--connectors must be exactly one of: {allowed}")

    connector = connectors[0]
    if connector == "ALL":
        return list(DEFAULT_CONNECTORS)

    if connector not in DEFAULT_CONNECTORS:
        raise ValueError(f"Unknown connector: {connector}. Use one of: {allowed}")
    return [connector]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value: true/false")


def make_dataset(args, split: str, connector: str):
    transform = make_transform(args.image_size, train=(split == "train"))
    print(
        "[Dataset] Loading "
        f"split={split}, root={Path(args.dataset_root).expanduser()}, "
        f"connector={connector}"
    )
    return AICVisionOffsetMultiView(
        args.dataset_root,
        split=split,
        connectors=(connector,),
        transform=transform,
        require_all_views=True,
        hf_repo_id=args.dataset_hf_repo_id,
        hf_revision=args.dataset_hf_revision,
    )


def to_device(batch, device):
    images, labels, _meta = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


class SeparateXYZRPYSmoothL1Loss(nn.Module):
    def __init__(
        self,
        *,
        xyz_scale_mm: float,
        rpy_scale_deg: float,
        xyz_weight: float,
        rpy_weight: float,
    ) -> None:
        super().__init__()
        if xyz_scale_mm <= 0.0:
            raise ValueError("--xyz-loss-scale-mm must be greater than 0")
        if rpy_scale_deg <= 0.0:
            raise ValueError("--rpy-loss-scale-deg must be greater than 0")
        if xyz_weight <= 0.0:
            raise ValueError("--xyz-loss-weight must be greater than 0")
        if rpy_weight <= 0.0:
            raise ValueError("--rpy-loss-weight must be greater than 0")

        xyz_scale_m = xyz_scale_mm / 1000.0
        rpy_scale_rad = math.radians(rpy_scale_deg)
        self.register_buffer(
            "xyz_scale",
            torch.tensor(xyz_scale_m, dtype=torch.float32),
        )
        self.register_buffer(
            "rpy_scale",
            torch.tensor(rpy_scale_rad, dtype=torch.float32),
        )
        self.xyz_weight = xyz_weight
        self.rpy_weight = rpy_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xyz_error = (pred[:, :3] - target[:, :3]) / self.xyz_scale
        rpy_error = (pred[:, 3:] - target[:, 3:]) / self.rpy_scale
        xyz_loss = nn.functional.smooth_l1_loss(
            xyz_error,
            torch.zeros_like(xyz_error),
        )
        rpy_loss = nn.functional.smooth_l1_loss(
            rpy_error,
            torch.zeros_like(rpy_error),
        )
        total_loss = self.xyz_weight * xyz_loss + self.rpy_weight * rpy_loss
        return total_loss, {
            "xyz_loss": xyz_loss.detach(),
            "rpy_loss": rpy_loss.detach(),
        }


def make_loss_fn(args, device):
    return SeparateXYZRPYSmoothL1Loss(
        xyz_scale_mm=args.xyz_loss_scale_mm,
        rpy_scale_deg=args.rpy_loss_scale_deg,
        xyz_weight=args.xyz_loss_weight,
        rpy_weight=args.rpy_loss_weight,
    ).to(device)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_xyz_loss = 0.0
    total_rpy_loss = 0.0
    total_count = 0
    abs_error_sum = torch.zeros(6, device=device)
    with torch.no_grad():
        for batch in loader:
            images, labels = to_device(batch, device)
            pred = model(images)
            loss, loss_parts = loss_fn(pred, labels)
            count = labels.size(0)
            total_loss += float(loss.item()) * count
            total_xyz_loss += float(loss_parts["xyz_loss"].item()) * count
            total_rpy_loss += float(loss_parts["rpy_loss"].item()) * count
            total_count += count
            abs_error_sum += torch.abs(pred - labels).sum(dim=0)
    mae = abs_error_sum / max(1, total_count)
    return {
        "loss": total_loss / max(1, total_count),
        "xyz_loss": total_xyz_loss / max(1, total_count),
        "rpy_loss": total_rpy_loss / max(1, total_count),
        "xyz_mae_mm": (mae[:3] * 1000.0).detach().cpu().tolist(),
        "rpy_mae_deg": torch.rad2deg(mae[3:]).detach().cpu().tolist(),
    }


def format_axis_metrics(values: list[float], names: tuple[str, ...]) -> str:
    values = [float(value) for value in values]
    if not values:
        return "n/a"

    parts = [f"{name}={value:.2f}" for name, value in zip(names, values)]
    parts.append(f"mean={sum(values) / len(values):.2f}")
    return " ".join(parts)


def mean_metric(values: list[float]) -> float | None:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def format_epoch_log(
    *,
    epoch: int,
    max_epochs: int,
    train_loss: float,
    train_xyz_loss: float,
    train_rpy_loss: float,
    train_metrics: dict,
    metrics: dict,
    epochs_without_improvement: int,
    early_stopping_patience: int,
    is_best: bool,
) -> str:
    marker = " best" if is_best else ""
    early_stop_text = (
        "off"
        if early_stopping_patience <= 0
        else f"{epochs_without_improvement}/{early_stopping_patience}"
    )
    return "\n".join(
        [
            f"[Epoch {epoch:03d}/{max_epochs:03d}]{marker}",
            "  loss    "
            f"train total={train_loss:.4f} xyz={train_xyz_loss:.4f} rpy={train_rpy_loss:.4f} | "
            f"val total={metrics['loss']:.4f} xyz={metrics['xyz_loss']:.4f} rpy={metrics['rpy_loss']:.4f}",
            "  train mae "
            f"xyz_mm {format_axis_metrics(train_metrics['xyz_mae_mm'], ('x', 'y', 'z'))} | "
            f"rpy_deg {format_axis_metrics(train_metrics['rpy_mae_deg'], ('roll', 'pitch', 'yaw'))}",
            "  val mae   "
            f"xyz_mm {format_axis_metrics(metrics['xyz_mae_mm'], ('x', 'y', 'z'))} | "
            f"rpy_deg {format_axis_metrics(metrics['rpy_mae_deg'], ('roll', 'pitch', 'yaw'))} | "
            f"early_stop {early_stop_text}",
        ]
    )


MODEL_NAMES = (
    "simple_cnn",
    "shared_bilinear",
    "multiview_bilinear",
    "cross_attention_bilinear",
)


def model_names_from_args(args) -> list[str]:
    if args.model == "all":
        return list(MODEL_NAMES)
    return [args.model]


def write_model_card(
    args,
    model_name: str,
    connector: str,
    output_dir: Path,
    best_loss: float,
    best_metrics: dict,
) -> None:
    card_path = output_dir / "README.md"
    card_path.write_text(
        "\n".join(
            [
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
                f"- connector: `{connector}`",
                f"- backbone: `{args.backbone_name}`",
                f"- pretrained backbone: `{args.pretrained}`",
                f"- feature_dim: `{args.feature_dim}`",
                f"- image_size: `{args.image_size}`",
                f"- share backbone weights (multiview only): `{args.share_backbone_weights}`",
                f"- attention heads: `{args.attention_heads}`",
                f"- attention layers: `{args.attention_layers}`",
                f"- attention dropout: `{args.attention_dropout}`",
                f"- attention pos grid: `{args.attention_pos_grid}`",
                f"- requested connectors: `{args.connectors}`",
                f"- dataset repo: `{args.dataset_hf_repo_id}`",
                f"- dataset revision: `{args.dataset_hf_revision}`",
                f"- xyz loss scale mm: `{args.xyz_loss_scale_mm}`",
                f"- rpy loss scale deg: `{args.rpy_loss_scale_deg}`",
                f"- xyz loss weight: `{args.xyz_loss_weight}`",
                f"- rpy loss weight: `{args.rpy_loss_weight}`",
                f"- seed: `{args.seed}`",
                "",
                "## Best Validation Metrics",
                "",
                f"- best val loss: `{best_loss:.8f}`",
                f"- xyz MAE mm: `{best_metrics.get('xyz_mae_mm', [])}`",
                f"- rpy MAE deg: `{best_metrics.get('rpy_mae_deg', [])}`",
                "",
                "## Artifacts",
                "",
                f"- best checkpoint: `{model_name}_best.pt`",
                "- training summary: `training_summary.json`",
                "- loss history: `loss_history.csv`",
                "- loss curve: `loss_curve.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_loss_artifacts(output_dir: Path, model_name: str, history: list[dict]) -> tuple[Path, Path]:
    history_path = output_dir / "loss_history.csv"
    curve_path = output_dir / "loss_curve.png"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_xyz_loss",
        "train_rpy_loss",
        "val_loss",
        "val_xyz_loss",
        "val_rpy_loss",
        "train_mean_xyz_mae_mm",
        "train_mean_rpy_mae_deg",
        "val_mean_xyz_mae_mm",
        "val_mean_rpy_mae_deg",
    ]

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    if not history:
        return history_path, curve_path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plots = [
        ("total loss", "train_loss", "val_loss"),
        ("xyz loss", "train_xyz_loss", "val_xyz_loss"),
        ("rpy loss", "train_rpy_loss", "val_rpy_loss"),
    ]

    fig, axes = plt.subplots(len(plots), 1, figsize=(8, 9), sharex=True)
    fig.suptitle(f"{model_name} train/val loss")
    for ax, (title, train_key, val_key) in zip(axes, plots):
        ax.plot(epochs, [row[train_key] for row in history], marker="o", linewidth=1.8, label="train")
        ax.plot(epochs, [row[val_key] for row in history], marker="o", linewidth=1.8, label="val")
        ax.set_title(title)
        ax.set_ylabel("SmoothL1")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    return history_path, curve_path


def write_training_curves_overview(output_dir: Path, summaries: list[dict]) -> Path | None:
    curve_path = output_dir / "model_training_curves.png"
    plot_summaries = [summary for summary in summaries if summary.get("history")]
    if not plot_summaries:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = [
        ("total loss", "train_loss", "val_loss", "SmoothL1"),
        ("mean xyz MAE", "train_mean_xyz_mae_mm", "val_mean_xyz_mae_mm", "mm"),
        ("mean rpy MAE", "train_mean_rpy_mae_deg", "val_mean_rpy_mae_deg", "deg"),
    ]
    fig, axes = plt.subplots(len(plots), 2, figsize=(13, 11), sharex=True)
    fig.suptitle("Model train/val curves")
    legend_handles = []
    legend_labels = []
    for row_idx, (title, train_key, val_key, ylabel) in enumerate(plots):
        train_ax = axes[row_idx][0]
        val_ax = axes[row_idx][1]
        for summary in plot_summaries:
            history = summary["history"]
            epochs = [row["epoch"] for row in history]
            train_values = [row.get(train_key) for row in history]
            val_values = [row.get(val_key) for row in history]
            if any(value is None for value in (*train_values, *val_values)):
                continue
            label = f"{summary.get('connector')}/{summary.get('model')}"
            (train_line,) = train_ax.plot(
                epochs,
                train_values,
                linewidth=1.5,
                alpha=0.8,
                label=label,
            )
            val_ax.plot(
                epochs,
                val_values,
                linewidth=1.8,
                alpha=0.8,
                label=label,
            )
            if label not in legend_labels:
                legend_handles.append(train_line)
                legend_labels.append(label)
        train_ax.set_title(f"{title} - train")
        val_ax.set_title(f"{title} - val")
        train_ax.set_ylabel(ylabel)
        val_ax.set_ylabel(ylabel)
        train_ax.grid(True, alpha=0.3)
        val_ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel("epoch")
    axes[-1][1].set_xlabel("epoch")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            fontsize=7,
            ncol=min(4, max(1, len(legend_labels))),
        )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    return curve_path


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
    if args.hub_revision:
        print(f"Ensuring Hub branch exists: {args.hub_repo_id}@{args.hub_revision}")
        api.create_branch(
            repo_id=args.hub_repo_id,
            repo_type="model",
            branch=args.hub_revision,
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
        "connector": summary.get("connector"),
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
        "loss_history": summary.get("loss_history"),
        "loss_curve": summary.get("loss_curve"),
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
        summary = next(
            summary
            for summary in summaries
            if summary.get("connector") == row["connector"] and summary.get("model") == row["model"]
        )
        summary = dict(summary)
        summary["selection_score"] = row["selection_score"]
        summary["mean_xyz_mae_mm"] = row["mean_xyz_mae_mm"]
        summary["mean_rpy_mae_deg"] = row["mean_rpy_mae_deg"]
        sorted_summaries.append(summary)
    overview_curve_path = write_training_curves_overview(output_dir, sorted_summaries)
    if overview_curve_path is not None:
        overview_curve_text = str(overview_curve_path)
        for row in rows:
            row["training_curves_overview"] = overview_curve_text
        for summary in sorted_summaries:
            summary["training_curves_overview"] = overview_curve_text
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
            f"{row['connector']}/{row['model']}: score={score_text}, "
            f"mean_xyz_mae_mm={row['mean_xyz_mae_mm']}, "
            f"mean_rpy_mae_deg={row['mean_rpy_mae_deg']}, "
            f"loss={loss_text}, "
            f"xyz_mae_mm=({row['x_mae_mm']}, {row['y_mae_mm']}, {row['z_mae_mm']}), "
            f"rpy_mae_deg=({row['roll_mae_deg']}, {row['pitch_mae_deg']}, {row['yaw_mae_deg']})"
        )
    print(f"Saved comparison: {csv_path}")
    if overview_curve_path is not None:
        print(f"Saved training curves overview: {overview_curve_path}")


def train_one_model(args, model_name: str, connector: str, output_dir: Path) -> dict:
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator = make_generator(args.seed)

    train_data = make_dataset(args, "train", connector)
    val_data = make_dataset(args, "val", connector)
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
        share_backbone_weights=args.share_backbone_weights,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        attention_dropout=args.attention_dropout,
        attention_pos_grid=args.attention_pos_grid,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    loss_fn = make_loss_fn(args, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_metrics = {}
    best_epoch = 0
    stopped_epoch = None
    epochs_without_improvement = 0
    history = []
    best_checkpoint_path = output_dir / f"{model_name}_best.pt"
    summary_path = output_dir / "training_summary.json"

    if args.skip_existing and best_checkpoint_path.exists() and summary_path.exists():
        print(f"[Skip] Existing model found: {best_checkpoint_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_xyz_loss = 0.0
        running_rpy_loss = 0.0
        running_count = 0
        running_abs_error_sum = torch.zeros(6, device=device)
        for batch in train_loader:
            images, labels = to_device(batch, device)
            pred = model(images)
            loss, loss_parts = loss_fn(pred, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = labels.size(0)
            running_loss += float(loss.item()) * count
            running_xyz_loss += float(loss_parts["xyz_loss"].item()) * count
            running_rpy_loss += float(loss_parts["rpy_loss"].item()) * count
            running_abs_error_sum += torch.abs(pred.detach() - labels).sum(dim=0)
            running_count += count

        train_loss = running_loss / max(1, running_count)
        train_xyz_loss = running_xyz_loss / max(1, running_count)
        train_rpy_loss = running_rpy_loss / max(1, running_count)
        train_mae = running_abs_error_sum / max(1, running_count)
        train_metrics = {
            "xyz_mae_mm": (train_mae[:3] * 1000.0).detach().cpu().tolist(),
            "rpy_mae_deg": torch.rad2deg(train_mae[3:]).detach().cpu().tolist(),
        }
        metrics = evaluate(model, val_loader, device, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_xyz_loss": train_xyz_loss,
                "train_rpy_loss": train_rpy_loss,
                "val_loss": metrics["loss"],
                "val_xyz_loss": metrics["xyz_loss"],
                "val_rpy_loss": metrics["rpy_loss"],
                "train_mean_xyz_mae_mm": mean_metric(train_metrics["xyz_mae_mm"]),
                "train_mean_rpy_mae_deg": mean_metric(train_metrics["rpy_mae_deg"]),
                "val_mean_xyz_mae_mm": mean_metric(metrics["xyz_mae_mm"]),
                "val_mean_rpy_mae_deg": mean_metric(metrics["rpy_mae_deg"]),
            }
        )
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
                    "connector": connector,
                    "backbone_name": args.backbone_name,
                    "pretrained": args.pretrained,
                    "feature_dim": args.feature_dim,
                    "image_size": args.image_size,
                    "share_backbone_weights": args.share_backbone_weights,
                    "attention_heads": args.attention_heads,
                    "attention_layers": args.attention_layers,
                    "attention_dropout": args.attention_dropout,
                    "attention_pos_grid": args.attention_pos_grid,
                    "xyz_loss_scale_mm": args.xyz_loss_scale_mm,
                    "rpy_loss_scale_deg": args.rpy_loss_scale_deg,
                    "xyz_loss_weight": args.xyz_loss_weight,
                    "rpy_loss_weight": args.rpy_loss_weight,
                    "label_order": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
                    "dataset_root": str(Path(args.dataset_root).expanduser()),
                },
                best_checkpoint_path,
            )
            best_metrics = metrics
        else:
            epochs_without_improvement += 1
        print(
            format_epoch_log(
                epoch=epoch,
                max_epochs=args.epochs,
                train_loss=train_loss,
                train_xyz_loss=train_xyz_loss,
                train_rpy_loss=train_rpy_loss,
                train_metrics=train_metrics,
                metrics=metrics,
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_patience=args.early_stopping_patience,
                is_best=is_best,
            )
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

    loss_history_path, loss_curve_path = write_loss_artifacts(output_dir, model_name, history)
    summary = {
        "model": model_name,
        "connector": connector,
        "backbone_name": args.backbone_name,
        "pretrained": args.pretrained,
        "feature_dim": args.feature_dim,
        "image_size": args.image_size,
        "share_backbone_weights": args.share_backbone_weights,
        "attention_heads": args.attention_heads,
        "attention_layers": args.attention_layers,
        "attention_dropout": args.attention_dropout,
        "attention_pos_grid": args.attention_pos_grid,
        "xyz_loss_scale_mm": args.xyz_loss_scale_mm,
        "rpy_loss_scale_deg": args.rpy_loss_scale_deg,
        "xyz_loss_weight": args.xyz_loss_weight,
        "rpy_loss_weight": args.rpy_loss_weight,
        "label_order": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "dataset_hf_repo_id": args.dataset_hf_repo_id,
        "dataset_hf_revision": args.dataset_hf_revision,
        "requested_connectors": args.connectors,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "rpy_score_weight": args.rpy_score_weight,
        "best_checkpoint": str(best_checkpoint_path),
        "loss_history": str(loss_history_path),
        "loss_curve": str(loss_curve_path),
        "history": history,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_model_card(args, model_name, connector, output_dir, best_loss, best_metrics)
    return summary


def train(args):
    root_output_dir = Path(args.output_dir).expanduser()
    selected_models = model_names_from_args(args)
    selected_connectors = connector_names_from_args(args)
    summaries = []

    for connector in selected_connectors:
        for model_name in selected_models:
            model_output_dir = root_output_dir / connector / model_name
            print(f"\n=== Training {connector}/{model_name} ===")
            summaries.append(train_one_model(args, model_name, connector, model_output_dir))

    write_comparison_files(root_output_dir, summaries)
    push_output_to_hub(args, root_output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIC vision-offset CNN models.")
    parser.add_argument("--dataset-root", default="/home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset")
    parser.add_argument("--dataset-hf-repo-id", default="aic-sejong-team/aic-vision-offset-dataset")
    parser.add_argument("--dataset-hf-revision", default=None)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument(
        "--model",
        choices=[
            "simple_cnn",
            "shared_bilinear",
            "multiview_bilinear",
            "cross_attention_bilinear",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--connectors",
        default="all",
        metavar="{all|SFP|SC}",
        help="connector selection; all trains SFP and SC as separate jobs",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-name", default="efficientnetv2_rw_s")
    parser.add_argument(
        "--pretrained",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="{true|false}",
    )
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument(
        "--share-backbone-weights",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="{true|false}",
    )
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--attention-pos-grid", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--xyz-loss-scale-mm", type=float, default=10.0)
    parser.add_argument("--rpy-loss-scale-deg", type=float, default=1.0)
    parser.add_argument("--xyz-loss-weight", type=float, default=2.0)
    parser.add_argument("--rpy-loss-weight", type=float, default=1.0)
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
