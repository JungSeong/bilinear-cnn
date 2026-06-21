from __future__ import annotations

"""Dataset loader for AIC vision-offset regression samples."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CAMERAS = ("left", "center", "right")
DEFAULT_CONNECTORS = ("SFP", "SC")
LABEL_KEYS = ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")
DEFAULT_HF_DATASET_REPO_ID = "aic-sejong-team/aic-vision-offset-dataset"


def _has_requested_metadata(
    root: Path,
    split: str,
    connectors: tuple[str, ...],
    cameras: tuple[str, ...],
) -> bool:
    if not (root / "images").is_dir() or not (root / "metadata").is_dir():
        return False
    for connector in connectors:
        for camera in cameras:
            metadata_dir = root / "metadata" / split / connector / camera
            if not metadata_dir.is_dir() or not any(metadata_dir.glob("*.json")):
                return False
    return True


def _download_command(
    root: Path,
    connectors: tuple[str, ...],
    hf_repo_id: str,
    hf_revision: str | None,
) -> str:
    connector_arg = "all" if connectors == DEFAULT_CONNECTORS else ",".join(connectors)
    command = [
        "python3 download_dataset.py",
        f"--dataset-root {root}",
        f"--dataset-hf-repo-id {hf_repo_id}",
        f"--connectors {connector_arg}",
    ]
    if hf_revision:
        command.append(f"--dataset-hf-revision {hf_revision}")
    return " ".join(command)


def download_dataset(
    root: str | Path,
    hf_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    hf_revision: str | None = None,
) -> Path:
    """Download the dataset snapshot from Hugging Face into root."""
    dataset_root = Path(root).expanduser()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the dataset. "
            "Install requirements.txt first."
        ) from exc

    dataset_root.mkdir(parents=True, exist_ok=True)
    print(
        "[Dataset] Downloading "
        f"{hf_repo_id}"
        f"{'@' + hf_revision if hf_revision else ''} "
        f"to {dataset_root}"
    )
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        revision=hf_revision,
        local_dir=str(dataset_root),
    )
    print(f"[Dataset] Download complete: {dataset_root}")
    return dataset_root


def require_dataset_available(
    root: str | Path,
    split: str,
    connectors: tuple[str, ...],
    cameras: tuple[str, ...],
    hf_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    hf_revision: str | None = None,
) -> Path:
    """Return the local dataset root, or fail with a download-first message."""
    dataset_root = Path(root).expanduser()
    if _has_requested_metadata(dataset_root, split, connectors, cameras):
        return dataset_root

    expected = dataset_root / "metadata" / split
    command = _download_command(dataset_root, connectors, hf_repo_id, hf_revision)
    print(
        "[Dataset] Local vision-offset dataset is missing or incomplete.\n"
        f"[Dataset] Missing split/connectors under: {expected}\n"
        "[Dataset] Training does not download datasets automatically. "
        "Download it first with:\n"
        f"[Dataset]   {command}"
    )
    raise FileNotFoundError(
        "Vision-offset dataset is missing or incomplete. "
        f"Run `{command}` before training."
    )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _label_tensor(record: dict) -> torch.Tensor:
    label = record.get("label", {})
    return torch.tensor([float(label[key]) for key in LABEL_KEYS], dtype=torch.float32)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class AICVisionOffsetMultiView(Dataset):
    """One item per sample_id containing left/center/right images."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
        connectors: tuple[str, ...] = DEFAULT_CONNECTORS,
        cameras: tuple[str, ...] = CAMERAS,
        transform=None,
        require_all_views: bool = True,
        hf_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
        hf_revision: str | None = None,
    ) -> None:
        self.root = require_dataset_available(
            root,
            split=split,
            connectors=connectors,
            cameras=cameras,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
        )
        self.split = split
        self.cameras = cameras
        self.transform = transform
        groups: dict[tuple[str, str], dict[str, Path]] = {}
        for connector in connectors:
            for camera in cameras:
                metadata_dir = self.root / "metadata" / split / connector / camera
                for metadata_path in sorted(metadata_dir.glob("*.json")):
                    record = _read_json(metadata_path)
                    key = (str(record.get("connector", connector)), str(record["sample_id"]))
                    groups.setdefault(key, {})[camera] = metadata_path

        self.groups: list[tuple[tuple[str, str], dict[str, Path]]] = []
        for key, camera_paths in sorted(groups.items()):
            has_required = all(camera in camera_paths for camera in cameras)
            if require_all_views and not has_required:
                continue
            if camera_paths:
                self.groups.append((key, camera_paths))
        if not self.groups:
            raise FileNotFoundError(
                f"No complete multiview samples found under {self.root / 'metadata' / split}"
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        (connector, sample_id), camera_paths = self.groups[index]
        images = []
        label = None
        image_paths = {}
        for camera in self.cameras:
            record = _read_json(camera_paths[camera])
            if label is None:
                label = _label_tensor(record)
            image_paths[camera] = record["image"]
            image = Image.open(self.root / record["image"]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = _pil_to_tensor(image)
            images.append(image)
        meta = {
            "sample_id": sample_id,
            "connector": connector,
            "cameras": list(self.cameras),
            "images": image_paths,
        }
        return torch.stack(images, dim=0), label, meta
