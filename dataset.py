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


def ensure_dataset_available(
    root: str | Path,
    split: str,
    connectors: tuple[str, ...],
    cameras: tuple[str, ...],
    hf_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    hf_revision: str | None = None,
) -> Path:
    """Use local dataset if valid; otherwise download it from Hugging Face."""
    dataset_root = Path(root).expanduser()
    if _has_requested_metadata(dataset_root, split, connectors, cameras):
        return dataset_root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Local vision-offset dataset was not found and huggingface_hub is "
            "not installed. Install requirements.txt or provide --dataset-root "
            "with an existing dataset."
        ) from exc

    dataset_root.mkdir(parents=True, exist_ok=True)
    print(
        "Local vision-offset dataset is missing or incomplete; "
        f"downloading {hf_repo_id} to {dataset_root}"
    )
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        revision=hf_revision,
        local_dir=str(dataset_root),
    )
    return dataset_root


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
        connectors: tuple[str, ...] = ("SFP", "SC"),
        cameras: tuple[str, ...] = CAMERAS,
        transform=None,
        require_all_views: bool = True,
        hf_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
        hf_revision: str | None = None,
    ) -> None:
        self.root = ensure_dataset_available(
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
