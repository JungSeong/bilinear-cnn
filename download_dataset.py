from __future__ import annotations

import argparse
from pathlib import Path

from dataset import (
    CAMERAS,
    DEFAULT_CONNECTORS,
    DEFAULT_HF_DATASET_REPO_ID,
    download_dataset,
    require_dataset_available,
)


def parse_connectors(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    allowed = ", ".join(("all", *DEFAULT_CONNECTORS))
    if len(items) != 1:
        raise argparse.ArgumentTypeError(f"use exactly one of: {allowed}")

    connector = items[0]
    if connector == "ALL":
        return DEFAULT_CONNECTORS
    if connector not in DEFAULT_CONNECTORS:
        raise argparse.ArgumentTypeError(f"unknown connector: {connector}. Use one of: {allowed}")
    return (connector,)


def format_connectors(connectors: tuple[str, ...]) -> str:
    return "all" if connectors == DEFAULT_CONNECTORS else ",".join(connectors)


def parse_args():
    parser = argparse.ArgumentParser(description="Download the AIC vision-offset dataset.")
    parser.add_argument("--dataset-root", default="data/vision_offset_dataset")
    parser.add_argument("--dataset-hf-repo-id", default=DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--dataset-hf-revision", default=None)
    parser.add_argument(
        "--connectors",
        type=parse_connectors,
        default="all",
        metavar="{all|SFP|SC}",
        help="connector selection; all downloads/verifies SFP and SC separately",
    )
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    download_dataset(
        dataset_root,
        hf_repo_id=args.dataset_hf_repo_id,
        hf_revision=args.dataset_hf_revision,
    )

    for split in args.splits:
        require_dataset_available(
            dataset_root,
            split=split,
            connectors=args.connectors,
            cameras=CAMERAS,
            hf_repo_id=args.dataset_hf_repo_id,
            hf_revision=args.dataset_hf_revision,
        )
        print(
            "[Dataset] Verified "
            f"split={split}, connectors={format_connectors(args.connectors)}, cameras={','.join(CAMERAS)}"
        )


if __name__ == "__main__":
    main()
