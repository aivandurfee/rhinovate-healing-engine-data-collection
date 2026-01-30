"""
Rhinovate Healing Engine – Healing-only training helper.
Reads aligned_manifest.csv, filters phase == "Healing", and builds PyTorch
DataLoaders grouped by timeframe (e.g. 01_1_week, 02_6_month) for healing-model training.

CLI:
  python scripts/healing_dataloaders.py [--manifest PATH] [--aligned PATH] [--batch-size N]

Python:
  from scripts.healing_dataloaders import get_healing_dataloaders, get_healing_dataloader_all
  loaders, tf2id, id2tf = get_healing_dataloaders("data/aligned_manifest.csv", "data/aligned_dataset")
  loader_all, tf2id, id2tf = get_healing_dataloader_all(...)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = _ROOT / "data" / "aligned_manifest.csv"
DEFAULT_ALIGNED_ROOT = _ROOT / "data" / "aligned_dataset"


def _require_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError as e:
        raise ImportError(
            "healing_dataloaders requires torch and torchvision. "
            "Install with: pip install torch torchvision"
        ) from e


def load_healing_manifest(
    manifest_path: Path | str,
    aligned_root: Path | str,
    min_samples_per_timeframe: int = 1,
) -> tuple[list[dict], dict[str, int], dict[int, str]]:
    """
    Load manifest CSV, keep only Healing rows, group by timeframe.
    Returns (rows, timeframe_to_id, id_to_timeframe).
    """
    manifest_path = Path(manifest_path)
    aligned_root = Path(aligned_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") != "Healing":
                continue
            tf = (row.get("timeframe") or "").strip()
            row["timeframe"] = tf or "unknown"
            p = (row.get("path") or "").strip().replace("\\", "/")
            if not p:
                continue
            full = aligned_root / p
            if not full.is_file():
                continue
            row["path"] = p
            rows.append(row)

    timeframes = sorted({r["timeframe"] for r in rows})
    counts = Counter(r["timeframe"] for r in rows)
    timeframes = [t for t in timeframes if counts[t] >= min_samples_per_timeframe]
    rows = [r for r in rows if r["timeframe"] in timeframes]

    timeframe_to_id = {t: i for i, t in enumerate(timeframes)}
    id_to_timeframe = {i: t for t, i in timeframe_to_id.items()}
    return rows, timeframe_to_id, id_to_timeframe


class HealingDataset:
    """PyTorch Dataset over Healing-only manifest rows. Yields (image, timeframe_id, path)."""

    def __init__(
        self,
        rows: list[dict],
        aligned_root: Path | str,
        timeframe_to_id: dict[str, int],
        transform=None,
    ):
        self.rows = rows
        self.root = Path(aligned_root)
        self.tf2id = timeframe_to_id
        self.transform = transform
        self._tv = _require_torch()[1]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        torch, torchvision = _require_torch()
        row = self.rows[idx]
        path = self.root / row["path"]
        img = self._tv.io.read_image(str(path))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0
        tid = self.tf2id[row["timeframe"]]
        if self.transform is not None:
            img = self.transform(img)
        return img, tid, row["path"]


def get_healing_dataloaders(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    aligned_root: Path | str = DEFAULT_ALIGNED_ROOT,
    batch_size: int = 32,
    num_workers: int = 0,
    min_samples_per_timeframe: int = 1,
    transform=None,
) -> tuple[dict[str, "torch.utils.data.DataLoader"], dict[str, int], dict[int, str]]:
    """
    Build Healing-only DataLoaders grouped by timeframe.
    Returns (loaders_by_timeframe, timeframe_to_id, id_to_timeframe).
    """
    torch, _ = _require_torch()

    rows, timeframe_to_id, id_to_timeframe = load_healing_manifest(
        manifest_path, aligned_root, min_samples_per_timeframe
    )
    if not rows:
        return {}, timeframe_to_id, id_to_timeframe

    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for tf in timeframe_to_id:
        sub = [r for r in rows if r["timeframe"] == tf]
        ds = HealingDataset(sub, aligned_root, timeframe_to_id, transform=transform)
        loaders[tf] = torch.utils.data.DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=True,
            num_workers=num_workers,
            drop_last=len(ds) >= batch_size,
        )
    return loaders, timeframe_to_id, id_to_timeframe


def get_healing_dataloader_all(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    aligned_root: Path | str = DEFAULT_ALIGNED_ROOT,
    batch_size: int = 32,
    num_workers: int = 0,
    min_samples_per_timeframe: int = 1,
    transform=None,
) -> tuple["torch.utils.data.DataLoader | None", dict[str, int], dict[int, str]]:
    """
    Single DataLoader over all Healing images (with timeframe id per sample).
    Returns (loader, timeframe_to_id, id_to_timeframe). loader is None if no data.
    """
    torch, _ = _require_torch()

    rows, timeframe_to_id, id_to_timeframe = load_healing_manifest(
        manifest_path, aligned_root, min_samples_per_timeframe
    )
    if not rows:
        return None, timeframe_to_id, id_to_timeframe

    ds = HealingDataset(rows, aligned_root, timeframe_to_id, transform=transform)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(ds) >= batch_size,
    )
    return loader, timeframe_to_id, id_to_timeframe


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build Healing-only dataloaders from aligned_manifest.csv")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest CSV path")
    ap.add_argument("--aligned", type=Path, default=DEFAULT_ALIGNED_ROOT, help="Aligned dataset root")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-samples", type=int, default=1, help="Min samples per timeframe to include")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}")
        print("Run: python scripts/align_faces.py --manifest data/aligned_manifest.csv")
        raise SystemExit(1)

    loaders, tf2id, id2tf = get_healing_dataloaders(
        args.manifest, args.aligned,
        batch_size=args.batch_size,
        min_samples_per_timeframe=args.min_samples,
    )
    if not loaders:
        print("No Healing rows in manifest (or all filtered out).")
        return
    print("Timeframes:", tf2id)
    for tf, dl in loaders.items():
        print(f"  {tf}: {len(dl.dataset)} samples, {len(dl)} batches")
    loader_all, _, _ = get_healing_dataloader_all(
        args.manifest, args.aligned,
        batch_size=args.batch_size,
        min_samples_per_timeframe=args.min_samples,
    )
    if loader_all is not None:
        print(f"Combined: {len(loader_all.dataset)} samples, {len(loader_all)} batches")


if __name__ == "__main__":
    main()
