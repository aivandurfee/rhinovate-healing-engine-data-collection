"""
Rhinovate – Remove duplicate images by content hash.

Scans folders for duplicate images (same pixel content, different filenames)
and removes duplicates, keeping the first occurrence.

Usage:
  python scripts/dedupe_images.py [--input data/clean_dataset] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def content_hash(path: Path) -> str | None:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


def dedupe_folder(folder: Path, root: Path, dry_run: bool) -> tuple[int, int]:
    """Dedupe within a single folder (e.g. Before/ or After/). Returns (kept, removed)."""
    seen: dict[str, Path] = {}
    removed = 0
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        h = content_hash(p)
        if h is None:
            continue
        if h in seen:
            if not dry_run:
                p.unlink()
            removed += 1
            print(f"  Removed: {p.relative_to(root)} (= {seen[h].name})")
        else:
            seen[h] = p
    return len(seen), removed


def dedupe_dir(root: Path, dry_run: bool = False) -> tuple[int, int]:
    """Scan directory, dedupe within each Before/ and After/ folder. Returns (total_kept, total_removed)."""
    root = Path(root).resolve()
    if not root.is_dir():
        return 0, 0

    total_kept = 0
    total_removed = 0

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for phase_dir in ("Before", "After"):
            folder = patient_dir / phase_dir
            if not folder.is_dir():
                continue
            kept, removed = dedupe_folder(folder, root, dry_run)
            total_kept += kept
            total_removed += removed

    return total_kept, total_removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove duplicate images by content hash.")
    ap.add_argument("--input", "-i", type=Path, default=Path("data/clean_dataset"), help="Root directory to scan")
    ap.add_argument("--raw", type=Path, default=Path("data/raw_downloads/asps"), help="Also dedupe raw ASPS folder")
    ap.add_argument("--dry-run", action="store_true", help="List duplicates without deleting")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_dir = root / args.input if not args.input.is_absolute() else args.input

    print(f"Scanning: {input_dir}")
    if args.dry_run:
        print("(dry-run: no files will be deleted)")
    kept, removed = dedupe_dir(input_dir, dry_run=args.dry_run)
    print(f"Kept: {kept} | Removed: {removed}")

    if args.raw:
        raw_dir = root / args.raw if not args.raw.is_absolute() else args.raw
        if raw_dir.is_dir():
            print(f"\nScanning: {raw_dir}")
            k, r = dedupe_dir(raw_dir, dry_run=args.dry_run)
            print(f"Kept: {k} | Removed: {r}")


if __name__ == "__main__":
    main()
