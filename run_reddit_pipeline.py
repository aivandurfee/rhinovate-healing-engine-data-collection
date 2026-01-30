#!/usr/bin/env python3
"""
Run the Reddit pipeline:
  1. Download from Reddit (gallery-dl) + sort into Before/After/Healing
  2. Align faces, delete garbage, write manifest
  3. Optionally verify Healing dataloaders

Overwrite vs new data (default = accumulate):
  - raw_downloads:  gallery-dl skips existing files → you get NEW data only.
  - clean_dataset:  sort copies from raw; same paths overwritten, new posts add new folders.
  - aligned_dataset + manifest:  align overwrites them each run from current clean_dataset.
  So each run adds new downloads, merges into clean, then overwrites aligned/manifest.
  Use --fresh to clear clean + aligned + manifest and start over before running.

Usage:
  python run_reddit_pipeline.py                  # rhinoplasty-only search (default), limit 100, then align + manifest
  python run_reddit_pipeline.py --limit 50
  python run_reddit_pipeline.py --full           # no post limit
  python run_reddit_pipeline.py --all-subreddit   # use r/PlasticSurgery/new (all plastic surgery; not just rhinoplasty)
  python run_reddit_pipeline.py --skip-download   # only align + manifest (existing clean_dataset)
  python run_reddit_pipeline.py --fresh           # clear clean + aligned + manifest, then run
  python run_reddit_pipeline.py --check-dataloaders  # also run healing_dataloaders at the end
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean_dataset"
ALIGNED_DIR = ROOT / "data" / "aligned_dataset"
MANIFEST_FILE = ROOT / "data" / "aligned_manifest.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Reddit pipeline (download + sort + align + manifest)")
    ap.add_argument("--limit", type=int, default=100, help="Max posts per query (default 100)")
    ap.add_argument("--full", action="store_true", help="No limit on posts")
    ap.add_argument("--all-subreddit", action="store_true", help="Use r/PlasticSurgery/new (all plastic surgery); default is rhinoplasty-only search")
    ap.add_argument("--skip-download", action="store_true", help="Skip gallery-dl + sort; only align + manifest")
    ap.add_argument("--fresh", action="store_true", help="Clear clean_dataset, aligned_dataset, manifest; then run")
    ap.add_argument("--no-delete", action="store_true", help="Do not delete bad images in align step")
    ap.add_argument("--check-dataloaders", action="store_true", help="Run healing_dataloaders after align")
    args = ap.parse_args()

    if args.fresh:
        for d in (CLEAN_DIR, ALIGNED_DIR):
            if d.is_dir():
                shutil.rmtree(d)
                print(f"Removed {d}")
        if MANIFEST_FILE.is_file():
            MANIFEST_FILE.unlink()
            print(f"Removed {MANIFEST_FILE}")
        print()

    if args.skip_download:
        pass
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "reddit_download.py")]
        if args.full:
            cmd.append("--full")
        else:
            cmd.extend(["--limit", str(args.limit)])
        if args.all_subreddit:
            cmd.append("--no-search")
        print(">>> Reddit download + sort")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("Pipeline stopped (download/sort failed).")
            sys.exit(r.returncode)

    print("\n>>> Align faces + manifest")
    align_cmd = [
        sys.executable, str(ROOT / "scripts" / "align_faces.py"),
        "--manifest", str(MANIFEST_FILE),
    ]
    if args.no_delete:
        align_cmd.append("--no-delete")
    r = subprocess.run(align_cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    if args.check_dataloaders:
        print("\n>>> Healing dataloaders")
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "healing_dataloaders.py"),
            "--manifest", str(MANIFEST_FILE),
            "--aligned", str(ALIGNED_DIR),
        ], cwd=ROOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
