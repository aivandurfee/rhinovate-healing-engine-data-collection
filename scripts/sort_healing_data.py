"""
Rhinovate Healing Engine – Data sorting pipeline.
Filters r/PlasticSurgery (and similar) downloads for rhinoplasty content,
then organizes images into Before / After / Healing with optional timeframe subdirs.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION (align with gallery-dl.conf)
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _ROOT / "data" / "raw_downloads" / "reddit"
CLEAN_DIR = _ROOT / "data" / "clean_dataset"

# Image extensions to process (Reddit often uses webp/gif)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Keywords: post is considered rhinoplasty-related if any match (case-insensitive)
RHINOPLASTY_KEYWORDS = [
    "rhinoplasty", "rhino", "nose job", "nosejob", "septorhinoplasty",
    "nose surgery", "nasal surgery", "nose op", "nose surgery results",
]

# Patterns for phase classification (title or per-image caption)
BEFORE_PATTERNS = [
    re.compile(r"\bbefore\b", re.I),
    re.compile(r"\bpre[- ]?op\b", re.I),
    re.compile(r"\bpre[- ]?surgery\b", re.I),
]
AFTER_PATTERNS = [
    re.compile(r"\bafter\b", re.I),
    re.compile(r"\bpost[- ]?op\b", re.I),
    re.compile(r"\bpost[- ]?surgery\b", re.I),
    re.compile(r"\bfinal\s+result", re.I),
    re.compile(r"\bresults?\b", re.I),
]
HEALING_PATTERNS = [
    re.compile(r"\bhealing\b", re.I),
    re.compile(r"\bprogress\b", re.I),
    re.compile(r"\bswelling\b", re.I),
    re.compile(r"\b(\d+)\s*week", re.I),
    re.compile(r"\b(\d+)\s*month", re.I),
    re.compile(r"\b(\d+)\s*year", re.I),
    re.compile(r"\b(\d+)\s*day", re.I),
    re.compile(r"\b\d+\s*(?:weeks?|months?|years?|days?)\s*(?:post|after)", re.I),
    re.compile(r"\b(?:1|2|3|4|5|6)\s*month", re.I),
]

# Extract timeframe for Healing subdirs (e.g. "1 week" -> "01_1_week")
TIMEFRAME_PATTERN = re.compile(
    r"(\d+)\s*(day|week|month|year)s?\s*(?:post|after|out)?",
    re.I,
)


def _sanitize_path(s: str) -> str:
    """Replace filesystem-unsafe chars. Keep alphanumeric, space, hyphen, underscore."""
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80] if len(s) > 80 else s or "unnamed"


def _extract_timeframe(text: str) -> str | None:
    m = TIMEFRAME_PATTERN.search(text)
    if not m:
        return None
    n, unit = m.group(1), m.group(2).lower()
    return f"{int(n):02d}_{n}_{unit}"


def _classify_phase(text: str) -> tuple[str, str | None]:
    """
    Classify text (title or caption) into phase and optional Healing timeframe.
    Returns (phase, timeframe_or_none) where phase in ("Before", "After", "Healing", "Unlabeled").
    """
    t = (text or "").strip()
    if not t:
        return "Unlabeled", None

    before = any(p.search(t) for p in BEFORE_PATTERNS)
    after = any(p.search(t) for p in AFTER_PATTERNS)
    healing = any(p.search(t) for p in HEALING_PATTERNS)
    timeframe = _extract_timeframe(t)

    if before and not (healing or timeframe):
        return "Before", None
    if healing or timeframe:
        return "Healing", timeframe or "unknown"
    if after:
        return "After", None
    if before:
        return "Before", None

    return "Unlabeled", None


def _is_rhinoplasty_post(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in RHINOPLASTY_KEYWORDS)


def _gallery_items(meta: dict) -> list[dict]:
    g = meta.get("gallery_data") or {}
    return g.get("items") or []


def _caption_for_index(meta: dict, index: int) -> str | None:
    items = _gallery_items(meta)
    if index < len(items):
        c = items[index].get("caption")
        if isinstance(c, str):
            return c
    return None


def _parse_num_from_filename(name: str) -> int | None:
    """Parse leading zero-padded number from e.g. '01_xxx.jpg' or '1_xxx.jpg'."""
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def categorize_images() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting intelligent sort (Before / After / Healing)...")

    for author in sorted(RAW_DIR.iterdir()):
        if not author.is_dir():
            continue

        for post_dir in sorted(author.iterdir()):
            if not post_dir.is_dir():
                continue

            meta_path = post_dir / "metadata.json"
            if not meta_path.is_file():
                meta_path = post_dir / "info.json"
            if not meta_path.is_file():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Skip {post_dir.name}: invalid metadata – {e}")
                continue

            # Support both single-dict and list-of-dicts metadata
            if isinstance(data, list):
                meta = data[0] if data else {}
            else:
                meta = data

            title = (meta.get("title") or "").strip()
            if not _is_rhinoplasty_post(title):
                print(f"  Skip (not rhino): {title[:50]}...")
                continue

            safe_author = _sanitize_path(author.name)
            safe_post = _sanitize_path(post_dir.name)
            patient_id = f"patient_{safe_author}_{safe_post[:40]}"
            patient_root = CLEAN_DIR / patient_id
            patient_root.mkdir(parents=True, exist_ok=True)
            print(f"  Patient: {patient_id}")

            # Collect image paths and sort by num (gallery-dl uses 01_, 02_, ...)
            images: list[tuple[int, Path]] = []
            for i, f in enumerate(sorted(post_dir.iterdir(), key=lambda p: p.name)):
                if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
                    n = _parse_num_from_filename(f.name)
                    order = n if n is not None else 999_000 + i
                    images.append((order, f))

            images.sort(key=lambda x: (x[0], x[1].name))
            if not images:
                print(f"  Skip (no images): {post_dir.name}")
                continue

            for idx, (num, img_path) in enumerate(images):
                caption = _caption_for_index(meta, idx)
                text = (caption or title).strip()
                phase, timeframe = _classify_phase(text)

                if phase == "Unlabeled" and len(images) == 2 and "before" in title.lower() and "after" in title.lower():
                    phase = "Before" if idx == 0 else "After"
                    timeframe = None

                if phase == "Healing" and timeframe:
                    subdir = patient_root / "Healing" / _sanitize_path(timeframe)
                elif phase == "Unlabeled":
                    subdir = patient_root / "Unlabeled"
                else:
                    subdir = patient_root / phase

                subdir.mkdir(parents=True, exist_ok=True)
                dest = subdir / img_path.name
                shutil.copy2(img_path, dest)
                rel = f"{phase}/{timeframe or ''}".strip("/")
                print(f"    -> {rel}  {img_path.name}")

    print("Done.")


if __name__ == "__main__":
    categorize_images()
