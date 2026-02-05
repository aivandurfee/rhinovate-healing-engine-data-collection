"""
Rhinovate Healing Engine – Production Preprocessing Pipeline.

A robust face alignment pipeline for rhinoplasty ML training:
- Detects face + 3 landmarks (Left Eye, Right Eye, Nose Tip) via MediaPipe
- Applies affine transformation to make eyes perfectly horizontal
- Crops 512×512 centered on nose tip
- Filters: blur, no face, extreme close-ups, invalid crops
- Bad images → review folder (categorized) instead of deletion
- Output: PyTorch ImageFolder-ready structure

Usage:
  python scripts/preprocess_pipeline.py [--input data/clean_dataset] [--output data/aligned_dataset]

Input structure (flexible):
  - clean_dataset/patient_id/Before|After|Healing/<timeframe>|Unlabeled/*.jpg
  - raw_downloads/author/post/*.jpg  (phase inferred from path or → Unlabeled)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _ROOT / "data" / "clean_dataset"
DEFAULT_OUTPUT = _ROOT / "data" / "aligned_dataset"
DEFAULT_REVIEW = _ROOT / "data" / "manual_review"
MODEL_DIR = _ROOT / "data" / "models"
IMG_SIZE = 512

# Quality thresholds
BLUR_VAR_THRESHOLD = 60.0  # Laplacian variance; lower = blurrier
MIN_INTER_OCULAR_PX = 30   # Reject extreme close-ups (eyes too close)
MAX_INTER_OCULAR_PX = 600  # Reject tiny faces (eyes too far apart, likely low-res)
MIN_FACE_CONFIDENCE = 0.5  # MediaPipe face presence

# MediaPipe Face Landmarker: 478 landmarks (Face Mesh V2)
# Nose tip: 1, Left eye: 468, Right eye: 473
_NOSE_IDX = 1
_LEFT_EYE_IDX = 468
_RIGHT_EYE_IDX = 473

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"

# Lazy-init
_landmarker = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class Landmarks(NamedTuple):
    nose: tuple[int, int]
    left_eye: tuple[int, int]
    right_eye: tuple[int, int]


class RejectReason:
    UNREADABLE = "unreadable"
    BLURRY = "blurry"
    NO_FACE = "no_face"
    EXTREME_CLOSEUP = "extreme_closeup"
    TINY_FACE = "tiny_face"
    INVALID_CROP = "invalid_crop"


def _ensure_model() -> Path:
    if FACE_LANDMARKER_PATH.is_file():
        return FACE_LANDMARKER_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Face Landmarker model to %s ...", FACE_LANDMARKER_PATH)
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    return FACE_LANDMARKER_PATH


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    import mediapipe as mp  # noqa: PLC0415

    model_path = _ensure_model()
    base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_presence_confidence=MIN_FACE_CONFIDENCE,
    )
    _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)  # noqa: PLW0603
    return _landmarker


def _blur_variance(image: np.ndarray) -> float:
    """Laplacian variance; lower = blurrier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _inter_ocular_distance(left_eye: tuple[int, int], right_eye: tuple[int, int]) -> float:
    return math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])


def get_landmarks(image: np.ndarray) -> Landmarks | None:
    """Returns Landmarks or None if no face. Uses MediaPipe Face Landmarker."""
    import mediapipe as mp  # noqa: PLC0415

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    n = len(lm)
    if n <= max(_NOSE_IDX, _LEFT_EYE_IDX, _RIGHT_EYE_IDX):
        return None
    nose = (int(lm[_NOSE_IDX].x * w), int(lm[_NOSE_IDX].y * h))
    left_eye = (int(lm[_LEFT_EYE_IDX].x * w), int(lm[_LEFT_EYE_IDX].y * h))
    right_eye = (int(lm[_RIGHT_EYE_IDX].x * w), int(lm[_RIGHT_EYE_IDX].y * h))
    return Landmarks(nose=nose, left_eye=left_eye, right_eye=right_eye)


def align_and_crop(image: np.ndarray, lm: Landmarks) -> np.ndarray:
    """
    Rotate around eye center, scale to standardize eye distance, place nose at 40% from top.
    Single affine warp; more stable and consistent than rotate-then-crop.
    """
    nose, left_eye, right_eye = lm.nose, lm.left_eye, lm.right_eye

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    current_eye_dist = math.sqrt(dx**2 + dy**2)
    target_eye_dist = IMG_SIZE * 0.35
    scale = target_eye_dist / current_eye_dist
    scale = max(0.25, min(4.0, scale))  # Clamp to avoid extreme zooms

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    nose_pt = np.array([[nose[0], nose[1], 1.0]])
    rotated_nose = (M @ nose_pt.T).flatten()[:2]

    target_nose = (IMG_SIZE / 2, IMG_SIZE * 0.4)
    M[0, 2] += target_nose[0] - rotated_nose[0]
    M[1, 2] += target_nose[1] - rotated_nose[1]

    aligned = cv2.warpAffine(
        image, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def _phase_and_class_from_path(rel_path: Path) -> tuple[str, str]:
    """
    Infer phase and ImageFolder class from relative path.
    clean_dataset: patient_id/Before/img.jpg -> ("Before", "Before")
                   patient_id/Healing/01_1_week/img.jpg -> ("Healing", "Healing_01_1_week")
    """
    parts = rel_path.parts
    phase = "Unlabeled"
    timeframe = ""
    for i, p in enumerate(parts):
        if p == "Before":
            phase = "Before"
            break
        if p == "After":
            phase = "After"
            break
        if p == "Healing" and i + 1 < len(parts):
            phase = "Healing"
            timeframe = parts[i + 1]
            break
        if p == "Unlabeled":
            phase = "Unlabeled"
            break
    if phase == "Healing" and timeframe:
        class_name = f"Healing_{timeframe}"
    else:
        class_name = phase
    return phase, class_name


def _collect_images(input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Recursively collect (absolute_path, relative_path) for all images.
    relative_path is used to infer phase/class (e.g. Before, Healing/01_1_week).
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        return []
    results: list[tuple[Path, Path]] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                rel = p.relative_to(input_dir)
            except ValueError:
                rel = p.name
            results.append((p, rel))
    return results


def process_image(
    img_path: Path,
    output_dir: Path,
    review_dir: Path,
    rel_path: Path,
    skip_blur: bool = True,
) -> tuple[bool, str | None]:
    """
    Process one image: align, crop, save to output, or copy to review.
    Returns (success, reject_reason_or_None).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, RejectReason.UNREADABLE

    if skip_blur:
        var = _blur_variance(img)
        if var < BLUR_VAR_THRESHOLD:
            return False, RejectReason.BLURRY

    landmarks = get_landmarks(img)
    if landmarks is None:
        return False, RejectReason.NO_FACE

    iod = _inter_ocular_distance(landmarks.left_eye, landmarks.right_eye)
    if iod < MIN_INTER_OCULAR_PX:
        return False, RejectReason.EXTREME_CLOSEUP
    if iod > MAX_INTER_OCULAR_PX:
        return False, RejectReason.TINY_FACE

    try:
        aligned = align_and_crop(img, landmarks)
    except (cv2.error, ValueError) as e:
        log.warning("Invalid crop for %s: %s", img_path.name, e)
        return False, RejectReason.INVALID_CROP

    if aligned.size == 0:
        return False, RejectReason.INVALID_CROP

    phase, class_name = _phase_and_class_from_path(rel_path)
    out_class_dir = output_dir / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)
    # Unique filename: parent_folder_image_name to avoid collisions
    safe_name = rel_path.as_posix().replace("/", "_").replace("\\", "_")
    out_file = out_class_dir / safe_name
    cv2.imwrite(str(out_file), aligned)
    return True, None


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    review_dir: Path,
    skip_blur: bool = True,
    manifest_path: Path | None = None,
    copy_to_review: bool = True,
) -> dict[str, int]:
    """
    Run full preprocessing pipeline.
    Returns stats: {kept, unreadable, blurry, no_face, extreme_closeup, tiny_face, invalid_crop}
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    review_dir = Path(review_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_to_review:
        for reason in [
            RejectReason.UNREADABLE,
            RejectReason.BLURRY,
            RejectReason.NO_FACE,
            RejectReason.EXTREME_CLOSEUP,
            RejectReason.TINY_FACE,
            RejectReason.INVALID_CROP,
        ]:
            (review_dir / reason).mkdir(parents=True, exist_ok=True)

    images = _collect_images(input_dir)
    log.info("Found %d images in %s", len(images), input_dir)

    stats: dict[str, int] = {
        "kept": 0,
        RejectReason.UNREADABLE: 0,
        RejectReason.BLURRY: 0,
        RejectReason.NO_FACE: 0,
        RejectReason.EXTREME_CLOSEUP: 0,
        RejectReason.TINY_FACE: 0,
        RejectReason.INVALID_CROP: 0,
    }
    manifest_rows: list[dict] = []

    for i, (abs_path, rel_path) in enumerate(images):
        if (i + 1) % 50 == 0 or i == 0:
            log.info("Processing %d / %d ...", i + 1, len(images))

        success, reason = process_image(
            abs_path, output_dir, review_dir, rel_path, skip_blur=skip_blur
        )

        if success:
            stats["kept"] += 1
            phase, class_name = _phase_and_class_from_path(rel_path)
            out_rel = f"{class_name}/{rel_path.as_posix().replace('/', '_').replace(chr(92), '_')}"
            manifest_rows.append({
                "path": out_rel,
                "phase": phase,
                "class": class_name,
                "source": str(rel_path),
            })
        else:
            if reason:
                stats[reason] = stats.get(reason, 0) + 1
                if copy_to_review:
                    review_sub = review_dir / reason
                    dest = review_sub / f"{rel_path.as_posix().replace('/', '_').replace(chr(92), '_')}"
                    try:
                        shutil.copy2(abs_path, dest)
                    except OSError as e:
                        log.warning("Could not copy to review: %s", e)

    if manifest_path and manifest_rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "phase", "class", "source"])
            w.writeheader()
            w.writerows(manifest_rows)
        log.info("Wrote manifest: %s", manifest_path)

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rhinovate preprocessing: align faces (512×512), filter quality, save bad → review.",
    )
    ap.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input root (clean_dataset, raw_downloads, or custom)",
    )
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output root (ImageFolder structure)",
    )
    ap.add_argument(
        "--review", "-r",
        type=Path,
        default=DEFAULT_REVIEW,
        help="Review folder for rejected images (by reason)",
    )
    ap.add_argument(
        "--no-blur-check",
        action="store_true",
        help="Do not reject blurry images",
    )
    ap.add_argument(
        "--no-copy-review",
        action="store_true",
        help="Do not copy rejected images to review (just log)",
    )
    ap.add_argument(
        "--manifest", "-m",
        type=Path,
        default=None,
        help="Write manifest CSV (path, phase, class, source)",
    )
    args = ap.parse_args()

    input_dir = args.input if args.input.is_absolute() else _ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else _ROOT / args.output
    review_dir = args.review if args.review.is_absolute() else _ROOT / args.review
    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = _ROOT / manifest

    log.info("Input:  %s", input_dir)
    log.info("Output: %s", output_dir)
    log.info("Review: %s", review_dir)
    log.info("Blur check: %s | Copy to review: %s", not args.no_blur_check, not args.no_copy_review)
    if manifest:
        log.info("Manifest: %s", manifest)
    log.info("")

    stats = run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        review_dir=review_dir,
        skip_blur=not args.no_blur_check,
        manifest_path=manifest,
        copy_to_review=not args.no_copy_review,
    )

    log.info("")
    log.info("=== Summary ===")
    log.info("Kept (aligned): %d", stats["kept"])
    for k, v in stats.items():
        if k != "kept" and v > 0:
            log.info("  %s: %d", k, v)
