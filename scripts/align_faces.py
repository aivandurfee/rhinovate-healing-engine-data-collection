"""
Rhinovate Healing Engine – Face alignment & garbage filtering.
Reads from clean_dataset (Before / After / Healing / Unlabeled), deletes bad images,
aligns faces (eyes horizontal, nose centered), crops to 512×512 for U‑Net training.
Preserves phase + healing timeframe structure so you know when during healing each
photo was taken. Optional --manifest CSV for training (patient_id, phase, timeframe, path).
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION (align with sort_healing_data output)
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = _ROOT / "data" / "clean_dataset"
OUTPUT_DIR = _ROOT / "data" / "aligned_dataset"
MODEL_DIR = _ROOT / "data" / "models"
IMG_SIZE = 512

# Reject images with Laplacian variance below this (blur / low detail)
BLUR_VAR_THRESHOLD = 60.0

# Face Landmarker model (MediaPipe tasks API)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"

# Nose tip: 1. Eyes: 468 (left), 473 (right) — same as legacy Face Mesh topology
_NOSE_IDX = 1
_LEFT_EYE_IDX = 468
_RIGHT_EYE_IDX = 473

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Lazy-init Face Landmarker
_landmarker = None


def _ensure_model() -> Path:
    if FACE_LANDMARKER_PATH.is_file():
        return FACE_LANDMARKER_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker model to {FACE_LANDMARKER_PATH} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    return FACE_LANDMARKER_PATH


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    import mediapipe as mp

    model_path = _ensure_model()
    base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)  # noqa: PLW0603
    return _landmarker


def _blur_variance(image: np.ndarray) -> float:
    """Laplacian variance; lower = blurrier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def get_landmarks(image: np.ndarray):
    """Returns (nose, left_eye, right_eye) or None if no face. Uses MediaPipe Face Landmarker."""
    import mediapipe as mp

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    lm_task = _get_landmarker()
    result = lm_task.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    n = len(lm)
    if n <= max(_NOSE_IDX, _LEFT_EYE_IDX, _RIGHT_EYE_IDX):
        return None
    nose = (int(lm[_NOSE_IDX].x * w), int(lm[_NOSE_IDX].y * h))
    left_eye = (int(lm[_LEFT_EYE_IDX].x * w), int(lm[_LEFT_EYE_IDX].y * h))
    right_eye = (int(lm[_RIGHT_EYE_IDX].x * w), int(lm[_RIGHT_EYE_IDX].y * h))
    return nose, left_eye, right_eye


def align_and_crop(image: np.ndarray, nose, left_eye, right_eye) -> np.ndarray:
    """Rotate around eye center, scale to standardize eye distance, place nose at 40% from top."""
    h, w = image.shape[:2]

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


def _phase_timeframe_from_rel(rel: Path) -> tuple[str, str]:
    """(phase, timeframe) from e.g. Before/img.jpg, Healing/01_1_week/img.jpg."""
    parts = rel.parts
    if len(parts) < 2:
        return "Unlabeled", ""
    phase = parts[0]
    timeframe = parts[1] if phase == "Healing" and len(parts) > 2 else ""
    return phase, str(timeframe)


def process_dataset(
    delete_bad: bool = True,
    skip_blur: bool = True,
    manifest_path: Path | None = None,
) -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    deleted = 0
    kept = 0
    manifest_rows: list[dict] = []

    for patient_dir in sorted(INPUT_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        out_patient = OUTPUT_DIR / patient_id
        any_kept = False

        # Walk all images under patient (Before/, After/, Healing/<timeframe>/, Unlabeled/)
        for img_path in sorted(patient_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = img_path.relative_to(patient_dir)
            out_file = out_patient / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(str(img_path))
            if img is None:
                if delete_bad:
                    img_path.unlink(missing_ok=True)
                    deleted += 1
                    print(f"   🗑 Deleted (unreadable): {patient_id} / {rel}")
                else:
                    print(f"   ⏭ Skip (unreadable): {patient_id} / {rel}")
                continue

            if skip_blur:
                var = _blur_variance(img)
                if var < BLUR_VAR_THRESHOLD:
                    if delete_bad:
                        img_path.unlink(missing_ok=True)
                        deleted += 1
                        print(f"   🗑 Deleted (blurry, var={var:.0f}): {patient_id} / {rel}")
                    else:
                        print(f"   ⏭ Skip blurry (var={var:.0f}): {patient_id} / {rel}")
                    continue

            landmarks = get_landmarks(img)
            if landmarks is None:
                if delete_bad:
                    img_path.unlink(missing_ok=True)
                    deleted += 1
                    print(f"   🗑 Deleted (no face): {patient_id} / {rel}")
                else:
                    print(f"   ⏭ Skip (no face): {patient_id} / {rel}")
                continue

            nose, le, re = landmarks
            aligned = align_and_crop(img, nose, le, re)
            cv2.imwrite(str(out_file), aligned)
            kept += 1
            any_kept = True
            phase, timeframe = _phase_timeframe_from_rel(rel)
            manifest_rows.append({
                "patient_id": patient_id,
                "phase": phase,
                "timeframe": timeframe,
                "path": str(out_file.relative_to(OUTPUT_DIR)),
            })
            print(f"   ✅ {patient_id} / {rel}")

        if any_kept:
            print(f"Processed {patient_id}")

    if manifest_path and manifest_rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "phase", "timeframe", "path"])
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"Wrote manifest: {manifest_path}")

    print(f"\nDone. Kept & aligned: {kept} | Deleted: {deleted}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Align faces (512×512, nose-centered) and delete bad images. Preserves Before/After/Healing structure.",
    )
    ap.add_argument(
        "--no-delete",
        action="store_true",
        help="Skip bad images but do not delete them from clean_dataset.",
    )
    ap.add_argument(
        "--no-blur-check",
        action="store_true",
        help="Do not reject blurry images (Laplacian variance).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write CSV of aligned images (patient_id, phase, timeframe, path) for training.",
    )
    args = ap.parse_args()

    delete_bad = not args.no_delete
    skip_blur = not args.no_blur_check
    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = _ROOT / manifest

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Delete bad: {delete_bad} | Blur check: {skip_blur}")
    if manifest:
        print(f"Manifest: {manifest}")
    print()

    process_dataset(delete_bad=delete_bad, skip_blur=skip_blur, manifest_path=manifest)


if __name__ == "__main__":
    main()
