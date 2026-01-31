"""
Prepare training data: angle detection (Front/Side), auto-crop on nose, move
no-face images to manual_review. Reads from clean_dataset, writes to training_ready.
Uses MediaPipe Tasks API (Face Landmarker), same as align_faces.py.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import shutil

# -----------------------------------------------------------------------------
# CONFIG (paths relative to project root so script works from any cwd)
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = _ROOT / "data" / "clean_dataset"
OUTPUT_DIR = _ROOT / "data" / "training_ready"
REVIEW_DIR = _ROOT / "data" / "manual_review"
MODEL_DIR = _ROOT / "data" / "models"
IMG_SIZE = 512
CROP_FACTOR = 2.5  # How "zoomed in" on the nose? (Higher = more face, Lower = just nose)
MIN_CROP_RADIUS = 32  # Avoid empty or tiny crops when face is very small

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Face Landmarker (MediaPipe 0.10+ Tasks API) — same model as align_faces.py
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"

# Landmark indices: Nose 1, Left eye 468, Right eye 473, Cheeks 234/454 (Face Landmarker topology)
_NOSE_IDX = 1
_LEFT_EYE_IDX = 468
_RIGHT_EYE_IDX = 473
_LEFT_CHEEK_IDX = 234
_RIGHT_CHEEK_IDX = 454

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
    _ensure_model()
    base = mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def get_face_info(image: np.ndarray) -> tuple[list | None, str | None]:
    """Returns (landmarks_list, view) or (None, None) if no face. Uses Face Landmarker."""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None, None
    lm = result.face_landmarks[0]
    n = len(lm)
    if n <= max(_NOSE_IDX, _LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_CHEEK_IDX, _RIGHT_CHEEK_IDX):
        return None, None
    nose = np.array([lm[_NOSE_IDX].x * w, lm[_NOSE_IDX].y * h])
    left_eye = np.array([lm[_LEFT_EYE_IDX].x * w, lm[_LEFT_EYE_IDX].y * h])
    right_eye = np.array([lm[_RIGHT_EYE_IDX].x * w, lm[_RIGHT_EYE_IDX].y * h])
    d_left = np.linalg.norm(left_eye - nose)
    d_right = np.linalg.norm(right_eye - nose)
    ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-6)
    if ratio < 1.5:
        view = "front"
    elif d_left < d_right:
        view = "side_right"
    else:
        view = "side_left"
    return lm, view

def smart_crop(image: np.ndarray, landmarks: list) -> np.ndarray:
    h, w, _ = image.shape
    nose_x = int(landmarks[_NOSE_IDX].x * w)
    nose_y = int(landmarks[_NOSE_IDX].y * h)
    left_cheek = landmarks[_LEFT_CHEEK_IDX].x * w
    right_cheek = landmarks[_RIGHT_CHEEK_IDX].x * w
    face_width = abs(right_cheek - left_cheek)
    radius = int(face_width / CROP_FACTOR)
    radius = max(radius, MIN_CROP_RADIUS)  # Avoid empty or tiny crops

    y1 = max(0, nose_y - radius)
    y2 = min(h, nose_y + radius)
    x1 = max(0, nose_x - radius)
    x2 = min(w, nose_x + radius)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        # Fallback: center crop around nose with min size (clamp to image bounds)
        side = max(min(MIN_CROP_RADIUS * 2, w, h), 1)
        x1 = max(0, min(nose_x - side // 2, w - side))
        y1 = max(0, min(nose_y - side // 2, h - side))
        x2 = min(x1 + side, w)
        y2 = min(y1 + side, h)
        cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

def process_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.is_dir():
        print(f"Input directory not found: {INPUT_DIR}")
        return

    cases = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"Processing {len(cases)} cases...")

    for case in cases:
        for phase in ["Before", "After"]:
            phase_dir = case / phase
            if not phase_dir.exists():
                continue

            for img_file in sorted(phase_dir.iterdir()):
                if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                landmarks, view = get_face_info(img)

                if landmarks is None:
                    save_path = REVIEW_DIR / f"{case.name}_{phase}_{img_file.name}"
                    shutil.copy2(img_file, save_path)
                    print(f"  Moved to Review (No Face Detected): {img_file.name}")
                    continue

                final_img = smart_crop(img, landmarks)
                # One folder per person (case): training_ready/{case}/Before/{view}/ and .../After/{view}/
                # So Person 1's before and after stay together; pair by case, then by view/filename
                save_dir = OUTPUT_DIR / case.name / phase / view
                save_dir.mkdir(parents=True, exist_ok=True)
                save_name = img_file.name
                cv2.imwrite(str(save_dir / save_name), final_img)

if __name__ == "__main__":
    process_dataset()