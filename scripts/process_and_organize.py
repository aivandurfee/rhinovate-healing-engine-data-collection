"""
Rhinovate – Strict Standardization & Flexible Time-Series Organization.

For training a Continuous Spatiotemporal Deformation Model:
  - Input: Before photo (Day 0) + time scalar t (days post-op)
  - Output: Predicted face at day t

Part 1 – Iron Fist Standardization:
  - Side profiles ONLY (inter-ocular + nose-to-eye ratio; rejects 3/4 and frontal)
  - Crop-only: NO rotation, NO affine transform (preserves orientation, avoids mutilation)
  - 512×512 square crop around face landmarks
  - Discard blur, no face

Part 2 – Flexible Time-Series:
  - Sparse longitudinal: patient_001/day_000.png, day_002.png, day_180.png
  - metadata.csv: patient_id, file_path, t_value, original_filename

Usage:
  python scripts/process_and_organize.py [--input data/clean_dataset] [--output data/dataset_v1]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _ROOT / "data" / "clean_dataset"
DEFAULT_OUTPUT = _ROOT / "data" / "dataset_v1"
MODEL_DIR = _ROOT / "data" / "models"
IMG_SIZE = 512

# Profile filter: STRICT side profiles only (rejects 3/4 views).
# True side profile: one eye occluded, nose perpendicular to camera.
# 3/4 profile: both eyes visible, yaw ~45° → REJECT these.
MAX_INTER_OCULAR_RATIO = 0.22   # Side: <0.22 (eyes overlap/occluded). 3/4: 0.30-0.45
MIN_NOSE_TO_EYE_RATIO = 2.5     # Side: >2.5 (far eye hidden). 3/4: 1.3-1.8
MIN_YAW_DEG = 55.0              # Side: 70-90°. 3/4: 30-50°. Threshold at 55°.
MAX_IO_FALLBACK = 0.20          # MediaPipe-only fallback (stricter)
MIN_NOSE_EYE_FALLBACK = 2.8     # MediaPipe-only fallback (stricter)
MIN_YAW_FALLBACK = 60.0         # MediaPipe-only fallback (stricter)

# Quality
BLUR_VAR_THRESHOLD = 5.0
# Reject ear-only / partial crops (profile cascade can fire on ears)
# MIN_FACE_IN_CROP=False: MediaPipe often fails on side profiles (frontal bias) → was causing Kept:0
MIN_PROFILE_BBOX_PX = 70    # Cascade bbox min dimension; ears give tiny bboxes (~50); real profiles ~100+
MIN_FACE_IN_CROP = False    # MediaPipe fails on side-profile crops; use bbox size filter instead
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"

# MediaPipe Face Landmarker 478 – indices for Frankfurt plane & profile
_NOSE_IDX = 1
_LEFT_EYE_IDX = 468
_RIGHT_EYE_IDX = 473
_LEFT_CHEEK_IDX = 234   # Tragus proxy (lateral face)
_RIGHT_CHEEK_IDX = 454
_LEFT_INFRAORBITAL = 263
_RIGHT_INFRAORBITAL = 362
_CHIN_IDX = 152
_FOREHEAD_IDX = 10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_landmarker = None
_eye_cascade = None
_profile_cascade = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Time Parsing
# -----------------------------------------------------------------------------
_DAY_PATTERNS = [
    (re.compile(r"\b(\d+)\s*hours?\s*(?:post|after|out)?", re.I), lambda m: int(m.group(1)) / 24),
    (re.compile(r"\b(\d+)\s*days?\s*(?:post|after|out)?", re.I), lambda m: float(m.group(1))),
    (re.compile(r"\b(\d+)\s*weeks?\s*(?:post|after|out)?", re.I), lambda m: float(m.group(1)) * 7),
    (re.compile(r"\b(\d+)\s*months?\s*(?:post|after|out)?", re.I), lambda m: float(m.group(1)) * 30),
    (re.compile(r"\b(\d+)\s*years?\s*(?:post|after|out)?", re.I), lambda m: float(m.group(1)) * 365),
    (re.compile(r"\b(\d+)\s*mo\b", re.I), lambda m: float(m.group(1)) * 30),
    (re.compile(r"\b(\d+)\s*wk\b", re.I), lambda m: float(m.group(1)) * 7),
    (re.compile(r"\b(\d+)\s*yr\b", re.I), lambda m: float(m.group(1)) * 365),
    (re.compile(r"\b(\d+)\s*(?:week|month|year)s?\b", re.I), lambda m: (
        float(m.group(1)) * (7 if "week" in m.group(0).lower() else 30 if "month" in m.group(0).lower() else 365)
    )),
]

BEFORE_KEYWORDS = {"before", "pre", "preop", "pre-op", "day 0", "day0"}
AFTER_KEYWORDS = {"after", "post", "postop", "post-op", "final", "result", "results", "healed", "6 month", "6 mo", "1 year"}


def parse_day_from_text(text: str) -> float | None:
    """Parse day number from filename/caption. Returns None if not found."""
    t = (text or "").strip().lower()
    if not t:
        return None

    for pattern, extractor in _DAY_PATTERNS:
        m = pattern.search(t)
        if m:
            return extractor(m)

    for kw in BEFORE_KEYWORDS:
        if kw in t:
            return 0.0
    for kw in AFTER_KEYWORDS:
        if kw in t:
            return 180.0  # Default "after" = 6 months

    return None


def infer_patient_id(path: Path, input_root: Path) -> str:
    """Infer patient_id from folder structure (e.g. asps_case_123, patient_foo_bar)."""
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return "unknown"
    parts = rel.parts
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


# -----------------------------------------------------------------------------
# MediaPipe
# -----------------------------------------------------------------------------
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

    _ensure_model()
    base = mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_presence_confidence=0.3,
    )
    _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)  # noqa: PLW0603
    return _landmarker


def _blur_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _to_px(lm, w: int, h: int, idx: int) -> tuple[float, float]:
    return (lm[idx].x * w, lm[idx].y * h)


def estimate_yaw_deg(lm, w: int, h: int) -> float:
    """Estimate yaw (degrees) from nose-to-eye distances. 0=front, 90=profile."""
    nose = np.array([lm[_NOSE_IDX].x * w, lm[_NOSE_IDX].y * h])
    left_eye = np.array([lm[_LEFT_EYE_IDX].x * w, lm[_LEFT_EYE_IDX].y * h])
    right_eye = np.array([lm[_RIGHT_EYE_IDX].x * w, lm[_RIGHT_EYE_IDX].y * h])
    d_left = np.linalg.norm(left_eye - nose)
    d_right = np.linalg.norm(right_eye - nose)
    ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-6)
    # ratio 1 -> 0°, ratio 2 -> ~60°, ratio 4+ -> ~80°+
    yaw = 90 * (1 - 1 / max(ratio, 1.0))
    if d_right > d_left:
        yaw = -yaw
    return abs(yaw)


def _face_size(lm, w: int, h: int) -> float:
    """Diagonal of face bbox in pixels."""
    xs = [lm[_CHIN_IDX].x, lm[_FOREHEAD_IDX].x, lm[_NOSE_IDX].x,
          lm[_LEFT_CHEEK_IDX].x, lm[_RIGHT_CHEEK_IDX].x]
    ys = [lm[_CHIN_IDX].y, lm[_FOREHEAD_IDX].y, lm[_NOSE_IDX].y,
          lm[_LEFT_CHEEK_IDX].y, lm[_RIGHT_CHEEK_IDX].y]
    bw = (max(xs) - min(xs)) * w
    bh = (max(ys) - min(ys)) * h
    return float(np.hypot(bw, bh)) if (bw > 0 and bh > 0) else 1.0


def nose_to_eye_ratio(lm, w: int, h: int) -> float:
    """Far/near eye distance from nose. Side: high. 3/4: low."""
    left_eye = np.array([lm[_LEFT_EYE_IDX].x * w, lm[_LEFT_EYE_IDX].y * h])
    right_eye = np.array([lm[_RIGHT_EYE_IDX].x * w, lm[_RIGHT_EYE_IDX].y * h])
    nose = np.array([lm[_NOSE_IDX].x * w, lm[_NOSE_IDX].y * h])
    d_left = np.linalg.norm(left_eye - nose)
    d_right = np.linalg.norm(right_eye - nose)
    return max(d_left, d_right) / (min(d_left, d_right) + 1e-6)


def inter_ocular_ratio(lm, w: int, h: int) -> float:
    """Distance between eyes / face size. Side: low. 3/4: high."""
    left_eye = np.array([lm[_LEFT_EYE_IDX].x * w, lm[_LEFT_EYE_IDX].y * h])
    right_eye = np.array([lm[_RIGHT_EYE_IDX].x * w, lm[_RIGHT_EYE_IDX].y * h])
    io_dist = float(np.linalg.norm(left_eye - right_eye))
    face_sz = _face_size(lm, w, h)
    return io_dist / (face_sz + 1e-6)


def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_eye.xml"
        _eye_cascade = cv2.CascadeClassifier(path)
    return _eye_cascade


def _get_profile_cascade():
    global _profile_cascade
    if _profile_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        _profile_cascade = cv2.CascadeClassifier(str(path))
    return _profile_cascade


def detect_profile_face(image: np.ndarray, high_recall: bool = False) -> tuple[int, int, int, int] | None:
    """
    Detect side profile face using OpenCV's profile cascade.
    Trained for left profiles; also checks flipped (right profiles).
    Returns largest (x, y, w, h) bbox or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    cascade = _get_profile_cascade()
    sf, nn, ms = (1.05, 3, (50, 50)) if high_recall else (1.06, 4, (55, 55))
    faces = cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=nn, minSize=ms)
    if len(faces) == 0:
        flipped = cv2.flip(gray, 1)
        faces = cascade.detectMultiScale(flipped, scaleFactor=sf, minNeighbors=nn, minSize=ms)
        if len(faces) == 0:
            return None
        # Convert coords back from flipped
        w_img = gray.shape[1]
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        x = w_img - (fx + fw)
        return (x, fy, fw, fh)
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return (x, y, w, h)


def has_face_in_crop(cropped: np.ndarray) -> bool:
    """True if MediaPipe detects a face in the crop. Rejects ear-only / partial crops."""
    lm, _ = detect_face_and_landmarks(cropped)
    return lm is not None


def count_visible_eyes(image: np.ndarray) -> int:
    """
    Count eyes detected by Haar cascade. Side profile: 0-1. 3/4: 2+.
    Stricter params to reduce false positives (glasses, shadows).
    """
    cascade = _get_eye_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Use stricter params: only clear eye detections count. 3/4 typically gets 2.
    eyes = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=8, minSize=(25, 25))
    return len(eyes)


def is_side_profile(
    lm, w: int, h: int,
    image: np.ndarray,
    max_io_ratio: float,
    min_yaw: float,
    min_nose_eye_ratio: float = 0.0,
    require_single_eye: bool = False,
) -> bool:
    """
    True only if STRICT side profile: only one eye visible.
    Uses: (1) low inter-ocular, (2) high nose-to-eye ratio, (3) optional eye count.
    """
    if inter_ocular_ratio(lm, w, h) > max_io_ratio:
        return False
    yaw = estimate_yaw_deg(lm, w, h)
    if yaw < min_yaw:
        return False
    if min_nose_eye_ratio > 0 and nose_to_eye_ratio(lm, w, h) < min_nose_eye_ratio:
        return False
    if require_single_eye:
        n_eyes = count_visible_eyes(image)
        if n_eyes > 1:
            return False
    return True


def crop_from_bbox(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Crop square around bbox, resize to IMG_SIZE. No rotation."""
    h_img, w_img = image.shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * 1.25
    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = int(cx + side / 2)
    y2 = int(cy + side / 2)
    pad_l = max(0, -x1)
    pad_r = max(0, x2 - w_img)
    pad_t = max(0, -y1)
    pad_b = max(0, y2 - h_img)
    if pad_l or pad_r or pad_t or pad_b:
        image = cv2.copyMakeBorder(
            image, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(128, 128, 128),
        )
        x1, x2 = x1 + pad_l, x2 + pad_l
        y1, y2 = y1 + pad_t, y2 + pad_t
        w_img, h_img = image.shape[1], image.shape[0]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = min(w_img, max(x2, x1 + 1))
    y2 = min(h_img, max(y2, y1 + 1))
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    hc, wc = cropped.shape[:2]
    side_len = min(wc, hc)
    x_off, y_off = (wc - side_len) // 2, (hc - side_len) // 2
    cropped = cropped[y_off : y_off + side_len, x_off : x_off + side_len]
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def crop_minimal(image: np.ndarray, lm, w: int, h: int) -> np.ndarray | None:
    """
    Crop square around face landmarks. NO rotation, NO affine transform.
    Preserves original image orientation to avoid mutilation.
    """
    pts = [
        (lm[_CHIN_IDX].x * w, lm[_CHIN_IDX].y * h),
        (lm[_FOREHEAD_IDX].x * w, lm[_FOREHEAD_IDX].y * h),
        (lm[_NOSE_IDX].x * w, lm[_NOSE_IDX].y * h),
        (lm[_LEFT_CHEEK_IDX].x * w, lm[_LEFT_CHEEK_IDX].y * h),
        (lm[_RIGHT_CHEEK_IDX].x * w, lm[_RIGHT_CHEEK_IDX].y * h),
    ]
    x_min = min(p[0] for p in pts)
    x_max = max(p[0] for p in pts)
    y_min = min(p[1] for p in pts)
    y_max = max(p[1] for p in pts)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = x_max - x_min
    bh = y_max - y_min
    side = max(bw, bh) * 1.25
    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = int(cx + side / 2)
    y2 = int(cy + side / 2)

    # Pad if crop extends beyond image (constant color)
    pad_l = max(0, -x1)
    pad_r = max(0, x2 - w)
    pad_t = max(0, -y1)
    pad_b = max(0, y2 - h)
    if pad_l or pad_r or pad_t or pad_b:
        image = cv2.copyMakeBorder(
            image, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )
        x1 += pad_l
        x2 += pad_l
        y1 += pad_t
        y2 += pad_t
        w = image.shape[1]
        h = image.shape[0]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = min(w, max(x2, x1 + 1))
    y2 = min(h, max(y2, y1 + 1))
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    hc, wc = cropped.shape[:2]
    side_len = min(wc, hc)
    x_off = (wc - side_len) // 2
    y_off = (hc - side_len) // 2
    cropped = cropped[y_off : y_off + side_len, x_off : x_off + side_len]
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def detect_face_and_landmarks(image: np.ndarray):
    """Returns (landmarks_list, yaw_deg) or (None, None)."""
    import mediapipe as mp  # noqa: PLC0415

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None, None
    lm = result.face_landmarks[0]
    n = len(lm)
    required = {
        _NOSE_IDX, _LEFT_EYE_IDX, _RIGHT_EYE_IDX,
        _LEFT_CHEEK_IDX, _RIGHT_CHEEK_IDX,
        _LEFT_INFRAORBITAL, _RIGHT_INFRAORBITAL,
        _CHIN_IDX, _FOREHEAD_IDX,
    }
    if n <= max(required):
        return None, None
    yaw = estimate_yaw_deg(lm, w, h)
    return lm, yaw


def align_frontal(image: np.ndarray, lm, w: int, h: int, apply_flip: bool = True) -> np.ndarray:
    """Eye-based alignment for frontal faces (eyes horizontal, nose at 40% from top)."""
    nose = _to_px(lm, w, h, _NOSE_IDX)
    left_eye = _to_px(lm, w, h, _LEFT_EYE_IDX)
    right_eye = _to_px(lm, w, h, _RIGHT_EYE_IDX)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    current_eye_dist = math.sqrt(dx**2 + dy**2)
    target_eye_dist = IMG_SIZE * 0.35
    scale = target_eye_dist / (current_eye_dist + 1e-6)
    scale = max(0.25, min(4.0, scale))

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    nose_pt = np.array([[nose[0], nose[1], 1.0]])
    rotated_nose = (M @ nose_pt.T).flatten()[:2]
    target_nose = (IMG_SIZE / 2, IMG_SIZE * 0.4)
    M[0, 2] += target_nose[0] - rotated_nose[0]
    M[1, 2] += target_nose[1] - rotated_nose[1]

    result = cv2.warpAffine(
        image, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if apply_flip:
        result = cv2.flip(result, 1)
    return result


def align_frankfurt_and_crop(image: np.ndarray, lm, profile_side: str, w: int, h: int, apply_flip: bool = True) -> np.ndarray | None:
    """
    Frankfurt plane: line tragus–infraorbital horizontal.
    Crop chin-to-eyebrow, nose-to-ear. Resize to IMG_SIZE x IMG_SIZE.
    Standardizes orientation: nose points right, ear on left.
    """
    # side_right = left face visible (left ear) -> use LEFT landmarks
    # side_left = right face visible (right ear) -> use RIGHT landmarks
    if profile_side == "side_right":
        tragus = _to_px(lm, w, h, _LEFT_CHEEK_IDX)
        infraorbital = _to_px(lm, w, h, _LEFT_INFRAORBITAL)
    else:
        tragus = _to_px(lm, w, h, _RIGHT_CHEEK_IDX)
        infraorbital = _to_px(lm, w, h, _RIGHT_INFRAORBITAL)

    # Rotate so tragus->infraorbital is horizontal; use -atan2 to align correctly
    dx = infraorbital[0] - tragus[0]
    dy = infraorbital[1] - tragus[1]
    angle = -math.degrees(math.atan2(dy, dx))
    center = ((tragus[0] + infraorbital[0]) / 2, (tragus[1] + infraorbital[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate all landmarks to get bbox in rotated space
    pts = np.array([
        [lm[_CHIN_IDX].x * w, lm[_CHIN_IDX].y * h, 1],
        [lm[_FOREHEAD_IDX].x * w, lm[_FOREHEAD_IDX].y * h, 1],
        [lm[_NOSE_IDX].x * w, lm[_NOSE_IDX].y * h, 1],
        [tragus[0], tragus[1], 1],
    ], dtype=np.float32)
    rotated_pts = (M @ pts.T).T[:, :2]  # affine: (2x3) @ (3x4) -> 2x4, then 4x2

    x_min = float(np.min(rotated_pts[:, 0]))
    x_max = float(np.max(rotated_pts[:, 0]))
    y_min = float(np.min(rotated_pts[:, 1]))
    y_max = float(np.max(rotated_pts[:, 1]))

    # Square crop to avoid distortion (resize of non-square would stretch)
    margin = 0.2
    bw = x_max - x_min
    bh = y_max - y_min
    side = max(bw, bh) * (1 + margin)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = int(cx + side / 2)
    y2 = int(cy + side / 2)

    # Use BORDER_CONSTANT to avoid mirrored artifacts (BORDER_REFLECT_101 caused mutilation)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128),
    )

    # Pad if crop extends beyond image (constant color, no mirroring)
    pad_l = max(0, -x1)
    pad_r = max(0, x2 - w)
    pad_t = max(0, -y1)
    pad_b = max(0, y2 - h)
    if pad_l or pad_r or pad_t or pad_b:
        rotated = cv2.copyMakeBorder(
            rotated, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )
        x1 += pad_l
        x2 += pad_l
        y1 += pad_t
        y2 += pad_t

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(rotated.shape[1], x2)
    y2 = min(rotated.shape[0], y2)
    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    # Guarantee square crop (avoids stretch distortion when resizing)
    hc, wc = cropped.shape[:2]
    side = min(wc, hc)
    x_off = (wc - side) // 2
    y_off = (hc - side) // 2
    cropped = cropped[y_off : y_off + side, x_off : x_off + side]
    result = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    # Flip disabled by default (was causing mutilation); use --flip to enable if needed
    if apply_flip:
        result = cv2.flip(result, 1)
        if profile_side == "side_left":
            result = cv2.flip(result, 1)
    return result


def process_image(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    allow_frontal: bool = False,
    blur_threshold: float = BLUR_VAR_THRESHOLD,
    min_yaw: float = MIN_YAW_DEG,
    max_io_ratio: float = MAX_INTER_OCULAR_RATIO,
    min_nose_eye_ratio: float = MIN_NOSE_TO_EYE_RATIO,
    require_single_eye: bool = False,
    apply_flip: bool = False,
    high_recall: bool = False,
    use_loose_fallback: bool = False,
) -> tuple[bool, float | None, str | None, str | None]:
    """
    Process one image. Returns (success, t_value, reject_reason, rel_file_path).
    On success: rel_file_path is e.g. "patient_001/day_007.png" for metadata.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None, "unreadable", None

    if blur_threshold > 0 and _blur_variance(img) < blur_threshold:
        return False, None, "blurry", None

    if not allow_frontal:
        profile_bbox = detect_profile_face(img)
        if profile_bbox is not None:
            x, y, bw, bh = profile_bbox
            if min(bw, bh) < MIN_PROFILE_BBOX_PX:
                return False, None, "not_profile", None  # Ear-only; bbox too small
        lm, _ = detect_face_and_landmarks(img)
        if profile_bbox is not None:
            # Path 1: cascade found – lenient landmark filter
            if lm is not None and not is_side_profile(
                lm, img.shape[1], img.shape[0], img, max_io_ratio, min_yaw, min_nose_eye_ratio, require_single_eye
            ):
                return False, None, "not_profile", None
        else:
            # Path 2: cascade missed – MediaPipe fallback (strict unless loose)
            io_fb = max_io_ratio if use_loose_fallback else MAX_IO_FALLBACK
            yaw_fb = min_yaw if use_loose_fallback else MIN_YAW_FALLBACK
            ratio_fb = min_nose_eye_ratio if use_loose_fallback else MIN_NOSE_EYE_FALLBACK
            if lm is None or not is_side_profile(
                lm, img.shape[1], img.shape[0], img, io_fb, yaw_fb, ratio_fb, require_single_eye,
            ):
                return False, None, "not_profile", None

    text = img_path.stem + " " + img_path.parent.name
    t_val = parse_day_from_text(text)
    if t_val is None:
        t_val = parse_day_from_text(str(img_path))
    if t_val is None:
        t_val = parse_day_from_text(img_path.parent.name)
    if t_val is None:
        return False, None, "no_time_parsed", None

    w, h = img.shape[1], img.shape[0]
    if allow_frontal:
        lm, _ = detect_face_and_landmarks(img)
        if lm is None:
            return False, None, "no_face", None
        aligned = align_frontal(img, lm, w, h, apply_flip=apply_flip)
    else:
        profile_bbox = detect_profile_face(img)
        if profile_bbox is not None:
            x, y, bw, bh = profile_bbox
            aligned = crop_from_bbox(img, x, y, bw, bh)
        else:
            lm, _ = detect_face_and_landmarks(img)
            if lm is None:
                return False, None, "no_face", None
            aligned = crop_minimal(img, lm, w, h)
        if aligned is not None and apply_flip:
            aligned = cv2.flip(aligned, 1)
    if aligned is None:
        return False, None, "crop_failed", None
    if MIN_FACE_IN_CROP and not allow_frontal and not has_face_in_crop(aligned):
        return False, None, "no_face_in_crop", None  # Ear-only / partial; reject

    patient_id = infer_patient_id(img_path, input_root)
    out_dir = output_root / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t_int = int(round(t_val))
    t_int = max(0, min(999, t_int))
    out_name = f"day_{t_int:03d}.png"
    out_path = out_dir / out_name

    # Handle collisions: same patient, same day (e.g. multiple "1 week" images)
    counter = 0
    while out_path.exists() and counter < 100:
        counter += 1
        out_name = f"day_{t_int:03d}_{counter}.png"
        out_path = out_dir / out_name
    cv2.imwrite(str(out_path), aligned)

    rel_path = f"{patient_id}/{out_name}"
    return True, t_val, None, rel_path


def run(
    input_dir: Path,
    output_dir: Path,
    allow_frontal: bool = False,
    blur_threshold: float = BLUR_VAR_THRESHOLD,
    min_yaw: float = MIN_YAW_DEG,
    max_io_ratio: float = MAX_INTER_OCULAR_RATIO,
    min_nose_eye_ratio: float = MIN_NOSE_TO_EYE_RATIO,
    require_single_eye: bool = False,
    apply_flip: bool = False,
    high_recall: bool = False,
    use_loose_fallback: bool = False,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images: list[tuple[Path, Path]] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                rel = p.relative_to(input_dir)
            except ValueError:
                rel = Path(p.name)
            images.append((p, rel))

    log.info("Found %d images in %s", len(images), input_dir)
    if len(images) == 0:
        log.warning("No images found. Check input path. Try: ./data/clean_dataset or ./data/raw_downloads/asps")
    metadata_rows: list[dict] = []
    stats: dict[str, int] = {"kept": 0, "unreadable": 0, "blurry": 0, "no_face": 0, "not_profile": 0, "no_time_parsed": 0, "crop_failed": 0, "no_face_in_crop": 0}

    for i, (abs_path, rel_path) in enumerate(images):
        if (i + 1) % 25 == 0 or i == 0:
            log.info("Processing %d / %d ...", i + 1, len(images))

        success, t_val, reason, file_path = process_image(
            abs_path, input_dir, output_dir,
            allow_frontal=allow_frontal,
            blur_threshold=blur_threshold,
            min_yaw=min_yaw,
            max_io_ratio=max_io_ratio,
            min_nose_eye_ratio=min_nose_eye_ratio,
            require_single_eye=require_single_eye,
            apply_flip=apply_flip,
            high_recall=high_recall,
            use_loose_fallback=use_loose_fallback,
        )

        if success and file_path:
            stats["kept"] += 1
            patient_id = infer_patient_id(abs_path, input_dir)
            metadata_rows.append({
                "patient_id": patient_id,
                "file_path": file_path,
                "t_value": f"{t_val:.1f}",
                "original_filename": str(rel_path),
            })
        elif reason:
            stats[reason] = stats.get(reason, 0) + 1

    manifest_path = output_dir / "metadata.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "file_path", "t_value", "original_filename"])
        w.writeheader()
        w.writerows(metadata_rows)
    log.info("Wrote %s", manifest_path)

    log.info("")
    log.info("=== Summary ===")
    log.info("Kept: %d", stats["kept"])
    for k, v in stats.items():
        if k != "kept" and v > 0:
            log.info("  %s: %d", k, v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standardization + flexible time-series organization. Use --allow-frontal for ASPS/frontal data.",
    )
    ap.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input root (clean_dataset)")
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output root (dataset_v1)")
    ap.add_argument(
        "--allow-frontal",
        action="store_true",
        help="Allow frontal faces. Uses eye alignment instead of side-profile Frankfurt.",
    )
    ap.add_argument(
        "--blur-threshold",
        type=float,
        default=BLUR_VAR_THRESHOLD,
        help="Laplacian variance threshold; lower = stricter. Default 20. Use 0 to disable.",
    )
    ap.add_argument(
        "--min-yaw",
        type=float,
        default=MIN_YAW_DEG,
        help="Min yaw (degrees). Default 55. True side profiles are 70-90°.",
    )
    ap.add_argument(
        "--max-io-ratio",
        type=float,
        default=MAX_INTER_OCULAR_RATIO,
        help="Max inter-ocular / face size. Default 0.22. 3/4 views are 0.30-0.45.",
    )
    ap.add_argument(
        "--min-nose-eye-ratio",
        type=float,
        default=MIN_NOSE_TO_EYE_RATIO,
        help="Min nose-to-eye ratio (far/near). Default 2.5. 3/4 views are 1.3-1.8.",
    )
    ap.add_argument(
        "--flip",
        action="store_true",
        help="Apply L-R flip (disabled by default; was causing mutilation).",
    )
    ap.add_argument(
        "--loose",
        action="store_true",
        help="Loosen profile filter (max-io 0.30, min-nose-eye 2.0, yaw 40°, no eye check). May include some 3/4.",
    )
    ap.add_argument(
        "--high-recall",
        action="store_true",
        help="Loosen profile cascade to capture more side profiles (~100). Verify results manually.",
    )
    args = ap.parse_args()

    min_yaw = 40.0 if args.loose else args.min_yaw
    max_io_ratio = 0.30 if args.loose else args.max_io_ratio
    min_nose_eye_ratio = 2.0 if args.loose else args.min_nose_eye_ratio
    # Eye count unreliable (shadows, ear shadows). Disable to restore side-profile recall.
    require_single_eye = False
    high_recall = args.high_recall
    use_loose_fallback = args.loose

    input_dir = args.input if args.input.is_absolute() else _ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else _ROOT / args.output

    log.info("Input:  %s", input_dir)
    log.info("Output: %s", output_dir)
    log.info("Allow frontal: %s | Blur: %.0f | Min yaw: %.0f° | Max IO: %.2f | Min nose-eye: %.1f | Flip: %s%s%s",
             args.allow_frontal, args.blur_threshold, min_yaw, max_io_ratio, min_nose_eye_ratio, args.flip,
             " [LOOSE]" if args.loose else "",
             " [HIGH-RECALL]" if args.high_recall else "")
    log.info("")
    run(
        input_dir, output_dir,
        allow_frontal=args.allow_frontal,
        blur_threshold=args.blur_threshold,
        min_yaw=min_yaw,
        max_io_ratio=max_io_ratio,
        min_nose_eye_ratio=min_nose_eye_ratio,
        require_single_eye=require_single_eye,
        apply_flip=args.flip,
        high_recall=high_recall,
        use_loose_fallback=use_loose_fallback,
    )


if __name__ == "__main__":
    main()
