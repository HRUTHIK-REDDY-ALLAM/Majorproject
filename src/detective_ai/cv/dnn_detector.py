"""Object detection using OpenCV DNN with MobileNet-SSD (Caffe).

Auto-downloads the pre-trained MobileNet-SSD model on first use (~23 MB).
Detects 20 COCO-class objects including persons, cars, buses, bicycles, etc.
Runs entirely on CPU — no GPU or extra pip packages required.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Model file URLs (hosted by OpenCV maintainers) ───────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent / "models"

_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://drive.google.com/uc?export=download&id="
    "0B3gersZ2cHIxRm5PMWRoTkdHdHc"
)

# Fallback: use the VOC-trained model from opencv_extra (more reliable download)
_PROTOTXT_URL_ALT = (
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/"
    "master/MobileNetSSD_deploy.prototxt"
)
_CAFFEMODEL_URL_ALT = (
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/"
    "master/MobileNetSSD_deploy.caffemodel"
)

_PROTOTXT_FILE = _MODELS_DIR / "MobileNetSSD_deploy.prototxt"
_CAFFEMODEL_FILE = _MODELS_DIR / "MobileNetSSD_deploy.caffemodel"

# MobileNet-SSD PASCAL VOC class labels (21 classes, index 0 = background)
_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# Map class names to our simplified labels
_LABEL_MAP = {
    "person": "person",
    "bicycle": "vehicle",
    "bus": "vehicle",
    "car": "vehicle",
    "motorbike": "vehicle",
    "train": "vehicle",
    "boat": "vehicle",
    "aeroplane": "vehicle",
    # Other objects we track for richer scene descriptions
    "chair": "furniture",
    "sofa": "furniture",
    "diningtable": "furniture",
    "tvmonitor": "electronics",
    "bottle": "object",
    "pottedplant": "object",
    "cat": "animal",
    "dog": "animal",
    "cow": "animal",
    "horse": "animal",
    "sheep": "animal",
    "bird": "animal",
}

# Singleton model
_net = None


def _ensure_model_downloaded() -> bool:
    """Download model files if not present. Returns True on success."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if _PROTOTXT_FILE.exists() and _CAFFEMODEL_FILE.exists():
        # Verify caffemodel is large enough (not a partial download)
        if _CAFFEMODEL_FILE.stat().st_size > 10_000_000:
            return True
        logger.warning("Caffemodel file seems truncated, re-downloading…")
        _CAFFEMODEL_FILE.unlink(missing_ok=True)

    # Download prototxt
    if not _PROTOTXT_FILE.exists():
        for url in [_PROTOTXT_URL_ALT, _PROTOTXT_URL]:
            try:
                logger.info(f"Downloading MobileNet-SSD prototxt from {url}…")
                urllib.request.urlretrieve(url, str(_PROTOTXT_FILE))
                if _PROTOTXT_FILE.exists() and _PROTOTXT_FILE.stat().st_size > 1000:
                    break
            except Exception as e:
                logger.warning(f"Failed to download prototxt from {url}: {e}")
                _PROTOTXT_FILE.unlink(missing_ok=True)

    # Download caffemodel
    if not _CAFFEMODEL_FILE.exists():
        for url in [_CAFFEMODEL_URL_ALT, _CAFFEMODEL_URL]:
            try:
                logger.info(f"Downloading MobileNet-SSD caffemodel from {url}…")
                urllib.request.urlretrieve(url, str(_CAFFEMODEL_FILE))
                if _CAFFEMODEL_FILE.exists() and _CAFFEMODEL_FILE.stat().st_size > 10_000_000:
                    break
            except Exception as e:
                logger.warning(f"Failed to download caffemodel from {url}: {e}")
                _CAFFEMODEL_FILE.unlink(missing_ok=True)

    ok = (
        _PROTOTXT_FILE.exists()
        and _CAFFEMODEL_FILE.exists()
        and _CAFFEMODEL_FILE.stat().st_size > 10_000_000
    )
    if ok:
        logger.info("MobileNet-SSD model files ready.")
    else:
        logger.error("Failed to download MobileNet-SSD model files.")
    return ok


def _get_net():
    """Lazy-load the DNN model."""
    global _net
    if _net is not None:
        return _net

    if not _ensure_model_downloaded():
        return None

    try:
        _net = cv2.dnn.readNetFromCaffe(
            str(_PROTOTXT_FILE), str(_CAFFEMODEL_FILE)
        )
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("MobileNet-SSD DNN model loaded successfully.")
        return _net
    except Exception as e:
        logger.error(f"Failed to load DNN model: {e}")
        return None


def detect_objects_dnn(
    frame: np.ndarray,
    confidence_threshold: float = 0.35,
) -> list[dict]:
    """Detect objects in a frame using MobileNet-SSD via OpenCV DNN.

    Args:
        frame: BGR image as numpy array.
        confidence_threshold: Minimum detection confidence (0-1).

    Returns:
        List of detection dicts with keys:
        bbox_x, bbox_y, bbox_w, bbox_h, confidence, label, class_id, class_name
    """
    net = _get_net()
    if net is None:
        return []

    h, w = frame.shape[:2]

    # MobileNet-SSD expects 300×300 input
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=0.007843, size=(300, 300),
        mean=(127.5, 127.5, 127.5), swapRB=False, crop=False,
    )
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id < 0 or class_id >= len(_CLASS_NAMES):
            continue

        class_name = _CLASS_NAMES[class_id]
        if class_name == "background":
            continue

        label = _LABEL_MAP.get(class_name, "object")

        # Convert from normalized coordinates to pixel coordinates
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        results.append({
            "bbox_x": float(x1),
            "bbox_y": float(y1),
            "bbox_w": float(x2 - x1),
            "bbox_h": float(y2 - y1),
            "confidence": round(confidence, 3),
            "label": label,
            "class_id": class_id,
            "class_name": class_name,
        })

    return results


def is_available() -> bool:
    """Check if the DNN detector is available (model downloaded and loadable)."""
    return _get_net() is not None
