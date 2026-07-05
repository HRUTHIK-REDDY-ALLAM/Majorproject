"""Person and vehicle detection using YOLOv8 nano (CPU-optimized).

Detects people and vehicles in video frames, returning bounding boxes
with detection confidence scores.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from detective_ai.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None

# COCO class IDs we care about
_PERSON_CLASS_ID = 0
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
_TARGET_CLASSES = {_PERSON_CLASS_ID} | _VEHICLE_CLASS_IDS


def _get_model():
    """Lazy-load YOLOv8 nano model."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        model_path = settings.yolo_model
        logger.info(f"Loading YOLOv8 model: {model_path} (CPU)")
        _model = YOLO(model_path)
        logger.info("YOLOv8 model loaded.")
    return _model


def detect_objects(
    frame: np.ndarray,
    confidence_threshold: float | None = None,
    target_size: tuple[int, int] = (640, 480),
) -> list[dict]:
    """Detect people and vehicles in a single frame.

    Args:
        frame: BGR image as numpy array.
        confidence_threshold: Minimum detection confidence.
        target_size: Resize frame to this size for faster inference.

    Returns:
        List of detection dicts with keys:
        bbox_x, bbox_y, bbox_w, bbox_h, confidence, label, class_id
    """
    confidence_threshold = confidence_threshold or settings.yolo_confidence_threshold
    model = _get_model()

    # Resize for CPU speed
    h, w = frame.shape[:2]
    if w > target_size[0] or h > target_size[1]:
        frame_resized = cv2.resize(frame, target_size)
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]
    else:
        frame_resized = frame
        scale_x = 1.0
        scale_y = 1.0

    # Run detection
    results = model(frame_resized, conf=confidence_threshold, device="cpu", verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in _TARGET_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Scale back to original coordinates
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            label = "person" if class_id == _PERSON_CLASS_ID else "vehicle"

            detections.append({
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
                "confidence": conf,
                "label": label,
                "class_id": class_id,
            })

    return detections


def detect_in_video_frames(
    frames: list[dict],
    camera_id: str,
    confidence_threshold: float | None = None,
) -> list[dict]:
    """Run detection on a list of extracted video frames.

    Args:
        frames: List of frame dicts from video_processor.extract_frames().
        camera_id: Camera identifier.
        confidence_threshold: Minimum detection confidence.

    Returns:
        List of detection dicts enriched with camera_id and frame_number.
    """
    all_detections = []

    for frame_data in frames:
        frame = frame_data["frame"]
        frame_number = frame_data["frame_number"]

        detections = detect_objects(frame, confidence_threshold)

        for det in detections:
            det["camera_id"] = camera_id
            det["frame_number"] = frame_number
            det["timestamp_offset"] = frame_data.get("timestamp_offset", 0)
            all_detections.append(det)

    logger.info(
        f"Detected {len(all_detections)} objects across "
        f"{len(frames)} frames from camera {camera_id}"
    )
    return all_detections


def crop_detection(frame: np.ndarray, detection: dict) -> np.ndarray:
    """Crop a detected person/vehicle from the frame.

    Args:
        frame: Original frame.
        detection: Detection dict with bbox_x, bbox_y, bbox_w, bbox_h.

    Returns:
        Cropped image as numpy array.
    """
    x = int(detection["bbox_x"])
    y = int(detection["bbox_y"])
    w = int(detection["bbox_w"])
    h = int(detection["bbox_h"])

    # Clamp to frame boundaries
    x = max(0, x)
    y = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    crop = frame[y:y2, x:x2]
    return crop
