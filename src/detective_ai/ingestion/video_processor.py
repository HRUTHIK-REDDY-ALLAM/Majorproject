"""Video processing: frame extraction, quality scoring, and embedding.

Extracts frames at configurable FPS, scores capture quality,
generates embeddings, and stores everything in PostgreSQL + pgvector.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from detective_ai.config import settings
from detective_ai.core.enums import EvidenceType
from detective_ai.core.models import Evidence
from detective_ai.ingestion.embeddings import embed_text
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    fps: int | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """Extract frames from a video file at a given FPS rate.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract. Defaults to config value.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of dicts with 'frame' (numpy array), 'frame_number', 'timestamp_offset'.
    """
    fps = fps or settings.frame_extraction_fps
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_offset = frame_idx / video_fps
            frames.append({
                "frame": frame,
                "frame_number": frame_idx,
                "timestamp_offset": timestamp_offset,
            })

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {len(frames)} frames from {video_path.name} "
        f"(total: {total_frames}, interval: {frame_interval})"
    )
    return frames


def compute_capture_confidence(frame: np.ndarray) -> float:
    """Score the quality of a captured frame (0-1).

    Considers resolution, brightness, and contrast as proxies for
    how reliable visual evidence from this frame would be.
    """
    h, w = frame.shape[:2]

    # Resolution score (higher is better, normalized to 1080p)
    resolution_score = min(1.0, (h * w) / (1920 * 1080))

    # Brightness score (histogram analysis)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray) / 255.0
    # Penalize very dark or very bright (washed out)
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2

    # Contrast score (standard deviation of pixel values)
    contrast = np.std(gray) / 128.0
    contrast_score = min(1.0, contrast)

    # Weighted combination
    confidence = (
        0.3 * resolution_score
        + 0.4 * brightness_score
        + 0.3 * contrast_score
    )
    return round(max(0.0, min(1.0, confidence)), 3)


def process_video(
    video_path: str | Path,
    camera_id: str,
    start_time: datetime,
    case_id: str = "",
    fps: int | None = None,
    max_frames: int | None = None,
) -> list[Evidence]:
    """Process a video file: extract frames, score quality, create evidence records.

    Args:
        video_path: Path to the video file.
        camera_id: Identifier for the source camera.
        start_time: Absolute timestamp of the video start.
        case_id: Investigation case ID.
        fps: Extraction rate.
        max_frames: Limit on extracted frames.

    Returns:
        List of Evidence objects created.
    """
    frames = extract_frames(video_path, fps=fps, max_frames=max_frames)
    evidence_items = []

    for frame_data in frames:
        timestamp = start_time + timedelta(seconds=frame_data["timestamp_offset"])
        confidence = compute_capture_confidence(frame_data["frame"])

        # Create a text description for embedding
        description = (
            f"Video frame from camera {camera_id} at {timestamp.isoformat()}, "
            f"frame {frame_data['frame_number']}, quality score {confidence:.2f}"
        )
        embedding = embed_text(description)

        evidence = Evidence(
            type=EvidenceType.VIDEO_FRAME,
            source=camera_id,
            timestamp=timestamp,
            confidence_score=confidence,
            description=description,
            metadata={
                "camera_id": camera_id,
                "frame_number": frame_data["frame_number"],
                "case_id": case_id,
                "resolution": f"{frame_data['frame'].shape[1]}x{frame_data['frame'].shape[0]}",
            },
        )

        # Store in database
        with db.session() as session:
            db.insert_evidence(
                session,
                id=evidence.id,
                type=evidence.type.value,
                source=evidence.source,
                timestamp=evidence.timestamp,
                confidence_score=evidence.confidence_score,
                description=evidence.description,
                metadata_=evidence.metadata,
                embedding=embedding,
            )

        evidence_items.append(evidence)

    logger.info(
        f"Processed video {Path(video_path).name}: "
        f"{len(evidence_items)} evidence items created for camera {camera_id}"
    )
    return evidence_items
