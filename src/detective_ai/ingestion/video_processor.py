"""Video processing: frame extraction, scene analysis, and evidence creation.

Extracts frames at configurable FPS, runs object detection using the best
available detector (YOLO → DNN MobileNet-SSD → HOG fallback), builds rich
scene descriptions with movement tracking, and stores everything in the DB.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from detective_ai.config import settings
from detective_ai.core.enums import EvidenceType
from detective_ai.core.models import Evidence
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

# ── HOG Person Detector (last-resort fallback) ────────────────────────────────

_hog_detector = None


def _get_hog_detector():
    """Lazy-load the OpenCV HOG person detector."""
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("OpenCV HOG person detector initialized.")
    return _hog_detector


def _try_yolo_detect(frame: np.ndarray) -> list[dict] | None:
    """Try YOLO detection if ultralytics is installed; return None otherwise."""
    try:
        from detective_ai.cv.detector import detect_objects
        return detect_objects(frame)
    except (ImportError, Exception):
        return None


def _try_dnn_detect(frame: np.ndarray) -> list[dict] | None:
    """Try DNN MobileNet-SSD detection; return None if unavailable."""
    try:
        from detective_ai.cv.dnn_detector import detect_objects_dnn, is_available
        if not is_available():
            return None
        return detect_objects_dnn(frame, confidence_threshold=0.35)
    except Exception as e:
        logger.warning(f"DNN detection failed: {e}")
        return None


def detect_persons_hog(
    frame: np.ndarray,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """Detect persons in a frame using OpenCV HOG (last-resort fallback).

    Returns:
        List of detection dicts with bbox and confidence info.
    """
    hog = _get_hog_detector()

    # Resize for speed if needed
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 640:
        scale = 640 / w
        frame_resized = cv2.resize(frame, (640, int(h * scale)))
    else:
        frame_resized = frame

    boxes, weights = hog.detectMultiScale(
        frame_resized,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.02,
    )

    detections = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        conf = float(weights[i]) if i < len(weights) else 0.5
        if conf < confidence_threshold:
            continue

        # Scale back to original coordinates
        detections.append({
            "bbox_x": x / scale,
            "bbox_y": y / scale,
            "bbox_w": bw / scale,
            "bbox_h": bh / scale,
            "confidence": min(conf, 1.0),
            "label": "person",
            "class_id": 0,
            "class_name": "person",
        })

    return detections


def detect_objects_best(frame: np.ndarray) -> list[dict]:
    """Detect objects using the best available detector.

    Priority: YOLO → DNN MobileNet-SSD → HOG (person-only fallback).
    """
    # 1. Try YOLO (best, but requires ultralytics)
    detections = _try_yolo_detect(frame)
    if detections is not None:
        logger.debug(f"YOLO detected {len(detections)} objects")
        return detections

    # 2. Try DNN MobileNet-SSD (good, no extra deps)
    detections = _try_dnn_detect(frame)
    if detections is not None:
        logger.debug(f"DNN detected {len(detections)} objects")
        return detections

    # 3. Fall back to HOG (person-only, unreliable)
    detections = detect_persons_hog(frame)
    logger.debug(f"HOG detected {len(detections)} objects")
    return detections


# ── Scene Analysis ────────────────────────────────────────────────────────────


def _classify_frame_region(bbox_x: float, bbox_y: float,
                           bbox_w: float, bbox_h: float,
                           frame_w: int, frame_h: int) -> str:
    """Classify where in the frame a detection is located using bbox center."""
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2

    # Horizontal thirds
    if cx < frame_w * 0.33:
        h_pos = "left side"
    elif cx < frame_w * 0.66:
        h_pos = "center"
    else:
        h_pos = "right side"

    # Vertical thirds
    if cy < frame_h * 0.33:
        v_pos = "upper"
    elif cy < frame_h * 0.66:
        v_pos = "middle"
    else:
        v_pos = "lower"

    # Simplify
    if v_pos == "middle" and h_pos == "center":
        return "center of frame"
    return f"{v_pos}-{h_pos} of frame"


def _estimate_relative_size(bbox_w: float, bbox_h: float,
                            frame_w: int, frame_h: int) -> str:
    """Estimate if a detected object is close/far based on bounding box size."""
    area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)
    if area_ratio > 0.15:
        return "close to camera"
    elif area_ratio > 0.04:
        return "at moderate distance"
    else:
        return "far from camera"


def _describe_scene_brightness(frame: np.ndarray) -> str:
    """Describe the overall lighting of the scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    if brightness < 0.15:
        return "very dark/nighttime"
    elif brightness < 0.3:
        return "dimly lit"
    elif brightness > 0.85:
        return "very bright/overexposed"
    elif brightness > 0.7:
        return "brightly lit"
    else:
        return "normally lit"


def _compute_motion_score(prev_frame: np.ndarray | None,
                          curr_frame: np.ndarray) -> float:
    """Compute a motion score between two consecutive frames (0-1)."""
    if prev_frame is None:
        return 0.0

    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Resize for speed
        small_size = (160, 120)
        prev_small = cv2.resize(prev_gray, small_size)
        curr_small = cv2.resize(curr_gray, small_size)

        # Frame difference
        diff = cv2.absdiff(prev_small, curr_small)
        motion = np.mean(diff) / 255.0
        return min(1.0, motion * 5)  # Scale up for sensitivity
    except Exception:
        return 0.0


def analyze_frame(
    frame: np.ndarray,
    frame_number: int,
    timestamp: datetime,
    camera_id: str,
    prev_frame: np.ndarray | None = None,
) -> dict:
    """Analyze a single frame: detect objects, describe the scene richly.

    Returns a dict with detections list and a human-readable scene description.
    """
    h, w = frame.shape[:2]

    # Run detection using best available detector
    detections = detect_objects_best(frame)

    # Categorize detections
    person_count = sum(1 for d in detections if d.get("label") == "person")
    vehicle_count = sum(1 for d in detections if d.get("label") == "vehicle")
    animal_count = sum(1 for d in detections if d.get("label") == "animal")
    other_objects = [d for d in detections if d.get("label") not in ("person", "vehicle", "animal")]

    # Scene lighting
    lighting = _describe_scene_brightness(frame)

    # Motion between frames
    motion_score = _compute_motion_score(prev_frame, frame)
    motion_desc = ""
    if motion_score > 0.3:
        motion_desc = "significant movement detected"
    elif motion_score > 0.1:
        motion_desc = "some movement detected"

    # Build rich description parts
    parts = []

    # Describe people
    if person_count > 0:
        person_details = []
        for d in detections:
            if d.get("label") == "person":
                region = _classify_frame_region(
                    d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"], w, h
                )
                distance = _estimate_relative_size(d["bbox_w"], d["bbox_h"], w, h)
                conf_pct = int(d["confidence"] * 100)
                person_details.append(f"{region} ({distance}, {conf_pct}% conf)")

        if person_count == 1:
            parts.append(f"1 person detected at {person_details[0]}")
        else:
            details_str = "; ".join(person_details[:5])
            parts.append(f"{person_count} people detected: {details_str}")

    # Describe vehicles
    if vehicle_count > 0:
        vehicle_details = []
        for d in detections:
            if d.get("label") == "vehicle":
                class_name = d.get("class_name", "vehicle")
                region = _classify_frame_region(
                    d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"], w, h
                )
                vehicle_details.append(f"{class_name} at {region}")

        parts.append(f"{vehicle_count} vehicle(s): {', '.join(vehicle_details[:5])}")

    # Describe animals
    if animal_count > 0:
        animal_details = []
        for d in detections:
            if d.get("label") == "animal":
                class_name = d.get("class_name", "animal")
                animal_details.append(class_name)
        parts.append(f"Animal(s) detected: {', '.join(animal_details[:3])}")

    # Describe other notable objects
    if other_objects:
        obj_names = [d.get("class_name", d.get("label", "object")) for d in other_objects[:3]]
        parts.append(f"Other objects: {', '.join(obj_names)}")

    # No detections fallback
    if not detections:
        parts.append(f"no people or vehicles detected ({lighting} scene)")

    # Add motion info
    if motion_desc:
        parts.append(motion_desc)

    # Compute frame quality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 128.0
    resolution_score = min(1.0, (h * w) / (1920 * 1080))
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
    contrast_score = min(1.0, contrast)
    quality = round(0.3 * resolution_score + 0.4 * brightness_score + 0.3 * contrast_score, 3)

    time_str = timestamp.strftime("%H:%M:%S")
    description = f"[{time_str}] Camera {camera_id}, frame {frame_number}: {'; '.join(parts)}"

    return {
        "detections": detections,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "animal_count": animal_count,
        "other_object_count": len(other_objects),
        "motion_score": motion_score,
        "lighting": lighting,
        "description": description,
        "quality": quality,
        "timestamp": timestamp,
        "frame_number": frame_number,
    }


def build_video_narrative(frame_analyses: list[dict], camera_id: str) -> str:
    """Build a consolidated narrative from all frame analyses.

    Summarizes what was observed across all frames — people counts,
    movement patterns (appearing/disappearing between frames), vehicle
    activity, and notable events.
    """
    if not frame_analyses:
        return f"Camera {camera_id}: No frames analyzed."

    total_frames = len(frame_analyses)
    frames_with_people = [f for f in frame_analyses if f["person_count"] > 0]
    frames_with_vehicles = [f for f in frame_analyses if f["vehicle_count"] > 0]
    frames_with_animals = [f for f in frame_analyses if f.get("animal_count", 0) > 0]
    max_people = max((f["person_count"] for f in frame_analyses), default=0)
    max_vehicles = max((f["vehicle_count"] for f in frame_analyses), default=0)

    # Compute overall motion
    motion_scores = [f.get("motion_score", 0) for f in frame_analyses]
    avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0

    parts = [f"Camera {camera_id} footage analysis ({total_frames} frames examined):"]

    # Overall scene description
    lighting_modes = [f.get("lighting", "unknown") for f in frame_analyses]
    most_common_lighting = max(set(lighting_modes), key=lighting_modes.count)
    parts.append(f"Scene lighting: {most_common_lighting}.")

    if avg_motion > 0.2:
        parts.append("Overall: significant activity and movement throughout the footage.")
    elif avg_motion > 0.05:
        parts.append("Overall: moderate activity detected in the footage.")
    else:
        parts.append("Overall: relatively static scene with minimal movement.")

    if not frames_with_people and not frames_with_vehicles:
        parts.append("No people or vehicles were detected in any frame.")
        if frames_with_animals:
            animal_types = set()
            for f in frames_with_animals:
                for d in f["detections"]:
                    if d.get("label") == "animal":
                        animal_types.add(d.get("class_name", "animal"))
            parts.append(f"Animals detected: {', '.join(animal_types)}")
        return "\n".join(parts)

    # ── People summary ────────────────────────────────────────────────────
    if frames_with_people:
        first_seen = frames_with_people[0]
        last_seen = frames_with_people[-1]
        parts.append(
            f"People detected in {len(frames_with_people)} of {total_frames} frames. "
            f"Maximum {max_people} person(s) visible at once."
        )
        parts.append(
            f"First person appeared at {first_seen['timestamp'].strftime('%H:%M:%S')} "
            f"(frame {first_seen['frame_number']})."
        )
        if last_seen != first_seen:
            parts.append(
                f"Last person visible at {last_seen['timestamp'].strftime('%H:%M:%S')} "
                f"(frame {last_seen['frame_number']})."
            )

        # Track movement: person count changes over time
        person_timeline = [
            (f["timestamp"].strftime("%H:%M:%S"), f["person_count"])
            for f in frame_analyses
        ]
        changes = []
        for i in range(1, len(person_timeline)):
            prev_count = person_timeline[i - 1][1]
            curr_count = person_timeline[i][1]
            if curr_count > prev_count:
                changes.append(
                    f"At {person_timeline[i][0]}: {curr_count - prev_count} person(s) "
                    f"entered the scene (total now {curr_count})"
                )
            elif curr_count < prev_count:
                changes.append(
                    f"At {person_timeline[i][0]}: {prev_count - curr_count} person(s) "
                    f"left the scene (total now {curr_count})"
                )

        if changes:
            parts.append("Movement timeline:")
            for ch in changes[:15]:
                parts.append(f"  - {ch}")

        # Track where people were seen (location changes)
        location_changes = []
        for i in range(1, len(frames_with_people)):
            prev_frame = frames_with_people[i - 1]
            curr_frame = frames_with_people[i]

            prev_locations = set()
            for d in prev_frame["detections"]:
                if d.get("label") == "person":
                    loc = _classify_frame_region(
                        d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"],
                        640, 480,  # approximate — we don't have frame dimensions here
                    )
                    prev_locations.add(loc)

            curr_locations = set()
            for d in curr_frame["detections"]:
                if d.get("label") == "person":
                    loc = _classify_frame_region(
                        d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"],
                        640, 480,
                    )
                    curr_locations.add(loc)

            if prev_locations and curr_locations and prev_locations != curr_locations:
                time_str = curr_frame["timestamp"].strftime("%H:%M:%S")
                from_locs = ", ".join(prev_locations)
                to_locs = ", ".join(curr_locations)
                location_changes.append(
                    f"At {time_str}: person(s) moved from {from_locs} to {to_locs}"
                )

        if location_changes:
            parts.append("Position tracking:")
            for lc in location_changes[:10]:
                parts.append(f"  - {lc}")

    # ── Vehicle summary ───────────────────────────────────────────────────
    if frames_with_vehicles:
        parts.append(
            f"Vehicles detected in {len(frames_with_vehicles)} of {total_frames} frames "
            f"(max {max_vehicles} at once)."
        )

        # List vehicle types seen
        vehicle_types = set()
        for f in frames_with_vehicles:
            for d in f["detections"]:
                if d.get("label") == "vehicle":
                    vehicle_types.add(d.get("class_name", "vehicle"))
        if vehicle_types:
            parts.append(f"Vehicle types observed: {', '.join(sorted(vehicle_types))}.")

        # Vehicle timeline
        vehicle_timeline = []
        for i in range(1, len(frame_analyses)):
            prev_v = frame_analyses[i - 1]["vehicle_count"]
            curr_v = frame_analyses[i]["vehicle_count"]
            if curr_v > prev_v:
                vehicle_timeline.append(
                    f"At {frame_analyses[i]['timestamp'].strftime('%H:%M:%S')}: "
                    f"{curr_v - prev_v} vehicle(s) appeared"
                )
            elif curr_v < prev_v:
                vehicle_timeline.append(
                    f"At {frame_analyses[i]['timestamp'].strftime('%H:%M:%S')}: "
                    f"{prev_v - curr_v} vehicle(s) left"
                )
        if vehicle_timeline:
            parts.append("Vehicle activity:")
            for vt in vehicle_timeline[:10]:
                parts.append(f"  - {vt}")

    # ── Animal summary ────────────────────────────────────────────────────
    if frames_with_animals:
        animal_types = set()
        for f in frames_with_animals:
            for d in f["detections"]:
                if d.get("label") == "animal":
                    animal_types.add(d.get("class_name", "animal"))
        parts.append(f"Animals seen: {', '.join(sorted(animal_types))} "
                     f"(in {len(frames_with_animals)} frames).")

    # ── Key frame details ─────────────────────────────────────────────────
    key_frames = []
    if frames_with_people:
        key_frames.append(frames_with_people[0])
        peak_frame = max(frames_with_people, key=lambda f: f["person_count"])
        if peak_frame != frames_with_people[0]:
            key_frames.append(peak_frame)
        if frames_with_people[-1] not in key_frames:
            key_frames.append(frames_with_people[-1])
    elif frames_with_vehicles:
        key_frames.append(frames_with_vehicles[0])

    # Also include high-motion frames
    high_motion_frames = [
        f for f in frame_analyses
        if f.get("motion_score", 0) > 0.3 and f not in key_frames
    ]
    key_frames.extend(high_motion_frames[:3])

    if key_frames:
        parts.append("Key observations:")
        for kf in key_frames:
            parts.append(f"  - {kf['description']}")

    return "\n".join(parts)


# ── Frame Extraction ──────────────────────────────────────────────────────────


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


# ── Main Processing Pipeline ─────────────────────────────────────────────────


def process_video(
    video_path: str | Path,
    camera_id: str,
    start_time: datetime,
    case_id: str = "",
    fps: int | None = None,
    max_frames: int | None = None,
) -> list[Evidence]:
    """Process a video file: extract frames, detect people, create evidence.

    This is the main entry point. It:
    1. Extracts frames at the configured FPS
    2. Runs object detection on each frame (YOLO → DNN → HOG fallback)
    3. Builds per-frame scene descriptions with motion analysis
    4. Creates a consolidated video narrative
    5. Stores individual frame evidence + one summary evidence item

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
    from detective_ai.ingestion.embeddings import embed_texts

    frames = extract_frames(video_path, fps=fps, max_frames=max_frames)
    if not frames:
        logger.warning(f"No frames extracted from {video_path}")
        return []

    # ── Step 1: Analyze each frame ────────────────────────────────────────
    logger.info(f"Analyzing {len(frames)} frames from {Path(video_path).name}…")
    frame_analyses = []
    all_detections = []
    prev_frame_data = None

    for frame_data in frames:
        timestamp = start_time + timedelta(seconds=frame_data["timestamp_offset"])
        analysis = analyze_frame(
            frame=frame_data["frame"],
            frame_number=frame_data["frame_number"],
            timestamp=timestamp,
            camera_id=camera_id,
            prev_frame=prev_frame_data,
        )
        frame_analyses.append(analysis)
        prev_frame_data = frame_data["frame"]

        # Enrich detections with frame-level metadata
        for det in analysis["detections"]:
            det["camera_id"] = camera_id
            det["frame_number"] = frame_data["frame_number"]
            det["timestamp"] = timestamp
            all_detections.append(det)

    # Log detection summary
    total_persons = sum(a["person_count"] for a in frame_analyses)
    total_vehicles = sum(a["vehicle_count"] for a in frame_analyses)
    logger.info(
        f"Detection summary: {total_persons} person detections, "
        f"{total_vehicles} vehicle detections across {len(frames)} frames"
    )

    # ── Step 2: Build video narrative ─────────────────────────────────────
    narrative = build_video_narrative(frame_analyses, camera_id)
    logger.info(f"Video narrative:\n{narrative}")

    # ── Step 3: Build descriptions for embedding ─────────────────────────
    descriptions = [a["description"] for a in frame_analyses]
    descriptions.append(narrative)  # also embed the narrative

    logger.info(f"Batch-embedding {len(descriptions)} descriptions…")
    embeddings = embed_texts(descriptions)

    # ── Step 4: Store individual frame evidence ──────────────────────────
    evidence_items = []

    for i, analysis in enumerate(frame_analyses):
        evidence = Evidence(
            type=EvidenceType.VIDEO_FRAME,
            source=camera_id,
            timestamp=analysis["timestamp"],
            confidence_score=analysis["quality"],
            description=analysis["description"],
            metadata={
                "camera_id": camera_id,
                "frame_number": analysis["frame_number"],
                "case_id": case_id,
                "person_count": analysis["person_count"],
                "vehicle_count": analysis["vehicle_count"],
                "animal_count": analysis.get("animal_count", 0),
                "motion_score": analysis.get("motion_score", 0),
                "lighting": analysis.get("lighting", "unknown"),
                "resolution": f"{frames[i]['frame'].shape[1]}x{frames[i]['frame'].shape[0]}",
            },
        )

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
                embedding=embeddings[i] if i < len(embeddings) else None,
            )

        evidence_items.append(evidence)

    # ── Step 5: Store visual detections ───────────────────────────────────
    det_count = 0
    for det in all_detections:
        try:
            det_id = str(uuid.uuid4())
            # Find the evidence item for this frame
            frame_evidence = next(
                (e for e in evidence_items
                 if e.metadata.get("frame_number") == det["frame_number"]),
                evidence_items[0] if evidence_items else None,
            )
            if frame_evidence:
                with db.session() as session:
                    db.insert_detection(
                        session,
                        id=det_id,
                        evidence_id=frame_evidence.id,
                        camera_id=det["camera_id"],
                        frame_number=det["frame_number"],
                        timestamp=det["timestamp"],
                        bbox_x=det["bbox_x"],
                        bbox_y=det["bbox_y"],
                        bbox_w=det["bbox_w"],
                        bbox_h=det["bbox_h"],
                        detection_confidence=det["confidence"],
                        label=det.get("label", "person"),
                    )
                det_count += 1
        except Exception as e:
            logger.warning(f"Failed to store detection: {e}")

    # ── Step 6: Store video analysis summary ─────────────────────────────
    summary_evidence = Evidence(
        type=EvidenceType.VIDEO_FRAME,  # reuse type; metadata marks it as summary
        source=camera_id,
        timestamp=start_time,
        confidence_score=0.9,
        description=narrative,
        metadata={
            "camera_id": camera_id,
            "case_id": case_id,
            "is_video_summary": True,
            "total_frames_analyzed": len(frame_analyses),
            "total_persons_detected": total_persons,
            "total_vehicles_detected": total_vehicles,
            "total_detections_stored": det_count,
        },
    )
    with db.session() as session:
        db.insert_evidence(
            session,
            id=summary_evidence.id,
            type=summary_evidence.type.value,
            source=summary_evidence.source,
            timestamp=summary_evidence.timestamp,
            confidence_score=summary_evidence.confidence_score,
            description=summary_evidence.description,
            metadata_=summary_evidence.metadata,
            embedding=embeddings[-1] if embeddings else None,
        )
    evidence_items.append(summary_evidence)

    logger.info(
        f"Processed video {Path(video_path).name}: "
        f"{len(evidence_items)} evidence items, {det_count} detections stored "
        f"for camera {camera_id}"
    )
    return evidence_items
