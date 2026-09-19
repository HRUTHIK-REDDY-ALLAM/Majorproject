"""Lightweight IoU-based multi-object tracker for surveillance frames.

Detectors drop objects for a frame or two. Treating each frame's raw count
as ground truth turns that flicker into fictitious "person entered / person
left" events. This tracker links detections across frames and tolerates
short gaps, so occupancy and movement reflect actual people rather than
detector noise.

Deliberately simple (greedy IoU association, no appearance model) so it has
no heavy dependencies and runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two bbox dicts."""
    ax1, ay1 = a["bbox_x"], a["bbox_y"]
    ax2, ay2 = ax1 + a["bbox_w"], ay1 + a["bbox_h"]
    bx1, by1 = b["bbox_x"], b["bbox_y"]
    bx2, by2 = bx1 + b["bbox_w"], by1 + b["bbox_h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A single object followed across frames."""

    track_id: int
    label: str
    first_frame: int
    last_frame: int
    first_time: datetime
    last_time: datetime
    bbox: dict
    positions: list[tuple[int, float, float]] = field(default_factory=list)
    hits: int = 1
    misses: int = 0

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame


def track_detections(
    frame_analyses: list[dict],
    label: str = "person",
    iou_threshold: float = 0.3,
    max_gap: int = 3,
    min_hits: int = 2,
) -> list[Track]:
    """Link per-frame detections into tracks.

    Args:
        frame_analyses: Ordered per-frame analysis dicts (must contain
            'detections', 'frame_number', 'timestamp').
        label: Which detection label to track.
        iou_threshold: Minimum IoU to associate a detection with a track.
        max_gap: How many consecutive frames a track may go undetected
            before it is closed. This is what absorbs detector flicker.
        min_hits: Tracks seen fewer times than this are discarded as noise.

    Returns:
        Confirmed tracks, ordered by first appearance.
    """
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1

    for idx, analysis in enumerate(frame_analyses):
        detections = [
            d for d in analysis.get("detections", []) if d.get("label") == label
        ]
        frame_number = analysis.get("frame_number", idx)
        timestamp = analysis.get("timestamp")

        unmatched = list(detections)
        # Greedy association: best IoU match wins, each detection used once.
        for track in active:
            best, best_iou = None, iou_threshold
            for det in unmatched:
                score = _iou(track.bbox, det)
                if score >= best_iou:
                    best, best_iou = det, score
            if best is not None:
                unmatched.remove(best)
                track.bbox = best
                track.last_frame = frame_number
                track.last_time = timestamp
                track.hits += 1
                track.misses = 0
                cx = best["bbox_x"] + best["bbox_w"] / 2
                cy = best["bbox_y"] + best["bbox_h"] / 2
                track.positions.append((frame_number, cx, cy))
            else:
                track.misses += 1

        # Any detection left over starts a new track.
        for det in unmatched:
            cx = det["bbox_x"] + det["bbox_w"] / 2
            cy = det["bbox_y"] + det["bbox_h"] / 2
            active.append(
                Track(
                    track_id=next_id,
                    label=label,
                    first_frame=frame_number,
                    last_frame=frame_number,
                    first_time=timestamp,
                    last_time=timestamp,
                    bbox=det,
                    positions=[(frame_number, cx, cy)],
                )
            )
            next_id += 1

        # Retire tracks that have been missing too long.
        still_active = []
        for track in active:
            if track.misses > max_gap:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

    finished.extend(active)
    confirmed = [t for t in finished if t.hits >= min_hits]
    confirmed.sort(key=lambda t: (t.first_frame, t.track_id))

    # Renumber so reports read Person 1, 2, 3… in order of appearance.
    for i, track in enumerate(confirmed, start=1):
        track.track_id = i
    return confirmed


def smoothed_occupancy(
    frame_analyses: list[dict],
    tracks: list[Track],
) -> list[int]:
    """Per-frame count of people, derived from tracks rather than raw
    detections, so a one-frame miss does not read as someone leaving."""
    counts = []
    for idx, analysis in enumerate(frame_analyses):
        frame_number = analysis.get("frame_number", idx)
        counts.append(
            sum(
                1
                for t in tracks
                if t.first_frame <= frame_number <= t.last_frame
            )
        )
    return counts
