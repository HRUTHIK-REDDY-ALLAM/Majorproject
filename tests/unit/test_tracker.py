"""Tests for the IoU tracker that stabilises per-frame detections."""

from __future__ import annotations

from datetime import datetime, timedelta

from detective_ai.cv.tracker import track_detections
from detective_ai.ingestion.video_processor import build_video_narrative

START = datetime(2026, 9, 18, 11, 0, 0)


def _person(x: float, y: float, conf: float = 0.95) -> dict:
    return {
        "bbox_x": x, "bbox_y": y, "bbox_w": 200.0, "bbox_h": 500.0,
        "confidence": conf, "label": "person", "class_id": 15,
        "class_name": "person",
    }


def _frames(drop_second_person_at: set[int], count: int = 30) -> list[dict]:
    frames = []
    for i in range(count):
        dets = [_person(200 + i * 30, 400)]
        if i not in drop_second_person_at:
            dets.append(_person(900 + i * 25, 420))
        frames.append({
            "detections": dets,
            "person_count": sum(1 for d in dets if d["label"] == "person"),
            "vehicle_count": 0,
            "animal_count": 0,
            "motion_score": 0.3,
            "lighting": "normally lit",
            "description": f"frame {i}",
            "quality": 0.7,
            "timestamp": START + timedelta(seconds=i),
            "frame_number": i * 15,
            "frame_w": 1920,
            "frame_h": 1080,
        })
    return frames


class TestTracking:
    def test_flicker_does_not_create_extra_tracks(self):
        """A detector missing someone for one frame must not read as that
        person leaving and a new person arriving."""
        frames = _frames(drop_second_person_at={5, 12, 19})
        tracks = track_detections(frames, label="person")
        assert len(tracks) == 2

    def test_tracks_survive_gap_shorter_than_max(self):
        frames = _frames(drop_second_person_at={10, 11})
        tracks = track_detections(frames, label="person")
        assert len(tracks) == 2

    def test_single_frame_noise_is_discarded(self):
        """A one-off spurious box should not become a tracked person."""
        frames = _frames(drop_second_person_at=set())
        frames[7]["detections"].append(_person(50, 50, conf=0.4))
        tracks = track_detections(frames, label="person")
        assert len(tracks) == 2

    def test_empty_input(self):
        assert track_detections([], label="person") == []


class TestNarrative:
    def test_narrative_uses_real_frame_dimensions(self):
        frames = _frames(drop_second_person_at=set())
        tracks = track_detections(frames, label="person")
        narrative = build_video_narrative(frames, "cam_1", tracks=tracks)
        assert "1920x1080" in narrative

    def test_narrative_reports_stable_occupancy(self):
        frames = _frames(drop_second_person_at={5, 12, 19})
        tracks = track_detections(frames, label="person")
        narrative = build_video_narrative(frames, "cam_1", tracks=tracks)
        assert "formed 2 movement track(s)" in narrative
        # The old implementation emitted an "entered/left the scene" line per
        # flicker; nothing should claim movement in and out of frame now.
        assert "entered the scene" not in narrative
        assert "left the scene" not in narrative

    def test_track_count_is_not_presented_as_a_people_count(self):
        """The detector splits one person into several tracks; the fallback
        narrative must never let that read as a number of people."""
        frames = _frames(drop_second_person_at=set())
        tracks = track_detections(frames, label="person")
        narrative = build_video_narrative(frames, "cam_1", tracks=tracks)
        assert "A track is NOT a person" in narrative
        assert "upper bound" in narrative
        assert "Do NOT report the track count as a number of people" in narrative

    def test_narrative_always_states_limitations(self):
        frames = _frames(drop_second_person_at=set())
        tracks = track_detections(frames, label="person")
        narrative = build_video_narrative(frames, "cam_1", tracks=tracks)
        assert "Limitations of this analysis" in narrative
        assert "does NOT identify who anyone is" in narrative

    def test_brief_detection_flagged_as_unreliable(self):
        frames = _frames(drop_second_person_at=set())
        frames[11]["detections"].append({
            "bbox_x": 50.0, "bbox_y": 50.0, "bbox_w": 60.0, "bbox_h": 60.0,
            "confidence": 0.36, "label": "vehicle", "class_id": 14,
            "class_name": "motorbike",
        })
        tracks = track_detections(frames, label="person")
        narrative = build_video_narrative(frames, "cam_1", tracks=tracks)
        assert "likely a false detection" in narrative

    def test_narrative_handles_no_frames(self):
        assert "No frames analyzed" in build_video_narrative([], "cam_1")
