"""Tests for LLM-driven video understanding (segmentation, montage, rendering)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from detective_ai.config import settings
from detective_ai.ingestion import video_understanding as vu

START = datetime(2026, 9, 18, 15, 30, 0)


def _observation(index: int, start_s: float, end_s: float, text: str):
    return vu.SegmentObservation(
        index=index,
        start_time=START + timedelta(seconds=start_s),
        end_time=START + timedelta(seconds=end_s),
        start_offset=start_s,
        end_offset=end_s,
        description=text,
        frame_count=6,
    )


class TestPlanSegments:
    def test_covers_every_frame_without_gaps_or_overlap(self):
        bounds = vu.plan_segments(100, 20)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 100
        for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
            assert prev_end == next_start

    def test_long_video_is_capped(self):
        bounds = vu.plan_segments(5000, 3600)
        assert len(bounds) == settings.video_max_segments
        assert bounds[-1][1] == 5000

    def test_short_video_gets_one_segment(self):
        assert vu.plan_segments(3, 1) == [(0, 3)]

    def test_empty_video(self):
        assert vu.plan_segments(0, 0) == []


class TestMontage:
    def test_grid_shape_and_labels(self):
        tiles = [(float(i), np.full((180, 320, 3), 40, np.uint8)) for i in range(6)]
        sheet = vu.build_montage(tiles, cols=3, cell_width=440)
        # 6 tiles at 3 per row -> 2 rows
        assert sheet.shape[1] == 440 * 3
        assert sheet.ndim == 3

    def test_partial_last_row_is_padded(self):
        tiles = [(float(i), np.full((180, 320, 3), 40, np.uint8)) for i in range(4)]
        sheet = vu.build_montage(tiles, cols=3, cell_width=200)
        assert sheet.shape[1] == 200 * 3


class TestRenderNarrative:
    def _obs(self):
        return [
            _observation(0, 0, 6, "One person walks in from the left."),
            _observation(1, 6, 12, "The same person reaches toward a shelf."),
        ]

    def test_uses_synthesised_people_count(self):
        synthesis = {
            "distinct_people": 1,
            "people_reasoning": "the same person continues across both windows",
            "summary": "A person enters and takes an item.",
            "setting": "shop aisle",
            "timeline": [{"time": "15:30:08", "event": "item taken from shelf"}],
            "object_events": ["a small item is taken from the shelf"],
            "uncertainties": ["the item is not clearly identifiable"],
        }
        text = vu.render_narrative(synthesis, self._obs(), "cam_1")
        assert "1 person(s) are the subject" in text
        assert "shop aisle" in text
        assert "item taken from shelf" in text
        assert "not clearly identifiable" in text

    def test_background_people_are_kept_out_of_the_subject_count(self):
        """A busy shop must not turn bystanders into participants."""
        text = vu.render_narrative(
            {"distinct_people": 1, "background_people": "a few passers-by"},
            self._obs(), "cam_1",
        )
        assert "1 person(s) are the subject" in text
        assert "a few passers-by" in text
        assert "NOT part of the subject count" in text

    def test_detector_count_is_marked_subordinate(self):
        text = vu.render_narrative(
            {"distinct_people": 1}, self._obs(), "cam_1",
            detector_note="The frame-by-frame object detector produced 4 person track(s); "
                          "this is an approximate machine count.",
        )
        assert "approximate machine count" in text

    def test_missing_synthesis_still_reports_observations(self):
        text = vu.render_narrative(None, self._obs(), "cam_1")
        assert "reaches toward a shelf" in text
        assert "did not complete" in text

    def test_always_states_limitations(self):
        text = vu.render_narrative({"distinct_people": 2}, self._obs(), "cam_1")
        assert "does NOT identify anyone" in text
        assert "do not establish intent" in text.lower()

    def test_window_observations_are_included(self):
        text = vu.render_narrative({"distinct_people": 1}, self._obs(), "cam_1")
        assert "Window-by-window observations" in text
        assert "walks in from the left" in text


class TestDailyQuotaDetection:
    class _Resp:
        def __init__(self, text):
            self.text = text
            self.headers = {}

    def test_per_day_limit_detected(self):
        r = self._Resp('{"error":{"message":"Rate limit reached ... on tokens per day (TPD): Limit 200000"}}')
        assert vu._is_daily_limit(r) is True

    def test_per_minute_limit_not_treated_as_daily(self):
        r = self._Resp('{"error":{"message":"Rate limit reached ... on input tokens per minute (ITPM): Limit 7000"}}')
        assert vu._is_daily_limit(r) is False
