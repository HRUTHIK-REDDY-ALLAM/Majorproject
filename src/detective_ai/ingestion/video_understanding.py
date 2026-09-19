"""LLM-driven video understanding.

Instead of captioning a handful of isolated frames, the video is split into
consecutive time segments covering its whole duration. Each segment is
rendered as a labelled contact sheet and shown to a vision model as one
image, so the model sees motion and can describe what actually happens
rather than guessing from a single still.

A final synthesis pass reads every segment description and reconciles them
into one account — in particular deciding how many distinct people appear,
which per-frame object detection is bad at (it counts the same person
repeatedly whenever tracking breaks).

Every step fails soft: callers fall back to detection-only evidence.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

from detective_ai.config import settings

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SEGMENT_PROMPT = """This image is a contact sheet: {n} frames taken in
chronological order from ONE security camera during a single {duration:.0f}
second window. Each tile is labelled with its timestamp. Read them in order
(left to right, top to bottom) as one continuous piece of footage.

Report, using only what is visible:

1. SUBJECTS: how many people are clearly visible in the foreground and
   actively doing something. The same person shown in several tiles is ONE
   person, not several. If unsure whether two tiles show the same person,
   say so.
2. BACKGROUND: separately, note whether other people are merely passing by
   or partly visible in the background (e.g. only legs or a torso). Count
   these SEPARATELY from the subjects — never merge the two numbers.
3. APPEARANCE: for each subject, what they look like (clothing colour,
   build, anything distinguishing). Do not guess age, gender, ethnicity or
   identity.
4. ACTIONS: what each subject does across the window, referencing the
   timestamps. Be concrete: walking, standing, reaching toward a shelf,
   bending down, handling an object, putting something into a bag or pocket.
5. OBJECTS: any object that is picked up, carried, concealed, put down or
   exchanged, and which subject handles it.
6. SETTING: where this appears to be (shop aisle, corridor, street, room).

Rules: describe only what is visible. Never state or imply intent, motive,
or that a crime occurred. If the footage is blurry or ambiguous, say so
plainly. Be concise - at most 140 words."""

_SYNTHESIS_PROMPT = """You are consolidating automated observations of a
single security camera recording. Below are descriptions of consecutive time
windows from the SAME camera, in chronological order.

Your job is to merge them into ONE coherent account. The critical task is
deciding how many DISTINCT people are actually involved. Two rules:

- A person described in several consecutive windows is almost certainly the
  SAME person continuing to move, not a new one. Only count a new person when
  the descriptions clearly show someone different (different clothing, or two
  people visible at the same moment).
- Count SUBJECTS separately from BACKGROUND people. Subjects are clearly
  visible in the foreground and doing something. Bystanders merely passing
  by, or visible only as legs/torsos at the edge, are background - they are
  NOT subjects. A recording of one shopper in a busy store has ONE subject,
  however many people drift through the background.

Respond with ONLY a JSON object:
{
  "distinct_people": <integer: SUBJECTS only, for the whole recording>,
  "background_people": "<none | a few passers-by | busy> - background only",
  "people_reasoning": "<one sentence on how you arrived at the subject count>",
  "people": [
    {"label": "Person 1", "appearance": "<what they look like>",
     "first_seen": "<timestamp>", "last_seen": "<timestamp>",
     "actions": "<what they did across the recording>"}
  ],
  "timeline": [
    {"time": "<timestamp>", "event": "<what was observed>"}
  ],
  "object_events": [
    "<any object picked up, carried, concealed, put down, or exchanged>"
  ],
  "setting": "<where this takes place>",
  "summary": "<3-5 sentence plain-language account of what the footage shows>",
  "uncertainties": ["<anything unclear, ambiguous or not visible>"]
}

Describe only what the observations support. Do NOT assert intent, motive,
or that a crime occurred - that judgement belongs to a later stage."""


@dataclass
class SegmentObservation:
    """A vision model's description of one time window of footage."""

    index: int
    start_time: datetime
    end_time: datetime
    start_offset: float
    end_offset: float
    description: str
    frame_count: int


class DailyQuotaError(RuntimeError):
    """Raised when the API's per-day token budget is spent.

    Distinct from a per-minute limit: waiting will not help, so the whole
    vision stage is abandoned at once instead of retrying every segment.
    """


def _is_daily_limit(response) -> bool:
    body = (response.text or "").lower()
    return "per day" in body or "tpd" in body or "rpd" in body


def _post_groq(payload: dict, what: str) -> dict | None:
    """POST to Groq with retry on per-minute token-budget exhaustion."""
    import httpx

    for attempt in range(3):
        try:
            response = httpx.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                if _is_daily_limit(response):
                    # A daily budget will not free up by waiting; retrying
                    # just stalls ingestion for minutes before failing anyway.
                    logger.error(
                        f"{what}: Groq daily token limit reached. Vision "
                        f"analysis unavailable until the quota resets."
                    )
                    raise DailyQuotaError(
                        "Groq daily token quota exhausted"
                    )
                if attempt < 2:
                    wait = _retry_after_seconds(response) or 20.0
                    logger.info(
                        f"{what}: rate limited, waiting {wait:.0f}s "
                        f"(attempt {attempt + 1}/3)."
                    )
                    time.sleep(wait)
                    continue

            logger.warning(
                f"{what}: HTTP {response.status_code} {response.text[:200]}"
            )
            return None
        except DailyQuotaError:
            raise
        except Exception as e:
            logger.warning(f"{what}: request failed: {e}")
            return None
    return None


def _retry_after_seconds(response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        raw = response.headers.get(header)
        if not raw:
            continue
        try:
            return min(45.0, max(1.0, float(str(raw).rstrip("s"))))
        except ValueError:
            continue
    return None


def is_available() -> bool:
    """Whether LLM video understanding is configured and enabled."""
    return bool(settings.vision_enabled and settings.groq_api_key)


def plan_segments(frame_count: int, duration: float) -> list[tuple[int, int]]:
    """Split frame indices into consecutive segments covering the whole video.

    Segment count scales with duration but is capped so a long recording
    stays within the API's per-minute token budget.
    """
    if frame_count <= 0:
        return []

    target = max(1.0, settings.video_segment_seconds)
    wanted = max(1, math.ceil(duration / target)) if duration > 0 else 1
    n_segments = max(1, min(settings.video_max_segments, wanted, frame_count))

    bounds = []
    per = frame_count / n_segments
    for i in range(n_segments):
        lo = int(round(i * per))
        hi = int(round((i + 1) * per)) if i < n_segments - 1 else frame_count
        if hi > lo:
            bounds.append((lo, hi))
    return bounds


def build_montage(
    tiles: list[tuple[float, np.ndarray]],
    cols: int = 3,
    cell_width: int = 440,
) -> np.ndarray:
    """Render timestamped frames as a single labelled contact sheet."""
    cells = []
    for offset, frame in tiles:
        h, w = frame.shape[:2]
        cell = cv2.resize(frame, (cell_width, max(1, int(h * cell_width / w))))
        # Timestamp banner so the model can anchor events in time.
        cv2.rectangle(cell, (0, 0), (cell_width, 30), (0, 0, 0), -1)
        cv2.putText(
            cell, f"t={offset:0.1f}s", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.rectangle(cell, (0, 0), (cell_width - 1, cell.shape[0] - 1),
                      (0, 255, 255), 2)
        cells.append(cell)

    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _encode(image: np.ndarray) -> str | None:
    try:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception as e:
        logger.warning(f"Montage encoding failed: {e}")
        return None


def describe_segment(
    montage: np.ndarray,
    tile_count: int,
    duration: float,
) -> str | None:
    """Ask the vision model to describe one segment's contact sheet."""
    b64 = _encode(montage)
    if b64 is None:
        return None

    prompt = _SEGMENT_PROMPT.format(n=tile_count, duration=max(duration, 1.0))
    payload = {
        "model": settings.vision_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 420,
        "temperature": 0.0,
    }

    data = _post_groq(payload, "Segment description")
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip() or None
    except (KeyError, IndexError):
        return None


def analyze_segments(
    frames: list[dict],
    start_time: datetime,
    delay_seconds: float = 1.0,
) -> list[SegmentObservation]:
    """Describe every segment of the video with the vision model.

    Args:
        frames: Extracted frames, each with 'frame' and 'timestamp_offset'.
        start_time: Absolute timestamp of the video start.
    """
    if not is_available() or not frames:
        return []

    duration = frames[-1]["timestamp_offset"] - frames[0]["timestamp_offset"]
    bounds = plan_segments(len(frames), duration)
    if not bounds:
        return []

    per_segment = max(1, settings.video_frames_per_segment)
    logger.info(
        f"Video understanding: {len(bounds)} segment(s) covering "
        f"{duration:.0f}s, up to {per_segment} frames each."
    )

    observations: list[SegmentObservation] = []
    for i, (lo, hi) in enumerate(bounds):
        window = frames[lo:hi]
        if not window:
            continue

        # Even spread within the segment so the montage shows progression.
        picks = np.linspace(0, len(window) - 1, min(per_segment, len(window)))
        tiles = [
            (window[int(p)]["timestamp_offset"], window[int(p)]["frame"])
            for p in picks
        ]

        seg_start = window[0]["timestamp_offset"]
        seg_end = window[-1]["timestamp_offset"]
        montage = build_montage(tiles)
        try:
            description = describe_segment(
                montage, len(tiles), max(0.0, seg_end - seg_start)
            )
        except DailyQuotaError:
            logger.error(
                f"Abandoning vision analysis after {len(observations)} of "
                f"{len(bounds)} segment(s): daily token quota exhausted."
            )
            break

        if description:
            observations.append(SegmentObservation(
                index=i,
                start_time=start_time + timedelta(seconds=seg_start),
                end_time=start_time + timedelta(seconds=seg_end),
                start_offset=seg_start,
                end_offset=seg_end,
                description=description,
                frame_count=len(tiles),
            ))
            logger.info(
                f"Segment {i + 1}/{len(bounds)} "
                f"({seg_start:.0f}-{seg_end:.0f}s): {description[:110]}"
            )
        else:
            logger.warning(f"Segment {i + 1}/{len(bounds)} could not be described.")

        if i < len(bounds) - 1:
            time.sleep(delay_seconds)

    logger.info(
        f"Video understanding: {len(observations)}/{len(bounds)} segments described."
    )
    return observations


def synthesize(observations: list[SegmentObservation]) -> dict | None:
    """Merge segment descriptions into one reconciled account."""
    if not observations:
        return None

    blocks = []
    for obs in observations:
        blocks.append(
            f"### Window {obs.index + 1} "
            f"({obs.start_time.strftime('%H:%M:%S')} - "
            f"{obs.end_time.strftime('%H:%M:%S')})\n{obs.description}"
        )

    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": _SYNTHESIS_PROMPT},
            {"role": "user", "content": "\n\n".join(blocks)},
        ],
        "max_tokens": 1800,
        "temperature": 0.0,
    }

    try:
        data = _post_groq(payload, "Video synthesis")
    except DailyQuotaError:
        return None
    if not data:
        return None

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return None

    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Could not parse video synthesis JSON: {e}")
        return None


def render_narrative(
    synthesis: dict | None,
    observations: list[SegmentObservation],
    camera_id: str,
    detector_note: str = "",
) -> str:
    """Render the analysis as the evidence text agents read.

    `synthesis` may be None when the reconciliation pass could not run; the
    per-window observations are still far better evidence than detections
    alone, so they are reported without a consolidated people count.
    """
    synthesis = synthesis or {}
    parts: list[str] = []

    first, last = observations[0], observations[-1]
    covered = last.end_offset - first.start_offset
    parts.append(
        f"Camera {camera_id} footage analysis: the full recording was reviewed "
        f"by a vision model in {len(observations)} consecutive time windows, "
        f"covering {first.start_time.strftime('%H:%M:%S')} to "
        f"{last.end_time.strftime('%H:%M:%S')} ({covered:.0f} seconds)."
    )

    if synthesis.get("setting"):
        parts.append(f"Setting: {synthesis['setting']}")

    if synthesis.get("summary"):
        parts.append("")
        parts.append("## What the footage shows")
        parts.append(str(synthesis["summary"]))

    count = synthesis.get("distinct_people")
    if isinstance(count, int):
        parts.append("")
        parts.append("## People")
        parts.append(
            f"- {count} person(s) are the subject of this recording — that is, "
            f"clearly visible and doing something."
        )
        if synthesis.get("background_people"):
            parts.append(
                f"- Background: {synthesis['background_people']}. Bystanders "
                f"passing through are NOT part of the subject count above and "
                f"should not be described as participants."
            )
        if synthesis.get("people_reasoning"):
            parts.append(f"- Basis for this count: {synthesis['people_reasoning']}")
    else:
        parts.append("")
        parts.append("## People")
        parts.append(
            "- The consolidation step did not complete, so no overall count of "
            "distinct people is available. Read the window observations below "
            "and note that the same person usually continues across "
            "consecutive windows."
        )

    for person in synthesis.get("people") or []:
        if not isinstance(person, dict):
            continue
        label = person.get("label", "Person")
        bits = []
        if person.get("appearance"):
            bits.append(str(person["appearance"]))
        seen = " to ".join(
            str(person[k]) for k in ("first_seen", "last_seen") if person.get(k)
        )
        if seen:
            bits.append(f"seen {seen}")
        if person.get("actions"):
            bits.append(str(person["actions"]))
        parts.append(f"- {label}: {'; '.join(bits)}")

    timeline = synthesis.get("timeline") or []
    if timeline:
        parts.append("")
        parts.append("## Timeline of observed events")
        for entry in timeline:
            if isinstance(entry, dict):
                parts.append(
                    f"- {entry.get('time', '?')}: {entry.get('event', '')}"
                )

    objects = synthesis.get("object_events") or []
    if objects:
        parts.append("")
        parts.append("## Object handling observed")
        for event in objects:
            parts.append(f"- {event}")

    parts.append("")
    parts.append("## Window-by-window observations")
    for obs in observations:
        parts.append(
            f"- [{obs.start_time.strftime('%H:%M:%S')}-"
            f"{obs.end_time.strftime('%H:%M:%S')}] {obs.description}"
        )

    parts.append("")
    parts.append("## Limitations of this analysis")
    parts.append(
        "- Descriptions come from a vision model reading sampled frames. It "
        "reports what is visible; it does NOT identify anyone. Person labels "
        "carry no identity."
    )
    parts.append(
        "- Observed actions do not establish intent, ownership of objects, or "
        "whether an offence occurred."
    )
    if detector_note:
        parts.append(f"- {detector_note}")
    for item in synthesis.get("uncertainties") or []:
        parts.append(f"- {item}")
    parts.append(
        "- Timestamps derive from the video start time given at upload and are "
        "only as accurate as that value."
    )

    return "\n".join(parts)
