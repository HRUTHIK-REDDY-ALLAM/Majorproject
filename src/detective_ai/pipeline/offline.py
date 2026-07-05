"""Deterministic offline investigation pipeline.

Runs the full investigative reasoning loop — hypothesis generation,
evidence weighing, blind-spot gap inference, adversarial critique,
verification, and report assembly — using the same core components as
the LLM pipeline (HypothesisTracker, GapFiller, ConfidenceEngine) but
with rule-based agents instead of LLM calls.

Used for: benchmark evaluation without API rate limits, CI, and as a
reproducible baseline the LLM pipeline is compared against.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from detective_ai.core.enums import MovementType
from detective_ai.core.models import Hypothesis, HypothesisEvidence
from detective_ai.hypothesis.confidence import ConfidenceEngine
from detective_ai.hypothesis.tracker import HypothesisTracker
from detective_ai.trajectory.camera_topology import CameraTopology
from detective_ai.trajectory.gap_filler import GapFiller

logger = logging.getLogger(__name__)

# A witness statement below this reliability, with no camera corroboration,
# triggers a critic objection instead of supporting a hypothesis outright.
LOW_RELIABILITY = 0.6


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


class OfflineInvestigationPipeline:
    """Rule-based end-to-end investigation over structured evidence.

    Input evidence is the same structure the ingestion layer produces:
    camera sightings (with resolved identity clusters), access logs, and
    witness statements, plus the camera topology.
    """

    def __init__(
        self,
        topology: dict[str, Any] | CameraTopology,
        crime_scene_camera: str = "cam_server_room",
        crime_window_minutes: int = 90,
    ) -> None:
        if isinstance(topology, CameraTopology):
            self.topology = topology
        else:
            self.topology = CameraTopology.from_dict(topology)
        self.crime_scene_camera = crime_scene_camera
        self.crime_window_minutes = crime_window_minutes
        self.confidence_engine = ConfidenceEngine()

    # ── Main entry point ──────────────────────────────────────

    def investigate(
        self,
        case_id: str,
        case_title: str,
        camera_sightings: list[dict],
        access_logs: list[dict],
        witness_statements: list[dict],
        case_description: str = "",
    ) -> dict[str, Any]:
        """Run the full investigation and return a structured report."""
        tracker = HypothesisTracker(prune_threshold=0.15, max_active=5)

        people = self._build_people_directory(camera_sightings, access_logs, witness_statements)
        sightings_by_person = self._group_sightings(camera_sightings)
        crime_time = self._estimate_crime_time(camera_sightings, access_logs)

        # Phase 1 — Investigator: propose hypotheses from every lead
        self._propose_hypotheses(
            tracker, case_id, people, sightings_by_person,
            access_logs, witness_statements, crime_time,
        )

        # Phase 2 — Trajectory: reconstruct the leading suspect's movement,
        # inferring blind-spot gaps with bounded uncertainty
        leading = tracker.get_leading_hypothesis()
        trajectory_segments = []
        if leading and leading.suspect_identity:
            suspect_sightings = sorted(
                sightings_by_person.get(leading.suspect_identity, []),
                key=lambda s: _parse_ts(s["timestamp"]),
            )
            gap_filler = GapFiller(self.topology)
            trajectory_segments = gap_filler.fill_gaps_in_timeline(
                suspect_sightings, identity_cluster_id=leading.suspect_identity,
                case_id=case_id,
            )

        # Phase 3 — Critic: adversarial review of every active hypothesis
        objections = self._critic_review(
            tracker, people, sightings_by_person, witness_statements,
            trajectory_segments, crime_time,
        )

        # Phase 4 — Verifier + Reporter
        leading = tracker.get_leading_hypothesis()
        report = self._assemble_report(
            case_id, case_title, case_description, tracker, leading,
            trajectory_segments, objections, people, sightings_by_person,
            camera_sightings, access_logs,
        )
        return report

    # ── Phase 1: hypothesis generation ────────────────────────

    def _build_people_directory(
        self, sightings: list[dict], access_logs: list[dict], statements: list[dict],
    ) -> dict[str, str]:
        """Map person_id → display name from evidence (no ground truth)."""
        people: dict[str, str] = {}
        for log in access_logs:
            if log.get("person_id"):
                people[log["person_id"]] = log.get("person_name", log["person_id"])
        for s in sightings:
            pid = s.get("person_id") or ""
            people.setdefault(pid, pid)
        people.pop("", None)
        return people

    def _group_sightings(self, sightings: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for s in sightings:
            if s.get("person_id"):
                grouped[s["person_id"]].append(s)
        return grouped

    def _estimate_crime_time(
        self, sightings: list[dict], access_logs: list[dict]
    ) -> datetime | None:
        """Estimate the incident time from activity at the crime-scene camera."""
        times = [
            _parse_ts(s["timestamp"]) for s in sightings
            if s.get("camera_id") == self.crime_scene_camera
        ]
        times += [
            _parse_ts(log["timestamp"]) for log in access_logs
            if self.crime_scene_camera.replace("cam_", "").replace("_", " ")
            in log.get("location", "").lower()
        ]
        return min(times) if times else None

    def _propose_hypotheses(
        self,
        tracker: HypothesisTracker,
        case_id: str,
        people: dict[str, str],
        sightings_by_person: dict[str, list[dict]],
        access_logs: list[dict],
        witness_statements: list[dict],
        crime_time: datetime | None,
    ) -> None:
        """One hypothesis per person with any link to the crime scene.

        Evidence is attached to the hypothesis and its confidence computed
        BEFORE registering it with the tracker, so pruning decisions are
        always made on a fully-evidenced branch.
        """
        scene_location = self.crime_scene_camera.replace("cam_", "").replace("_", " ")

        for person_id, name in people.items():
            supporting: list[HypothesisEvidence] = []
            contradicting: list[HypothesisEvidence] = []

            # Camera sightings at the crime scene: strong support
            for s in sightings_by_person.get(person_id, []):
                if s.get("camera_id") == self.crime_scene_camera:
                    supporting.append(HypothesisEvidence(
                        evidence_id=f"sighting:{s['camera_id']}:{s['timestamp']}",
                        relationship="supports",
                        weight=2.0 * float(s.get("detection_confidence", 0.8)),
                        reasoning=f"Camera {s['camera_id']} detected {name} at the scene.",
                    ))

            # Blind-spot avoidance behaviour (e.g. stairwell instead of elevator)
            for s in sightings_by_person.get(person_id, []):
                if "stairwell" in s.get("camera_id", ""):
                    supporting.append(HypothesisEvidence(
                        evidence_id=f"sighting:{s['camera_id']}:{s['timestamp']}",
                        relationship="supports",
                        weight=0.8,
                        reasoning=f"{name} used a route through a camera blind spot.",
                    ))

            # Access logs at the scene: strong support
            for log in access_logs:
                if log.get("person_id") == person_id and scene_location in log.get(
                    "location", ""
                ).lower():
                    supporting.append(HypothesisEvidence(
                        evidence_id=f"access:{log['location']}:{log['timestamp']}",
                        relationship="supports",
                        weight=2.0,
                        reasoning=f"Badge log places {name} at {log['location']}.",
                    ))

            # Witness statements naming this person
            for stmt in witness_statements:
                text_lower = stmt.get("text", "").lower()
                if name.lower() in text_lower:
                    reliability = float(stmt.get("reliability_score", 0.7))
                    supporting.append(HypothesisEvidence(
                        evidence_id=f"statement:{stmt.get('source', 'unknown')}",
                        relationship="supports",
                        weight=reliability,
                        reasoning=f"Witness {stmt.get('source')} mentioned {name}.",
                    ))

            # Alibi: sighted somewhere else at the crime time contradicts placement
            if crime_time is not None:
                for s in sightings_by_person.get(person_id, []):
                    ts = _parse_ts(s["timestamp"])
                    if (
                        s.get("camera_id") != self.crime_scene_camera
                        and abs((ts - crime_time).total_seconds()) < 300
                        and not self._can_reach(s["camera_id"], self.crime_scene_camera,
                                                abs((ts - crime_time).total_seconds()))
                    ):
                        contradicting.append(HypothesisEvidence(
                            evidence_id=f"sighting:{s['camera_id']}:{s['timestamp']}",
                            relationship="contradicts",
                            weight=1.5,
                            reasoning=(
                                f"{name} was seen at {s['camera_id']} too close to the "
                                f"incident time to have reached the scene."
                            ),
                        ))

            h = Hypothesis(
                case_id=case_id,
                title=f"{name} ({person_id}) accessed the {scene_location}",
                description=(
                    f"Theory that {name} is responsible, based on placement "
                    f"at or near the {scene_location} around the incident."
                ),
                suspect_identity=person_id,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
            )
            n_items = len(supporting) + len(contradicting)
            h.confidence = self.confidence_engine.compute_confidence(
                [e.weight for e in supporting],
                [e.weight for e in contradicting],
                [0.85] * n_items,
                [0.9] * n_items,
            ) if n_items else 0.2
            tracker.add_hypothesis(h)

        # No-evidence hypotheses decay: anyone never linked to the scene at all
        for h in list(tracker.get_active_hypotheses()):
            if not h.supporting_evidence:
                tracker.prune_hypothesis(
                    h.id, "No evidence places this person at or near the scene.",
                )

    def _can_reach(self, from_cam: str, to_cam: str, seconds_available: float) -> bool:
        path = self.topology.get_shortest_path(from_cam, to_cam)
        if not path:
            return False
        return self.topology.get_path_travel_time(path) <= seconds_available

    # ── Phase 3: adversarial critic ───────────────────────────

    def _critic_review(
        self,
        tracker: HypothesisTracker,
        people: dict[str, str],
        sightings_by_person: dict[str, list[dict]],
        witness_statements: list[dict],
        trajectory_segments: list,
        crime_time: datetime | None,
    ) -> list[dict[str, Any]]:
        """Attack every active hypothesis; prune the ones that fail review."""
        objections: list[dict[str, Any]] = []

        for h in list(tracker.get_active_hypotheses()):
            name = people.get(h.suspect_identity or "", "")
            scene_sightings = [
                s for s in sightings_by_person.get(h.suspect_identity or "", [])
                if s.get("camera_id") == self.crime_scene_camera
            ]
            witness_support = [
                e for e in h.supporting_evidence if e.evidence_id.startswith("statement:")
            ]
            hard_support = [
                e for e in h.supporting_evidence if not e.evidence_id.startswith("statement:")
            ]

            # Attack 1: hypothesis rests only on witness memory, and the witness
            # is unreliable — the classic planted false lead.
            if witness_support and not scene_sightings and not hard_support:
                low_rel = [
                    stmt for stmt in witness_statements
                    if name and name.lower() in stmt.get("text", "").lower()
                    and float(stmt.get("reliability_score", 0.7)) < LOW_RELIABILITY
                ]
                objection = {
                    "hypothesis_id": h.id,
                    "severity": "CRITICAL",
                    "target_claim": h.title,
                    "objection_text": (
                        f"The case against {name} rests entirely on witness memory"
                        + (
                            f" of low assessed reliability "
                            f"({low_rel[0].get('reliability_score')})" if low_rel else ""
                        )
                        + f"; no camera or badge evidence places {name} at the scene."
                    ),
                    "alternative_explanation": (
                        f"The witness misidentified the person; {name} was never "
                        f"observed by any camera at the crime scene."
                    ),
                    "resolved": True,
                    "resolution_text": (
                        f"Resolved against the hypothesis: camera coverage of the scene "
                        f"never recorded {name}. Hypothesis rejected as a false lead."
                    ),
                }
                objections.append(objection)
                tracker.prune_hypothesis(
                    h.id,
                    f"False lead: only uncorroborated witness testimony implicates "
                    f"{name}; no camera or access-log evidence supports it.",
                )
                continue

            # Attack 2: low-confidence inferred movement in the leading theory
            weak_inferred = [
                s for s in trajectory_segments
                if h.suspect_identity == getattr(s, "identity_cluster_id", None)
                and s.movement_type == MovementType.INFERRED
                and s.confidence < 0.5
            ]
            for seg in weak_inferred:
                objections.append({
                    "hypothesis_id": h.id,
                    "severity": "MAJOR",
                    "target_claim": (
                        f"Movement {seg.from_camera} → {seg.to_camera} is inferred, "
                        f"not observed."
                    ),
                    "objection_text": (
                        f"The route between {seg.from_camera} and {seg.to_camera} "
                        f"passes through a blind spot and is inferred with only "
                        f"{seg.confidence:.0%} confidence."
                    ),
                    "alternative_explanation": (
                        "The person may have taken a different unobserved route or "
                        "left coverage entirely."
                    ),
                    "resolved": False,
                    "resolution_text": "",
                })

            # Attack 3: no badge corroboration for the leading suspect
            badge_support = [
                e for e in h.supporting_evidence if e.evidence_id.startswith("access:")
            ]
            if scene_sightings and not badge_support:
                objections.append({
                    "hypothesis_id": h.id,
                    "severity": "MINOR",
                    "target_claim": h.title,
                    "objection_text": (
                        f"Camera evidence places {name} at the scene but no badge "
                        f"entry was logged — entry method is unexplained."
                    ),
                    "alternative_explanation": "Tailgating or a propped door.",
                    "resolved": False,
                    "resolution_text": "",
                })

        return objections

    # ── Phase 4: verification + report ────────────────────────

    def _assemble_report(
        self,
        case_id: str,
        case_title: str,
        case_description: str,
        tracker: HypothesisTracker,
        leading: Hypothesis | None,
        trajectory_segments: list,
        objections: list[dict[str, Any]],
        people: dict[str, str],
        sightings_by_person: dict[str, list[dict]],
        camera_sightings: list[dict],
        access_logs: list[dict],
    ) -> dict[str, Any]:
        # Timeline: every observed sighting of the leading suspect plus
        # inferred blind-spot segments, each citing its evidence
        timeline: list[dict[str, Any]] = []
        if leading and leading.suspect_identity:
            for s in sorted(
                sightings_by_person.get(leading.suspect_identity, []),
                key=lambda x: _parse_ts(x["timestamp"]),
            ):
                timeline.append({
                    "time": str(s["timestamp"]),
                    "camera_id": s["camera_id"],
                    "event": s.get("description", f"Sighted at {s['camera_id']}"),
                    "movement_type": "observed",
                    "confidence": float(s.get("detection_confidence", 0.8)),
                    "evidence_ids": [f"sighting:{s['camera_id']}:{s['timestamp']}"],
                })
            for seg in trajectory_segments:
                if seg.movement_type == MovementType.INFERRED:
                    timeline.append({
                        "time": str(seg.from_time),
                        "camera_id": f"{seg.from_camera}→{seg.to_camera}",
                        "event": (
                            f"INFERRED movement through blind spot "
                            f"({seg.from_camera} → {seg.to_camera}); "
                            f"{len(seg.possible_routes)} candidate routes"
                        ),
                        "movement_type": "inferred",
                        "confidence": seg.confidence,
                        "possible_routes": seg.possible_routes,
                        "route_probabilities": seg.route_probabilities,
                        "evidence_ids": [f"trajectory:{seg.id}"],
                    })
            timeline.sort(key=lambda t: t["time"])

        # Calibrated overall confidence for the leading hypothesis
        overall_confidence = 0.0
        if leading:
            sup_w = [e.weight for e in leading.supporting_evidence]
            con_w = [e.weight for e in leading.contradicting_evidence]
            reliabilities = [0.85] * (len(sup_w) + len(con_w))
            proximities = [0.9] * (len(sup_w) + len(con_w))
            overall_confidence = self.confidence_engine.compute_confidence(
                sup_w, con_w, reliabilities, proximities,
            )
            # Unresolved objections lower confidence, weighted by severity
            for o in objections:
                if o["hypothesis_id"] == leading.id and not o["resolved"]:
                    penalty = {"CRITICAL": 0.15, "MAJOR": 0.07, "MINOR": 0.03}.get(
                        o["severity"], 0.03
                    )
                    overall_confidence = max(0.05, overall_confidence - penalty)
            overall_confidence = round(overall_confidence, 3)
            tracker.set_confidence(leading.id, overall_confidence)

        # Verification: every claim must cite at least one evidence ID
        verification_results = []
        for entry in timeline:
            verification_results.append({
                "claim_text": entry["event"],
                "status": "verified" if entry.get("evidence_ids") else "unsupported",
                "evidence_ids": entry.get("evidence_ids", []),
                "notes": (
                    "Inferred claim — flagged distinctly from observation."
                    if entry["movement_type"] == "inferred" else ""
                ),
            })
        verification_passed = all(v["status"] == "verified" for v in verification_results)

        suspect_name = people.get(leading.suspect_identity or "", "") if leading else ""
        unresolved = [o for o in objections if not o["resolved"]]
        resolved = [o for o in objections if o["resolved"]]

        report = {
            "case_id": case_id,
            "title": f"Investigation Report — {case_title}",
            "summary": (
                f"The investigation identifies {suspect_name} "
                f"({leading.suspect_identity}) as the primary suspect with "
                f"{overall_confidence:.0%} calibrated confidence."
                if leading else
                "The investigation could not identify a suspect from the "
                "available evidence."
            ),
            "primary_conclusion": {
                "hypothesis": leading.title if leading else None,
                "suspect_id": leading.suspect_identity if leading else None,
                "suspect_name": suspect_name,
                "confidence": overall_confidence,
                "supporting_evidence": [
                    {"evidence_id": e.evidence_id, "weight": e.weight,
                     "reasoning": e.reasoning}
                    for e in (leading.supporting_evidence if leading else [])
                ],
                "contradicting_evidence": [
                    {"evidence_id": e.evidence_id, "weight": e.weight,
                     "reasoning": e.reasoning}
                    for e in (leading.contradicting_evidence if leading else [])
                ],
            },
            "timeline": timeline,
            "confidence_assessment": {
                "overall_confidence": overall_confidence,
                "is_confirmed": overall_confidence >= 0.6,
                "note": (
                    "Confidence below threshold — conclusion is UNCONFIRMED."
                    if overall_confidence < 0.6 else
                    "Confidence above the 0.6 reporting threshold."
                ),
            },
            "alternative_hypotheses": [
                {
                    "title": h.title,
                    "suspect_id": h.suspect_identity,
                    "confidence": h.confidence,
                    "status": h.status.value,
                    "rejection_reason": h.rejection_reason,
                }
                for h in tracker.get_all_hypotheses()
                if not leading or h.id != leading.id
            ],
            "unresolved_objections": unresolved,
            "resolved_objections": resolved,
            "verification": {
                "passed": verification_passed,
                "results": verification_results,
            },
            "metadata": {
                "pipeline": "offline-deterministic",
                "hypotheses_considered": len(tracker.get_all_hypotheses()),
                "hypotheses_rejected": len(tracker.get_pruned_hypotheses()),
                "critic_objections_total": len(objections),
                "unresolved_objections": len(unresolved),
                "verification_passed": verification_passed,
                "inferred_segments": sum(
                    1 for s in trajectory_segments
                    if s.movement_type == MovementType.INFERRED
                ),
                "observed_segments": sum(
                    1 for s in trajectory_segments
                    if s.movement_type == MovementType.OBSERVED
                ),
            },
        }
        logger.info(
            f"Offline investigation complete: suspect={suspect_name or 'none'}, "
            f"confidence={overall_confidence}, "
            f"rejected={report['metadata']['hypotheses_rejected']}"
        )
        return report
