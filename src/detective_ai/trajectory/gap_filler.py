"""Gap filler: infers probable paths when individuals leave camera coverage.

Combines the Markov trajectory model with temporal constraints to produce
trajectory segments marked as inferred with bounded uncertainty.
"""

from __future__ import annotations

import logging
from datetime import datetime

from detective_ai.core.enums import MovementType
from detective_ai.core.models import TrajectorySegment
from detective_ai.trajectory.camera_topology import CameraTopology
from detective_ai.trajectory.markov_model import MarkovTrajectoryModel

logger = logging.getLogger(__name__)


class GapFiller:
    """Fills camera blind spot gaps with probabilistic trajectory inference."""

    def __init__(
        self,
        topology: CameraTopology,
        markov_model: MarkovTrajectoryModel | None = None,
    ) -> None:
        self.topology = topology
        self.markov = markov_model or MarkovTrajectoryModel(topology)

    def infer_trajectory(
        self,
        from_camera: str,
        to_camera: str,
        from_time: datetime,
        to_time: datetime,
        identity_cluster_id: str,
        case_id: str = "",
        max_routes: int = 3,
    ) -> TrajectorySegment:
        """Infer a trajectory segment through a camera blind spot.

        Args:
            from_camera: Camera where the person was last seen.
            to_camera: Camera where the person reappeared.
            from_time: Timestamp of last sighting.
            to_time: Timestamp of reappearance.
            identity_cluster_id: Which identity cluster this belongs to.
            case_id: Investigation case ID.
            max_routes: Maximum number of routes to consider.

        Returns:
            TrajectorySegment marked as INFERRED with route distribution.
        """
        # Calculate time gap
        time_gap = (to_time - from_time).total_seconds()

        # Get possible routes with probabilities
        routes = self.markov.predict_route_distribution(
            from_camera, to_camera, max_paths=max_routes
        )

        # Filter routes by temporal feasibility
        feasible_routes = []
        for route in routes:
            expected_travel = route["travel_time"]
            # Allow 2x margin for deviations
            if expected_travel <= time_gap * 2.0:
                feasible_routes.append(route)

        if not feasible_routes and routes:
            # If nothing is feasible, keep the most probable anyway but lower confidence
            feasible_routes = routes[:1]

        # Extract routes and probabilities
        possible_paths = [r["path"] for r in feasible_routes]
        probabilities = [r["probability"] for r in feasible_routes]

        # Confidence is based on:
        # 1) Whether feasible routes exist
        # 2) How concentrated the probability is (less spread = more confident)
        # 3) Whether the time gap is reasonable
        if feasible_routes:
            max_prob = max(probabilities) if probabilities else 0
            route_confidence = max_prob
            time_reasonability = min(1.0, routes[0]["travel_time"] / max(1, time_gap))
            confidence = round(0.6 * route_confidence + 0.4 * time_reasonability, 3)
        else:
            confidence = 0.1

        segment = TrajectorySegment(
            case_id=case_id,
            identity_cluster_id=identity_cluster_id,
            from_camera=from_camera,
            to_camera=to_camera,
            from_time=from_time,
            to_time=to_time,
            movement_type=MovementType.INFERRED,
            confidence=confidence,
            possible_routes=possible_paths,
            route_probabilities=probabilities,
        )

        logger.info(
            f"Inferred trajectory: {from_camera} → {to_camera}, "
            f"{len(feasible_routes)} feasible routes, confidence={confidence:.2f}"
        )
        return segment

    def fill_gaps_in_timeline(
        self,
        sightings: list[dict],
        identity_cluster_id: str,
        case_id: str = "",
    ) -> list[TrajectorySegment]:
        """Fill all gaps in a timeline of camera sightings.

        Args:
            sightings: List of dicts with 'camera_id' and 'timestamp',
                      sorted chronologically.
            identity_cluster_id: Which identity cluster.
            case_id: Investigation case ID.

        Returns:
            List of TrajectorySegments (both observed and inferred).
        """
        if len(sightings) < 2:
            return []

        segments = []

        for i in range(len(sightings) - 1):
            current = sightings[i]
            next_sighting = sightings[i + 1]

            from_cam = current["camera_id"]
            to_cam = next_sighting["camera_id"]
            from_time = current["timestamp"]
            to_time = next_sighting["timestamp"]

            if isinstance(from_time, str):
                from_time = datetime.fromisoformat(from_time)
            if isinstance(to_time, str):
                to_time = datetime.fromisoformat(to_time)

            if from_cam == to_cam:
                # Same camera = observed, no gap
                segments.append(
                    TrajectorySegment(
                        case_id=case_id,
                        identity_cluster_id=identity_cluster_id,
                        from_camera=from_cam,
                        to_camera=to_cam,
                        from_time=from_time,
                        to_time=to_time,
                        movement_type=MovementType.OBSERVED,
                        confidence=0.95,
                    )
                )
            elif self.topology.has_blind_spot(from_cam, to_cam):
                # Different cameras with blind spot = inferred
                segment = self.infer_trajectory(
                    from_camera=from_cam,
                    to_camera=to_cam,
                    from_time=from_time,
                    to_time=to_time,
                    identity_cluster_id=identity_cluster_id,
                    case_id=case_id,
                )
                segments.append(segment)
            else:
                # Different cameras, direct connection = observed transition
                segments.append(
                    TrajectorySegment(
                        case_id=case_id,
                        identity_cluster_id=identity_cluster_id,
                        from_camera=from_cam,
                        to_camera=to_cam,
                        from_time=from_time,
                        to_time=to_time,
                        movement_type=MovementType.OBSERVED,
                        confidence=0.9,
                    )
                )

        logger.info(
            f"Filled {len(segments)} trajectory segments for cluster "
            f"{identity_cluster_id} ({sum(1 for s in segments if s.movement_type == MovementType.INFERRED)} inferred)"
        )
        return segments
