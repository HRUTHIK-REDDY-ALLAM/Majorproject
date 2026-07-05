"""Markov chain model for trajectory prediction through blind spots.

Builds a transition probability matrix over the camera topology and
returns distributions over possible routes rather than single guesses.
"""

from __future__ import annotations

import logging

import numpy as np

from detective_ai.trajectory.camera_topology import CameraTopology

logger = logging.getLogger(__name__)


class MarkovTrajectoryModel:
    """Probabilistic trajectory model using Markov chain over camera topology."""

    def __init__(self, topology: CameraTopology) -> None:
        self.topology = topology
        self._cameras = topology.cameras
        self._cam_to_idx = {c: i for i, c in enumerate(self._cameras)}
        self._n = len(self._cameras)
        self._transition_matrix = self._build_default_matrix()

    def _build_default_matrix(self) -> np.ndarray:
        """Build default transition matrix from topology (uniform over neighbors)."""
        matrix = np.zeros((self._n, self._n))

        for cam in self._cameras:
            neighbors = self.topology.get_adjacent_cameras(cam)
            if neighbors:
                i = self._cam_to_idx[cam]
                for neighbor in neighbors:
                    j = self._cam_to_idx.get(neighbor)
                    if j is not None:
                        # Weight inversely by travel time
                        tt = self.topology.get_travel_time(cam, neighbor) or 60.0
                        matrix[i][j] = 1.0 / max(1.0, tt)

        # Normalize rows to sum to 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        matrix = matrix / row_sums

        return matrix

    def update_from_observations(
        self, transitions: list[tuple[str, str]], learning_rate: float = 0.3
    ) -> None:
        """Update transition matrix from observed movement data.

        Args:
            transitions: List of (from_camera, to_camera) pairs.
            learning_rate: How much to weight new observations vs. default.
        """
        observed = np.zeros((self._n, self._n))
        for from_cam, to_cam in transitions:
            i = self._cam_to_idx.get(from_cam)
            j = self._cam_to_idx.get(to_cam)
            if i is not None and j is not None:
                observed[i][j] += 1

        # Normalize observed
        row_sums = observed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        observed_norm = observed / row_sums

        # Blend with existing matrix
        self._transition_matrix = (
            (1 - learning_rate) * self._transition_matrix
            + learning_rate * observed_norm
        )

        # Re-normalize
        row_sums = self._transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._transition_matrix = self._transition_matrix / row_sums

        logger.info(f"Updated transition matrix with {len(transitions)} observations")

    def predict_next(self, current_camera: str) -> dict[str, float]:
        """Predict the distribution over next cameras from current position.

        Returns:
            Dict mapping camera_id → probability.
        """
        i = self._cam_to_idx.get(current_camera)
        if i is None:
            return {}

        probs = self._transition_matrix[i]
        result = {}
        for j, p in enumerate(probs):
            if p > 0.001:
                result[self._cameras[j]] = round(float(p), 4)

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def predict_path_probability(self, path: list[str]) -> float:
        """Calculate the probability of a specific path through the topology.

        Args:
            path: List of camera IDs representing a route.

        Returns:
            Probability of the path (product of transition probabilities).
        """
        if len(path) < 2:
            return 1.0

        prob = 1.0
        for k in range(len(path) - 1):
            i = self._cam_to_idx.get(path[k])
            j = self._cam_to_idx.get(path[k + 1])
            if i is None or j is None:
                return 0.0
            prob *= self._transition_matrix[i][j]

        return float(prob)

    def predict_route_distribution(
        self,
        from_camera: str,
        to_camera: str,
        max_paths: int = 5,
    ) -> list[dict]:
        """Get a distribution over possible routes between two cameras.

        Args:
            from_camera: Start camera.
            to_camera: End camera.
            max_paths: Maximum routes to return.

        Returns:
            List of dicts with 'path', 'probability', 'travel_time'.
        """
        all_paths = self.topology.get_all_paths(from_camera, to_camera)

        if not all_paths:
            return []

        routes = []
        for path in all_paths:
            prob = self.predict_path_probability(path)
            tt = self.topology.get_path_travel_time(path)
            routes.append({
                "path": path,
                "probability": round(prob, 6),
                "travel_time": round(tt, 1),
                "has_blind_spot": any(
                    self.topology.has_blind_spot(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                ),
            })

        # Normalize probabilities
        total_prob = sum(r["probability"] for r in routes)
        if total_prob > 0:
            for r in routes:
                r["probability"] = round(r["probability"] / total_prob, 4)

        # Sort by probability descending
        routes.sort(key=lambda r: r["probability"], reverse=True)
        return routes[:max_paths]

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix.copy()

    @property
    def camera_labels(self) -> list[str]:
        return list(self._cameras)
