"""Camera topology: graph of camera positions and spatial adjacency.

Represents the physical layout of cameras, including which cameras
can see which areas and the typical travel times between them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class CameraTopology:
    """Graph of camera positions and spatial relationships."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_camera(
        self,
        camera_id: str,
        location: str = "",
        position: tuple[float, float] | None = None,
        coverage_area: str = "",
    ) -> None:
        """Add a camera to the topology."""
        self._graph.add_node(
            camera_id,
            location=location,
            position=position or (0.0, 0.0),
            coverage_area=coverage_area,
        )

    def add_connection(
        self,
        from_camera: str,
        to_camera: str,
        travel_time_seconds: float,
        distance_meters: float = 0.0,
        is_direct: bool = True,
        via_blind_spot: bool = False,
    ) -> None:
        """Add a directed connection between two cameras.

        Args:
            from_camera: Source camera ID.
            to_camera: Target camera ID.
            travel_time_seconds: Typical travel time between cameras.
            distance_meters: Physical distance.
            is_direct: Whether the path is a direct connection.
            via_blind_spot: Whether the path passes through a blind spot.
        """
        self._graph.add_edge(
            from_camera,
            to_camera,
            travel_time=travel_time_seconds,
            distance=distance_meters,
            is_direct=is_direct,
            via_blind_spot=via_blind_spot,
        )

    def get_adjacent_cameras(self, camera_id: str) -> list[str]:
        """Get cameras directly reachable from a given camera."""
        if camera_id not in self._graph:
            return []
        return list(self._graph.successors(camera_id))

    def get_travel_time(self, from_camera: str, to_camera: str) -> float | None:
        """Get the expected travel time between two cameras."""
        if self._graph.has_edge(from_camera, to_camera):
            return self._graph[from_camera][to_camera]["travel_time"]
        return None

    def get_all_paths(
        self,
        from_camera: str,
        to_camera: str,
        max_length: int = 5,
    ) -> list[list[str]]:
        """Get all simple paths between two cameras up to max_length."""
        if from_camera not in self._graph or to_camera not in self._graph:
            return []
        try:
            return list(
                nx.all_simple_paths(self._graph, from_camera, to_camera, cutoff=max_length)
            )
        except nx.NetworkXError:
            return []

    def get_shortest_path(
        self, from_camera: str, to_camera: str
    ) -> list[str] | None:
        """Get the shortest path between two cameras."""
        try:
            return nx.shortest_path(
                self._graph, from_camera, to_camera, weight="travel_time"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_path_travel_time(self, path: list[str]) -> float:
        """Calculate total travel time for a given path."""
        total = 0.0
        for i in range(len(path) - 1):
            tt = self.get_travel_time(path[i], path[i + 1])
            if tt is not None:
                total += tt
            else:
                total += float("inf")
        return total

    def has_blind_spot(self, from_camera: str, to_camera: str) -> bool:
        """Check if the path between two cameras passes through a blind spot."""
        if self._graph.has_edge(from_camera, to_camera):
            return self._graph[from_camera][to_camera].get("via_blind_spot", False)
        return True  # No direct connection = definitely a blind spot

    @property
    def cameras(self) -> list[str]:
        return list(self._graph.nodes())

    @property
    def camera_count(self) -> int:
        return self._graph.number_of_nodes()

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": [
                {"id": n, **data} for n, data in self._graph.nodes(data=True)
            ],
            "connections": [
                {"from": u, "to": v, **data}
                for u, v, data in self._graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraTopology:
        topo = cls()
        for cam in data.get("cameras", []):
            cam = dict(cam)  # never mutate the caller's data
            cam_id = cam.pop("id")
            topo.add_camera(camera_id=cam_id, **cam)
        for conn in data.get("connections", []):
            topo.add_connection(
                conn["from"],
                conn["to"],
                travel_time_seconds=conn.get("travel_time", 0.0),
                distance_meters=conn.get("distance", 0.0),
                is_direct=conn.get("is_direct", True),
                via_blind_spot=conn.get("via_blind_spot", False),
            )
        return topo

    @classmethod
    def from_json(cls, path: str | Path) -> CameraTopology:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
