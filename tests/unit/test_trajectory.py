"""Unit tests for trajectory inference."""

from datetime import datetime

from detective_ai.core.enums import MovementType
from detective_ai.trajectory.camera_topology import CameraTopology
from detective_ai.trajectory.gap_filler import GapFiller
from detective_ai.trajectory.markov_model import MarkovTrajectoryModel


class TestCameraTopology:
    def setup_method(self):
        self.topo = CameraTopology()
        self.topo.add_camera("cam_a", location="Room A")
        self.topo.add_camera("cam_b", location="Room B")
        self.topo.add_camera("cam_c", location="Room C")
        self.topo.add_connection("cam_a", "cam_b", travel_time_seconds=30)
        self.topo.add_connection("cam_b", "cam_c", travel_time_seconds=45)
        self.topo.add_connection("cam_a", "cam_c", travel_time_seconds=90, via_blind_spot=True)

    def test_adjacent_cameras(self):
        adj = self.topo.get_adjacent_cameras("cam_a")
        assert "cam_b" in adj
        assert "cam_c" in adj

    def test_travel_time(self):
        tt = self.topo.get_travel_time("cam_a", "cam_b")
        assert tt == 30

    def test_blind_spot_detection(self):
        assert self.topo.has_blind_spot("cam_a", "cam_c") is True
        assert self.topo.has_blind_spot("cam_a", "cam_b") is False

    def test_shortest_path(self):
        path = self.topo.get_shortest_path("cam_a", "cam_c")
        assert path is not None
        assert path[0] == "cam_a"
        assert path[-1] == "cam_c"

    def test_serialization(self):
        data = self.topo.to_dict()
        restored = CameraTopology.from_dict(data)
        assert restored.camera_count == 3


class TestMarkovModel:
    def setup_method(self):
        self.topo = CameraTopology()
        self.topo.add_camera("cam_a")
        self.topo.add_camera("cam_b")
        self.topo.add_camera("cam_c")
        self.topo.add_connection("cam_a", "cam_b", travel_time_seconds=30)
        self.topo.add_connection("cam_b", "cam_c", travel_time_seconds=45)
        self.topo.add_connection("cam_a", "cam_c", travel_time_seconds=90)
        self.model = MarkovTrajectoryModel(self.topo)

    def test_predict_next(self):
        predictions = self.model.predict_next("cam_a")
        assert len(predictions) > 0
        # Probabilities should sum to ~1
        total = sum(predictions.values())
        assert abs(total - 1.0) < 0.01

    def test_path_probability(self):
        prob = self.model.predict_path_probability(["cam_a", "cam_b", "cam_c"])
        assert 0 < prob <= 1

    def test_route_distribution(self):
        routes = self.model.predict_route_distribution("cam_a", "cam_c")
        assert len(routes) > 0
        # Probabilities should sum to ~1
        total = sum(r["probability"] for r in routes)
        assert abs(total - 1.0) < 0.01


class TestGapFiller:
    def setup_method(self):
        self.topo = CameraTopology()
        self.topo.add_camera("cam_a")
        self.topo.add_camera("cam_b")
        self.topo.add_camera("cam_c")
        self.topo.add_connection("cam_a", "cam_b", travel_time_seconds=30)
        self.topo.add_connection("cam_b", "cam_c", travel_time_seconds=45, via_blind_spot=True)
        self.filler = GapFiller(self.topo)

    def test_infer_trajectory(self):
        seg = self.filler.infer_trajectory(
            from_camera="cam_a",
            to_camera="cam_c",
            from_time=datetime(2025, 1, 15, 9, 0),
            to_time=datetime(2025, 1, 15, 9, 5),
            identity_cluster_id="cluster_0",
        )
        assert seg.movement_type == MovementType.INFERRED
        assert seg.confidence > 0
        assert seg.confidence < 1

    def test_fill_gaps_timeline(self):
        sightings = [
            {"camera_id": "cam_a", "timestamp": datetime(2025, 1, 15, 9, 0)},
            {"camera_id": "cam_b", "timestamp": datetime(2025, 1, 15, 9, 2)},
            {"camera_id": "cam_c", "timestamp": datetime(2025, 1, 15, 9, 5)},
        ]
        segments = self.filler.fill_gaps_in_timeline(
            sightings, "cluster_0"
        )
        assert len(segments) == 2  # two transitions
