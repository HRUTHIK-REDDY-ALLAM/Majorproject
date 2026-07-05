"""In-memory evidence graph built on top of NetworkX.

Nodes represent evidence items, hypotheses, and inferred events.
Edges represent relationships: supports, contradicts, temporally-follows, spatially-adjacent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class GraphNode:
    """A node in the evidence graph."""

    id: str
    node_type: str  # "evidence" | "hypothesis" | "detection" | "trajectory"
    label: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the evidence graph."""

    source_id: str
    target_id: str
    relationship: str  # "supports" | "contradicts" | "temporally_follows" | "spatially_adjacent"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceGraph:
    """Directed graph of evidence relationships.

    Provides methods to add evidence, link items, query support/contradiction,
    and prune hypotheses while preserving audit trail.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._pruned_nodes: dict[str, dict[str, Any]] = {}  # audit trail

    # ── Node operations ───────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        """Add an evidence node to the graph."""
        self._graph.add_node(
            node.id,
            node_type=node.node_type,
            label=node.label,
            data=node.data,
        )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a node's data by ID."""
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    # ── Edge operations ───────────────────────────────────────

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a relationship edge between two nodes."""
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship=edge.relationship,
            weight=edge.weight,
            metadata=edge.metadata,
        )

    def link(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Convenience method to link two nodes."""
        self.add_edge(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                weight=weight,
                metadata=metadata,
            )
        )

    # ── Queries ───────────────────────────────────────────────

    def get_supporting(self, hypothesis_id: str) -> list[dict[str, Any]]:
        """Get all evidence nodes that support a hypothesis."""
        results = []
        for source, _, data in self._graph.in_edges(hypothesis_id, data=True):
            if data.get("relationship") == "supports":
                node_data = self.get_node(source)
                if node_data:
                    results.append({"node_id": source, **node_data, "edge": data})
        return results

    def get_contradicting(self, hypothesis_id: str) -> list[dict[str, Any]]:
        """Get all evidence nodes that contradict a hypothesis."""
        results = []
        for source, _, data in self._graph.in_edges(hypothesis_id, data=True):
            if data.get("relationship") == "contradicts":
                node_data = self.get_node(source)
                if node_data:
                    results.append({"node_id": source, **node_data, "edge": data})
        return results

    def get_timeline(self, node_type: str = "evidence") -> list[dict[str, Any]]:
        """Get temporally ordered nodes of a given type."""
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") == node_type:
                nodes.append({"node_id": nid, **data})
        # Sort by timestamp if available in data
        nodes.sort(key=lambda n: n.get("data", {}).get("timestamp", ""))
        return nodes

    def get_neighbors(self, node_id: str, relationship: str | None = None) -> list[str]:
        """Get neighboring node IDs, optionally filtered by relationship type."""
        neighbors = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append(target)
        for source, _, data in self._graph.in_edges(node_id, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append(source)
        return neighbors

    # ── Hypothesis management ─────────────────────────────────

    def prune_hypothesis(self, hypothesis_id: str, reason: str) -> None:
        """Prune a hypothesis and log it in the audit trail."""
        node_data = self.get_node(hypothesis_id)
        if node_data:
            self._pruned_nodes[hypothesis_id] = {
                **node_data,
                "rejection_reason": reason,
                "connected_evidence": list(self._graph.predecessors(hypothesis_id)),
            }
            self._graph.remove_node(hypothesis_id)

    def get_pruned_hypotheses(self) -> dict[str, dict[str, Any]]:
        """Return all pruned hypotheses with their rejection reasons."""
        return dict(self._pruned_nodes)

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "nodes": [
                {"id": nid, **data} for nid, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": s, "target": t, **data}
                for s, t, data in self._graph.edges(data=True)
            ],
            "pruned": self._pruned_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceGraph:
        """Deserialize a graph from a dictionary."""
        graph = cls()
        for node in data.get("nodes", []):
            nid = node.pop("id")
            graph._graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            s = edge.pop("source")
            t = edge.pop("target")
            graph._graph.add_edge(s, t, **edge)
        graph._pruned_nodes = data.get("pruned", {})
        return graph

    # ── Stats ─────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def summary(self) -> dict[str, int]:
        """Count nodes by type."""
        counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("node_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        counts["pruned_hypotheses"] = len(self._pruned_nodes)
        return counts
