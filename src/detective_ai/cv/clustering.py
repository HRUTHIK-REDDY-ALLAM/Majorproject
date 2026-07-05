"""Appearance-based identity clustering across cameras.

Groups visual detections into identity clusters using cosine similarity
on ReID embeddings. Returns similarity scores, not binary matches.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from detective_ai.cv.reid import compute_reid_similarity

logger = logging.getLogger(__name__)


class IdentityClusterer:
    """Clusters detections across cameras into identity groups."""

    def __init__(self, distance_threshold: float = 0.4) -> None:
        """
        Args:
            distance_threshold: Maximum cosine distance (1 - similarity) to
                consider two detections as the same person. Lower = stricter.
        """
        self.distance_threshold = distance_threshold

    def cluster_detections(
        self,
        detections: list[dict],
        embedding_key: str = "appearance_embedding",
    ) -> list[dict]:
        """Cluster detections by appearance similarity.

        Args:
            detections: List of detection dicts, each with an embedding.
            embedding_key: Key in the dict containing the embedding vector.

        Returns:
            Same detections enriched with 'cluster_id' and 'cluster_similarities'.
        """
        if not detections:
            return []

        # Extract embeddings
        embeddings = []
        valid_indices = []

        for i, det in enumerate(detections):
            emb = det.get(embedding_key)
            if emb and len(emb) > 0 and any(v != 0 for v in emb):
                embeddings.append(emb)
                valid_indices.append(i)

        if len(embeddings) < 2:
            # Not enough embeddings to cluster
            for det in detections:
                det["cluster_id"] = "cluster_0"
                det["cluster_similarities"] = {}
            return detections

        embedding_matrix = np.array(embeddings)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(embedding_matrix)
        distance_matrix = 1 - sim_matrix

        # Agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Assign cluster IDs
        label_map = {}
        for idx, valid_idx in enumerate(valid_indices):
            cluster_label = int(labels[idx])
            cluster_id = f"cluster_{cluster_label}"
            detections[valid_idx]["cluster_id"] = cluster_id
            label_map[valid_idx] = cluster_label

        # Assign unmatched detections to their own clusters
        next_cluster = max(labels) + 1 if len(labels) > 0 else 0
        for i, det in enumerate(detections):
            if "cluster_id" not in det:
                det["cluster_id"] = f"cluster_{next_cluster}"
                next_cluster += 1

        n_clusters = len(set(labels))
        logger.info(
            f"Clustered {len(embeddings)} detections into {n_clusters} identities "
            f"(threshold={self.distance_threshold})"
        )

        return detections

    def find_matches(
        self,
        query_embedding: list[float],
        gallery_embeddings: list[dict],
        top_k: int = 5,
        min_similarity: float = 0.6,
    ) -> list[dict]:
        """Find the top-k most similar detections to a query embedding.

        Args:
            query_embedding: The query appearance embedding.
            gallery_embeddings: List of dicts with 'id' and 'embedding'.
            top_k: Number of top matches to return.
            min_similarity: Minimum similarity to include.

        Returns:
            List of match dicts with 'id', 'similarity', sorted by similarity.
        """
        matches = []
        for item in gallery_embeddings:
            emb = item.get("embedding", [])
            if not emb:
                continue
            sim = compute_reid_similarity(query_embedding, emb)
            if sim >= min_similarity:
                matches.append({
                    "id": item.get("id"),
                    "similarity": round(sim, 4),
                    "metadata": {k: v for k, v in item.items() if k not in ("id", "embedding")},
                })

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]

    def compute_similarity_matrix(
        self, embeddings: list[list[float]]
    ) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if not embeddings:
            return np.array([])
        return cosine_similarity(np.array(embeddings))
