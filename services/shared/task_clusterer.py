"""
Task Clusterer for Cost Optimization

Clusters similar prompts/tasks using semantic similarity to identify opportunities
for caching and model downgrades.

Uses sentence-transformers when available, falls back to simpler similarity methods.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Try to import sentence-transformers (optional dependency)
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

# Try to import sklearn for clustering (optional dependency)
try:
    from sklearn.cluster import KMeans
    import numpy as np

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None  # type: ignore
    np = None  # type: ignore


@dataclass
class PromptCluster:
    """A cluster of similar prompts"""

    cluster_id: str
    representative_text: str  # Most common or central prompt
    member_texts: list[str]
    member_count: int
    avg_tokens: float
    total_cost: float
    frequency_per_day: float
    potential_cache_savings: float


class TaskClusterer:
    """
    Clusters similar prompts/tasks using semantic similarity.

    Supports two modes:
    1. With sentence-transformers: High-quality semantic clustering
    2. Fallback mode: Simple hash-based and text similarity clustering
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 3,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the task clusterer.

        Args:
            similarity_threshold: Minimum similarity to consider prompts as duplicates (0.0-1.0)
            min_cluster_size: Minimum number of prompts to form a cluster
            model_name: Sentence-transformers model name (if available)
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self._encoder = None
        self.encoder_available = False

        # Try to initialize sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(model_name)
                self.encoder_available = True
                logger.info(f"Initialized TaskClusterer with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}")
                logger.info("Falling back to simpler similarity methods")
        else:
            logger.info("sentence-transformers not available, using fallback clustering")

    def cluster_prompts(
        self,
        prompts: list[dict[str, Any]],
        n_clusters: int | None = None,
    ) -> list[PromptCluster]:
        """
        Cluster similar prompts for cache analysis.

        Args:
            prompts: List of dicts with keys: text, tokens, cost, timestamp
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            List of PromptCluster objects
        """
        if not prompts:
            return []

        if len(prompts) < self.min_cluster_size:
            logger.debug(f"Not enough prompts ({len(prompts)}) for clustering")
            return []

        # Use semantic clustering if available
        if self.encoder_available and SKLEARN_AVAILABLE:
            return self._cluster_semantic(prompts, n_clusters)
        else:
            return self._cluster_fallback(prompts)

    def _cluster_semantic(
        self,
        prompts: list[dict[str, Any]],
        n_clusters: int | None = None,
    ) -> list[PromptCluster]:
        """Cluster prompts using sentence-transformers + K-means"""
        try:
            # Extract texts
            texts = [p["text"][:500] for p in prompts]  # Truncate to 500 chars

            # Generate embeddings
            embeddings = self._encoder.encode(texts, show_progress_bar=False)

            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                # Use rule of thumb: sqrt(n/2)
                n_clusters = max(2, int((len(prompts) / 2) ** 0.5))
                n_clusters = min(n_clusters, len(prompts) // 2)  # Cap at half

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Build clusters
            clusters_dict: dict[int, list[int]] = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(idx)

            # Convert to PromptCluster objects
            result = []
            for label, indices in clusters_dict.items():
                if len(indices) < self.min_cluster_size:
                    continue  # Skip small clusters

                members = [prompts[i] for i in indices]
                member_texts = [prompts[i]["text"] for i in indices]

                # Find representative (most central)
                cluster_embeddings = embeddings[indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = [
                    np.linalg.norm(emb - centroid) for emb in cluster_embeddings
                ]
                representative_idx = indices[distances.index(min(distances))]
                representative_text = prompts[representative_idx]["text"]

                # Calculate statistics
                avg_tokens = sum(m["tokens"] for m in members) / len(members)
                total_cost = sum(m["cost"] for m in members)

                # Calculate frequency (prompts per day)
                if members:
                    timestamps = [m.get("timestamp") for m in members if m.get("timestamp")]
                    if len(timestamps) >= 2:
                        from datetime import datetime

                        ts_objs = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
                        time_span = (max(ts_objs) - min(ts_objs)).total_seconds() / 86400  # days
                        frequency_per_day = len(members) / max(time_span, 1.0)
                    else:
                        frequency_per_day = len(members)
                else:
                    frequency_per_day = 0.0

                # Estimate cache savings (all hits except first = saved)
                cache_savings = total_cost * (len(members) - 1) / len(members)

                cluster = PromptCluster(
                    cluster_id=f"cluster_{label}",
                    representative_text=representative_text[:200],  # Truncate for display
                    member_texts=member_texts,
                    member_count=len(members),
                    avg_tokens=avg_tokens,
                    total_cost=total_cost,
                    frequency_per_day=frequency_per_day,
                    potential_cache_savings=cache_savings,
                )
                result.append(cluster)

            return sorted(result, key=lambda c: c.potential_cache_savings, reverse=True)

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return self._cluster_fallback(prompts)

    def _cluster_fallback(self, prompts: list[dict[str, Any]]) -> list[PromptCluster]:
        """Simple fallback clustering using exact matches and text similarity"""
        # Group by exact hash first
        exact_groups: dict[str, list[dict[str, Any]]] = {}
        for prompt in prompts:
            text = prompt["text"]
            hash_key = hashlib.md5(text.encode()).hexdigest()
            if hash_key not in exact_groups:
                exact_groups[hash_key] = []
            exact_groups[hash_key].append(prompt)

        # Build clusters from exact matches
        result = []
        cluster_id = 0
        for members in exact_groups.values():
            if len(members) < self.min_cluster_size:
                continue

            representative_text = members[0]["text"]
            member_texts = [m["text"] for m in members]
            avg_tokens = sum(m["tokens"] for m in members) / len(members)
            total_cost = sum(m["cost"] for m in members)

            # Calculate frequency
            timestamps = [m.get("timestamp") for m in members if m.get("timestamp")]
            if len(timestamps) >= 2:
                from datetime import datetime

                ts_objs = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
                time_span = (max(ts_objs) - min(ts_objs)).total_seconds() / 86400
                frequency_per_day = len(members) / max(time_span, 1.0)
            else:
                frequency_per_day = len(members)

            cache_savings = total_cost * (len(members) - 1) / len(members)

            cluster = PromptCluster(
                cluster_id=f"cluster_{cluster_id}",
                representative_text=representative_text[:200],
                member_texts=member_texts,
                member_count=len(members),
                avg_tokens=avg_tokens,
                total_cost=total_cost,
                frequency_per_day=frequency_per_day,
                potential_cache_savings=cache_savings,
            )
            result.append(cluster)
            cluster_id += 1

        return sorted(result, key=lambda c: c.potential_cache_savings, reverse=True)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Returns:
            Similarity score (0.0-1.0)
        """
        if self.encoder_available:
            try:
                embeddings = self._encoder.encode([text1, text2])
                # Cosine similarity
                dot_product = np.dot(embeddings[0], embeddings[1])
                norm1 = np.linalg.norm(embeddings[0])
                norm2 = np.linalg.norm(embeddings[1])
                return float(dot_product / (norm1 * norm2))
            except Exception as e:
                logger.debug(f"Embedding similarity failed: {e}")

        # Fallback to difflib
        return difflib.SequenceMatcher(None, text1, text2).ratio()
