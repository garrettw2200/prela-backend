"""
Tests for TaskClusterer module.

Tests clustering functionality with both semantic (sentence-transformers)
and fallback (hash-based) modes.
"""

import pytest
from datetime import datetime, timedelta
from shared.task_clusterer import TaskClusterer, PromptCluster


class TestTaskClusterer:
    """Test suite for TaskClusterer"""

    def test_initialization_default(self):
        """Test clusterer initialization with defaults"""
        clusterer = TaskClusterer()
        assert clusterer.similarity_threshold == 0.8
        assert clusterer.min_cluster_size == 3
        # encoder_available depends on sentence-transformers installation

    def test_initialization_custom(self):
        """Test clusterer initialization with custom parameters"""
        clusterer = TaskClusterer(
            similarity_threshold=0.9,
            min_cluster_size=5,
        )
        assert clusterer.similarity_threshold == 0.9
        assert clusterer.min_cluster_size == 5

    def test_cluster_prompts_empty_list(self):
        """Test clustering with empty prompt list"""
        clusterer = TaskClusterer()
        result = clusterer.cluster_prompts([])
        assert result == []

    def test_cluster_prompts_too_few(self):
        """Test clustering with insufficient prompts"""
        clusterer = TaskClusterer(min_cluster_size=5)
        prompts = [
            {"text": "test", "tokens": 100, "cost": 0.01, "timestamp": datetime.utcnow().isoformat()},
            {"text": "test2", "tokens": 100, "cost": 0.01, "timestamp": datetime.utcnow().isoformat()},
        ]
        result = clusterer.cluster_prompts(prompts)
        assert result == []

    def test_cluster_prompts_exact_duplicates(self):
        """Test clustering with exact duplicate prompts"""
        clusterer = TaskClusterer(min_cluster_size=3)

        base_time = datetime.utcnow()
        prompts = [
            {
                "text": "What is the capital of France?",
                "tokens": 100,
                "cost": 0.01,
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            }
            for i in range(5)
        ]

        result = clusterer.cluster_prompts(prompts)

        # Should find at least one cluster with exact duplicates
        assert len(result) >= 1
        assert result[0].member_count >= 3
        assert result[0].potential_cache_savings > 0

    def test_cluster_prompts_different_texts(self):
        """Test clustering with completely different prompts"""
        clusterer = TaskClusterer(min_cluster_size=3)

        prompts = [
            {"text": "What is the capital of France?", "tokens": 100, "cost": 0.01, "timestamp": "2024-01-01T00:00:00"},
            {"text": "How do I bake a cake?", "tokens": 100, "cost": 0.01, "timestamp": "2024-01-01T01:00:00"},
            {"text": "Explain quantum physics", "tokens": 100, "cost": 0.01, "timestamp": "2024-01-01T02:00:00"},
        ]

        result = clusterer.cluster_prompts(prompts)

        # Different texts shouldn't form clusters (if using fallback exact matching)
        # Result depends on whether semantic clustering is available
        # In fallback mode, no clusters expected (different hashes)
        if not clusterer.encoder_available:
            assert len(result) == 0

    def test_cluster_prompts_cost_calculation(self):
        """Test that cost savings are calculated correctly"""
        clusterer = TaskClusterer(min_cluster_size=3)

        base_time = datetime.utcnow()
        prompts = [
            {
                "text": "Test prompt",
                "tokens": 200,
                "cost": 0.05,
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            for i in range(5)
        ]

        result = clusterer.cluster_prompts(prompts)

        assert len(result) >= 1
        cluster = result[0]

        # Total cost should equal sum of all costs
        assert cluster.total_cost == pytest.approx(0.25, rel=0.01)  # 5 * 0.05

        # Savings should be total cost * (n-1)/n (all but first call saved)
        expected_savings = 0.25 * (4/5)  # 4 hits out of 5
        assert cluster.potential_cache_savings == pytest.approx(expected_savings, rel=0.01)

    def test_cluster_prompts_frequency_calculation(self):
        """Test that frequency per day is calculated correctly"""
        clusterer = TaskClusterer(min_cluster_size=3)

        # Create prompts spanning 2 days
        base_time = datetime.utcnow()
        prompts = [
            {
                "text": "Test prompt",
                "tokens": 100,
                "cost": 0.01,
                "timestamp": (base_time + timedelta(hours=i*12)).isoformat(),  # 12 hour intervals
            }
            for i in range(5)  # 5 prompts over 2.5 days
        ]

        result = clusterer.cluster_prompts(prompts)

        assert len(result) >= 1
        cluster = result[0]

        # 5 prompts over 2.5 days = ~2 prompts/day
        assert cluster.frequency_per_day > 0
        assert cluster.frequency_per_day < 5  # Should be less than total prompts

    def test_compute_similarity_exact_match(self):
        """Test similarity computation for identical texts"""
        clusterer = TaskClusterer()

        text = "This is a test prompt"
        similarity = clusterer.compute_similarity(text, text)

        assert similarity == 1.0

    def test_compute_similarity_different_texts(self):
        """Test similarity computation for different texts"""
        clusterer = TaskClusterer()

        text1 = "What is the capital of France?"
        text2 = "How do I bake a cake?"

        similarity = clusterer.compute_similarity(text1, text2)

        # Different texts should have low similarity
        assert similarity < 0.5

    def test_compute_similarity_similar_texts(self):
        """Test similarity computation for similar texts"""
        clusterer = TaskClusterer()

        text1 = "What is the capital of France?"
        text2 = "What is the capital of Germany?"

        similarity = clusterer.compute_similarity(text1, text2)

        # Similar texts should have high similarity
        assert similarity > 0.7

    def test_prompt_cluster_dataclass(self):
        """Test PromptCluster dataclass creation"""
        cluster = PromptCluster(
            cluster_id="cluster_0",
            representative_text="Test prompt",
            member_texts=["Test prompt", "Test prompt", "Test prompt"],
            member_count=3,
            avg_tokens=100.0,
            total_cost=0.03,
            frequency_per_day=2.5,
            potential_cache_savings=0.02,
        )

        assert cluster.cluster_id == "cluster_0"
        assert cluster.member_count == 3
        assert cluster.avg_tokens == 100.0
        assert cluster.total_cost == 0.03
        assert cluster.frequency_per_day == 2.5
        assert cluster.potential_cache_savings == 0.02

    def test_cluster_prompts_sorting(self):
        """Test that clusters are sorted by savings potential"""
        clusterer = TaskClusterer(min_cluster_size=2)

        # Create two groups with different costs
        prompts = [
            # High cost group (5 duplicates)
            *[{"text": "expensive prompt", "tokens": 1000, "cost": 1.0, "timestamp": f"2024-01-01T{i:02d}:00:00"} for i in range(5)],
            # Low cost group (5 duplicates)
            *[{"text": "cheap prompt", "tokens": 100, "cost": 0.01, "timestamp": f"2024-01-01T{i:02d}:00:00"} for i in range(5)],
        ]

        result = clusterer.cluster_prompts(prompts)

        # Should have 2 clusters
        assert len(result) == 2

        # First cluster should have highest savings
        assert result[0].potential_cache_savings > result[1].potential_cache_savings

    def test_truncation(self):
        """Test that long prompts are truncated appropriately"""
        clusterer = TaskClusterer()

        # Create prompt with very long text
        long_text = "x" * 1000  # 1000 characters
        prompts = [
            {"text": long_text, "tokens": 100, "cost": 0.01, "timestamp": f"2024-01-01T{i:02d}:00:00"}
            for i in range(5)
        ]

        result = clusterer.cluster_prompts(prompts)

        # Clustering should still work (truncation happens during processing)
        assert len(result) >= 1

        # Representative text should be truncated to 200 chars
        assert len(result[0].representative_text) <= 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
