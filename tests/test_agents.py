from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.aggregator import ContextAggregator
from src.agents.crag_node import CRAGNode
from src.agents.router import AgenticRouter
from src.agents.state import AgentState, make_initial_state


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with (
        patch("src.agents.router.settings") as mock_s,
        patch("src.agents.crag_node.settings") as mock_c,
        patch("src.agents.generator.settings") as mock_g,
    ):
        mock_s.openai_api_key = "test-key"
        mock_c.openai_api_key = "test-key"
        mock_g.openai_api_key = "test-key"
        yield


class TestAgenticRouter:
    @pytest.mark.asyncio
    async def test_mode_override_skips_llm(self) -> None:
        router = AgenticRouter()
        state = make_initial_state("test query", mode="graph")
        result = await router.classify(state)
        assert result["router_decision"] == "graph"

    @pytest.mark.asyncio
    async def test_llm_classify_vector(self) -> None:
        router = AgenticRouter()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "vector"
        mock_llm.ainvoke.return_value = mock_response
        router._llm = mock_llm

        state = make_initial_state("What did Apple say about AI?")
        result = await router.classify(state)
        assert "vector" in result["router_decision"]

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        router = AgenticRouter()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API error")
        router._llm = mock_llm

        state = make_initial_state("test")
        result = await router.classify(state)
        assert result["router_decision"] == "vector"


class TestCRAGNode:
    @pytest.mark.asyncio
    async def test_evaluate_empty_context(self) -> None:
        crag = CRAGNode()
        state = make_initial_state("test")
        result = await crag.evaluate(state)
        assert result["crag_confidence_score"] == 0.0
        assert result["retrieval_attempts"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self) -> None:
        crag = CRAGNode()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "0.85"
        mock_llm.ainvoke.return_value = mock_response
        crag._llm = mock_llm

        state: AgentState = {
            **make_initial_state("test query"),
            "retrieval_results": [
                {
                    "source_type": "vector",
                    "content": "Apple revenue grew 14% in Q1 2026.",
                    "score": 0.92,
                    "metadata": {},
                }
            ],
        }
        result = await crag.evaluate(state)
        assert result["crag_confidence_score"] == 0.85

    @pytest.mark.asyncio
    async def test_should_retry_low_confidence(self) -> None:
        crag = CRAGNode()
        state: AgentState = {
            **make_initial_state("test"),
            "crag_confidence_score": 0.4,
            "retrieval_attempts": 1,
            "max_retrieval_attempts": 3,
        }
        assert crag.should_retry(state) == "rewrite"

    @pytest.mark.asyncio
    async def test_should_not_retry_high_confidence(self) -> None:
        crag = CRAGNode()
        state: AgentState = {
            **make_initial_state("test"),
            "crag_confidence_score": 0.85,
            "retrieval_attempts": 1,
            "max_retrieval_attempts": 3,
        }
        assert crag.should_retry(state) == "generate"

    @pytest.mark.asyncio
    async def test_should_not_exceed_max_attempts(self) -> None:
        crag = CRAGNode()
        state: AgentState = {
            **make_initial_state("test"),
            "crag_confidence_score": 0.3,
            "retrieval_attempts": 3,
            "max_retrieval_attempts": 3,
        }
        assert crag.should_retry(state) == "generate"


class TestContextAggregator:
    @pytest.mark.asyncio
    async def test_aggregate_empty(self) -> None:
        agg = ContextAggregator()
        state = make_initial_state("test")
        result = await agg.aggregate(state)
        assert result["aggregated_context"] == ""

    @pytest.mark.asyncio
    async def test_aggregate_multiple_sources(self) -> None:
        agg = ContextAggregator()
        state: AgentState = {
            **make_initial_state("test"),
            "retrieval_results": [
                {
                    "source_type": "vector",
                    "content": "Content A",
                    "score": 0.9,
                    "metadata": {"company": "AAPL"},
                },
                {
                    "source_type": "graph",
                    "content": "Content B",
                    "score": 0.85,
                    "metadata": {},
                },
            ],
        }
        result = await agg.aggregate(state)
        assert "Content A" in result["aggregated_context"]
        assert "Content B" in result["aggregated_context"]
        assert len(result["sources"]) == 2
