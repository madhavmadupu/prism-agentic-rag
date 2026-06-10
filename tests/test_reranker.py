from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.retrieval.reranker import CohereReranker
from src.utils.exceptions import RetrievalError


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with patch("src.retrieval.reranker.settings") as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        yield


def _make_reranker() -> CohereReranker:
    reranker = CohereReranker()
    reranker._client = AsyncMock()
    return reranker


@pytest.mark.asyncio
async def test_rerank_empty_documents() -> None:
    reranker = _make_reranker()
    result = await reranker.rerank("test query", [])
    assert result == []


@pytest.mark.asyncio
async def test_rerank_returns_scores() -> None:
    reranker = _make_reranker()

    mock_response = AsyncMock()
    mock_result_1 = AsyncMock()
    mock_result_1.index = 0
    mock_result_1.relevance_score = 0.95
    mock_result_2 = AsyncMock()
    mock_result_2.index = 1
    mock_result_2.relevance_score = 0.42
    mock_response.results = [mock_result_1, mock_result_2]
    reranker._client.rerank.return_value = mock_response

    result = await reranker.rerank("test query", ["doc a", "doc b"], top_n=5)
    assert result == [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.42},
    ]


@pytest.mark.asyncio
async def test_rerank_propagates_error() -> None:
    reranker = _make_reranker()
    reranker._client.rerank.side_effect = Exception("API error")

    with pytest.raises(RetrievalError, match="Cohere reranking failed"):
        await reranker.rerank("test", ["doc"])
