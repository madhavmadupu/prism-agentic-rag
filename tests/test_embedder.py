from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.embedder import CohereEmbedder
from src.utils.exceptions import EmbeddingError


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        yield


def _make_embedder() -> CohereEmbedder:
    embedder = CohereEmbedder()
    embedder._client = AsyncMock()
    return embedder


@pytest.mark.asyncio
async def test_embed_empty_texts() -> None:
    embedder = _make_embedder()
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_returns_float_vectors() -> None:
    embedder = _make_embedder()

    mock_response = AsyncMock()
    mock_response.embeddings.float_ = [[0.1, 0.2, 0.3]]
    embedder._client.embed.return_value = mock_response

    result = await embedder.embed(["test text"])
    assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_embed_raises_on_empty_response() -> None:
    embedder = _make_embedder()

    mock_response = AsyncMock()
    mock_response.embeddings.float_ = None
    embedder._client.embed.return_value = mock_response

    with pytest.raises(EmbeddingError, match="Empty embedding response"):
        await embedder.embed(["test"])
