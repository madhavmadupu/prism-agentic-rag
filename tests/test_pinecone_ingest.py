from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.pinecone_ingest import PineconeIngester


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with patch("src.ingestion.pinecone_ingest.settings") as mock_settings:
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"
        yield


def _make_ingester() -> PineconeIngester:
    ingester = PineconeIngester()
    ingester._embedder = AsyncMock()
    ingester._index = MagicMock()
    return ingester


@pytest.mark.asyncio
async def test_ingest_empty_chunks() -> None:
    ingester = _make_ingester()
    result = await ingester.ingest_chunks([])
    assert result == 0


@pytest.mark.asyncio
async def test_ingest_single_batch() -> None:
    ingester = _make_ingester()
    ingester._embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

    chunks = [
        Chunk(text="doc 1", metadata={"source": "test"}),
        Chunk(text="doc 2", metadata={"source": "test"}),
    ]

    result = await ingester.ingest_chunks(chunks)
    assert result == 2
    ingester._index.upsert.assert_called_once()
