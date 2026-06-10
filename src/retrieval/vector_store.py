from __future__ import annotations

from typing import Any

from pinecone import Pinecone

from src.config import settings
from src.ingestion.embedder import CohereEmbedder
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RetrievalError("PINECONE_API_KEY is not configured")

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._embedder = CohereEmbedder()
        self._index = self._pc.Index(settings.pinecone_index_name)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embedder.embed([query])
        if not query_embedding:
            raise RetrievalError("Failed to generate query embedding")

        try:
            results = self._index.query(
                vector=query_embedding[0],
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True,
            )

            matches = list(results.matches)
            logger.debug(
                "vector_search_complete",
                query=query[:60],
                matches=len(matches),
            )
            return matches

        except Exception as exc:
            logger.error("vector_search_failed", error=str(exc), query=query[:60])
            raise RetrievalError(f"Pinecone query failed: {exc}") from exc
