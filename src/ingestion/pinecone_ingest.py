from __future__ import annotations

from typing import Any

from pinecone import Pinecone

from src.config import settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import CohereEmbedder
from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PineconeIngester:
    BATCH_SIZE: int = 100

    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise IngestionError("PINECONE_API_KEY is not configured")

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._embedder = CohereEmbedder()
        self._index = self._pc.Index(settings.pinecone_index_name)

    async def ingest_chunks(
        self,
        chunks: list[Chunk],
        namespace: str = "",
    ) -> int:
        if not chunks:
            return 0

        ingested = 0
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            texts = [c.text for c in batch]
            embeddings = await self._embedder.embed(texts)

            vectors: list[dict[str, Any]] = []
            for chunk, embedding in zip(batch, embeddings):
                vectors.append(
                    {
                        "id": chunk.chunk_id or f"chunk_{i + ingested}",
                        "values": embedding,
                        "metadata": chunk.metadata,
                    }
                )

            try:
                self._index.upsert(
                    vectors=vectors,
                    namespace=namespace,
                )
                ingested += len(vectors)
                logger.debug(
                    "batch_ingested",
                    batch_size=len(vectors),
                    namespace=namespace,
                )
            except Exception as exc:
                logger.error(
                    "batch_ingest_failed",
                    error=str(exc),
                    batch_start=i,
                )
                raise IngestionError(f"Pinecone upsert failed: {exc}") from exc

        logger.info("ingestion_complete", total_chunks=ingested, namespace=namespace)
        return ingested
