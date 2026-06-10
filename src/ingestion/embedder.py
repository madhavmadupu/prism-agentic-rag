from __future__ import annotations

from typing import Any

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CohereEmbedder:
    MODEL_ID: str = "embed-english-v3.0"
    INPUT_TYPE: str = "search_document"

    def __init__(self) -> None:
        if not settings.cohere_api_key:
            raise EmbeddingError("COHERE_API_KEY is not configured")
        self._client: Any = cohere.AsyncClientV2(api_key=settings.cohere_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = await self._client.embed(
                texts=texts,
                model=self.MODEL_ID,
                input_type=self.INPUT_TYPE,
                embedding_types=["float"],
            )

            if not response.embeddings or not response.embeddings.float_:
                raise EmbeddingError("Empty embedding response from Cohere")

            return response.embeddings.float_

        except Exception as exc:
            logger.error("embedding_failed", error=str(exc), text_count=len(texts))
            raise EmbeddingError(f"Cohere embedding failed: {exc}") from exc
