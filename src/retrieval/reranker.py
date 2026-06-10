from __future__ import annotations

from typing import Any

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CohereReranker:
    MODEL_ID: str = "rerank-english-v3.0"

    def __init__(self) -> None:
        if not settings.cohere_api_key:
            raise RetrievalError("COHERE_API_KEY is not configured")
        self._client: Any = cohere.AsyncClientV2(api_key=settings.cohere_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        try:
            response = await self._client.rerank(
                model=self.MODEL_ID,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            results: list[dict[str, Any]] = []
            for result in response.results:
                results.append(
                    {
                        "index": result.index,
                        "relevance_score": result.relevance_score,
                    }
                )

            logger.debug(
                "rerank_complete",
                query=query[:60],
                input_docs=len(documents),
                output_docs=len(results),
            )
            return results

        except Exception as exc:
            logger.error("rerank_failed", error=str(exc), query=query[:60])
            raise RetrievalError(f"Cohere reranking failed: {exc}") from exc
