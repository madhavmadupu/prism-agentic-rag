from __future__ import annotations

from src.agents.state import AgentState, RetrievalResult
from src.retrieval.reranker import CohereReranker
from src.retrieval.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalNode:
    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._reranker = CohereReranker()

    async def run(self, state: AgentState) -> dict:
        rewritten = state.get("rewritten_query", "")
        original = state.get("query", "")
        query = rewritten or original

        raw_matches = await self._vector_store.search(
            query=query,
            top_k=10,
        )

        if not raw_matches:
            logger.warning("vector_search_empty", query=query[:60])
            return {"retrieval_results": []}

        doc_texts = [
            m.get("metadata", {}).get("chunk_text", "")
            or m.get("metadata", {}).get("text", "")
            for m in raw_matches
        ]
        doc_texts = [t for t in doc_texts if t]

        reranked = await self._reranker.rerank(
            query=query,
            documents=doc_texts,
            top_n=5,
        )

        results: list[dict] = []
        for rank in reranked:
            idx = rank["index"]
            match = raw_matches[idx]
            text = doc_texts[idx]
            results.append(
                RetrievalResult(
                    source_type="vector",
                    content=text,
                    score=rank["relevance_score"],
                    metadata=match.get("metadata", {}),
                ).__dict__
            )

        logger.info(
            "retrieval_complete",
            query=query[:60],
            raw_matches=len(raw_matches),
            reranked=len(results),
        )
        return {"retrieval_results": results}
