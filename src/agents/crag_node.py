from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIDENCE_PROMPT = """You are a fact-checking evaluator for a financial RAG system. Given a user query and the retrieved context, evaluate whether the context is sufficient to answer the query accurately.

Query: {query}
Retrieved Context: {context}

Score the retrieval confidence on a scale of 0.0 to 1.0 based on:
1. **Relevance** (0-0.4): Does the context directly address the query?
2. **Completeness** (0-0.3): Does the context contain all necessary information?
3. **Factual alignment** (0-0.3): Is the context internally consistent and specific?

A score below 0.7 means the context is insufficient and re-retrieval is needed.

Return ONLY a float number between 0.0 and 1.0. No explanations."""

REWRITE_PROMPT = """The initial retrieval for the following query did not return sufficient results (confidence < 0.7). Rewrite the query to improve retrieval precision.

Original query: {query}
Previous attempt context (if any): {previous_context}

Suggestions for improvement:
- Add company names, tickers, or specific fiscal periods
- Use more precise financial terminology
- Break down multi-part questions into simpler terms

Return ONLY the rewritten query. No explanations."""


class CRAGNode:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    async def evaluate(self, state: AgentState) -> dict:
        results = state.get("retrieval_results", [])
        context = self._build_context(results)
        if not context.strip():
            return {
                "crag_confidence_score": 0.0,
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            }

        query = state.get("query", "")
        try:
            response = await self._llm.ainvoke(
                CONFIDENCE_PROMPT.format(query=query, context=context)
            )
            score = float(response.content.strip())
            score = max(0.0, min(1.0, score))

            attempts = state.get("retrieval_attempts", 0) + 1

            logger.info(
                "crag_evaluation",
                confidence=score,
                attempts=attempts,
                max_attempts=state.get("max_retrieval_attempts", 3),
            )
            return {
                "crag_confidence_score": score,
                "retrieval_attempts": attempts,
            }

        except (ValueError, Exception) as exc:
            logger.error("crag_evaluation_failed", error=str(exc))
            return {
                "crag_confidence_score": 0.5,
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            }

    async def rewrite_query(self, state: AgentState) -> dict:
        query = state.get("query", "")
        try:
            response = await self._llm.ainvoke(
                REWRITE_PROMPT.format(
                    query=query,
                    previous_context=self._build_context(
                        state.get("retrieval_results", [])
                    ),
                )
            )
            rewritten = response.content.strip()
            rewritten = rewritten.removeprefix("```").removesuffix("```").strip()

            logger.info(
                "query_rewritten",
                original=query[:40],
                rewritten=rewritten[:60],
            )
            return {"rewritten_query": rewritten}

        except Exception as exc:
            logger.error("query_rewrite_failed", error=str(exc))
            return {"rewritten_query": query}

    def _build_context(self, results: list) -> str:
        parts = []
        for r in results:
            if isinstance(r, dict):
                parts.append(f"[{r.get('source_type', 'unknown')}] {r.get('content', '')[:500]}")
            else:
                parts.append(f"[{r.source_type}] {r.content[:500]}")
        return "\n\n".join(parts)

    def should_retry(self, state: AgentState) -> str:
        if (
            state.get("crag_confidence_score", 0.0) < 0.7
            and state.get("retrieval_attempts", 0) < state.get("max_retrieval_attempts", 3)
        ):
            return "rewrite"
        return "generate"
