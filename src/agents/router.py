from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTER_PROMPT = """You are a query router for a financial RAG system. Classify the user's query into one or more of the following categories:

- "graph": Query involves entity relationships, multi-hop reasoning, company comparisons, executive backgrounds, or sector analysis. (e.g., "Compare R&D spend of companies whose CEOs are MIT alumni")
- "vector": Query involves semantic search over financial documents, finding specific passages, or retrieving textual information. (e.g., "What did Apple say about AI in their latest 10-K?")
- "mcp": Query requires real-time or live financial data such as current stock prices, P/E ratios, or market data. (e.g., "What is Apple's current P/E ratio?")
- "multimodal": Query references charts, graphs, images, or asks for visual analysis of financial data. (e.g., "Show me the revenue trend chart from the latest earnings PDF")

If a query fits multiple categories, return a comma-separated list ordered by relevance.
If the query is a follow-up or clarification, consider the conversation context.

Query: {query}
Mode override (if not "auto"): {mode}

Return ONLY a comma-separated list of categories. No explanations."""


class AgenticRouter:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    async def classify(self, state: AgentState) -> dict:
        mode = state.get("mode", "auto")
        if mode != "auto":
            return {"router_decision": mode}

        query = state.get("query", "")
        try:
            response = await self._llm.ainvoke(
                ROUTER_PROMPT.format(
                    query=query,
                    mode=mode,
                )
            )
            decision = response.content.strip().lower()
            decision = decision.removeprefix("```").removesuffix("```").strip()

            logger.info(
                "router_classified",
                query=query[:60],
                decision=decision,
            )
            return {"router_decision": decision}

        except Exception as exc:
            logger.error("router_classification_failed", error=str(exc))
            return {"router_decision": "vector"}
