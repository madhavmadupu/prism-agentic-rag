from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ANSWER_PROMPT = """You are a financial analyst AI assistant. Answer the user's query based strictly on the provided context. If the context does not contain sufficient information, state that clearly — do not hallucinate.

Context:
{context}

Query: {query}

Instructions:
1. Cite specific sources where possible (e.g., "According to the Q1 2026 filing...").
2. If financial figures are provided, format them clearly (e.g., "$7.5B").
3. If the context includes graph data, mention the entity relationships found.
4. If data appears contradictory, flag the discrepancy.
5. If you cannot answer from the context, say "I cannot answer this from the available documents."

{disclaimer}

Answer:"""

DISCLAIMER = "Disclaimer: This is for informational purposes only and does not constitute financial advice."


class AnswerGenerator:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    async def generate(self, state: AgentState) -> dict:
        context = state.get("aggregated_context", "")
        query = state.get("query", "")

        if not context:
            return {
                "answer": (
                    "I could not find relevant information to answer your query. "
                    "Please try rephrasing or providing more specific details."
                )
            }

        try:
            response = await self._llm.ainvoke(
                ANSWER_PROMPT.format(
                    context=context,
                    query=query,
                    disclaimer=DISCLAIMER,
                )
            )
            answer = response.content.strip()

            usage_meta = getattr(response, "usage_metadata", {}) or {}
            tokens = (usage_meta.get("input_tokens", 0) if usage_meta else 0) + (
                usage_meta.get("output_tokens", 0) if usage_meta else 0
            )

            logger.info(
                "answer_generated",
                answer_length=len(answer),
                tokens_used=tokens,
            )
            return {"answer": answer, "tokens_used": tokens}

        except Exception as exc:
            logger.error("answer_generation_failed", error=str(exc))
            return {
                "answer": "An error occurred while generating the answer. Please try again."
            }
