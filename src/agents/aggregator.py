from __future__ import annotations

from src.agents.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContextAggregator:
    async def aggregate(self, state: AgentState) -> dict:
        results = state.get("retrieval_results", [])
        if not results:
            return {"aggregated_context": ""}

        sections: list[str] = []
        sources: list[dict] = []

        for i, result in enumerate(results):
            if isinstance(result, dict):
                source_type = result.get("source_type", "unknown")
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                score = result.get("score", 0.0)
            else:
                source_type = result.source_type
                content = result.content
                metadata = result.metadata
                score = result.score

            header = f"[Source {i + 1}] Type: {source_type.upper()}"
            if metadata:
                meta_preview = {
                    k: v
                    for k, v in metadata.items()
                    if k in ("company", "ticker", "document_type", "page", "quarter")
                }
                if meta_preview:
                    header += f" | {meta_preview}"
            sections.append(f"{header}\n{content}")

            sources.append(
                {
                    "type": source_type,
                    "score": score,
                    **metadata,
                }
            )

        context = "\n\n".join(sections)
        logger.info(
            "context_aggregated",
            sources=len(sources),
            total_chars=len(context),
        )

        return {
            "aggregated_context": context,
            "sources": sources,
        }
