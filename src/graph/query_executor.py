from __future__ import annotations

from typing import Any

from src.graph.client import Neo4jClient
from src.graph.cypher_generator import CypherGenerator
from src.utils.exceptions import GraphQueryError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphQueryExecutor:
    def __init__(
        self,
        client: Neo4jClient,
        generator: CypherGenerator,
    ) -> None:
        self._client = client
        self._generator = generator

    async def answer_question(self, question: str) -> str:
        try:
            cypher = await self._generator.generate(question)
            parameters = self._infer_parameters(question)
            records = await self._client.run_query(cypher, parameters)

            if not records:
                return "No matching data found in the knowledge graph."

            formatted = self._format_results(records)
            logger.info(
                "graph_query_completed",
                question=question[:60],
                records=len(records),
            )
            return formatted

        except GraphQueryError:
            raise
        except Exception as exc:
            logger.error("graph_query_execution_failed", error=str(exc))
            raise GraphQueryError(
                f"Graph query execution failed: {exc}"
            ) from exc

    def _infer_parameters(self, question: str) -> dict[str, Any]:
        params: dict[str, Any] = {}

        tickers_map = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "nvidia": "NVDA",
            "tesla": "TSLA",
        }
        q_lower = question.lower()
        matched_tickers = [
            ticker for word, ticker in tickers_map.items() if word in q_lower
        ]
        if matched_tickers:
            params["tickers"] = matched_tickers

        quarter_map = {
            "q1": "Q1",
            "q2": "Q2",
            "q3": "Q3",
            "q4": "Q4",
        }
        for word, q in quarter_map.items():
            if word in q_lower:
                params["quarter"] = q
                break

        if "sector" in q_lower or "operates in" in q_lower:
            sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
            for sector in sectors:
                if sector.lower() in q_lower:
                    params["sector"] = sector
                    break

        if "technology" in q_lower:
            params.setdefault("sector", "Technology")

        return params

    def _format_results(self, records: list[dict]) -> str:
        if not records:
            return "No results found."

        lines: list[str] = []
        for record in records:
            parts = []
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    if key in ("revenue", "rnd_spend", "net_income", "market_cap"):
                        parts.append(f"{key}: ${value:,.1f}")
                    else:
                        parts.append(f"{key}: {value}")
                else:
                    parts.append(f"{key}: {value}")
            lines.append(" | ".join(parts))

        context = "\n".join(lines)
        logger.debug(
            "graph_results_formatted",
            record_count=len(records),
            context_length=len(context),
        )
        return context
