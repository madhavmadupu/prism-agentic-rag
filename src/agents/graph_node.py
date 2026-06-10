from __future__ import annotations

from src.agents.state import AgentState, RetrievalResult
from src.graph.client import Neo4jClient
from src.graph.cypher_generator import CypherGenerator
from src.graph.query_executor import GraphQueryExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphNode:
    def __init__(self) -> None:
        self._client = Neo4jClient()
        self._generator = CypherGenerator()
        self._executor = GraphQueryExecutor(self._client, self._generator)

    async def run(self, state: AgentState) -> dict:
        query = state.get("query", "")
        try:
            await self._client.connect()
            result = await self._executor.answer_question(query)

            if not result or result.startswith("No matching"):
                return {"retrieval_results": []}

            retrieval = RetrievalResult(
                source_type="graph",
                content=result,
                score=0.9,
                metadata={"query": query},
            )

            logger.info("graph_node_complete", query=query[:60])
            return {"retrieval_results": [retrieval.__dict__]}

        except Exception as exc:
            logger.error("graph_node_failed", error=str(exc))
            return {"retrieval_results": []}
