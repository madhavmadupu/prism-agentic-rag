from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.cypher_generator import CypherGenerator
from src.graph.query_executor import GraphQueryExecutor
from src.graph.schema import (
    NodeLabel,
    NodeSchema,
    RelationSchema,
    RelationType,
    cypher_create_node,
    cypher_create_relation,
)
from src.utils.exceptions import GraphQueryError


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with (
        patch("src.graph.client.settings") as mock_client_settings,
        patch("src.graph.cypher_generator.settings") as mock_gen_settings,
    ):
        mock_client_settings.neo4j_uri = "bolt://localhost:7687"
        mock_client_settings.neo4j_user = "neo4j"
        mock_client_settings.neo4j_password = "password"
        mock_gen_settings.openai_api_key = "test-key"
        yield


class TestSchema:
    def test_cypher_create_node(self) -> None:
        cypher = cypher_create_node(
            NodeSchema(label=NodeLabel.COMPANY, properties={"ticker": str, "name": str}, required_properties=["ticker", "name"]),
            {"ticker": "AAPL", "name": "Apple Inc."},
        )
        assert "MERGE (company:Company {ticker: $ticker, name: $name})" in cypher

    def test_cypher_create_relation(self) -> None:
        cypher = cypher_create_relation(
            RelationSchema(type=RelationType.IS_CEO_OF, from_label=NodeLabel.EXECUTIVE, to_label=NodeLabel.COMPANY),
            {"name": "Tim Cook"},
            {"ticker": "AAPL"},
        )
        assert "MERGE (a)-[r:IS_CEO_OF]->(b)" in cypher


class TestCypherGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_cypher(self) -> None:
        generator = CypherGenerator()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "MATCH (e:Executive)-[:IS_CEO_OF]->(c:Company {ticker: $ticker}) RETURN e.name"
        mock_llm.ainvoke.return_value = mock_response
        generator._llm = mock_llm

        result = await generator.generate("Who is the CEO of Apple?")
        assert result.startswith("MATCH")

    @pytest.mark.asyncio
    async def test_generate_rejects_invalid_cypher(self) -> None:
        generator = CypherGenerator()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "SELECT * FROM users"
        mock_llm.ainvoke.return_value = mock_response
        generator._llm = mock_llm

        with pytest.raises(GraphQueryError, match="Generated invalid Cypher"):
            await generator.generate("test question")


class TestGraphQueryExecutor:
    @pytest.mark.asyncio
    async def test_format_results(self) -> None:
        mock_client = AsyncMock()
        mock_generator = AsyncMock()
        executor = GraphQueryExecutor(mock_client, mock_generator)
        mock_generator.generate.return_value = "MATCH (c:Company) RETURN c.ticker LIMIT 1"
        mock_client.run_query.return_value = [{"c.ticker": "AAPL"}]

        result = await executor.answer_question("List companies")
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        mock_client = AsyncMock()
        mock_generator = AsyncMock()
        executor = GraphQueryExecutor(mock_client, mock_generator)
        mock_generator.generate.return_value = "MATCH (c:Company) RETURN c.ticker"
        mock_client.run_query.return_value = []

        result = await executor.answer_question("List companies")
        assert "No matching data" in result
