from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.mcp_node import MCPNode
from src.agents.state import make_initial_state
from src.mcp_servers.base import MCPRegistry, MCPServer


class _TestServer(MCPServer):
    def get_tool_name(self) -> str:
        return "test_tool"

    def get_tool_description(self) -> str:
        return "Test server"

    def get_tool_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, params: dict) -> dict:
        return {"result": "success", "ticker": "AAPL", "timestamp": "2026-01-01"}


class TestMCPRegistry:
    def setup_method(self) -> None:
        if "test_tool" not in MCPRegistry._servers:
            MCPRegistry.register(_TestServer())

    def test_register_and_list(self) -> None:
        servers = MCPRegistry.list_servers()
        names = [s["name"] for s in servers]
        assert "test_tool" in names

    def test_get_server(self) -> None:
        server = MCPRegistry.get_server("test_tool")
        assert server is not None
        assert server.get_tool_name() == "test_tool"

    def test_get_unknown_server(self) -> None:
        server = MCPRegistry.get_server("nonexistent")
        assert server is None

    @pytest.mark.asyncio
    async def test_execute_tool(self) -> None:
        result = await MCPRegistry.execute_tool("test_tool", {})
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        result = await MCPRegistry.execute_tool("unknown", {})
        assert "error" in result


class TestMCPNode:
    @pytest.mark.asyncio
    async def test_no_ticker_returns_empty(self) -> None:
        node = MCPNode()
        state = make_initial_state("What is the weather today?")
        result = await node.run(state)
        assert result["retrieval_results"] == []

    @pytest.mark.asyncio
    async def test_extracts_ticker_from_query(self) -> None:
        node = MCPNode()
        ticker = node._extract_ticker("What is apple stock price?")
        assert ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_determines_price_metric(self) -> None:
        node = MCPNode()
        assert node._determine_metric("stock price") == "price"

    @pytest.mark.asyncio
    async def test_determines_pe_metric(self) -> None:
        node = MCPNode()
        assert node._determine_metric("what is the pe ratio") == "pe_ratio"

    @pytest.mark.asyncio
    async def test_determines_market_cap_metric(self) -> None:
        node = MCPNode()
        assert node._determine_metric("market cap of") == "market_cap"
