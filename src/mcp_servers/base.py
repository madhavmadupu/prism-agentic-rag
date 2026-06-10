from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MCPServer(ABC):
    @abstractmethod
    def get_tool_name(self) -> str:
        ...

    @abstractmethod
    def get_tool_description(self) -> str:
        ...

    @abstractmethod
    def get_tool_schema(self) -> dict[str, Any]:
        ...

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        ...


class MCPRegistry:
    _servers: dict[str, MCPServer] = {}

    @classmethod
    def register(cls, server: MCPServer) -> None:
        name = server.get_tool_name()
        cls._servers[name] = server

    @classmethod
    def get_server(cls, name: str) -> MCPServer | None:
        return cls._servers.get(name)

    @classmethod
    def list_servers(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": s.get_tool_name(),
                "description": s.get_tool_description(),
                "schema": s.get_tool_schema(),
            }
            for s in cls._servers.values()
        ]

    @classmethod
    async def execute_tool(
        cls, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        server = cls.get_server(tool_name)
        if not server:
            return {"error": f"Unknown tool: {tool_name}"}
        return await server.execute(params)
