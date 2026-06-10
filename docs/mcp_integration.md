# MCP Integration — Model Context Protocol

## Overview

P.R.I.S.M. implements the **Model Context Protocol (MCP)** to connect the agentic pipeline with live external data sources. MCP provides a standardized interface for LLMs to call external tools and APIs in real time.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│  LangGraph       │     │  MCP Registry         │
│  Agent           │────►│  (src/mcp_servers/)   │
│                  │     │                       │
│  retrieve_mcp    │     │  YahooFinanceServer   │
│  node            │     │  (yfinance library)   │
└─────────────────┘     └──────────────────────┘
```

---

## Available MCP Tools

### Yahoo Finance (`yahoo_finance`)

Fetches real-time and historical financial data.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | `string` | Yes | Stock ticker symbol (AAPL, MSFT, etc.) |
| `metric` | `string` | Yes | One of: `overview`, `price`, `pe_ratio`, `market_cap`, `financials` |

**Metrics supported:**

| Metric | Returns |
|--------|---------|
| `overview` | Full company profile: name, sector, price, P/E, market cap, 52-week range, 5-day price history |
| `price` | Current price, previous close, day change percentage |
| `pe_ratio` | Trailing P/E, Forward P/E, PEG ratio |
| `market_cap` | Market capitalization, enterprise value |
| `financials` | Latest quarterly financial data (revenue, net income, R&D) |

**Example usage:**

```json
{
  "tool": "yahoo_finance",
  "params": {
    "ticker": "AAPL",
    "metric": "overview"
  }
}
```

**Example response:**

```json
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "current_price": 198.50,
  "pe_ratio": 32.4,
  "market_cap": 3100000000000,
  "sector": "Technology",
  "timestamp": "2026-06-10T14:30:00Z",
  "source": "Yahoo Finance"
}
```

---

## Registering a New MCP Server

1. Create a class that extends `MCPServer` (from `src/mcp_servers/base.py`)
2. Implement the required abstract methods:

```python
from src.mcp_servers.base import MCPServer, MCPRegistry


class MyCustomServer(MCPServer):
    def get_tool_name(self) -> str:
        return "my_tool"

    def get_tool_description(self) -> str:
        return "Description of what this tool does"

    def get_tool_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
            },
            "required": ["param1"],
        }

    async def execute(self, params: dict) -> dict:
        # Your data fetching logic here
        return {"result": "success", "data": ...}


MCPRegistry.register(MyCustomServer())
```

3. The server is now available to the LangGraph MCP node automatically.

---

## How the MCP Node Works

The `MCPNode` in `src/agents/mcp_node.py`:

1. Parses the user query to detect a ticker symbol (e.g., "Apple" -> "AAPL")
2. Determines the requested metric from query keywords
3. Calls `MCPRegistry.execute_tool()` with the Yahoo Finance server
4. Formats the response into a context string
5. Returns it as a `RetrievalResult` with `source_type: "mcp"`

---

## Agent Routing

When the `AgenticRouter` classifies a query as `"mcp"`, the graph routes to `retrieve_mcp` node. For example:

- *"What is Apple's current P/E ratio?"* -> `mcp` route
- *"Show me Microsoft's latest stock price"* -> `mcp` route
- *"What is the market cap of Tesla?"* -> `mcp` route

---

## Future MCP Servers

Potential additions for future phases:

- **Alpha Vantage** — Alternative financial data provider
- **SEC EDGAR** — SEC filing retrieval
- **News API** — Real-time financial news sentiment
- **Bloomberg Terminal** — Premium financial data (requires license)
