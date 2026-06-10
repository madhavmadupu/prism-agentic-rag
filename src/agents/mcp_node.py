from __future__ import annotations

from src.agents.state import AgentState, RetrievalResult
from src.mcp_servers.base import MCPRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

MCP_TRIGGER_PATTERNS = [
    "stock price",
    "pe ratio",
    "market cap",
    "current price",
    "real-time",
    "live data",
    "trading",
    "dividend",
    "yahoo finance",
]


class MCPNode:
    async def run(self, state: AgentState) -> dict:
        query = state.get("query", "").lower()
        ticker = self._extract_ticker(query)

        if not ticker:
            logger.info("mcp_skipped_no_ticker", query=query[:60])
            return {"retrieval_results": []}

        metric = self._determine_metric(query)

        try:
            result = await MCPRegistry.execute_tool(
                "yahoo_finance",
                {"ticker": ticker, "metric": metric},
            )

            if "error" in result:
                logger.warning("mcp_error", error=result["error"])
                return {"retrieval_results": []}

            content = self._format_result(result, metric)

            retrieval = RetrievalResult(
                source_type="mcp",
                content=content,
                score=1.0,
                metadata={
                    "ticker": ticker,
                    "metric": metric,
                    "timestamp": result.get("timestamp", ""),
                    "source": "Yahoo Finance",
                },
            )

            logger.info("mcp_data_fetched", ticker=ticker, metric=metric)
            return {"retrieval_results": [retrieval.__dict__]}

        except Exception as exc:
            logger.error("mcp_node_failed", ticker=ticker, error=str(exc))
            return {"retrieval_results": []}

    def _extract_ticker(self, query: str) -> str | None:
        known_tickers = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "tesla": "TSLA",
            "netflix": "NFLX",
            "berkshire": "BRK-A",
            "jpmorgan": "JPM",
            "goldman": "GS",
            "visa": "V",
            "mastercard": "MA",
        }

        for name, ticker in known_tickers.items():
            if name in query:
                return ticker
        return None

    def _determine_metric(self, query: str) -> str:
        if "pe_ratio" in query or "price to earnings" in query or "p/e" in query:
            return "pe_ratio"
        if "market cap" in query or "market capitalization" in query:
            return "market_cap"
        if "financials" in query or "revenue" in query or "earnings" in query:
            return "financials"
        if "overview" in query or "summary" in query or "profile" in query:
            return "overview"
        return "price"

    def _format_result(self, result: dict, metric: str) -> str:
        ticker = result.get("ticker", "N/A")
        timestamp = result.get("timestamp", "")

        if metric == "price":
            price = result.get("current_price", "N/A")
            change = result.get("day_change_pct", "N/A")
            return (
                f"Real-time stock price for {ticker}: "
                f"${price} (day change: {change:.2%}) as of {timestamp}"
            )

        if metric == "pe_ratio":
            pe = result.get("pe_ratio", "N/A")
            fpe = result.get("forward_pe", "N/A")
            return f"{ticker} P/E Ratio: Trailing {pe}, Forward {fpe} (as of {timestamp})"

        if metric == "market_cap":
            mc = result.get("market_cap", "N/A")
            ev = result.get("enterprise_value", "N/A")
            mc_str = f"${mc:,.0f}" if isinstance(mc, (int, float)) else mc
            ev_str = f"${ev:,.0f}" if isinstance(ev, (int, float)) else ev
            return f"{ticker} Market Cap: {mc_str}, Enterprise Value: {ev_str} (as of {timestamp})"

        if metric == "financials":
            rev = result.get("total_revenue", "N/A")
            ni = result.get("net_income", "N/A")
            rd = result.get("research_and_development", "N/A")
            rev_str = f"${rev:,.0f}" if isinstance(rev, (int, float)) else rev
            ni_str = f"${ni:,.0f}" if isinstance(ni, (int, float)) else ni
            rd_str = f"${rd:,.0f}" if isinstance(rd, (int, float)) else rd
            return f"{ticker} Financials: Revenue {rev_str}, Net Income {ni_str}, R&D {rd_str}"

        return str(result)
