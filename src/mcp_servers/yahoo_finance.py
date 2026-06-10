from __future__ import annotations

from datetime import datetime
from typing import Any

import yfinance as yf

from src.mcp_servers.base import MCPServer, MCPRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceServer(MCPServer):
    def get_tool_name(self) -> str:
        return "yahoo_finance"

    def get_tool_description(self) -> str:
        return (
            "Fetch real-time and historical financial data for publicly traded "
            "companies including stock price, P/E ratio, market cap, and "
            "quarterly financial metrics."
        )

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
                },
                "metric": {
                    "type": "string",
                    "enum": [
                        "overview",
                        "price",
                        "pe_ratio",
                        "market_cap",
                        "financials",
                        "income_statement",
                        "balance_sheet",
                    ],
                    "description": "Financial metric to retrieve",
                },
            },
            "required": ["ticker", "metric"],
        }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        ticker = params.get("ticker", "").upper().strip()
        metric = params.get("metric", "overview")

        if not ticker:
            return {"error": "Missing required parameter: ticker"}

        try:
            stock = yf.Ticker(ticker)

            if metric == "price":
                return await self._get_price(stock, ticker)
            elif metric == "pe_ratio":
                return await self._get_pe_ratio(stock, ticker)
            elif metric == "market_cap":
                return await self._get_market_cap(stock, ticker)
            elif metric == "financials":
                return await self._get_financials(stock, ticker)
            elif metric == "overview":
                return await self._get_overview(stock, ticker)
            else:
                return {"error": f"Unknown metric: {metric}"}

        except Exception as exc:
            logger.error(
                "yahoo_finance_failed",
                ticker=ticker,
                metric=metric,
                error=str(exc),
            )
            return {
                "error": f"Failed to fetch Yahoo Finance data for {ticker}: {exc}",
                "ticker": ticker,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _get_price(self, stock: yf.Ticker, ticker: str) -> dict:
        info = stock.info or {}
        hist = stock.history(period="1d")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None and not hist.empty:
            price = float(hist["Close"].iloc[-1])

        return {
            "ticker": ticker,
            "current_price": price,
            "currency": info.get("currency", "USD"),
            "previous_close": info.get("previousClose"),
            "day_change_pct": info.get("regularMarketChangePercent"),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Yahoo Finance",
        }

    async def _get_pe_ratio(self, stock: yf.Ticker, ticker: str) -> dict:
        info = stock.info or {}
        return {
            "ticker": ticker,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Yahoo Finance",
        }

    async def _get_market_cap(self, stock: yf.Ticker, ticker: str) -> dict:
        info = stock.info or {}
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Yahoo Finance",
        }

    async def _get_financials(self, stock: yf.Ticker, ticker: str) -> dict:
        financials = stock.quarterly_financials
        if financials is None or financials.empty:
            financials = stock.financials

        summary: dict[str, Any] = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Yahoo Finance",
        }

        if financials is not None and not financials.empty:
            cols = financials.columns
            latest = financials[cols[0]] if len(cols) > 0 else None
            if latest is not None:
                summary["latest_quarter"] = str(cols[0])
                for field in ["Total Revenue", "Net Income", "Operating Income",
                              "Gross Profit", "Research and Development"]:
                    if field in latest.index:
                        summary[field.lower().replace(" ", "_")] = float(latest[field])

        return summary

    async def _get_overview(self, stock: yf.Ticker, ticker: str) -> dict:
        info = stock.info or {}
        hist = stock.history(period="5d")
        price_history = []
        if not hist.empty:
            price_history = [
                {
                    "date": str(idx.date()),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
                for idx, row in hist.iterrows()
            ]

        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": info.get("currentPrice"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "price_history_5d": price_history,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Yahoo Finance",
        }


MCPRegistry.register(YahooFinanceServer())
