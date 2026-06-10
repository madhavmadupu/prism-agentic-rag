from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import settings
from src.utils.exceptions import GraphQueryError
from src.utils.logger import get_logger

logger = get_logger(__name__)


CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher expert. Convert the given natural language question into a Cypher query.

### Graph Schema:
**Nodes:**
- (:Company {ticker, name, sector, description, founded_year})
- (:Executive {name, title, alma_mater, education, age})
- (:FinancialMetric {quarter, fiscal_year, revenue, rnd_spend, net_income, eps, pe_ratio, market_cap})
- (:Sector {name, description})

**Relationships:**
- (:Executive)-[:IS_CEO_OF]->(:Company)
- (:Executive)-[:IS_CFO_OF]->(:Company)
- (:Executive)-[:IS_CTO_OF]->(:Company)
- (:Company)-[:OPERATES_IN]->(:Sector)
- (:Company)-[:REPORTED]->(:FinancialMetric)
- (:Executive)-[:SUCCEEDED]->(:Executive)
- (:Executive)-[:BOARD_MEMBER_OF]->(:Company)

### Rules:
1. Always use parameterized queries with $param syntax for literal values.
2. Use OPTIONAL MATCH for non-critical paths.
3. Return only the properties needed to answer the question.
4. Limit results to 20 unless specified otherwise.
5. Use COLLECT and UNWIND for aggregations where appropriate.

### Examples:
Q: "Who is the CEO of Apple?"
Cypher: MATCH (e:Executive)-[:IS_CEO_OF]->(c:Company {{ticker: $ticker}}) RETURN e.name, e.title

Q: "Compare R&D spend of Apple and Microsoft in Q1 2026"
Cypher: MATCH (c:Company)-[:REPORTED]->(f:FinancialMetric) WHERE c.ticker IN $tickers AND f.quarter = $quarter RETURN c.name, f.rnd_spend, f.fiscal_year

Q: "What companies operate in the Technology sector?"
Cypher: MATCH (c:Company)-[:OPERATES_IN]->(s:Sector {{name: $sector}}) RETURN c.ticker, c.name

### Question:
{question}

Return ONLY the Cypher query. No explanations, no markdown formatting."""


class CypherGenerator:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    async def generate(self, question: str) -> str:
        if not question.strip():
            raise GraphQueryError("Empty question for Cypher generation")

        try:
            response = await self._llm.ainvoke(
                CYPHER_GENERATION_PROMPT.format(question=question)
            )
            cypher = response.content.strip()
            cypher = cypher.removeprefix("```cypher").removeprefix("```sql")
            cypher = cypher.removesuffix("```").strip()

            if not cypher.upper().startswith(("MATCH", "CALL", "RETURN")):
                raise GraphQueryError(
                    f"Generated invalid Cypher: {cypher[:100]}"
                )

            logger.debug("cypher_generated", query=cypher[:100])
            return cypher

        except GraphQueryError:
            raise
        except Exception as exc:
            logger.error("cypher_generation_failed", error=str(exc))
            raise GraphQueryError(f"Cypher generation failed: {exc}") from exc
