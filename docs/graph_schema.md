# Graph Schema — Structured Memory (Neo4j)

## Overview

P.R.I.S.M. uses a Neo4j knowledge graph for multi-hop entity reasoning. The graph stores financial entities (companies, executives, metrics, sectors) and their relationships, enabling complex traversals like *"Compare R&D spend of companies whose CEOs are MIT alumni"*.

---

## Node Types

### `:Company`
| Property | Type | Required | Unique | Description |
|----------|------|----------|--------|-------------|
| `ticker` | `str` | Yes | Yes | Stock ticker symbol (e.g., `AAPL`) |
| `name` | `str` | Yes | | Full company name |
| `sector` | `str` | | | Industry sector |
| `description` | `str` | | | Business description |
| `founded_year` | `int` | | | Year founded |

### `:Executive`
| Property | Type | Required | Unique | Description |
|----------|------|----------|--------|-------------|
| `name` | `str` | Yes | Yes | Full name |
| `title` | `str` | Yes | | Current title (CEO, CFO, etc.) |
| `alma_mater` | `str` | | | University attended |
| `education` | `str` | | | Education details |
| `age` | `int` | | | Age in years |

### `:FinancialMetric`
| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `quarter` | `str` | Yes | Quarter label (e.g., `Q1 2026`) |
| `fiscal_year` | `int` | Yes | Fiscal year |
| `revenue` | `float` | | Revenue in dollars |
| `rnd_spend` | `float` | | R&D spend in dollars |
| `net_income` | `float` | | Net income in dollars |
| `eps` | `float` | | Earnings per share |
| `pe_ratio` | `float` | | Price-to-earnings ratio |
| `market_cap` | `float` | | Market capitalization |

### `:Sector`
| Property | Type | Required | Unique | Description |
|----------|------|----------|--------|-------------|
| `name` | `str` | Yes | Yes | Sector name (e.g., `Technology`) |
| `description` | `str` | | | Sector description |

---

## Relationship Types

| Relationship | From | To | Description |
|-------------|------|----|-------------|
| `IS_CEO_OF` | `Executive` | `Company` | Executive is CEO of the company |
| `IS_CFO_OF` | `Executive` | `Company` | Executive is CFO of the company |
| `IS_CTO_OF` | `Executive` | `Company` | Executive is CTO of the company |
| `OPERATES_IN` | `Company` | `Sector` | Company operates in a sector |
| `REPORTED` | `Company` | `FinancialMetric` | Company reported a financial metric |
| `SUCCEEDED` | `Executive` | `Executive` | Executive succeeded another |
| `BOARD_MEMBER_OF` | `Executive` | `Company` | Executive is on the board |

---

## Example Queries

### Find the CEO of Apple
```cypher
MATCH (e:Executive)-[:IS_CEO_OF]->(c:Company {ticker: $ticker})
RETURN e.name, e.title
```

### Compare R&D spend of two companies
```cypher
MATCH (c:Company)-[:REPORTED]->(f:FinancialMetric)
WHERE c.ticker IN $tickers AND f.quarter = $quarter
RETURN c.name, f.rnd_spend, f.fiscal_year
```

### Find companies in Technology sector with high R&D spend
```cypher
MATCH (c:Company)-[:OPERATES_IN]->(s:Sector {name: $sector})
MATCH (c)-[:REPORTED]->(f:FinancialMetric)
WHERE f.rnd_spend > $min_rnd
RETURN c.ticker, c.name, f.rnd_spend, f.quarter
ORDER BY f.rnd_spend DESC
```

---

## Ingestion Pipeline

1. **Extract:** Financial document text is passed to `EntityExtractor` (GPT-4o) which returns structured entities and relationships.
2. **Transform:** `ExtractedEntity` and `ExtractedRelation` dataclasses are mapped to Cypher `MERGE` statements.
3. **Load:** Statements are executed via `Neo4jClient.run_query()` in batches.

See `src/graph/extractor.py` and `src/graph/client.py` for implementation details.
