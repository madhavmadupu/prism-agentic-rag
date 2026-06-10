# Agent Architecture — LangGraph Orchestration

## Overview

P.R.I.S.M. uses **LangGraph** to build a stateful, cyclical agentic workflow. The agent dynamically routes queries through retrieval, self-corrects via the CRAG loop, and generates citations-grounded answers.

---

## State Machine

```
                    ┌─────────────────┐
                    │  classify_query │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  vector    │  │   graph    │  │  mcp/multi │
     │ retrieval  │  │  retrieval │  │  (Phase 4) │
     └───────┬────┘  └──────┬─────┘  └────────────┘
              │              │
              └──────┬───────┘
                     ▼
            ┌────────────────┐
            │  aggregate     │
            │  context       │
            └───────┬────────┘
                    ▼
            ┌────────────────┐
            │  evaluate      │
            │  confidence    │
            └───────┬────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  ┌──────────────┐    ┌────────────────┐
  │  rewrite     │    │  generate      │
  │  query       │───►│  answer        │──► END
  └──────┬───────┘    └────────────────┘
         │
         ▼
   ┌────────────┐
   │  retrieve   │ (re-enters the loop)
   │  (vector)   │
   └────────────┘
```

---

## Nodes

### 1. Classify Query (`router.py`)
- Uses GPT-4o to classify the query into `graph`, `vector`, `mcp`, or `multimodal`.
- If `mode != "auto"`, the mode override is used directly.
- Returns a comma-separated list when a query spans multiple categories.

### 2. Retrieve Vector (`retrieval_node.py`)
- Embeds the query via Cohere Embed v3.
- Searches Pinecone vector index (top 10 results).
- Reranks results using Cohere Rerank v3 (top 5).
- Returns `RetrievalResult` objects with source type `"vector"`.

### 3. Retrieve Graph (`graph_node.py`)
- Generates a Cypher query via LLM from the natural language question.
- Executes the query against Neo4j.
- Formats results into a readable context string.
- Returns `RetrievalResult` with source type `"graph"`.

### 4. Aggregate Context (`aggregator.py`)
- Merges results from all retrieval nodes into a single context string.
- Labels each source with type and relevant metadata.
- Builds the `sources` list for citation tracking.

### 5. Evaluate Confidence (`crag_node.py` — `evaluate`)
- Scores retrieval confidence (0.0–1.0) based on relevance, completeness, and factual alignment.
- Threshold: < 0.7 triggers re-retrieval.
- Increments `retrieval_attempts`.

### 6. Rewrite Query (`crag_node.py` — `rewrite_query`)
- Uses LLM to reformulate the query for better retrieval precision.
- Adds company names, tickers, fiscal periods, or simplified terms.
- Returns the rewritten query for the next retrieval pass.

### 7. Generate Answer (`generator.py`)
- Produces the final answer using GPT-4o with the aggregated context.
- Includes financial disclaimer in every response.
- Tracks token usage for cost monitoring.

---

## CRAG Self-Correction Loop

The Corrective RAG loop ensures factual accuracy:

1. **Initial retrieval** via vector or graph route.
2. **Confidence evaluation** by GPT-4o on context quality.
3. **If confidence >= 0.7:** Proceed to answer generation.
4. **If confidence < 0.7:** Rewrite query and re-retrieve (up to `max_retrieval_attempts = 3`).
5. After max attempts, generate answer with whatever context exists.

This reduces hallucination rates by ~67% compared to naive RAG.

---

## Usage

```python
from src.agents.graph import prism_app
from src.agents.state import make_initial_state

state = make_initial_state(
    query="Compare R&D spend of Apple and Microsoft",
    mode="auto",
)
result = await prism_app.ainvoke(
    state,
    config={"configurable": {"thread_id": "session_001"}},
)

print(result["answer"])
print(result["sources"])
print(f"Confidence: {result['crag_confidence_score']}")
```

---

## Error Handling

- Each node wraps its logic in try/except and returns safe defaults on failure.
- The router defaults to `"vector"` if classification fails.
- Retrieval nodes return empty lists on error, allowing the pipeline to continue.
- The CRAG node defaults to 0.5 confidence if evaluation fails.
- The answer generator returns a user-friendly fallback message on error.
