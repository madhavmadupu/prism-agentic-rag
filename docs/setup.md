# Setup Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- API keys for: OpenAI, Cohere, Pinecone

---

## Environment Configuration

Copy `.env.example` to `.env` and populate:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | GPT-4o for inference, vision, routing |
| `COHERE_API_KEY` | Yes | — | Embed v3 & Rerank v3 |
| `PINECONE_API_KEY` | Yes | — | Vector index |
| `PINECONE_INDEX_NAME` | No | `prism-index` | Pinecone index name |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `REDIS_URL` | No | `redis://localhost:6379` | Cache connection |

---

## Docker Setup (Recommended)

```bash
docker-compose up --build
```

| Service | Port | Purpose |
|---------|------|---------|
| `prism-api` | 8000 | FastAPI application |
| `prism-redis` | 6379 | Semantic caching |
| `prism-neo4j` | 7687 / 7474 | Knowledge graph |

---

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

uvicorn src.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

---

## Streamlit Frontend

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`. Set `PRISM_API_URL` environment variable if the API is not at `localhost:8000`.

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Project Configuration

All settings flow through `src/config.py` via `pydantic-settings`:

| Setting | Default | Description |
|---------|---------|-------------|
| `APP_NAME` | `P.R.I.S.M.` | Application identifier |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEBUG` | `False` | Debug mode (console logging) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/query` | Main agentic query endpoint |
| POST | `/api/v1/query/multimodal` | Image + text query |
| GET | `/api/v1/graph/query?question=...` | Direct graph query |
| GET | `/health` | Service health + Neo4j status |

---

## Related Documentation

- [Architecture](agent_architecture.md) — LangGraph state machine, CRAG loop
- [Graph Schema](graph_schema.md) — Neo4j nodes, relationships, queries
- [MCP Integration](mcp_integration.md) — Yahoo Finance live data
- [Evaluation](evaluation.md) — RAGAS metrics, golden dataset
