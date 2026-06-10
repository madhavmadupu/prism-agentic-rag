# Setup Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- API keys for: OpenAI, Cohere, Pinecone

---

## Local Development Setup

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/prism-ai.git
cd prism-ai
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o |
| `COHERE_API_KEY` | Cohere API key for Embed v3 & Rerank v3 |
| `PINECONE_API_KEY` | Pinecone API key for vector index |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `prism-index`) |
| `NEO4J_URI` | Neo4j connection URI (default: `bolt://localhost:7687`) |
| `REDIS_URL` | Redis connection URL (default: `redis://localhost:6379`) |

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn src.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Docker Setup (Recommended)

```bash
docker-compose up --build
```

This starts:

| Service | Port | Purpose |
|---------|------|---------|
| `prism-api` | 8000 | FastAPI application |
| `prism-redis` | 6379 | Semantic caching |
| `prism-neo4j` | 7687 / 7474 | Knowledge graph |

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Project Configuration

All configuration flows through `src/config.py` using `pydantic-settings`. Environment variables are loaded from `.env` automatically.

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `APP_NAME` | `P.R.I.S.M.` | Application identifier |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEBUG` | `False` | Enable debug mode |

---

## Adding Data to the Vector Store

```python
from src.ingestion.pinecone_ingest import ingest_document

# Ingest a text document into Pinecone
await ingest_document(
    text="Company revenue grew 14% in Q1...",
    metadata={"company": "AAPL", "document_type": "10-K"},
)
```

See `/docs/ingestion.md` for the full ingestion guide.
