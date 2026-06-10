from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from src.agents.graph import prism_app
from src.agents.state import make_initial_state
from src.api.models import (
    AgentTrace,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    Source,
)
from src.cache.semantic import semantic_cache
from src.graph.client import Neo4jClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])

_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient | None:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


@router.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()

    logger.info(
        "query_received",
        query=request.query[:100],
        mode=request.mode,
        user_id=request.user_id,
    )

    cached = await semantic_cache.lookup(request.query)
    if cached:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            answer=cached.get("answer", ""),
            sources=[
                Source(**s) if isinstance(s, dict) else s
                for s in cached.get("sources", [])
            ],
            agent_trace=AgentTrace(
                router_decision=cached.get("router_decision", "cache"),
                crag_confidence_score=1.0,
                retrieval_attempts=cached.get("retrieval_attempts", 0),
            ),
            metrics=QueryMetrics(
                latency_ms=round(elapsed_ms, 2),
                tokens_used=0,
            ),
        )

    thread_id = str(uuid.uuid4())
    initial_state = make_initial_state(
        query=request.query,
        mode=request.mode,
        include_multimodal=request.include_multimodal,
    )

    try:
        result = await prism_app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        sources_raw = result.get("sources", [])
        sources = [
            Source(
                type=s.get("type", "unknown"),
                doc_id=s.get("doc_id") or s.get("id"),
                page=s.get("page"),
            )
            for s in sources_raw
        ]

        response_data = QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            agent_trace=AgentTrace(
                router_decision=result.get("router_decision", ""),
                crag_confidence_score=round(
                    result.get("crag_confidence_score", 0.0), 2
                ),
                retrieval_attempts=result.get("retrieval_attempts", 0),
            ),
            metrics=QueryMetrics(
                latency_ms=round(elapsed_ms, 2),
                tokens_used=result.get("tokens_used", 0),
            ),
        )

        await semantic_cache.store(
            request.query,
            {
                "answer": response_data.answer,
                "sources": [s.model_dump() for s in response_data.sources],
                "router_decision": response_data.agent_trace.router_decision,
                "retrieval_attempts": response_data.agent_trace.retrieval_attempts,
            },
        )

        return response_data

    except Exception as exc:
        logger.error("pipeline_failed", error=str(exc), query=request.query[:60])
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            answer="An error occurred while processing your query. Please try again.",
            agent_trace=AgentTrace(
                router_decision="error",
                crag_confidence_score=0.0,
                retrieval_attempts=0,
            ),
            metrics=QueryMetrics(
                latency_ms=round(elapsed_ms, 2),
                tokens_used=0,
            ),
        )


@router.post("/api/v1/query/multimodal", response_model=QueryResponse)
async def query_multimodal(
    query: str = Form(min_length=1, max_length=4096),
    file: UploadFile = File(...),
    mode: str = Form("auto"),
) -> QueryResponse:
    start_time = time.perf_counter()

    logger.info(
        "multimodal_query_received",
        query=query[:100],
        filename=file.filename,
        content_type=file.content_type,
    )

    image_bytes = await file.read()
    thread_id = str(uuid.uuid4())
    initial_state = make_initial_state(
        query=query,
        mode=mode,
        include_multimodal=True,
    )
    initial_state["multimodal_data"] = {
        "image_bytes": image_bytes,
        "mime_type": file.content_type or "image/png",
        "context": query,
    }

    try:
        result = await prism_app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        sources_raw = result.get("sources", [])

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=[
                Source(
                    type=s.get("type", "unknown"),
                    doc_id=s.get("doc_id") or s.get("id"),
                )
                for s in sources_raw
            ],
            agent_trace=AgentTrace(
                router_decision=result.get("router_decision", ""),
                crag_confidence_score=round(
                    result.get("crag_confidence_score", 0.0), 2
                ),
                retrieval_attempts=result.get("retrieval_attempts", 0),
            ),
            metrics=QueryMetrics(
                latency_ms=round(elapsed_ms, 2),
                tokens_used=result.get("tokens_used", 0),
            ),
        )
    except Exception as exc:
        logger.error("multimodal_pipeline_failed", error=str(exc))
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            answer="An error occurred while processing your image query.",
            agent_trace=AgentTrace(
                router_decision="error",
                crag_confidence_score=0.0,
                retrieval_attempts=0,
            ),
            metrics=QueryMetrics(
                latency_ms=round(elapsed_ms, 2),
                tokens_used=0,
            ),
        )


@router.get("/health")
async def health() -> dict:
    graph_status = "disconnected"
    try:
        client = get_neo4j_client()
        await client.connect()
        graph_status = "connected"
    except Exception:
        graph_status = "unavailable"

    return {
        "status": "ok",
        "service": "P.R.I.S.M.",
        "graph_db": graph_status,
    }


@router.get("/api/v1/graph/query")
async def graph_query(question: str) -> dict:
    from src.graph.cypher_generator import CypherGenerator
    from src.graph.query_executor import GraphQueryExecutor

    client = get_neo4j_client()
    await client.connect()
    generator = CypherGenerator()
    executor = GraphQueryExecutor(client, generator)
    result = await executor.answer_question(question)
    return {"answer": result}
