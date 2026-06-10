from __future__ import annotations

import time

from fastapi import APIRouter

from src.api.models import (
    AgentTrace,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    Source,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])


@router.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()

    logger.info(
        "query_received",
        query=request.query[:100],
        mode=request.mode,
        user_id=request.user_id,
    )

    # TODO: Phase 2+ will replace this with LangGraph agentic orchestration
    answer = (
        f"P.R.I.S.M. received your query: '{request.query}'. "
        "The agentic pipeline is under construction. "
        "Full implementation coming in Phase 3."
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return QueryResponse(
        answer=answer,
        sources=[
            Source(
                type="vector",
                doc_id="placeholder",
            )
        ],
        agent_trace=AgentTrace(
            router_decision="pending",
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
    return {"status": "ok", "service": "P.R.I.S.M."}
