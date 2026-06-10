from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    type: str = Field(description="Source type: graph, vector, mcp, multimodal")
    entity: str | None = None
    relation: str | None = None
    doc_id: str | None = None
    page: int | None = None
    tool: str | None = None
    timestamp: datetime | None = None


class AgentTrace(BaseModel):
    router_decision: str = Field(description="Route(s) selected by the agentic router")
    crag_confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score from CRAG loop"
    )
    retrieval_attempts: int = Field(
        ge=1, description="Number of retrieval attempts made"
    )


class QueryMetrics(BaseModel):
    latency_ms: float = Field(description="End-to-end query latency in milliseconds")
    tokens_used: int = Field(ge=0, description="Total LLM tokens consumed")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096, description="User query")
    mode: str = Field(
        default="auto",
        pattern="^(auto|graph|vector|mcp|multimodal)$",
        description="Routing mode override",
    )
    include_multimodal: bool = Field(
        default=False, description="Whether to include vision/PDF analysis"
    )
    user_id: str | None = Field(default=None, description="Optional user identifier")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer text")
    sources: list[Source] = Field(default_factory=list)
    agent_trace: AgentTrace | None = None
    metrics: QueryMetrics | None = None
    disclaimer: str = Field(
        default="Disclaimer: This is for informational purposes only and does not constitute financial advice."
    )
