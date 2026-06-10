from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    mode: str
    include_multimodal: bool

    router_decision: str
    retrieval_results: list
    aggregated_context: str

    crag_confidence_score: float
    retrieval_attempts: int
    max_retrieval_attempts: int
    rewritten_query: str

    multimodal_data: dict | None

    answer: str
    sources: list
    latency_ms: float
    tokens_used: int


def make_initial_state(query: str, mode: str = "auto", include_multimodal: bool = False) -> AgentState:
    return {
        "messages": [],
        "query": query,
        "mode": mode,
        "include_multimodal": include_multimodal,
        "router_decision": "",
        "retrieval_results": [],
        "aggregated_context": "",
        "crag_confidence_score": 0.0,
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 3,
        "rewritten_query": "",
        "multimodal_data": None,
        "answer": "",
        "sources": [],
        "latency_ms": 0.0,
        "tokens_used": 0,
    }


@dataclass
class RetrievalResult:
    source_type: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
