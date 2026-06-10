from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.aggregator import ContextAggregator
from src.agents.crag_node import CRAGNode
from src.agents.generator import AnswerGenerator
from src.agents.graph_node import GraphNode
from src.agents.retrieval_node import RetrievalNode
from src.agents.router import AgenticRouter
from src.agents.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_prism_graph() -> StateGraph:
    router = AgenticRouter()
    vector_retrieval = RetrievalNode()
    graph_retrieval = GraphNode()
    aggregator = ContextAggregator()
    crag = CRAGNode()
    generator = AnswerGenerator()

    workflow = StateGraph(AgentState)

    workflow.add_node("classify_query", router.classify)
    workflow.add_node("retrieve_vector", vector_retrieval.run)
    workflow.add_node("retrieve_graph", graph_retrieval.run)
    workflow.add_node("aggregate_context", aggregator.aggregate)
    workflow.add_node("evaluate_confidence", crag.evaluate)
    workflow.add_node("rewrite_query", crag.rewrite_query)
    workflow.add_node("generate_answer", generator.generate)

    workflow.set_entry_point("classify_query")

    def route_by_decision(state: AgentState) -> str:
        decision = state.get("router_decision", "vector").lower()

        if "graph" in decision and "vector" not in decision:
            return "graph"
        if "vector" in decision:
            return "vector"
        if "graph" in decision:
            return "graph"
        return "vector"

    workflow.add_conditional_edges(
        "classify_query",
        route_by_decision,
        {
            "vector": "retrieve_vector",
            "graph": "retrieve_graph",
        },
    )

    workflow.add_edge("retrieve_vector", "aggregate_context")
    workflow.add_edge("retrieve_graph", "aggregate_context")
    workflow.add_edge("aggregate_context", "evaluate_confidence")

    workflow.add_conditional_edges(
        "evaluate_confidence",
        crag.should_retry,
        {
            "rewrite": "rewrite_query",
            "generate": "generate_answer",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_vector")
    workflow.add_edge("generate_answer", END)

    return workflow


def compile_prism_graph() -> StateGraph:
    workflow = build_prism_graph()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logger.info("prism_graph_compiled")
    return app


prism_app = compile_prism_graph()
