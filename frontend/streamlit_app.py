import os
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_BASE = os.getenv("PRISM_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="P.R.I.S.M. — Agentic RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .main-header { color: #00d4ff; font-size: 2.5rem; font-weight: 700; }
    .sub-header { color: #8892b0; font-size: 1rem; }
    .source-box {
        background-color: #1a1d29; border-radius: 8px;
        padding: 12px; margin: 8px 0; border-left: 3px solid #00d4ff;
    }
    .metric-box {
        background-color: #1a1d29; border-radius: 8px;
        padding: 8px 16px; text-align: center;
    }
    .disclaimer {
        color: #666; font-size: 0.75rem; font-style: italic;
        margin-top: 16px; padding: 8px; border-top: 1px solid #333;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">P.R.I.S.M.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Pipeline for Retrieval, Inference, & Structured Memory</p>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Configuration")

mode = st.sidebar.selectbox(
    "Query Mode",
    options=["auto", "vector", "graph", "mcp", "multimodal"],
    index=0,
    help="auto: let the router decide. Override to force a specific route.",
)

include_multimodal = st.sidebar.checkbox(
    "Enable Multimodal (Vision)",
    value=False,
    help="Allow image/PDF analysis via Vision LLM",
)

st.sidebar.markdown("---")
st.sidebar.markdown("## System Status")

try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    status = r.json()
    st.sidebar.success(f"API: {status.get('status', 'ok')}")
    graph_db = status.get("graph_db", "unknown")
    if graph_db == "connected":
        st.sidebar.success(f"Neo4j: {graph_db}")
    else:
        st.sidebar.warning(f"Neo4j: {graph_db}")
except requests.ConnectionError:
    st.sidebar.error("API: disconnected")
except Exception:
    st.sidebar.warning("API: error")

st.sidebar.markdown("---")
st.sidebar.markdown("## Documentation")
st.sidebar.markdown("[Architecture](docs/agent_architecture.md)")
st.sidebar.markdown("[Graph Schema](docs/graph_schema.md)")
st.sidebar.markdown("[MCP Integration](docs/mcp_integration.md)")
st.sidebar.markdown("[Evaluation](docs/evaluation.md)")
st.sidebar.markdown("[Setup](docs/setup.md)")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "For multimodal queries",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded", use_container_width=True)

with col1:
    st.markdown("### Query")
    query = st.text_input(
        "Ask a financial question",
        placeholder="e.g., Compare R&D spend of Apple and Microsoft in Q1 2026",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        clear = st.button("Clear", use_container_width=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if clear:
    st.session_state.chat_history = []
    st.rerun()

if send and query:
    with st.spinner("P.R.I.S.M. is analyzing..."):
        start = time.perf_counter()

        try:
            if uploaded_file and (mode == "multimodal" or include_multimodal):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"query": query, "mode": "multimodal"}
                resp = requests.post(
                    f"{API_BASE}/api/v1/query/multimodal",
                    files=files,
                    data=data,
                    timeout=120,
                )
            else:
                resp = requests.post(
                    f"{API_BASE}/api/v1/query",
                    json={
                        "query": query,
                        "mode": mode,
                        "include_multimodal": include_multimodal,
                    },
                    timeout=120,
                )

            elapsed = time.perf_counter() - start
            result = resp.json()

            st.session_state.chat_history.append(
                {"query": query, "result": result, "latency": elapsed}
            )

        except requests.ConnectionError:
            st.error("Cannot connect to P.R.I.S.M. API. Is the server running?")
        except Exception as e:
            st.error(f"Request failed: {e}")

for entry in reversed(st.session_state.chat_history):
    with st.container():
        st.markdown(f"**You:** {entry['query']}")

        result = entry["result"]
        answer = result.get("answer", "")
        st.markdown(f"**P.R.I.S.M.:** {answer}")

        trace = result.get("agent_trace", {})
        metrics = result.get("metrics", {})

        tab_sources, tab_trace, tab_metrics = st.tabs(
            ["Sources", "Agent Trace", "Metrics"]
        )

        with tab_sources:
            sources = result.get("sources", [])
            if sources:
                for s in sources:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>{s.get("type", "unknown").upper()}</strong>'
                        f'{" — " + str(s.get("doc_id", "")) if s.get("doc_id") else ""}'
                        f'{" — Page " + str(s.get("page")) if s.get("page") else ""}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No sources recorded for this query.")

        with tab_trace:
            st.json(trace)

        with tab_metrics:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="color:#00d4ff;font-size:1.5rem;margin:0">'
                    f'{metrics.get("latency_ms", 0):.0f}ms</p>'
                    f'<p style="color:#8892b0;font-size:0.8rem;margin:0">Latency</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="color:#00d4ff;font-size:1.5rem;margin:0">'
                    f'{metrics.get("tokens_used", 0)}</p>'
                    f'<p style="color:#8892b0;font-size:0.8rem;margin:0">Tokens</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m_col3:
                confidence = trace.get("crag_confidence_score", 0)
                color = "#00ff88" if confidence >= 0.7 else "#ffaa00"
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="color:{color};font-size:1.5rem;margin:0">'
                    f'{confidence:.0%}</p>'
                    f'<p style="color:#8892b0;font-size:0.8rem;margin:0">Confidence</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m_col4:
                attempts = trace.get("retrieval_attempts", 0)
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="color:#00d4ff;font-size:1.5rem;margin:0">'
                    f'{attempts}</p>'
                    f'<p style="color:#8892b0;font-size:0.8rem;margin:0">Retries</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<p class="disclaimer">Disclaimer: This is for informational purposes '
            "only and does not constitute financial advice.</p>",
            unsafe_allow_html=True,
        )

        st.divider()

st.markdown("---")
st.caption(
    "P.R.I.S.M. v0.1.0 — Built with LangGraph, Neo4j, Pinecone, Cohere, GPT-4o"
)
