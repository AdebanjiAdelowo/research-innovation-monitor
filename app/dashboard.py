"""Streamlit dashboard over the HTTP API.

    uvicorn rim.api.app:app --port 8000
    streamlit run app/dashboard.py

The dashboard holds no logic of its own: every number it shows comes from the API or
from result files written by an experiment run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("RIM_API_URL", "http://localhost:8000")
RESULTS = Path(os.environ.get("RIM_RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))

st.set_page_config(page_title="Research and Innovation Monitor", layout="wide")
st.title("Research and Innovation Monitor")


def call(method: str, path: str, **kwargs) -> dict | None:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.warning(f"{exc.response.status_code}: {exc.response.json().get('detail', exc)}")
    except httpx.HTTPError as exc:
        st.error(f"API not reachable at {API_URL}: {exc}")
    return None


def show_evidence(passages: list[dict]) -> None:
    for passage in passages:
        with st.expander(f"[{passage['index']}] {passage['title']}  ({passage['doc_id']})"):
            st.write(passage["text"])


health = call("GET", "/health")
if health:
    corpora = health["corpora"]
    st.caption(
        f"Corpora: {corpora} · encoder: {health['encoder'].get('model_id', health['encoder']['kind'])} · "
        f"generator: {health['generator'].get('model', health['generator']['kind'])} · "
        f"commit: {(health['git'].get('commit') or 'unknown')[:7]}"
    )
else:
    corpora = {}
    st.stop()

search_tab, ask_tab, trends_tab, report_tab, results_tab = st.tabs(
    ["Search", "Ask", "Trends", "Report", "Experiment results"]
)

with search_tab:
    left, mid, right = st.columns([4, 1, 1])
    query = left.text_input("Query", "neural networks that solve partial differential equations")
    corpus = mid.selectbox("Corpus", list(corpora), key="search_corpus")
    mode = right.selectbox("Mode", ["hybrid", "bm25", "dense"])
    if st.button("Search") and query:
        body = call("POST", "/search", json={"query": query, "corpus": corpus, "mode": mode})
        if body:
            st.caption(" · ".join(f"{k}: {v:.1f}" for k, v in body["timings_ms"].items()))
            table = pd.DataFrame(body["hits"])[["rank", "score", "doc_id", "title", "text"]]
            st.dataframe(table, hide_index=True, width="stretch")

with ask_tab:
    question = st.text_input("Question", "What do dual encoders embed?")
    ask_corpus = st.selectbox("Corpus", list(corpora), key="ask_corpus")
    if st.button("Answer") and question:
        body = call("POST", "/ask", json={"question": question, "corpus": ask_corpus})
        if body:
            if body["abstained"]:
                st.info("The retrieved passages do not contain enough evidence to answer.")
            for sentence in body["sentences"]:
                marks = "".join(f"[{i}]" for i in sentence["citations"]) or "(no citation)"
                st.markdown(f"- {sentence['text']} **{marks}**")
                if sentence["invalid_markers"]:
                    st.caption(f"cites passages that do not exist: {sentence['invalid_markers']}")
            show_evidence(body["passages"])
            st.caption(json.dumps(body["usage"]))

with trends_tab:
    direction = st.radio("Direction", ["rising", "falling"], horizontal=True)
    body = call("GET", "/trends", params={"direction": direction, "limit": 15})
    if body and body["topics"]:
        st.caption(f"Source run: {body['run_id']}. q-values are Benjamini-Hochberg adjusted.")
        table = pd.DataFrame(body["topics"])
        st.dataframe(table, hide_index=True, width="stretch")
        topic = st.selectbox(
            "Topic",
            table["topic"],
            format_func=lambda t: f"{t}: {table.set_index('topic').loc[t, 'terms']}",
        )
        series = call("GET", f"/trends/{topic}")
        if series:
            chart = pd.DataFrame(
                {"share of documents (%)": [100 * s for s in series["shares"]]},
                index=pd.Index(series["years"], name="year"),
            )
            st.line_chart(chart, color="#2a78d6")
            with st.expander("Table view"):
                st.dataframe(chart.assign(documents=series["counts"]))

with report_tab:
    request = st.text_area(
        "Request", "Find papers on neural operators and say whether the topic is growing."
    )
    if st.button("Build report") and request:
        body = call("POST", "/report", json={"request": request})
        if body:
            st.caption(f"Tools called: {body['tools_called']} · completed: {body['completed']}")
            for sentence in body["sentences"]:
                marks = "".join(f"[{i}]" for i in sentence["citations"]) or "(no citation)"
                st.markdown(f"- {sentence['text']} **{marks}**")
            if body["trends"]:
                st.markdown("**Trend statistics** (computed, not generated)")
                st.dataframe(pd.DataFrame(body["trends"]), hide_index=True)
            if body["errors"]:
                st.warning(body["errors"])
            show_evidence(body["evidence"])

with results_tab:
    st.caption("Read from result files written by experiment runs. Nothing here is recomputed.")
    for summary_path in sorted(RESULTS.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        with st.expander(f"{summary_path.parent.name}  ·  {summary.get('run_id', '')}"):
            if "systems" in summary:
                rows = {name: row["test"] for name, row in summary["systems"].items()}
                st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")
            else:
                st.json(summary, expanded=False)
