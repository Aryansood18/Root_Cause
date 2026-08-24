"""
RootCause dashboard — a Streamlit demo/inspection tool on top of core.py.

This is NOT a replacement for the MCP server; it's a way to eyeball the same pipeline
(`core.analyze_bug`) interactively — embed -> retrieve -> gate -> rerank -> generate —
and to see the retrieved candidates and their scores that the MCP tool hides.

Run:
    streamlit run src/dashboard.py
"""

import json
import os

import streamlit as st

import core

st.set_page_config(page_title="RootCause", page_icon="🔍", layout="wide")

# ---------------------------------------------------------
# Sidebar — configuration & loaded index
# ---------------------------------------------------------
with st.sidebar:
    st.header("RootCause")
    st.caption("RAG debugging assistant — inspection dashboard")

    st.subheader("Models")
    st.markdown(
        f"- **Embedding:** `{core.EMBEDDING_MODEL}`\n"
        f"- **Generation:** `{core.CHAT_MODEL}`\n"
        f"- **Rerank:** `{core.RERANK_MODEL}`"
    )

    st.subheader("Retrieval")
    try:
        path, ntotal, dim = core.index_info()
        st.markdown(
            f"- **Index:** `{os.path.relpath(path, core.BASE_DIR)}`\n"
            f"- **Vectors:** {ntotal:,} (dim {dim})\n"
            f"- **Top-K:** {core.TOP_K}\n"
            f"- **Confidence gate:** {core.CONFIDENCE_THRESHOLD}"
        )
    except Exception as e:
        st.error(f"Could not load index: {e}")

# ---------------------------------------------------------
# Main — analyze a bug
# ---------------------------------------------------------
st.title("Analyze a bug")
st.write(
    "Describe a bug, paste a traceback, or drop in a code snippet. RootCause retrieves "
    "similar historical bug-fixes, gates on retrieval confidence, optionally reranks, and "
    "asks the LLM for a grounded root cause + fix."
)

query = st.text_area(
    "Bug description",
    height=140,
    placeholder="e.g. TypeError: 'NoneType' object has no attribute 'get' when a user "
    "record is missing from the cache",
)

if st.button("Analyze", type="primary") and query.strip():
    with st.spinner("Embedding, retrieving, and generating…"):
        try:
            result = core.analyze_bug(query)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    top_score = result["top_score"]
    used = result["use_rag"]
    st.markdown(
        f"**Top retrieval score:** {top_score:.3f} "
        f"({'above' if used else 'below'} the {result['confidence_threshold']} gate → "
        f"{'grounded on retrieved examples' if used else 'answering from reasoning only'})"
    )

    # Structured answer
    st.subheader("Answer")
    try:
        st.json(json.loads(result["answer"]))
    except (json.JSONDecodeError, TypeError):
        st.code(result["answer"])

    # Retrieved candidates
    st.subheader("Retrieved candidates")
    rows = [
        {
            "score": round(r["score"], 3),
            "bug_type": r["bug_type"],
            "issue": r["issue"],
            "fix": r["fix"],
        }
        for r in result["retrieved"]
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No candidates retrieved.")

# ---------------------------------------------------------
# Routing — calls diverted to LLM vs RAG-based retrieval
# ---------------------------------------------------------
st.divider()
st.header("Routing — calls diverted to LLM vs RAG-based retrieval")
st.write(
    f"Every call is routed by the confidence gate. If the top retrieval score is at or "
    f"above **{core.CONFIDENCE_THRESHOLD}**, the answer is grounded on **RAG-based "
    f"retrieval**; below it, the call is **diverted to the LLM** and answered from "
    f"reasoning alone. This runs a fixed set of representative bug queries through the "
    f"*currently loaded* index to show how it routes them — retrieval only, no generation."
)

# Representative probe queries spanning the bug-type distribution.
PROBE_QUERIES = [
    "TypeError: 'NoneType' object has no attribute 'get' when a record is missing",
    "AttributeError because a dict lookup returned None and I called .strip() on it",
    "TypeError: unsupported operand type(s) for +: 'int' and 'str' from parsed CSV",
    "My loop skips the last element because the range upper bound is wrong",
    "The conditional branch falls through when the coupon is empty, applying the discount twice",
    "Variable defined inside the if-block is referenced outside it and raises NameError",
    "A bare except swallows the real error and the program continues with corrupted state",
    "Calling requests without a timeout and the worker hangs forever on a slow endpoint",
    "Two threads increment the same counter without a lock and the total is wrong",
    "os.path.exists passes but the file is deleted before open() runs",
    "A cache dict grows unbounded because entries are never evicted",
    "User input is interpolated directly into an SQL string, enabling injection",
]

if st.button("Run routing analysis"):
    rag = 0
    rows = []
    progress = st.progress(0.0, text="Routing probe queries…")
    for i, q in enumerate(PROBE_QUERIES):
        try:
            cands = core.retrieve(q)
        except Exception as e:
            st.error(f"Routing failed: {e}")
            st.stop()
        top = cands[0]["score"] if cands else 0.0
        is_rag = top >= core.CONFIDENCE_THRESHOLD
        rag += is_rag
        rows.append({
            "query": q[:55] + ("…" if len(q) > 55 else ""),
            "top_score": round(top, 3),
            "routed_to": "RAG retrieval" if is_rag else "LLM (diverted)",
        })
        progress.progress((i + 1) / len(PROBE_QUERIES))
    progress.empty()

    total = len(PROBE_QUERIES)
    diverted = total - rag
    c1, c2 = st.columns(2)
    c1.metric("RAG-based retrieval", f"{rag}/{total}", f"{rag / total * 100:.0f}%")
    c2.metric("Diverted to LLM", f"{diverted}/{total}", f"{diverted / total * 100:.0f}%")
    st.bar_chart(
        {"RAG-based retrieval": [rag], "Diverted to LLM": [diverted]},
        color=["#2e7d32", "#c62828"],
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.caption("Click to route the probe set against the loaded index (embeddings only).")
