"""
RootCause — shared retrieval + rerank + generation logic.

Both the MCP server (`rootcause_server.py`) and the Streamlit dashboard
(`dashboard.py`) import from this module so the analysis path is defined exactly
once. The pipeline is:

    query -> embed -> FAISS top-K -> confidence gate -> (optional) LLM rerank
          -> build context -> LLM generate grounded JSON answer

Every model and threshold below was validated in step-2 (see CALIBRATION.md); do
not re-tune them here.
"""

import os
import re
import json
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. CONFIG (ENV + MODELS)
# ---------------------------------------------------------
load_dotenv()

# No hardcoded fallback: fail loudly at import rather than shipping a live key in source.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
        "(see the Quickstart in README.md), or export it in the environment."
    )

# MUST match the model the index was built with (scripts/build_index.py). Embedding a
# query with a different model than built the index does not error — it silently returns
# garbage nearest neighbors, because the two models don't share a vector space. This is
# the single easiest thing to get wrong; the bundled sample index is qwen/qwen3-embedding-8b
# (dim 4096).
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Final grounded-answer generation. glm-4.7-flash: cheap, strong on code, and the model
# the step-2 RAG-vs-LLM eval was run against — the eval only measures the system we ship
# if generation uses the same model it was evaluated with. It is a reasoning model, so
# GEN_REASONING disables reasoning outright: with effort caps it still burns the whole
# budget on hidden reasoning and emits an empty answer; off, it writes a clean answer.
CHAT_MODEL = "z-ai/glm-4.7-flash"

# Reranker, kept as its own knob so it can be swapped/downgraded independently of
# generation. deepseek-v4-flash honors effort "low" and stays cheap.
RERANK_MODEL = "deepseek/deepseek-v4-flash"

# Per-model reasoning control via OpenRouter's `reasoning` param.
GEN_REASONING = {"enabled": False}     # glm-4.7-flash generation
RERANK_REASONING = {"effort": "low"}   # deepseek-v4-flash reranker

# Retrieval confidence gate. Calibrated empirically against the qwen3-embedding-8b index
# (was a guessed 0.6 carried over from text-embedding-3-small). See CALIBRATION.md:
# genuinely-relevant hits scored >= ~0.516, no-good-match hits <= ~0.425; 0.47 splits them.
CONFIDENCE_THRESHOLD = 0.47

TOP_K = 5  # candidates retrieved from FAISS before reranking

# ---------------------------------------------------------
# 2. PATHS (env-overridable, default to the bundled sample)
# ---------------------------------------------------------
# The repo runs out of the box against the bundled ~2,000-vector sample index. To point at
# a full self-built corpus, set FAISS_INDEX_PATH / METADATA_PATH — no code edits. This is
# the "bring your own data" mechanism described in the README.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_INDEX = os.path.join(BASE_DIR, "data", "sample", "sample_corpus.faiss")
_DEFAULT_METADATA = os.path.join(BASE_DIR, "data", "sample", "sample_corpus_metadata.pkl")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", _DEFAULT_INDEX)
METADATA_PATH = os.getenv("METADATA_PATH", _DEFAULT_METADATA)

# ---------------------------------------------------------
# 3. CLIENT + LAZY ASSET LOADING
# ---------------------------------------------------------
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

_INDEX = None
_METADATA = None


def load_assets():
    """Lazy-load the FAISS index and metadata once, and return them. Raises a clear
    FileNotFoundError if either artifact is missing (build them with
    `python scripts/build_index.py`)."""
    global _INDEX, _METADATA

    if _INDEX is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. Build it with "
                f"`python scripts/build_index.py --input data/sample/sample_bug_corpus.jsonl "
                f"--output data/sample/sample_corpus.faiss`."
            )
        _INDEX = faiss.read_index(FAISS_INDEX_PATH)

    if _METADATA is None:
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
        with open(METADATA_PATH, "rb") as f:
            _METADATA = pickle.load(f)

    return _INDEX, _METADATA


def index_info():
    """(path, vector_count, dim) for the loaded index — used by the dashboard sidebar."""
    index, _ = load_assets()
    return FAISS_INDEX_PATH, index.ntotal, index.d


# ---------------------------------------------------------
# 4. EMBED + RETRIEVE
# ---------------------------------------------------------
def embed_query(query):
    """Embed a query with EMBEDDING_MODEL and L2-normalize it for cosine search."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=query[:8000])
    emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def retrieve(query, top_k=TOP_K):
    """Return up to `top_k` nearest corpus records as dicts with their similarity score,
    in FAISS order (most similar first)."""
    index, metadata = load_assets()
    emb = embed_query(query)
    scores, indices = index.search(emb, top_k)

    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        sample = metadata[idx]
        patches = sample.get("patches") or [""]
        retrieved.append({
            "score": float(scores[0][i]),
            "bug_type": sample.get("bug_type"),
            "issue": sample.get("issue"),
            "fix": sample.get("fix"),
            "patch": patches[0][:500] if patches else "",
        })
    return retrieved


# ---------------------------------------------------------
# 5. RERANK
# ---------------------------------------------------------
def _extract_index_list(text, n):
    """Pull the first JSON array of ints out of a model response, tolerating code fences
    and any leading reasoning prose. Returns a de-duplicated, in-range permutation of
    candidate indices, or None if nothing parseable is found."""
    if not text:
        return None
    match = re.search(r"\[[\s\d,]*\]", text)
    if not match:
        return None
    try:
        order = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(order, list):
        return None
    seen = []
    for i in order:
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            seen.append(i)
    return seen or None


def rerank_with_llm(query, retrieved):
    """Ask RERANK_MODEL to reorder the retrieved candidates by relevance. Reordering
    only — candidates the model omits are appended in FAISS order, never dropped. Any
    failure falls back to the original order."""
    if not retrieved:
        return []

    prompt = f"""
Rank these bug candidates by relevance.

Query:
{query}

Candidates:
"""
    for i, r in enumerate(retrieved):
        prompt += f"\n[{i}] Type: {r['bug_type']}\nIssue: {r['issue'][:200]}\n"

    prompt += "\nReturn ONLY a JSON list of indices, most relevant first, e.g. [1,0,2]"

    try:
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            extra_body={"reasoning": RERANK_REASONING},
        )
        order = _extract_index_list(response.choices[0].message.content, len(retrieved))
        if order:
            ranked = [retrieved[i] for i in order]
            ranked += [r for j, r in enumerate(retrieved) if j not in order]
            return ranked
    except Exception:
        pass

    return retrieved


# ---------------------------------------------------------
# 6. CONTEXT + GENERATION
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are RootCause, a debugging agent used by coding assistants.

Your job:
1. Identify the root cause of the bug precisely
2. Provide a concrete fix (code or clear instruction)
3. Base your answer on the most relevant historical examples if provided

Rules:
- Be specific, not generic
- Do NOT hallucinate APIs or behavior
- If examples are weak, rely on reasoning instead

Return ONLY JSON:
{
"root_cause": "...",
"fix": "...",
"confidence": 0-1,
"examples_used": []
}
"""


def build_context(retrieved, limit=3):
    """Format the top `limit` retrieved records into the grounding block that goes into
    the generation prompt."""
    context = ""
    for r in retrieved[:limit]:
        context += f"""
--- EXAMPLE ---
Type: {r['bug_type']}
Issue: {r['issue']}
Fix: {r['fix']}
Patch: {r['patch']}
"""
    return context


def generate_answer(query, context, use_rag):
    """Call CHAT_MODEL and return its raw JSON-string answer. When `use_rag` is False the
    prompt omits examples and tells the model no strong matches were found."""
    user_prompt = (
        f"{context}\nBug: {query}"
        if use_rag
        else f"Bug: {query}\n(No strong matches found)"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        extra_body={"reasoning": GEN_REASONING},
    )
    return response.choices[0].message.content


# ---------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------
def analyze_bug(query):
    """Run the full retrieve -> gate -> rerank -> generate pipeline.

    Returns a dict with both the final answer and the intermediate state, so the MCP
    server can return just the answer string while the dashboard can show the retrieved
    candidates and the gate decision:

        {
          "answer": "<raw JSON string from CHAT_MODEL>",
          "retrieved": [ {score, bug_type, issue, fix, patch}, ... ],  # post-rerank
          "top_score": float,
          "use_rag": bool,
          "confidence_threshold": float,
        }
    """
    retrieved = retrieve(query)

    top_score = retrieved[0]["score"] if retrieved else 0.0
    use_rag = top_score >= CONFIDENCE_THRESHOLD

    if use_rag:
        retrieved = rerank_with_llm(query, retrieved)
        context = build_context(retrieved)
    else:
        context = ""

    answer = generate_answer(query, context, use_rag)

    return {
        "answer": answer,
        "retrieved": retrieved,
        "top_score": top_score,
        "use_rag": use_rag,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }
