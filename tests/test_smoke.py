"""
Smoke tests for the bundled sample index.

Two levels, so a clone with no API key and no CI still gets a meaningful check:

  * test_index_metadata_aligned — pure local: loads data/sample/*, asserts the FAISS
    index and metadata are the same length and the vectors are the expected dimension,
    and that metadata records carry the fields the server reads. No network, no key.
  * test_retrieve_returns_scored_candidates — imports core and runs a single real
    embedding call through retrieve(). Skipped automatically if OPENROUTER_API_KEY is
    unset (core raises at import without one), so key-less runs still pass green.

Run:  pytest tests/            (or: python tests/test_smoke.py)
"""

import os
import pickle
import sys

import faiss
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
INDEX_PATH = os.path.join(SAMPLE_DIR, "sample_corpus.faiss")
METADATA_PATH = os.path.join(SAMPLE_DIR, "sample_corpus_metadata.pkl")

EXPECTED_DIM = 4096  # qwen/qwen3-embedding-8b
REQUIRED_FIELDS = ("bug_type", "issue", "fix")


def test_index_metadata_aligned():
    """The index and metadata must be 1:1 and the right shape — a mismatch means
    retrieval would return records that don't line up with the vectors."""
    assert os.path.exists(INDEX_PATH), (
        f"{INDEX_PATH} missing — build it with "
        f"`python scripts/build_index.py --input data/sample/sample_bug_corpus.jsonl "
        f"--output data/sample/sample_corpus.faiss`"
    )
    assert os.path.exists(METADATA_PATH), f"{METADATA_PATH} missing"

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    assert index.ntotal == len(metadata), (
        f"index has {index.ntotal} vectors but metadata has {len(metadata)} records"
    )
    assert index.ntotal > 0, "sample index is empty"
    assert index.d == EXPECTED_DIM, f"expected dim {EXPECTED_DIM}, got {index.d}"

    for rec in metadata:
        for field in REQUIRED_FIELDS:
            assert rec.get(field), f"record missing required field '{field}': {rec!r}"


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping the live embedding call",
)
def test_retrieve_returns_scored_candidates():
    """One real embedding call: retrieve() should return TOP_K scored, descending
    candidates for an on-domain query."""
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    import core

    results = core.retrieve("TypeError: 'NoneType' object has no attribute 'get'")
    assert results, "retrieve() returned no candidates"
    assert len(results) <= core.TOP_K
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "candidates not in descending score order"
    for r in results:
        assert r["bug_type"] and r["issue"] and r["fix"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
