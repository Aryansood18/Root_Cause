"""
Confidence-gate calibration.

Runs a held-out set of realistic, bug-report-style queries through a FAISS index and
prints each query's top-1 similarity plus its top-3 retrieved examples, so the score
distribution of "genuinely relevant" hits can be compared against "no-good-match" hits and
a separating threshold picked empirically instead of asserted. This is how
CONFIDENCE_THRESHOLD = 0.47 was chosen for the shipped qwen3-embedding-8b index; see
CALIBRATION.md.

Re-run this after any change to the embedding model or the corpus — the separating band,
not the specific number, is what to preserve.

By default it calibrates against the bundled sample index. Point it at a full self-built
index with --index / --metadata (or the FAISS_INDEX_PATH / METADATA_PATH env vars).

    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --index data/my_corpus.faiss --metadata data/my_corpus_metadata.pkl

Needs OPENROUTER_API_KEY in .env (or the environment).
"""

import os
import argparse
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Must match the model the index was built with. Embedding a query with a different model
# than built the index silently returns garbage neighbors.
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BASE_DIR, "data", "sample", "sample_corpus.faiss")
DEFAULT_METADATA = os.path.join(BASE_DIR, "data", "sample", "sample_corpus_metadata.pkl")

# Held-out query set. `bug_type` is the category being probed; `expect_relevant` is a prior
# about whether a genuinely useful match should exist in the corpus. Off-domain queries
# (expect_relevant=False) exist to map where "no good match" scores land.
QUERIES = [
    # --- on-domain: should have relevant matches ---
    ("null_check",        "TypeError: 'NoneType' object has no attribute 'get' when a user record is missing from the cache", True),
    ("null_check",        "My function crashes with AttributeError because a dictionary lookup returned None and I called .strip() on it", True),
    ("type_error",        "TypeError: unsupported operand type(s) for +: 'int' and 'str' when summing values from a parsed CSV", True),
    ("type_error",        "Passing a string where an integer was expected causes the comparison to behave incorrectly", True),
    ("off_by_one",        "My loop skips the last element of the list because the range upper bound is wrong", True),
    ("off_by_one",        "Slicing with array[i:i] returns empty and the pagination drops one row per page", True),
    ("logic_bug",         "The discount is applied twice because the conditional branch falls through when the coupon is empty", True),
    ("logic_bug",         "Boolean condition uses 'or' where it should use 'and', so the validation passes when it shouldn't", True),
    ("scope_error",       "Variable defined inside the if-block is referenced outside it and raises NameError", True),
    ("exception_handling","A bare except swallows the real error and the program continues with corrupted state", True),
    ("api_misuse",        "Calling the requests library without a timeout and the worker hangs forever on a slow endpoint", True),
    ("api_misuse",        "Using dict.get() with a mutable default argument that is shared across calls", True),
    ("concurrency",       "Two threads increment the same counter without a lock and the total comes out wrong", True),
    ("race_condition",    "Check-then-act on a file: os.path.exists passes but the file is deleted before open()", True),
    ("memory_leak",       "A cache dict grows unbounded because entries are never evicted and memory keeps climbing", True),
    ("performance",       "N+1 query problem: the ORM issues one SELECT per row inside a loop and the endpoint is slow", True),
    ("security",          "User input is interpolated directly into an SQL string enabling injection", True),
    ("security",          "Deserializing untrusted data with pickle allows arbitrary code execution", True),
    # --- off-domain / weak-match: should NOT have a genuinely useful match ---
    ("_offdomain",        "What is the best pizza topping combination for a party of ten people?", False),
    ("_offdomain",        "How do I set up a Kubernetes ingress with TLS termination on GKE?", False),
    ("_offdomain",        "My cat keeps knocking my coffee mug off the desk every morning", False),
    ("_offdomain",        "Explain the plot of Hamlet in three sentences", False),
]


def embed(client, text):
    res = client.embeddings.create(model=EMBEDDING_MODEL, input=text[:8000])
    return np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.getenv("FAISS_INDEX_PATH", DEFAULT_INDEX),
                    help="FAISS index to calibrate against (default: bundled sample).")
    ap.add_argument("--metadata", default=os.getenv("METADATA_PATH", DEFAULT_METADATA),
                    help="Metadata pickle aligned to the index.")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Put it in a .env at the repo root.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    index = faiss.read_index(args.index)
    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)
    print(f"Index: {index.ntotal} vectors, dim={index.d}\n")

    rows = []
    for bug_type, query, expect in QUERIES:
        emb = embed(client, query)
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, 5)
        top1 = float(scores[0][0])
        rows.append((bug_type, expect, top1, query))

        tag = "REL?" if expect else "OFF "
        print(f"[{tag}] top1={top1:.3f}  ({bug_type})  {query[:70]}")
        for s, idx in zip(scores[0][:3], indices[0][:3]):
            if idx == -1:
                continue
            rec = metadata[idx]
            print(f"        [{s:.3f}] {rec.get('bug_type')}: {str(rec.get('issue'))[:90]}")
        print()

    rel = sorted(t for _, e, t, _ in rows if e)
    off = sorted(t for _, e, t, _ in rows if not e)
    print("=" * 70)
    print("TOP-1 SCORE DISTRIBUTION")
    print(f"  on-domain (expect relevant): n={len(rel)}  "
          f"min={min(rel):.3f}  median={rel[len(rel)//2]:.3f}  max={max(rel):.3f}")
    print(f"  off-domain (expect weak):    n={len(off)}  "
          f"min={min(off):.3f}  median={off[len(off)//2]:.3f}  max={max(off):.3f}")
    print(f"\n  separating band: off-domain max = {max(off):.3f}, "
          f"on-domain min = {min(rel):.3f}")
    print("  -> pick a threshold inside this band (or just above off-domain max).")


if __name__ == "__main__":
    main()
