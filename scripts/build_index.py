"""
build_index.py — build the FAISS retrieval artifacts from a JSONL bug corpus.

Script form of the step-1 notebook (`step1_parse_and_embed.ipynb` in the research repo),
reshaped for clone-and-run use. Same two-source pipeline, auto-detected per row:

  * rows that are already labeled (have bug_type + issue + fix, e.g. the bundled sample
    or synthetic bugs) are normalized directly — no LLM call;
  * rows that are raw diffs (title + patches, no labels) are parsed by PARSER_MODEL into
    {bug_type, issue, fix, severity, confidence}.

Both are then merged, normalized, confidence-gated, embedded with EMBED_MODEL, and written
to a FAISS IndexFlatIP (cosine similarity on L2-normalized vectors) plus an aligned
metadata pickle.

Examples
--------
Rebuild the bundled sample index from its JSONL (~2,000 embedding calls, no parsing):

    python scripts/build_index.py \
        --input data/sample/sample_bug_corpus.jsonl \
        --output data/sample/sample_corpus.faiss

Bring your own data (raw PR diffs + pre-labeled synthetic, first 200 raw rows for a cheap
dry run), then point the server/dashboard at it via env vars:

    python scripts/build_index.py \
        --input data/raw_bug_corpus.jsonl data/synthetic_bugs.jsonl \
        --output data/my_corpus.faiss --max-samples 200
    # then: set FAISS_INDEX_PATH / METADATA_PATH to the new files

Needs OPENROUTER_API_KEY in .env (or the environment).
"""

import os
import sys
import json
import time
import pickle
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; degrade to a no-op iterator
    def tqdm(it, **kwargs):
        return it

# ---------------------------------------------------------
# MODEL SELECTION — change these to swap models
# ---------------------------------------------------------
# Chat model that turns a raw title+diff into structured JSON. Only runs on UNLABELED
# rows; labeled rows (sample/synthetic) skip it entirely. It's a reasoning model, so the
# request caps reasoning effort — step-1 found 300-token replies were fully consumed by
# hidden reasoning before any JSON was written.
PARSER_MODEL = "deepseek/deepseek-v4-flash"

# Embedding model for every record from every source, so they share one vector space.
# MUST match core.EMBEDDING_MODEL, or the server will query the index with the wrong model
# and silently get garbage neighbors.
EMBED_MODEL = "qwen/qwen3-embedding-8b"

# Run controls
PARSE_WORKERS = 50
PARSE_RETRIES = 3
PARSE_MAX_TOKENS = 1000
# Per-model reasoning control for the parser (OpenRouter `reasoning` param). deepseek-v4-flash
# honors {"effort": "low"} and stays cheap; reasoning models like glm-4.7-flash IGNORE effort
# caps and burn the whole budget on hidden reasoning, emitting empty JSON — those need
# reasoning DISABLED ({"enabled": False}). Override with --parser-reasoning.
PARSE_REASONING = {"effort": "low"}
DIFF_TRUNCATE_CHARS = 3000
EMBED_WORKERS = 10
EMBED_TRUNCATE_CHARS = 8000
EMBED_RETRIES = 3
MIN_CONFIDENCE = "medium"  # keep medium+high, drop low (matches step-1)

# ---------------------------------------------------------
# BUG-TYPE ENUM & NORMALIZATION (reused unchanged from step-1)
# ---------------------------------------------------------
CANONICAL_BUGS = {
    "logic_bug", "api_misuse", "type_error", "exception_handling",
    "null_check", "scope_error", "off_by_one", "concurrency",
    "memory_leak", "race_condition", "security", "performance",
}

NORMALIZATION_MAP = {
    "logics_bug": "logic_bug", "algorithmic_bug": "logic_bug", "out_of_bounds": "off_by_one",
    "infinite_recursion": "logic_bug", "division_by_zero": "logic_bug", "api_ misuse": "api_misuse",
    "API_MISUSE": "api_misuse", "deprecated_api_misuse": "api_misuse", "timezone_misuse": "api_misuse",
    "data_type": "type_error", "encoding": "type_error", "key_error": "type_error",
    "error_handling": "exception_handling", "null": "null_check", "none": "null_check",
    "invalid_reference": "null_check", "name_error": "scope_error", "security_issue": "security",
    "memory_error": "memory_leak", "performance_bug": "performance", "caching_error": "performance",
    "serialization": "api_misuse", "attribute_error": "type_error", "bug": "logic_bug",
}

PARSER_PROMPT = """You are an expert Python Bug Analyst and Data Extraction Tool.
Your singular task is to analyze a GitHub code diff and commit message, and output a strict JSON object mapping the bug to a predefined schema.

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON.
2. Do NOT wrap the JSON in markdown code blocks (no ```json).
3. Do NOT include greetings, explanations, or any conversational text.
4. Your response must begin exactly with `{` and end exactly with `}`.
5. You MUST map the 'bug_type' strictly to ONE of the allowed ENUM values below. Do not invent new types.

BUG TYPE ENUMERATIONS (Choose exactly one):
- "null_check": Missing or incorrect 'None' checks.
- "type_error": Type mismatches, casting errors, KeyError, AttributeError.
- "logic_bug": Incorrect algorithms, wrong math, flawed boolean logic, or bad state management.
- "off_by_one": Loop boundary errors, list index out of range.
- "scope_error": Variable scoping issues, NameError, shadowing.
- "exception_handling": Bad try/except blocks, catching wrong exceptions.
- "api_misuse": Incorrect library function usage, wrong parameters passed to an API.
- "concurrency": Race conditions, missing thread locks, async/await issues.
- "performance": Inefficient loops, N+1 queries, memory bloat.
- "security": Unsafe deserialization, SQL injection, unsafe regex.

SCHEMA DEFINITION:
{
  "bug_type": "<string> (Must be exact match from the enum list above)",
  "issue": "<string> (A clear, technical, single-sentence explanation of what was broken)",
  "fix": "<string> (A clear, technical, single-sentence explanation of how the patch fixed it)",
  "severity": "<string> (Must be 'low', 'medium', or 'high')",
  "confidence": "<string> (Must be 'low', 'medium', or 'high' based on how clearly you understand the diff)"
}"""

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def normalize_bug_type(bug_type):
    if not bug_type:
        return "logic_bug"
    bt_clean = bug_type.strip()
    if bt_clean in CANONICAL_BUGS:
        return bt_clean
    if bt_clean in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[bt_clean]
    bt_lower = bt_clean.lower().replace(" ", "_").replace("-", "_")
    for canonical in CANONICAL_BUGS:
        if canonical in bt_lower or bt_lower in canonical:
            return canonical
    return "logic_bug"


def normalize_severity(v):
    v = str(v or "").strip().lower()
    return v if v in {"low", "medium", "high"} else "medium"


def normalize_confidence(v):
    v = str(v or "").strip().lower()
    return v if v in {"low", "medium", "high"} else "low"


# ---------------------------------------------------------
# LOADING & CLASSIFICATION
# ---------------------------------------------------------
def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def is_labeled(row):
    """A row is already labeled if it carries bug_type + issue + fix."""
    return bool(row.get("bug_type")) and bool(row.get("issue")) and bool(row.get("fix"))


def to_labeled_record(row):
    """Bring an already-labeled row to the unified schema. Handles both the parsed-corpus
    shape and the synthetic (buggy_code/fixed_code) shape."""
    patches = row.get("patches")
    if not patches and (row.get("buggy_code") or row.get("fixed_code")):
        buggy = row.get("buggy_code") or ""
        fixed = row.get("fixed_code") or ""
        patches = [f"--- buggy ---\n{buggy}\n--- fixed ---\n{fixed}"]
    return {
        "source": row.get("source", "labeled"),
        "repo": row.get("repo") or row.get("base_repo"),
        "title": row.get("title"),
        "patches": patches or [""],
        "bug_type": row.get("bug_type"),
        "issue": row.get("issue"),
        "fix": row.get("fix"),
        "severity": row.get("severity"),
        "confidence": row.get("confidence"),
    }


def row_hash(row):
    return hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# PHASE 1 — LLM parse (unlabeled rows only)
# ---------------------------------------------------------
def parse_raw_sample(client, sample):
    title = sample.get("title") or sample.get("message", "")
    patches = "\n".join(sample.get("patches", []))
    user_content = f"Title: {title}\n\nDiff:\n{patches[:DIFF_TRUNCATE_CHARS]}"

    last_error = None
    for attempt in range(PARSE_RETRIES):
        try:
            response = client.chat.completions.create(
                model=PARSER_MODEL,
                messages=[
                    {"role": "system", "content": PARSER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=PARSE_MAX_TOKENS,
                extra_body={
                    "reasoning": PARSE_REASONING,
                    "provider": {"sort": "throughput"},
                },
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = json.loads(raw[: raw.rfind("}") + 1])
            return {**sample, **parsed, "source": sample.get("source", "pr")}
        except Exception as e:
            last_error = str(e)
            if attempt < PARSE_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {**sample, "parse_error": last_error, "source": sample.get("source", "pr")}


def parse_unlabeled(client, unlabeled, checkpoint_path):
    """Parse unlabeled rows with PARSER_MODEL, resuming from a content-hash checkpoint and
    flushing each result to disk as it completes — a crash mid-run doesn't re-bill the
    calls that already succeeded."""
    parsed = []
    done_hashes = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        for rec in load_jsonl(checkpoint_path):
            parsed.append(rec)
            if "_row_hash" in rec:
                done_hashes.add(rec["_row_hash"])
        print(f"  resuming: {len(parsed)} rows already parsed in a previous run")

    remaining = [(row_hash(r), r) for r in unlabeled if row_hash(r) not in done_hashes]
    print(f"  {len(remaining)} unlabeled rows to parse with {PARSER_MODEL}")
    if not remaining:
        return parsed

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
        with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as executor:
            futures = {executor.submit(parse_raw_sample, client, r): h for h, r in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing"):
                result = future.result()
                result["_row_hash"] = futures[future]
                parsed.append(result)
                ckpt.write(json.dumps(result) + "\n")
                ckpt.flush()

    n_errors = sum(1 for r in parsed if r.get("parse_error"))
    print(f"  parsed {len(parsed)} rows, {n_errors} parse errors (filtered next)")
    return parsed


# ---------------------------------------------------------
# PHASE 2 — merge, normalize, confidence-gate
# ---------------------------------------------------------
def merge_and_gate(records):
    merged = []
    for rec in records:
        if rec.get("parse_error"):
            continue
        if not is_labeled(rec):
            continue
        rec = dict(rec)
        rec.pop("_row_hash", None)
        rec["severity"] = normalize_severity(rec.get("severity"))
        rec["confidence"] = normalize_confidence(rec.get("confidence"))
        rec["bug_type"] = normalize_bug_type(rec.get("bug_type"))
        if CONFIDENCE_ORDER[rec["confidence"]] < CONFIDENCE_ORDER[MIN_CONFIDENCE]:
            continue
        merged.append(rec)
    return merged


# ---------------------------------------------------------
# PHASE 3 — embed
# ---------------------------------------------------------
def build_embedding_text(rec):
    context = rec.get("title") or rec.get("repo") or ""
    return (
        f"Bug type: {rec.get('bug_type')}\n"
        f"Issue: {rec.get('issue')}\n"
        f"Fix: {rec.get('fix')}\n"
        f"Context: {context}\n"
    )


def fetch_embedding(client, text):
    for attempt in range(EMBED_RETRIES):
        try:
            res = client.embeddings.create(model=EMBED_MODEL, input=text[:EMBED_TRUNCATE_CHARS])
            return res.data[0].embedding
        except Exception as e:
            if attempt == EMBED_RETRIES - 1:
                print(f"  embedding failed after {EMBED_RETRIES} attempts: {e}")
            else:
                time.sleep(2 ** attempt)
    return None


def embed_records(client, records):
    texts = [build_embedding_text(r) for r in records]
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {executor.submit(fetch_embedding, client, t): i for i, t in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            embeddings[futures[future]] = future.result()
    return [(e, r) for e, r in zip(embeddings, records) if e is not None]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    global PARSER_MODEL, PARSE_REASONING  # allow CLI flags to override the defaults
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more JSONL corpus files (labeled and/or raw-diff rows).")
    ap.add_argument("--output", required=True,
                    help="Output FAISS index path (e.g. data/sample/sample_corpus.faiss). "
                         "Metadata is written alongside as <stem>_metadata.pkl.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap rows loaded PER input file (None = all). Use a small value "
                         "for a cheap dry run when a file needs LLM parsing.")
    ap.add_argument("--parsed-output", default=None,
                    help="Optional path to also write the merged parsed corpus as JSONL "
                         "(human-readable audit trail).")
    ap.add_argument("--parser-model", default=PARSER_MODEL,
                    help=f"Chat model used to parse UNLABELED raw-diff rows "
                         f"(default: {PARSER_MODEL}). Labeled rows never hit it.")
    ap.add_argument("--parser-reasoning", default="low",
                    help="Reasoning control for the parser: 'low'/'medium'/'high' (effort "
                         "cap, for models that honor it like deepseek) or 'disabled' (for "
                         "reasoning models like glm-4.7-flash that ignore effort caps and "
                         "would otherwise return empty JSON). Default: low.")
    args = ap.parse_args()

    # Honor the CLI overrides (parse_raw_sample reads these globals).
    PARSER_MODEL = args.parser_model
    if args.parser_reasoning.lower() in {"disabled", "off", "none", "false"}:
        PARSE_REASONING = {"enabled": False}
    else:
        PARSE_REASONING = {"effort": args.parser_reasoning}

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Put it in a .env at the repo root.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if not args.output.endswith(".faiss"):
        sys.exit("--output must end in .faiss")
    index_path = args.output
    metadata_path = args.output[: -len(".faiss")] + "_metadata.pkl"
    checkpoint_path = args.output[: -len(".faiss")] + "_parse_checkpoint.jsonl"
    # Create the output dir up front — the parse checkpoint (Phase 1) writes here before
    # the index/metadata (Phase 4).
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    # Phase 0 — load & classify
    all_rows = []
    for path in args.input:
        rows = load_jsonl(path, args.max_samples)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {path}")

    labeled = [to_labeled_record(r) for r in all_rows if is_labeled(r)]
    unlabeled = [r for r in all_rows if not is_labeled(r)]
    print(f"Classified: {len(labeled)} already-labeled, {len(unlabeled)} need LLM parsing")

    # Phase 1 — parse unlabeled rows (skipped entirely if there are none)
    parsed = parse_unlabeled(client, unlabeled, checkpoint_path) if unlabeled else []

    # Phase 2 — merge + normalize + gate
    merged = merge_and_gate(labeled + parsed)
    print(f"Phase 2: {len(merged)} records survived normalization + confidence gate")
    if not merged:
        sys.exit("No records left to index after the confidence gate.")

    if args.parsed_output:
        with open(args.parsed_output, "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote parsed corpus: {args.parsed_output}")

    # Phase 3 — embed
    valid_pairs = embed_records(client, merged)
    print(f"Phase 3: {len(valid_pairs)}/{len(merged)} records embedded")
    if not valid_pairs:
        sys.exit("No embeddings were generated — check API key/model/network.")

    # Phase 4 — build & save FAISS
    matrix = np.array([e for e, _ in valid_pairs], dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    records = [r for _, r in valid_pairs]

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(records, f)

    print(f"\nSaved {index_path} ({index.ntotal} vectors, dim={matrix.shape[1]})")
    print(f"Saved {metadata_path} ({len(records)} records, aligned to index order)")


if __name__ == "__main__":
    main()
