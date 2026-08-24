# Confidence-gate calibration

**`CONFIDENCE_THRESHOLD = 0.47`** (was `0.6`, inherited from `text-embedding-3-small`).

The retrieval gate decides whether the top FAISS hit is good enough to ground the answer
on. Below the threshold, RootCause answers from reasoning alone and tells the model no
strong matches were found. The value was picked by running a held-out set of realistic,
bug-report-style queries through the `qwen/qwen3-embedding-8b` index and judging relevance
by eye — not asserted. Reproduce it with `python scripts/calibrate_threshold.py`.

## Why 0.47

Re-grouped by *actual retrieved relevance* rather than by category prior:

| group | top-1 similarity |
|---|---|
| relevant match retrieved | min ≈ **0.516** |
| no good match retrieved   | max ≈ **0.425** |

That's a clean ~0.09 gap. `0.47` sits in the middle of it (~0.045 margin on each side). At
the old `0.6`, genuinely-relevant retrievals would have been discarded and the LLM forced
to answer with "no strong matches" — e.g. null_check (0.598, 0.568), type_error (0.572,
0.546), concurrency (0.595), off_by_one (0.516). `qwen3-embedding-8b` simply produces
lower absolute cosine scores on this text than `text-embedding-3-small` did, so the old
constant never transferred.

## One global threshold, not per-`bug_type`

Every bug_type's genuinely-relevant hits clear a single global 0.47 line while every
no-good-match hit falls below it, so per-type thresholds would add complexity without
buying accuracy. Revisit only if a future corpus shows a category whose relevant hits
routinely score below the gate.

## When to re-run

Re-run `calibrate_threshold.py` after any change to the embedding model or the corpus. The
separating band — not the specific number — is what to preserve.
