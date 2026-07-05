"""Multi-query expansion (RAG Fusion) + Reciprocal Rank Fusion.

Generate several paraphrases of the user's question so retrieval casts a wider,
more robust net, then fuse the per-variant result lists into one ranking."""

import config

_VARIANTS_PROMPT = (
    "You are helping a search system. Rewrite the question below into {n} "
    "alternative search queries that capture different phrasings and angles. "
    "Return only the queries, one per line, no numbering.\n\nQUESTION: {q}"
)


def generate_variants(llm, query, n=None):
    """Return [original, variant1, ...]. Falls back to [query] on any failure."""
    n = n or config.NUM_QUERY_VARIANTS
    if not config.USE_MULTI_QUERY:
        return [query]
    try:
        raw = llm.invoke(_VARIANTS_PROMPT.format(n=n, q=query)).content
        variants = [ln.strip("-•* ").strip() for ln in raw.splitlines() if ln.strip()]
    except Exception:
        variants = []
    # Original first, dedup, cap at n+1.
    out, seen = [query], {query.lower()}
    for v in variants:
        if v.lower() not in seen:
            out.append(v)
            seen.add(v.lower())
    return out[: n + 1]


def _doc_key(doc):
    """Identity for fusion: prefer parent_id so children of the same parent fuse."""
    return doc.metadata.get("parent_id") or doc.page_content[:120]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked doc lists into one. Returns docs sorted by RRF score."""
    scores, docs_by_key = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_by_key.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ordered]
