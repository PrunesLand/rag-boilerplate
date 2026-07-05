"""Query-side orchestration:

  expand -> hybrid retrieve (+RRF) -> CRAG relevance check -> rerank
         -> Small2Big parent expansion

Every stage is guarded by a config toggle so the pipeline degrades cleanly to a
plain vector search when features are off. Returns parent Documents to generate."""

import config
from rag.expand import generate_variants, reciprocal_rank_fusion

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def _hybrid_retrieve(stores, query, k):
    """Vector search, fused with BM25 keyword search when hybrid is enabled."""
    vector_hits = stores.vector_retriever(k).invoke(query)
    if not config.USE_HYBRID_SEARCH:
        return vector_hits
    bm25 = stores.bm25_retriever(k)
    if bm25 is None:
        return vector_hits
    return reciprocal_rank_fusion([vector_hits, bm25.invoke(query)])


def _multi_retrieve(stores, llm, query, k):
    """Run every query variant through hybrid retrieval and RRF-merge the lists."""
    variants = generate_variants(llm, query, config.NUM_QUERY_VARIANTS)
    ranked_lists = [_hybrid_retrieve(stores, v, k) for v in variants]
    return reciprocal_rank_fusion(ranked_lists)


def _rerank(query, docs, top_n):
    if not config.USE_RERANKER or not docs:
        return docs[:top_n]
    model = _get_reranker()
    scored = model.predict([(query, d.page_content) for d in docs])
    ranked = [d for _, d in sorted(zip(scored, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:top_n]


_CRAG_PROMPT = (
    "You grade whether retrieved context is relevant to a question. "
    "Reply with exactly one word: HIGH, LOW, or NONE.\n"
    "HIGH = clearly answers it, LOW = loosely related, NONE = unrelated.\n\n"
    "QUESTION: {q}\n\nCONTEXT:\n{ctx}"
)


def _crag_grade(llm, query, docs):
    if not config.USE_CRAG or not docs:
        return "HIGH"
    ctx = "\n---\n".join(d.page_content for d in docs[:3])
    try:
        verdict = llm.invoke(_CRAG_PROMPT.format(q=query, ctx=ctx)).content.strip().upper()
    except Exception:
        return "HIGH"
    # Bias toward proceeding: only bail on a clear, standalone NONE. Weak models
    # otherwise false-negative and suppress good answers.
    if verdict.split() and verdict.split()[0] == "NONE":
        return "NONE"
    if "LOW" in verdict:
        return "LOW"
    return "HIGH"


def _expand_to_parents(stores, child_docs):
    """Map matched children to their unique parents, preserving rank order."""
    ordered_ids, seen = [], set()
    for d in child_docs:
        pid = d.metadata.get("parent_id")
        if pid and pid not in seen:
            ordered_ids.append(pid)
            seen.add(pid)
    parents = stores.get_parents(ordered_ids)
    by_id = {p.metadata.get("parent_id"): p for p in parents}
    return [by_id[pid] for pid in ordered_ids if pid in by_id]


def retrieve(query, stores, llm):
    """Return (parent_docs, status). status in {'ok','none'}."""
    children = _multi_retrieve(stores, llm, query, config.RETRIEVAL_K)

    verdict = _crag_grade(llm, query, children)
    if verdict == "LOW":
        # CRAG corrective step: reformulate once and retry.
        try:
            reformed = llm.invoke(
                f"Rewrite this question to be clearer and more searchable:\n{query}"
            ).content.strip()
            children = _multi_retrieve(stores, llm, reformed, config.RETRIEVAL_K)
            verdict = _crag_grade(llm, reformed, children)
        except Exception:
            pass
    if verdict == "NONE" or not children:
        return [], "none"

    top_children = _rerank(query, children, config.RERANK_TOP_N)
    parents = _expand_to_parents(stores, top_children)
    return parents, "ok"
