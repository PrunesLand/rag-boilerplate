from langchain_core.documents import Document

import config
from rag.store import get_llm

_QUESTIONS_PROMPT = (
    "Generate exactly 3 distinct questions that the following text directly answers. "
    "Return only the questions, one per line, no numbering.\n\nTEXT:\n{text}"
)
_SUMMARY_PROMPT = "Summarise the following text in one short sentence.\n\nTEXT:\n{text}"


def _lines(raw):
    return [ln.strip("-•* ").strip() for ln in raw.splitlines() if ln.strip()]


def enrich(parents):
    """Mutates parents (adds 'summary') and returns a list of question Documents."""
    if not config.USE_REVERSE_HYDE:
        print("Reverse-HyDE disabled; skipping enrichment.")
        return []

    llm = get_llm()
    question_docs = []

    for i, parent in enumerate(parents, 1):
        text = parent.page_content[:2000]
        try:
            questions = _lines(llm.invoke(_QUESTIONS_PROMPT.format(text=text)).content)[:3]
            summary = llm.invoke(_SUMMARY_PROMPT.format(text=text)).content.strip()
        except Exception as e:  # keep ingestion resilient to a bad LLM call
            print(f"  [warn] enrich failed on parent {i}: {e}")
            continue

        parent.metadata["summary"] = summary
        for q in questions:
            question_docs.append(
                Document(
                    page_content=q,
                    metadata={**parent.metadata, "kind": "question"},
                )
            )
        if i % 10 == 0:
            print(f"  enriched {i}/{len(parents)} parents")

    print(f"Enrichment produced {len(question_docs)} hypothetical-question docs.")
    return question_docs
