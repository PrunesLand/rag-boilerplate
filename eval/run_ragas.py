import json
import os

import config
from rag.generate import answer
from rag.retrieve import retrieve
from rag.store import get_llm, load_stores

TESTSET = os.path.join(os.path.dirname(__file__), "testset.jsonl")


def _load_testset():
    with open(TESTSET) as f:
        return [json.loads(line) for line in f if line.strip()]


def _collect():
    stores = load_stores()
    llm = get_llm()
    rows = []
    for item in _load_testset():
        q = item["question"]
        parents, _ = retrieve(q, stores, llm)
        contexts = [p.page_content for p in parents]
        result = answer(q, [], stores)
        rows.append(
            {
                "question": q,
                "answer": "".join(result.tokens),
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
        print(f"  scored inputs for: {q[:60]}")
    return rows


def _patch_ragas_import():
    """Make ragas 0.4.3 importable on langchain-community 0.4.x.

    ragas hard-imports the removed langchain_community...vertexai module (we never
    use Vertex AI). Register a placeholder so the import resolves; delete once fixed.
    """
    import sys
    import types

    missing = "langchain_community.chat_models.vertexai"
    if missing not in sys.modules:
        try:
            __import__(missing)
        except ModuleNotFoundError:
            placeholder = types.ModuleType(missing)
            placeholder.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[missing] = placeholder


def main():
    rows = _collect()
    try:
        _patch_ragas_import()
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        judge = ChatOllama(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL)
        embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.LLM_BASE_URL)
        scores = evaluate(
            Dataset.from_list(rows),
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge,
            embeddings=embeddings,
        )
        print("\nRAGAS scores:")
        print(scores)
    except Exception as e:
        print(f"\n[RAGAS unavailable or failed: {e}]")
        print("Collected inputs anyway; here are the raw answers:\n")
        for r in rows:
            print(f"Q: {r['question']}\nA: {r['answer'][:300]}\n")


if __name__ == "__main__":
    main()
