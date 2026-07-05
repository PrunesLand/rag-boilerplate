"""Offline quality harness. Runs the pipeline over eval/testset.jsonl and scores
it with RAGAS, using the local Ollama models as both generator and judge.

Run:  python -m eval.run_ragas

RAGAS with a small local judge model is directional, not gospel — use it to
compare config toggles against each other, not as an absolute grade."""

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


def main():
    rows = _collect()
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        judge = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
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
