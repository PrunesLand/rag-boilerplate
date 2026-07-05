"""Ingestion entry point: crawl -> chunk (Small2Big) -> enrich (Reverse-HyDE)
-> embed children + question docs into the vector store, write parents to the
docstore, and persist the child list for BM25.

Run:  python -m ingest.build_index
Re-run whenever your source pages change. Rebuilds all stores from scratch."""

import shutil

import config
from ingest.chunk import chunk
from ingest.crawl import crawl
from ingest.enrich import enrich
from rag.store import ParentStore, get_vectorstore, save_children


def _reset_stores():
    for path in (config.CHROMA_PERSIST_DIR, config.DOCSTORE_DIR):
        shutil.rmtree(path, ignore_errors=True)
    print("Cleared existing stores.")


def build():
    _reset_stores()

    docs = crawl()
    if not docs:
        print("No documents crawled. Check sources.txt / network. Aborting.")
        return

    parents, children = chunk(docs)
    question_docs = enrich(parents)  # mutates parents (adds summary), returns questions

    embed_docs = children + question_docs
    print(f"Embedding {len(embed_docs)} docs ({len(children)} children + "
          f"{len(question_docs)} questions) via {config.EMBEDDING_PROVIDER}...")

    vectorstore = get_vectorstore()
    # Batch to keep memory/HTTP payloads reasonable on a laptop.
    for i in range(0, len(embed_docs), 128):
        vectorstore.add_documents(embed_docs[i:i + 128])

    ParentStore().put(parents)
    save_children(embed_docs)  # provider-independent corpus for BM25
    print(f"Wrote {len(parents)} parents to docstore at {config.DOCSTORE_DIR}.")
    print("Ingestion complete.")


if __name__ == "__main__":
    build()
