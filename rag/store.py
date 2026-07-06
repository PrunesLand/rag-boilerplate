import json
import os

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config
from rag.providers import get_embeddings, get_llm, get_vectorstore

__all__ = ["get_llm", "get_embeddings", "get_vectorstore", "ParentStore", "Stores", "load_stores"]

_CHILDREN_FILE = "_children.jsonl"


def _doc_to_dict(doc):
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def _dict_to_doc(d):
    return Document(page_content=d["page_content"], metadata=d["metadata"])


class ParentStore:
    """On-disk store of parent Documents as one JSON file per parent_id.

    A deliberately tiny local docstore — the Small2Big parents live here, NOT in
    the vector DB. Persists across runs so retrieval can always expand a matched
    child to its full parent."""

    def __init__(self):
        self._dir = config.DOCSTORE_DIR
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, parent_id):
        return os.path.join(self._dir, f"{parent_id}.json")

    def put(self, parents):
        for p in parents:
            with open(self._path(p.metadata["parent_id"]), "w") as f:
                json.dump(_doc_to_dict(p), f)

    def get(self, parent_ids):
        out = []
        for pid in parent_ids:
            path = self._path(pid)
            if os.path.exists(path):
                with open(path) as f:
                    out.append(_dict_to_doc(json.load(f)))
        return out


def save_children(docs):
    """Persist the embedded child docs for provider-independent BM25 search."""
    os.makedirs(config.DOCSTORE_DIR, exist_ok=True)
    with open(os.path.join(config.DOCSTORE_DIR, _CHILDREN_FILE), "w") as f:
        for d in docs:
            f.write(json.dumps(_doc_to_dict(d)) + "\n")


def load_children():
    path = os.path.join(config.DOCSTORE_DIR, _CHILDREN_FILE)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [_dict_to_doc(json.loads(line)) for line in f if line.strip()]


class Stores:
    """Bundle of the loaded stores + lazily-built BM25 over child docs."""

    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.parents = ParentStore()
        self._child_docs = None

    def child_docs(self):
        if self._child_docs is None:
            self._child_docs = load_children()
        return self._child_docs

    def bm25_retriever(self, k):
        docs = self.child_docs()
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = k
        return retriever

    def vector_retriever(self, k):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_parents(self, parent_ids):
        return self.parents.get(parent_ids)


def load_stores():
    return Stores()
