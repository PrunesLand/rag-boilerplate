import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# RecursiveCharacterTextSplitter counts characters; config sizes are in tokens.
CHARS_PER_TOKEN = 4


def _splitter(size_tokens, overlap_tokens):
    return RecursiveCharacterTextSplitter(
        chunk_size=size_tokens * CHARS_PER_TOKEN,
        chunk_overlap=overlap_tokens * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk(docs):
    """Return (parents, children) as two lists of Documents."""
    parent_splitter = _splitter(config.PARENT_CHUNK_SIZE, config.CHUNK_OVERLAP)
    child_splitter = _splitter(config.CHILD_CHUNK_SIZE, config.CHUNK_OVERLAP)

    parents, children = [], []

    for doc in docs:
        for parent_text in parent_splitter.split_text(doc.page_content):
            parent_id = str(uuid.uuid4())
            parent_meta = {**doc.metadata, "parent_id": parent_id}
            parents.append(Document(page_content=parent_text, metadata=parent_meta))

            for child_text in child_splitter.split_text(parent_text):
                child_meta = {**doc.metadata, "parent_id": parent_id, "kind": "child"}
                children.append(Document(page_content=child_text, metadata=child_meta))

    print(f"Chunked into {len(parents)} parents / {len(children)} children.")
    return parents, children


if __name__ == "__main__":
    from ingest.crawl import crawl

    chunk(crawl())
