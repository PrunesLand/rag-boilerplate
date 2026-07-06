import os

# ── Identity ──────────────────────────────────────────────────────────────────
# Who the assistant is. Set these to make the bot yours — no code changes needed.
ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME", "Your Organization")
ASSISTANT_DESCRIPTION = os.getenv(
    "ASSISTANT_DESCRIPTION", f"a helpful assistant for {ORGANIZATION_NAME}"
)

# ── Providers ─────────────────────────────────────────────────────────────────
# Swap the engine without touching pipeline code. See rag/providers.py.
#   LLM_PROVIDER:        ollama | openai | anthropic
#   EMBEDDING_PROVIDER:  ollama | openai
#   VECTORSTORE_PROVIDER: chroma | pgvector
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
VECTORSTORE_PROVIDER = os.getenv("VECTORSTORE_PROVIDER", "chroma")

# ── Models ────────────────────────────────────────────────────────────────────
# Model names are provider-specific, e.g.
#   ollama -> "qwen2.5:0.5b-instruct" / "nomic-embed-text"
#   openai -> "gpt-4o-mini"           / "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Provider connection details (only the selected provider's settings are used).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
# OPENAI_API_KEY / ANTHROPIC_API_KEY are read from the environment by the SDKs.

# Local cross-encoder reranker (runs on CPU via sentence-transformers).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Stores ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCSTORE_DIR = os.getenv("DOCSTORE_DIR", "./docstore")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
PGVECTOR_CONNECTION = os.getenv("PGVECTOR_CONNECTION", "")  # used when VECTORSTORE_PROVIDER=pgvector

# ── Chunking (Small2Big) ──────────────────────────────────────────────────────
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "200"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

# ── Crawling ──────────────────────────────────────────────────────────────────
SOURCES_FILE = os.getenv("SOURCES_FILE", "sources.txt")
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "1"))
RESPECT_ROBOTS_TXT = os.getenv("RESPECT_ROBOTS_TXT", "true").lower() == "true"

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
NUM_QUERY_VARIANTS = int(os.getenv("NUM_QUERY_VARIANTS", "4"))

# ── Feature toggles (Modular RAG) ─────────────────────────────────────────────
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"
USE_REVERSE_HYDE = os.getenv("USE_REVERSE_HYDE", "true").lower() == "true"
# CRAG self-grading needs a capable judge; small models false-negative on it.
# Turn on with a 7b+ / cloud LLM.
USE_CRAG = os.getenv("USE_CRAG", "false").lower() == "true"

# ── Observability ─────────────────────────────────────────────────────────────
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "org-rag")
