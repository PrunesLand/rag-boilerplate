import os

# Load secrets from .env before anything imports LangChain.
from dotenv import load_dotenv

load_dotenv()

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
#   ollama -> "qwen3:0.6b"  / "nomic-embed-text"
#   openai -> "gpt-4o-mini" / "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:0.6b")
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
# CRAWL_SOURCE: "live" BFS-fetches each seed; "commoncrawl" reads each line as a
# URL pattern (e.g. example.org/*) and pulls archived pages from Common Crawl.
CRAWL_SOURCE = os.getenv("CRAWL_SOURCE", "live")
SOURCES_FILE = os.getenv("SOURCES_FILE", "sources.txt")

# Live-crawl only.
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "1"))
RESPECT_ROBOTS_TXT = os.getenv("RESPECT_ROBOTS_TXT", "true").lower() == "true"

# Common Crawl only. CC_RECENT_CRAWLS = number of latest snapshots to query;
# CC_CRAWL pins an explicit id (e.g. "CC-MAIN-2026-25") and takes priority.
CC_CRAWL = os.getenv("CC_CRAWL", "")
CC_RECENT_CRAWLS = int(os.getenv("CC_RECENT_CRAWLS", "1"))
CC_LIMIT_PER_PATTERN = int(os.getenv("CC_LIMIT_PER_PATTERN", "50"))
CC_RETRIES = int(os.getenv("CC_RETRIES", "3"))
# Keep only these capture languages (ISO-639-3, comma-separated); empty = any.
CC_LANGUAGES = [l for l in os.getenv("CC_LANGUAGES", "").split(",") if l]

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
NUM_QUERY_VARIANTS = int(os.getenv("NUM_QUERY_VARIANTS", "4"))
# Cross-encoder relevance floor; if nothing clears it, the bot says it doesn't
# know rather than cite unrelated pages (ms-marco logits: relevant ≈ 0 or above).
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-5.0"))

# ── Feature toggles (Modular RAG) ─────────────────────────────────────────────
USE_QUERY_ROUTER = os.getenv("USE_QUERY_ROUTER", "true").lower() == "true"
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"
USE_REVERSE_HYDE = os.getenv("USE_REVERSE_HYDE", "true").lower() == "true"
# CRAG self-grading needs a capable judge; small models false-negative on it.
USE_CRAG = os.getenv("USE_CRAG", "false").lower() == "true"

# ── Observability ─────────────────────────────────────────────────────────────
# LangSmith reads these LANGCHAIN_* vars from .env; the API key stays there only.
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "org-rag")
