import config


def _missing(pkg, provider):
    return ImportError(
        f"Provider '{provider}' needs the '{pkg}' package. Install it with: pip install {pkg}"
    )


def get_llm():
    """Return a LangChain chat model for config.LLM_PROVIDER."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise _missing("langchain-openai", provider)
        return ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise _missing("langchain-anthropic", provider)
        return ChatAnthropic(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def get_embeddings():
    """Return a LangChain embeddings object for config.EMBEDDING_PROVIDER."""
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.LLM_BASE_URL)
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise _missing("langchain-openai", provider)
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.EMBEDDING_PROVIDER}")


def get_vectorstore(embeddings=None):
    """Return a LangChain vector store for config.VECTORSTORE_PROVIDER."""
    embeddings = embeddings or get_embeddings()
    provider = config.VECTORSTORE_PROVIDER.lower()

    if provider == "chroma":
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
    if provider == "pgvector":
        try:
            from langchain_postgres import PGVector
        except ImportError:
            raise _missing("langchain-postgres", provider)
        if not config.PGVECTOR_CONNECTION:
            raise ValueError("Set PGVECTOR_CONNECTION when VECTORSTORE_PROVIDER=pgvector")
        return PGVector(
            embeddings=embeddings,
            collection_name=config.CHROMA_COLLECTION,
            connection=config.PGVECTOR_CONNECTION,
        )

    raise ValueError(f"Unknown VECTORSTORE_PROVIDER: {config.VECTORSTORE_PROVIDER}")
