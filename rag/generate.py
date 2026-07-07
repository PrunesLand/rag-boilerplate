from dataclasses import dataclass, field
from typing import Iterator

import config
from rag.retrieve import retrieve
from rag.store import get_llm

_SYSTEM_PROMPT = (
    f"You are {config.ASSISTANT_DESCRIPTION}. Answer ONLY using the context below. "
    "If the answer isn't in the context, say you don't know and suggest where to "
    "look. Always cite the source URL(s) you used."
)

_ANSWER_PROMPT = "{system}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

_NO_CONTEXT = (
    "I don't have information about that in the sources I've been given. "
    "Try rephrasing, or check the website directly."
)

_CHAT_PROMPT = (
    f"You are {config.ASSISTANT_DESCRIPTION}. The user is making small talk — "
    "reply briefly and warmly, and offer to help with questions about "
    f"{config.ORGANIZATION_NAME}. Do not cite sources.\n\nUser: {{query}}\n\nReply:"
)

# A message made up only of these words skips retrieval. Kept deterministic
# because small local models misclassify plain "hello"; extend per deployment.
_SMALL_TALK_WORDS = {
    "hello", "hi", "hey", "hiya", "howdy", "yo", "greetings",
    "thanks", "thank", "thx", "cheers", "appreciated",
    "good", "morning", "afternoon", "evening", "night", "day",
    "how", "are", "you", "doing", "whats", "up", "sup", "there",
    "bye", "goodbye", "later", "ok", "okay", "cool", "great", "nice", "awesome",
}


def _needs_retrieval(query):
    """Deterministic small-talk gate; anything not clearly small talk searches."""
    if not config.USE_QUERY_ROUTER:
        return True
    import re
    words = re.findall(r"\w+", query.lower())
    if not words or len(words) > 6:
        return True
    return not all(w in _SMALL_TALK_WORDS for w in words)


@dataclass
class AnswerResult:
    source_urls: list = field(default_factory=list)
    tokens: Iterator[str] = iter(())


def _rewrite_with_history(llm, query, history):
    """Turn a follow-up into a standalone query using recent turns."""
    if not history:
        return query
    convo = "\n".join(f"{role}: {text}" for role, text in history[-4:])
    try:
        return llm.invoke(
            "Given the conversation, rewrite the follow-up as a standalone question.\n"
            f"\n{convo}\n\nFollow-up: {query}\n\nStandalone question:"
        ).content.strip()
    except Exception:
        return query


def _format_context(parents):
    blocks = []
    for p in parents:
        url = p.metadata.get("url", "unknown source")
        blocks.append(f"[Source: {url}]\n{p.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer(query, history, stores):
    """Run the pipeline, resolve sources, return a streaming AnswerResult."""
    llm = get_llm()

    # Small talk never touches the index, so it can never surface citations.
    if not _needs_retrieval(query):
        def _chat_stream():
            for chunk in llm.stream(_CHAT_PROMPT.format(query=query)):
                yield chunk.content
        return AnswerResult(source_urls=[], tokens=_chat_stream())

    standalone = _rewrite_with_history(llm, query, history)
    parents, status = retrieve(standalone, stores, llm)

    if status == "none" or not parents:
        return AnswerResult(source_urls=[], tokens=iter([_NO_CONTEXT]))

    source_urls = list(dict.fromkeys(p.metadata.get("url") for p in parents if p.metadata.get("url")))
    prompt = _ANSWER_PROMPT.format(
        system=_SYSTEM_PROMPT, context=_format_context(parents), question=standalone
    )

    def _stream():
        for chunk in llm.stream(prompt):
            yield chunk.content

    return AnswerResult(source_urls=source_urls, tokens=_stream())
