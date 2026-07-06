import sys

from rag.generate import answer
from rag.store import load_stores


def main():
    print("Loading stores + connecting to Ollama...")
    stores = load_stores()
    if not stores.child_docs():
        print("The index is empty. Run `python -m ingest.build_index` first.")
        return

    print("Ready. Ask a question (Ctrl-C or 'exit' to quit).\n")
    history = []

    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not query or query.lower() in {"exit", "quit"}:
            print("bye")
            return

        print("bot > ", end="", flush=True)
        result = answer(query, history, stores)

        full = []
        for token in result.tokens:
            sys.stdout.write(token)
            sys.stdout.flush()
            full.append(token)
        print()

        if result.source_urls:
            print("\nsources:")
            for url in result.source_urls:
                print(f"  - {url}")
        print()

        history.append(("user", query))
        history.append(("assistant", "".join(full)))


if __name__ == "__main__":
    main()
