"""Streamlit chat UI. Thin shell over the same rag/ interface as cli.py.

Run:  streamlit run app.py
Make sure you've run `python -m ingest.build_index` first."""

import streamlit as st

import config
from rag.generate import answer
from rag.store import load_stores

st.set_page_config(page_title=f"{config.ORGANIZATION_NAME} Assistant", page_icon="🎓")
st.title(f"🎓 {config.ORGANIZATION_NAME} Assistant")


@st.cache_resource
def _stores():
    return load_stores()


stores = _stores()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))

if prompt := st.chat_input(f"Ask about {config.ORGANIZATION_NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Searching sources..."):
            result = answer(prompt, history, stores)
        text = st.write_stream(result.tokens)
        if result.source_urls:
            st.caption("Sources: " + " · ".join(result.source_urls))

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": result.source_urls}
    )
