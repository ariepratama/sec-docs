import streamlit as st

# Must be the first Streamlit call, before rag_core is imported.
st.set_page_config(
    page_title="SEC filing",
    layout="wide",
)

import rag_core  # noqa: E402,F401  (initialises env, logging, clients)

bm25_demo = st.Page("views/bm25_demo.py", title="BM25", icon="🔎")
search_embeding_demo = st.Page("views/search_embeding_demo.py", title="Embedding retrieval", icon="🔎")

ask = st.Page("views/ask.py", title="Ask", icon="🔎", default=True)
tickers = st.Page("views/tickers.py", title="Tickers", icon="📋")
search_embedding_docs = st.Page("views/search_embedding_docs.py", title="Search", icon="🔎")

sec_docs_pages = [ask, tickers, search_embedding_docs]

pg = st.navigation(
    {
        "Retrieval tutorial": [
            bm25_demo,
            search_embeding_demo,
        ],
        "SEC Docs retrieval": [
            ask,
            tickers,
            search_embedding_docs,
        ],
    }
)

# Preserve slider selection across page changes
if "saved_top_k" not in st.session_state:
    st.session_state["saved_top_k"] = 30

st.session_state["top_k"] = st.session_state["saved_top_k"]

# Passages to retrieve applies to SEC Docs retrieval pages
if pg in sec_docs_pages:
    with st.sidebar:
        top_k = st.slider(
            "Passages to retrieve",
            min_value=1,
            max_value=30,
            value=st.session_state["saved_top_k"],
            key="top_k_slider",
        )
        st.session_state["saved_top_k"] = top_k
        st.session_state["top_k"] = top_k

pg.run()
