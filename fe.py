import streamlit as st

# Must be the first Streamlit call, before rag_core is imported.
st.set_page_config(
    page_title="SEC filing",
    layout="wide",
)

import rag_core  # noqa: E402,F401  (initialises env, logging, clients)

ask = st.Page("views/ask.py", title="Ask", icon="🔎", default=True)
tickers = st.Page("views/tickers.py", title="Tickers", icon="📋")
search_embedding_docs = st.Page("views/search_embedding_docs.py", title="Search")

pg = st.navigation([ask, tickers, search_embedding_docs])

# Runs on every page, so the sidebar shows on home and subpages.
with st.sidebar:
    st.slider(
        "Passages to retrieve", min_value=1, max_value=30, value=30, key="top_k"
    )

pg.run()
