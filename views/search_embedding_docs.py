import logging
import time

import streamlit as st

from rag_core import (
    CostTracker, embed_text, retrieve_chunks, list_tickers
)

log = logging.getLogger("fe")
top_k = st.session_state.get("top_k", 30)
tickers = list_tickers("data")

st.title("Search embedding")

query = st.text_input(
    "Search something here"
)

ticker_dropdown = st.selectbox("ticker", options=tickers)

if st.button("Search"):
    if not query.strip():
        st.warning("please search something")
    else:
        cost_tracker = CostTracker()
        
        with st.spinner("embedding query ..."):
            query_vectors = embed_text(query, costs=cost_tracker)


        with st.spinner("retrieving documents"):
            chunks = retrieve_chunks(
                query_vectors=query_vectors, top_k=top_k, ticker=ticker_dropdown)
    
        for i, c in enumerate(chunks, 1):
            with st.expander(
                f"[Source {i}] {c['source_file']} | {c['ticker']} {c['year']} Q{c['quarter']} (score: {c['score']:.4f})"
            ):
                st.write(c["text"])
