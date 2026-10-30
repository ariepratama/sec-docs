import streamlit as st

from rag_core import list_tickers

st.title("Available tickers")
top_k = st.session_state.get("top_k", 30)
st.caption(f"Sidebar setting: retrieving {top_k} passages per question.")

tickers = list_tickers("data")
st.write(f"{len(tickers)} tickers indexed")
st.dataframe({"ticker": tickers}, use_container_width=True)

