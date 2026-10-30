import logging
import time

import streamlit as st

from rag_core import (
    CostTracker, EMBEDDING_PRICE_PER_M, LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M,
    ask_rag, embed_text, mmr, optimize_query, retrieve_chunks, split_stream,
)

log = logging.getLogger("fe")
top_k = st.session_state.get("top_k", 30)

st.title("SEC filing Q&A")

query = st.text_input(
    "Ask a question about quarterly reports"
)

if st.button("Analyze & verify"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        log.info("Request: query=%r top_k=%d", query, top_k)
        costs = CostTracker()
        with st.spinner("optimizing query..."):
            q = optimize_query(query, costs=costs)
            # decompose queries into specific topics / sub queries
            # sub_queries = decompose_query(query)

        with st.spinner("embedding query..."):
            # subs = sub_queries["sub_queries"]
            # subs = subs[:6]
            # pool = {}
            # per_sub_k = max(top_k // len(subs) * 3, 15)
            # for sq in subs:
                
            query_vectors = embed_text(query, costs=costs)

        with st.spinner("retrieving relevant documents..."):
            chunks = retrieve_chunks(query_vectors, max(top_k * 4, 30), q["ticker"],
                                     start_year=q["start_year"], end_year=q["end_year"],
                                     start_quarter=q["start_quarter"], end_quarter=q["end_quarter"])

        with st.spinner("diversifying retrieved documents (MMR)..."):
            if chunks:
                keep = mmr(query_vectors[0], [c["vector"] for c in chunks], k=top_k)
                chunks = [chunks[i] for i in keep]
            for c in chunks:
                c.pop("vector", None)

        with st.spinner("formulating answer..."):
            stream, chunks = ask_rag(query, chunks)

        think_box = st.status("Thinking...", expanded=True)
        think_area = think_box.empty()
        st.subheader("Answer")
        answer_area = st.empty()

        thinking_text, answer_text = "", ""
        stream_usage: dict = {}
        t_stream = time.perf_counter()
        first_token = None
        try:
            for kind, piece in split_stream(stream, usage_out=stream_usage):
                if first_token is None:
                    first_token = time.perf_counter() - t_stream
                    log.info("First %s token after %.3fs", kind, first_token)
                if kind == "thinking":
                    thinking_text += piece
                    think_area.markdown(thinking_text)
                else:
                    answer_text += piece
                    answer_area.markdown(answer_text)
        except Exception:
            log.exception("Streaming failed after %d thinking / %d answer chars",
                          len(thinking_text), len(answer_text))
            raise
        log.info(
            "Stream finished in %.3fs: thinking=%d chars, answer=%d chars",
            time.perf_counter() - t_stream, len(thinking_text), len(answer_text),
        )
        if thinking_text:
            think_box.update(label="Thought process", state="complete", expanded=False)
        else:
            think_box.update(label="No thinking output", state="complete", expanded=False)

        if stream_usage.get("prompt_tokens") is not None:
            costs.add("Answer generation", LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M,
                      stream_usage["prompt_tokens"], stream_usage.get("completion_tokens") or 0)
        else:
            prompt_chars = len(query) + sum(len(c["text"]) for c in chunks)
            costs.add("Answer generation", LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M,
                      prompt_chars // 4, (len(thinking_text) + len(answer_text)) // 4, True)
        # retrieval + MMR run locally (Milvus Lite / numpy): no API cost

        st.subheader("Cost")
        st.table(costs.rows())
        log.info("Request cost: %s total=$%.6f", costs.stages, costs.total)

        st.subheader("Retrieved filing evidence")
        for i, c in enumerate(chunks, 1):
            with st.expander(
                f"[Source {i}] {c['source_file']} | {c['ticker']} {c['year']} Q{c['quarter']} (score: {c['score']:.4f})"
            ):
                st.write(c["text"])
