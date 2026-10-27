import streamlit as st
from dotenv import load_dotenv
from index_to_milvus import *
from openai import OpenAI
import json
import logging
import time
import numpy as np

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("fe")

db_path = os.environ.get("DB_PATH", "vector/10qvector.db")


@st.cache_resource
def get_milvus_client(path: str) -> pymilvus.MilvusClient:
    log.info("Opening Milvus client at %s", path)
    return pymilvus.MilvusClient(path)


mclient = get_milvus_client(db_path)
EMBEDDING_URL = os.environ["EMBEDDING_URL"]
LLM_URL = os.environ["LLM_URL"]
log.debug("Config: db_path=%s embedding_url=%s llm_url=%s", db_path, EMBEDDING_URL, LLM_URL)

oai_client = OpenAI(
    base_url=EMBEDDING_URL,
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)
llm_client = OpenAI(base_url=LLM_URL, api_key=os.environ.get("LLM_API_KEY", "none"))


@st.cache_resource(ttl=300)
def get_llm_model() -> str:
    """Use LLM_MODEL if set, otherwise auto-detect the first model served at LLM_URL."""
    configured = os.environ.get("LLM_MODEL")
    if configured:
        log.info("Using configured LLM model: %s", configured)
        return configured
    model = llm_client.models.list().data[0].id
    log.info("Auto-detected LLM model: %s", model)
    return model


def embed_text(text: str, model_name: str = "bge-base-en-v1.5", batch_size: int = 16) -> list[list[float]]:
    chunks = chunk_text(text)
    log.debug("Embedding query (%d chars, %d chunks) with %s", len(text), len(chunks), model_name)
    query_vectors = get_embeddings(
        oai_client=oai_client,
        texts=chunks,
        model=model_name,
        batch_size=batch_size,
    )
    return query_vectors


def retrieve_chunks(query_vectors: list[list[float]], top_k: int, ticker: str, 
                    start_year: int = None, end_year: int = None, 
                    start_quarter: int = None, end_quarter: int = None,
                    collection_name: str = "10q_1024"):
    t0 = time.perf_counter()
    if not query_vectors:
        log.warning("No query vectors produced; returning no chunks")
        return []

    dim_collection = f"10q_{len(query_vectors[0])}"
    if mclient.has_collection(dim_collection):
        collection_name = dim_collection

    log.info("Loading collection %s", collection_name)
    mclient.load_collection(collection_name)

    def _to_int(v):
        # LLM output may be None, an int, or a numeric string like "2025"
        try:
            return None if v is None else int(v)
        except (TypeError, ValueError):
            log.warning("Ignoring non-integer filter value %r", v)
            return None

    start_year, end_year = _to_int(start_year), _to_int(end_year)
    start_quarter, end_quarter = _to_int(start_quarter), _to_int(end_quarter)

    conditions = []
    if ticker:
        safe_ticker = str(ticker).replace("\\", "").replace("'", "")
        conditions.append(f"ticker == '{safe_ticker}'")
    if start_year is not None:
        conditions.append(f"year >= {start_year}")
    if end_year is not None:
        conditions.append(f"year <= {end_year}")
    if start_quarter is not None:
        conditions.append(f"quarter >= {start_quarter}")
    if end_quarter is not None:
        conditions.append(f"quarter <= {end_quarter}")
    filter = " and ".join(conditions)

    log.info("Searching %s: top_k=%d filter=%s", collection_name, top_k, filter)
    results = mclient.search(
        collection_name=collection_name,
        data=query_vectors[:1],
        limit=top_k,
        filter=filter,
        output_fields=["text", "ticker", "quarter", "year", "source_file", "vector"],
    )
    chunks = []
    for hit in results[0]:
        chunks.append({
            "text": hit["entity"]["text"],
            "ticker": hit["entity"]["ticker"],
            "quarter": hit["entity"]["quarter"],
            "year": hit["entity"]["year"],
            "source_file": hit["entity"]["source_file"],
            "page": hit["entity"].get("page", "N/A"),
            "score": hit["distance"],
            "vector": hit["entity"]["vector"]
        })
    log.info(
        "Retrieved %d chunks in %.3fs: %s",
        len(chunks),
        time.perf_counter() - t0,
        [(c["source_file"], round(c["score"], 4)) for c in chunks],
    )
    return chunks


def optimize_query(query: str) -> dict:
    prompt = f""" you are an assistant preparing a search query for SEC financial filings (10-K, 10-Q, 8-K).
    Analyze the user's question, extract metadata, and rewrite the query using formal financial and accounting terminology.
    Additionally you must control and return start_year and end_year, start_quarter and end_quarter whenever applicable. 
    let those fields to be null if you do not want to filter out the years or quarters
    User Query: "{query}"
    Respond ONLY with a valid JSON object matching this schema:
{{
  "ticker": "string or null",
  "start_year": "integer or null",
  "end_year": "integer or null",
  "start_quarter": "integer or null",
  "end_quarter": "integer or null",
  "optimized_search_query": "string"
}}
"""
    log.info("Optimizing query: %r", query)
    t0 = time.perf_counter()
    response = llm_client.chat.completions.create(
        model=get_llm_model(),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    log.debug("optimize_query raw response: %s", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.exception("optimize_query returned invalid JSON: %r", raw)
        raise
    log.info("Optimized query in %.3fs: %s", time.perf_counter() - t0, parsed)
    return parsed

def ask_rag(query: str, chunks) -> tuple[str, list[dict]]:
    log.info("Building answer from %d chunks for query=%r", len(chunks), query)
    context_str = ""
    for i, c in enumerate(chunks, 1):
        context_str += (
            f"\n[Source {i}] source_file={c['source_file']} "
            f"(ticker={c['ticker']}, year={c['year']}, Q{c['quarter']})"
            f"<begin_text>{c['text']}</end_text>"
        )

    system_prompt = (
        "You are a strict financial filing research assistant. "
        "Answer the user's question using ONLY the provided context excerpts below. "
        "Cite your sources using [Source X] notation. If the numbers or facts "
        "are not directly mentioned in the context, state that the information is unavailable."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
    stream = llm_client.chat.completions.create(
        model=get_llm_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        # temperature=0.1,  # Low temperature to prevent hallucinations on numbers
    )
    return stream, chunks


def split_stream(stream):
    """Yield ("thinking", text) / ("answer", text) pieces from a streamed response.

    Handles both server-side reasoning fields (`reasoning_content` / `reasoning`,
    used by llama.cpp, vLLM, DeepSeek) and inline <think>...</think> tags.
    """
    in_think = False
    buf = ""
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            yield "thinking", reasoning
        content = delta.content
        if not content:
            continue
        buf += content
        while buf:
            tag = "</think>" if in_think else "<think>"
            idx = buf.find(tag)
            if idx != -1:
                if idx:
                    yield ("thinking" if in_think else "answer"), buf[:idx]
                buf = buf[idx + len(tag):]
                in_think = not in_think
                continue
            # keep a possible partial tag at the end of the buffer
            keep = 0
            for n in range(min(len(tag) - 1, len(buf)), 0, -1):
                if tag.startswith(buf[-n:]):
                    keep = n
                    break
            emit, buf = buf[: len(buf) - keep], buf[len(buf) - keep:]
            if emit:
                yield ("thinking" if in_think else "answer"), emit
            break
    if buf:
        yield ("thinking" if in_think else "answer"), buf



def mmr(query_vec, cand_vecs, k, lambda_mult=0.6):
    q = np.asarray(query_vec, dtype=np.float32)
    C = np.asarray(cand_vecs, dtype=np.float32)

    q = q / np.linalg.norm(q)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)

    rel = C @ q
    sim = C @ C.T

    selected = [
        int(np.argmax(rel))
    ]

    remaining = set(range(len(C))) - set(selected)


    while remaining and len(selected) < k:
        best, best_score = None, -np.inf

        for i in remaining:
            redundancy = sim[i, selected].max()
            score = lambda_mult * rel[i] - (1 - lambda_mult) * redundancy

            if score > best_score:
                best, best_score = i, score

        selected.append(best)
        remaining.remove(best)
    return selected

st.set_page_config(
    page_title="SEC filing",
    layout="wide",
)

@st.cache_data(ttl=60)
def list_tickers(data_dir: str) -> list[str]:
    """Distinct tickers parsed from filenames like '2019_3_AAPL.md' in data_dir."""
    tickers = {
        parse_filename_metadata(p)["ticker"] for p in Path(data_dir).rglob("*.md")
    }
    tickers.discard("UNKNOWN")
    log.debug("Found %d tickers in %s", len(tickers), data_dir)
    return sorted(tickers)


with st.sidebar:
    tickers = list_tickers(os.environ.get("FILINGS_DIR", "data/fillings"))
    if not tickers:
        st.error("No filings found. Check FILINGS_DIR (default: data/fillings).")
        st.stop()
    ticker = st.selectbox("Company", tickers)
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=30, value=3)

query = st.text_input(
    "Ask a question about quarterly reports"
)

if st.button("Analyze & verify"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        log.info("Request: query=%r sidebar_ticker=%s top_k=%d", query, ticker, top_k)
        with st.spinner("optimizing query..."):
            q = optimize_query(query)

        with st.spinner("embedding query..."):
            query_vectors = embed_text(query)

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
        t_stream = time.perf_counter()
        first_token = None
        try:
            for kind, piece in split_stream(stream):
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

        st.subheader("Retrieved filing evidence")
        for i, c in enumerate(chunks, 1):
            with st.expander(
                f"[Source {i}] {c['source_file']} | {c['ticker']} {c['year']} Q{c['quarter']} (score: {c['score']:.4f})"
            ):
                st.write(c["text"])
