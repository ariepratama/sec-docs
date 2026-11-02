import streamlit as st
import logging
from datasets import load_dataset
import pandas as pd
import bm25s

@st.cache_resource(show_spinner="Loading FinanceBench dataset...")
def get_financebench_dataset():
    return load_dataset("PatronusAI/financebench", split="train")

@st.cache_resource(show_spinner="Loading FinanceBench into index")
def transform_df(df: pd.DataFrame) -> tuple[list[set[int]], list[str], dict[tuple[str, str], int], dict[int, tuple[str, str]]]:
    evidences = df["evidence"]
    gold_indices = []
    doc_to_idx = {}
    id_to_doc_meta = {}
    corpus = []
    for evidence_list in evidences:
        gold_set = set()
        for ev in evidence_list:
            text = ev.get("evidence_text_full_page") or ev["evidence_text"]
            key = (ev["doc_name"], ev["evidence_page_num"])
            
            if key not in doc_to_idx:
                doc_to_idx[key] = len(corpus)
                corpus.append(text)

            gold_set.add(doc_to_idx[key])
        gold_indices.append(gold_set)
    id_to_doc_meta = {v: k for k, v in doc_to_idx.items()}
    logging.info(f"loaded {len(gold_indices)} gold_indices and {len(corpus)} corpus")
    return gold_indices, corpus, doc_to_idx, id_to_doc_meta

def get_all_questions(df: pd.DataFrame) -> list[str]:
    questions = df["question"].to_list()
    return questions

def get_retriever(corpus: list[str]) -> bm25s.BM25:
    retriever = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever.index(corpus_tokens)
    return retriever


def search(retriever: bm25s.BM25, queries: list[str], k=10):
    query_tokens = bm25s.tokenize(
        queries,
        stopwords="en"
    )

    return retriever.retrieve(query_tokens, k=k)

def on_template_change():
    if st.session_state.selected_template:
        st.session_state.query_text = st.session_state.selected_template

def get_avg_metrics():
    pass

log = logging.getLogger("fe")
df = get_financebench_dataset().to_pandas()
gold_indices, corpus, doc_to_idx, id_to_doc_meta = transform_df(df)

retriever = get_retriever(corpus)
all_questions = get_all_questions(df)

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""


st.markdown(f"""
# FinchBench with BM25 retrieval 

This page is pre-loaded with the FinchBench dataset, which contains {len(corpus)} documents from financial statements, with {len(all_questions)} queries.

You may load the dataset yourself with the following code snippet (using huggingface datasets library)
```
from datasets import load_dataset
load_dataset("PatronusAI/financebench", split="train")
```

Below is a simple simulation of corpus retrieval using [BM25 library](https://github.com/xhluca/bm25s).


""")
                                        
template_questions = st.selectbox(
    "select from a question in finchbase dataset", 
    options=[""] + all_questions,
    key="selected_template",
    on_change=on_template_change)
query = st.text_input(
    "query",
    placeholder="choose a template question to prefill or type your own question",
    key="query_text"
)

if st.button("search"):
    with st.spinner("searching docs"):
        retrieved_indices, scores = search(retriever, [query])

    for doc_id, score in zip(retrieved_indices[0], scores[0]):
        with st.expander(
            f"doc_id={doc_id} | doc_name={id_to_doc_meta[doc_id][0]} | doc_page={id_to_doc_meta[doc_id][1]} | score={score}"
        ):
            st.write(corpus[doc_id])
