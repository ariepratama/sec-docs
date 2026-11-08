import logging
import pandas as pd
import streamlit as st
from datasets import load_dataset


@st.cache_resource(show_spinner="Loading FinanceBench dataset...")
def get_financebench_dataset():
    return load_dataset("PatronusAI/financebench", split="train")


@st.cache_resource(show_spinner="Loading FinanceBench into index")
def transform_df(
    _df: pd.DataFrame,
) -> tuple[
    list[set[int]],
    list[str],
    dict[tuple[str, str], int],
    dict[int, tuple[str, str]],
]:
    evidences = _df["evidence"]
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

