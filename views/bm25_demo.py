import logging
import streamlit as st
import pandas as pd
import bm25s
from eval_metrics import calculate_metrics, get_avg_metrics

try:
    from views.financebench import get_all_questions, get_financebench_dataset, transform_df
except ImportError:
    from financebench import get_all_questions, get_financebench_dataset, transform_df

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

log = logging.getLogger("fe")
df = get_financebench_dataset().to_pandas()
gold_indices, corpus, doc_to_idx, id_to_doc_meta = transform_df(df)

retriever = get_retriever(corpus)
all_questions = get_all_questions(df)

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""


st.markdown(f"""
# FinanceBench with BM25 retrieval 

This page is pre-loaded with the FinanceBench dataset, which contains {len(corpus)} documents from financial statements, with {len(all_questions)} queries.

You may load the dataset yourself with the following code snippet (using huggingface datasets library)
```python
from datasets import load_dataset
load_dataset("PatronusAI/financebench", split="train")
```
""")

with st.expander("Understanding TF-IDF (Foundations & Limitations)", expanded=False):
    st.markdown(r"""
Traditional lexical search evaluates relevance using **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**:

### 1. Term Frequency (TF)
Measures how frequently query term $t$ appears in document $D$:
$$
\text{TF}(t, D) = \frac{f(t, D)}{|D|} \quad \text{or raw count} \quad f(t, D)
$$

### 2. Inverse Document Frequency (IDF)
Penalizes ubiquitous stop-words (like *"the"*, *"is"*) and boosts informative, discriminative terms that appear in few documents:
$$
\text{IDF}(t) = \ln\left(\frac{N}{n(t)}\right)
$$
where:
- $N$: Total number of documents in corpus $\mathcal{C}$.
- $n(t)$: Number of documents containing query term $t$.

### 3. Standard TF-IDF Score
$$
\text{Score}_{\text{TF-IDF}}(D, Q) = \sum_{t \in Q} \text{TF}(t, D) \times \text{IDF}(t)
$$

### 4. Limitations of Pure TF-IDF (Why BM25 is needed)
1. **Unbounded Term Frequency (No Saturation)**: In raw TF-IDF, a document mentioning a term 20 times receives 20× the TF weight of one mentioning it once. In reality, term relevance exhibits **diminishing marginal returns**.
2. **Length Bias**: Longer documents naturally contain higher word counts, unfairly skewing TF-IDF scores even when they are verbose and irrelevant.
""")

with st.expander("Okapi BM25: Mathematical Formulation & Scoring", expanded=False):
    st.markdown(r"""
**BM25 (Best Matching 25)** improves upon TF-IDF by introducing **term frequency saturation** and **document length normalization**.

### 1. The Okapi BM25 Formula
Given a query $Q = \{q_1, q_2, \dots, q_m\}$ and document $D$:
$$
\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^m \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

---

### 2. Component Breakdown

#### A. Probabilistic Inverse Document Frequency (IDF)
Calculated with Robertson-Spärck Jones probabilistic weighting:
$$
\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)
$$
- $N$: Total number of documents in the corpus.
- $n(q_i)$: Number of documents containing query term $q_i$.
- $+0.5$ and $+1$: Laplace smoothing terms that prevent division by zero and negative IDF scores for terms present in over half the corpus.

#### B. Saturated Term Frequency & $k_1$ Parameter
The fraction $\frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + \dots}$ controls how fast term frequency score saturates:
- As $f(q_i, D) \to \infty$, the fraction asymptotically approaches $k_1 + 1$.
- **$k_1$ (typically $1.2 \le k_1 \le 2.0$, default $1.5$)**: Determines saturation speed.
  - Lower $k_1$: Rapid saturation; repeated terms contribute very little additional score.
  - Higher $k_1$: Slower saturation; closer to linear term frequency.

#### C. Document Length Normalization & $b$ Parameter
The denominator scales by document length relative to average corpus length:
$$
B = 1 - b + b \cdot \frac{|D|}{\text{avgdl}}
$$
- $|D|$: Length of document $D$ (number of tokens/words).
- $\text{avgdl} = \frac{1}{N}\sum_{d \in \mathcal{C}} |d|$: Average document length across the corpus.
- **$b$ (typically $0.75$)**: Controls the strength of length normalization penalty ($0 \le b \le 1$).
  - $b = 1.0$: Full length normalization (heavily penalizes verbose documents).
  - $b = 0.0$: No length normalization (ignores document size).
  - When $|D| = \text{avgdl}$, $B = 1$ regardless of $b$.
""")

with st.expander("Evaluation Metrics (Recall@k, MRR@k, Hit Rate@k)", expanded=False):
    st.markdown(r"""
In this benchmark, we evaluate retrieval performance using standard Information Retrieval (IR) metrics:

### 1. Recall@k
$$ 
    \text{Recall}@k = \frac{|\text{Retrieved}_{1:k} \cap \text{Relevant}|}{|\text{Relevant}|}
$$
Measures the proportion of all relevant evidence documents retrieved within the top-$k$ results. It is effective for verifying that the retriever doesn't leave critical context behind when answers require multiple pages.

### 2. Reciprocal Rank (RR@k) & MRR@k

$$
    \text{RR}@k =  
        \begin{cases} 
            \frac{1}{\text{rank}_i} & \text{if } \text{rank}_i \le k \\
            0 & \text{otherwise}
        \end{cases}
$$

$$
    \text{MRR}@k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{RR}_i@k
$$
Measures how high up the *first* relevant document appears, awarding a higher score when it lands near the top. It is effective because LLMs and users heavily prioritize the highest-ranked items and suffer when correct context is buried.

### 3. Hit Rate / Success@k
$$
    \text{Hit}@k = 
        \begin{cases} 
            1 & \text{if } |\text{Retrieved}_{1:k} \cap \text{Relevant}| > 0 \\
            0 & \text{otherwise}
        \end{cases}
$$
Measures the percentage of queries that retrieved at least one relevant document in the top-$k$. It is effective as an intuitive baseline to confirm whether the retriever provides any usable answer context at all.
""")

st.subheader("Benchmark Evaluation")
if st.button("Calculate average metrics (k = 1, 3, 5, 10)"):
    with st.spinner("Calculating BM25 retrieval metrics across FinanceBench questions..."):
        k_values = [1, 3, 5, 10]
        retrieved_indices, _ = search(retriever, all_questions, k=max(k_values))
        metrics_df = calculate_metrics(retrieved_indices, gold_indices, k_values=k_values)
        st.session_state["benchmark_summary"] = get_avg_metrics(metrics_df)

if "benchmark_summary" in st.session_state:
    st.dataframe(
        st.session_state["benchmark_summary"].style.format({
            "Recall@k": "{:.4f}",
            "MRR@k": "{:.4f}",
            "Hit Rate@k": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

st.markdown("---")
st.subheader("Interactive BM25 Search Demo")
st.markdown("Below is a simple simulation of corpus retrieval using [BM25 library](https://github.com/xhluca/bm25s).")
                                        
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
