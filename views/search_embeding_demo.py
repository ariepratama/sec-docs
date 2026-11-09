import logging
import os
from pathlib import Path
import pandas as pd
import pymilvus
import streamlit as st

from eval_metrics import calculate_metrics, get_avg_metrics
from rag_core import CostTracker, embed_text

try:
    from views.financebench import get_all_questions, get_financebench_dataset, transform_df
except ImportError:
    from financebench import get_all_questions, get_financebench_dataset, transform_df

log = logging.getLogger("fe")


def retrieve_docs(
    mclient: pymilvus.MilvusClient,
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int = 10,
) -> list[dict]:
    """Retrieve top chunks matching the query vector."""
    if not query_vectors:
        return []

    results = mclient.search(
        collection_name=collection_name,
        data=query_vectors[:1],
        limit=limit,
        output_fields=[
            "id",
            "doc_id",
            "chunk_id",
            "text",
            "doc_name",
            "page_num",
            "vector",
        ],
    )

    docs = []
    if results and results[0]:
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append({
                "id": hit.get("id"),
                "doc_id": entity.get("doc_id"),
                "chunk_id": entity.get("chunk_id"),
                "text": entity.get("text"),
                "doc_name": entity.get("doc_name"),
                "page_num": entity.get("page_num"),
                "score": hit.get("distance"),
                "vector": entity.get("vector"),
            })
    return docs


def search_questions(
    mclient: pymilvus.MilvusClient,
    collection_name: str,
    questions: list[str],
    k: int = 10,
    progress_bar=None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Search for multiple questions and return deduplicated document indices & scores."""
    all_retrieved_doc_ids = []
    all_scores = []

    for i, q in enumerate(questions):
        query_vectors = embed_text(q)
        # Fetch extra chunks to ensure we obtain k unique documents after deduplication
        docs = retrieve_docs(mclient, collection_name, query_vectors, limit=k * 4)

        seen = set()
        doc_ids = []
        scores = []
        for d in docs:
            doc_id = d["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
                scores.append(d["score"])
                if len(doc_ids) >= k:
                    break

        all_retrieved_doc_ids.append(doc_ids)
        all_scores.append(scores)

        if progress_bar is not None:
            progress_bar.progress((i + 1) / len(questions))

    return all_retrieved_doc_ids, all_scores


def on_template_change():
    if st.session_state.get("selected_template_embed"):
        st.session_state["query_text_embed"] = st.session_state["selected_template_embed"]


# 1. Load dataset & corpus
df = get_financebench_dataset().to_pandas()
gold_indices, corpus, doc_to_idx, id_to_doc_meta = transform_df(df)
all_questions = get_all_questions(df)

HF_FINANCEBENCH_REPO = os.environ.get(
    "HF_FINANCEBENCH_REPO", "ariesutiono/financebench-vector-db"
)
HF_METRICS_REPO = os.environ.get(
    "HF_METRICS_REPO", "ariesutiono/financebench-vector-retrieval-metrics"
)


def ensure_financebench_milvus_db(path: str | Path) -> bool:
    """Download the FinanceBench Milvus Lite DB from Hugging Face if not on disk.

    Returns True if DB exists or was successfully downloaded, False otherwise.
    """
    path = Path(path)
    collections_dir = path / "collections"
    if collections_dir.is_dir() and any(collections_dir.iterdir()):
        return True

    try:
        from huggingface_hub import snapshot_download

        log.info(
            "FinanceBench Milvus DB not found at %s; downloading from HF dataset %s",
            path,
            HF_FINANCEBENCH_REPO,
        )
        with st.spinner(
            f"Downloading FinanceBench Milvus database from Hugging Face ({HF_FINANCEBENCH_REPO})..."
        ):
            snapshot_download(
                repo_id=HF_FINANCEBENCH_REPO,
                repo_type="dataset",
                allow_patterns=["collections/**"],
                local_dir=str(path),
                token=os.environ.get("HF_TOKEN")
                or os.environ.get("SEC_RAG_HF_TOKEN")
                or None,
            )
        log.info("Downloaded FinanceBench Milvus DB to %s", path)
        st.info(f"Downloaded FinanceBench Milvus DB from Hugging Face ({HF_FINANCEBENCH_REPO})")
        return collections_dir.is_dir() and any(collections_dir.iterdir())
    except Exception as e:
        log.warning(
            "Could not download FinanceBench Milvus DB from HF dataset %s: %s",
            HF_FINANCEBENCH_REPO,
            e,
        )
        st.warning(
            f"Could not download Milvus DB from Hugging Face ({HF_FINANCEBENCH_REPO}): {e}. Will build index locally if needed."
        )
        return False


@st.cache_resource
def get_financebench_milvus_client(path: str) -> pymilvus.MilvusClient:
    ensure_financebench_milvus_db(path)
    return pymilvus.MilvusClient(path)


# 2. Initialize local Milvus Lite
db_path = Path("vector/financebench.db")
collection_name = "financebench"
mclient = get_financebench_milvus_client(str(db_path))

# 3. Create collection and index if missing or empty
needs_indexing = False
if not mclient.has_collection(collection_name):
    needs_indexing = True
else:
    stats = mclient.get_collection_stats(collection_name)
    if stats.get("row_count", 0) == 0:
        mclient.drop_collection(collection_name)
        needs_indexing = True

if needs_indexing:
    st.info(f"Begin indexing dataset into {db_path}...")
    sample_vec = embed_text("test")[0]
    dimension = len(sample_vec)
    mclient.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        metric_type="COSINE",
        auto_id=False,
        id_type="string",
        max_length=64,
    )

    with st.spinner("Embedding corpus..."):
        rows = []
        for id, text in enumerate(corpus):
            doc_name, page_num = id_to_doc_meta.get(id, ("unknown", 0))
            vectors = embed_text(text)
            for chunk_id, vector in enumerate(vectors):
                rows.append({
                    "id": f"{id}_{chunk_id}",
                    "doc_id": id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "vector": vector,
                })

    with st.spinner("Upserting corpus to Milvus..."):
        if rows:
            mclient.upsert(collection_name=collection_name, data=rows)
    st.success(f"Indexed {len(rows)} chunks into '{collection_name}'!")

if "query_text_embed" not in st.session_state:
    st.session_state["query_text_embed"] = ""

# 4. Markdown introduction & evaluation metrics
st.markdown(f"""
# FinanceBench with Dense Vector / Embedding Retrieval

This page is pre-loaded with the FinanceBench dataset, which contains {len(corpus)} documents from financial statements, with {len(all_questions)} queries.
Indexed using local Milvus Lite with dense vector embeddings.
""")

st.markdown(r"""

Instead of relying on document statistics, like TF-IDF, which is in the heart of how BM25 algorithm works, now with neural network,
we can rely on vector embeddings from a language model. In this page, I use `BAAI/bge-large-en-v1.5` embedding model from `deepinfra.com`, which will results in `1024` embedding dimension. 

Additionally, I used local `Milvus` db to index the embedding vector from the financebench corpus. The rows are structured as follow
```yaml
row:
    id: 
        type: str
        description: "Unique chunk identifier formatted as '{doc_id}_{chunk_id}'"
    doc_id:
        type: int
        description: "Zero-based index referencing the parent document in the corpus"
    chunk_id:
        type: int
        description: "Zero-based index of the chunk within the document"
    doc_name:
        type: str
        description: "Source filing or financial statement document name"
    page_num:
        type: int
        description: "Page number from the original SEC document"
    text:
        type: str
        description: "Text passage content of the document/chunk"
    vector:
        type: list[float]
        description: "1024-dimensional dense embedding vector from BAAI/bge-large-en-v1.5"
```
""")

with st.expander("Why dense embedding retrieval outperforms BM25?", expanded=False):
    st.markdown(r"""
* **Semantic understanding vs exact keyword overlap**  
  BM25 requires exact lexical match. If a user asks *"How much money did Apple make in 2022?"* and the 10-K filing says *"Apple generated record net sales and operating income"*, BM25 scores low because the keywords *"money"* and *"make"* never appear in the document.

* **Resolving vocabulary mismatch problem**  
  It is hard for BM25 to "understand" the concept of synonyms or relations between words. For example, documents that describe *"borrowings"* or *"credit facilities"* might not contain the exact word *"debt"*, causing BM25 to miss them entirely. Dense embeddings project semantically related words into nearby regions of the continuous vector space, allowing the retriever to match concepts rather than exact surface tokens.

* **Context-aware disambiguation (Polysemy)**  
  BM25 treats identical words identically regardless of context. In financial filings, words like *"interest"* (*"interest expense on debt"* vs *"controlling interest in subsidiary"*) or *"share"* (*"common stock share"* vs *"market share"*) carry completely different meanings. Transformer encoders use self-attention to generate contextualized vectors based on surrounding words.

* **Handling complex, natural language queries**  
  When users ask conversational questions (e.g., *"What factors contributed to the decline in gross margin?"*), BM25 gives weight to filler words like *"factors"*, *"contributed"*, or *"decline"*. Dense embeddings encode the overall intent and semantic direction of the question rather than decomposing it into isolated word frequencies.

* **Conceptual and implicit matching**  
  Financial questions often ask about high-level themes (e.g., *"liquidity risk"* or *"regulatory headwinds"*). A filing passage describing *"drawdown on revolving credit lines"* or *"antitrust inquiries"* answers the question directly without ever explicitly containing the exact query phrase. Dense vectors naturally cluster these conceptually related discussions together.
""")

with st.expander("Mathematical formulation of dense retrieval & cosine similarity", expanded=False):
    st.markdown(r"""
### 1. Dense Vector Encoding
A neural transformer encoder $E_\theta(\cdot)$ (e.g., `BAAI/bge-large-en-v1.5`) maps an arbitrary-length query string $q$ and document passage $d \in \mathcal{D}$ into dense vector representations $\mathbf{q}, \mathbf{d} \in \mathbb{R}^D$ where $D = 1024$:

$$
\mathbf{q} = E_\theta(q), \quad \mathbf{d} = E_\theta(d)
$$

The vectors are normalized to unit length ($L_2$ norm):

$$
\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2} = \frac{\mathbf{q}}{\sqrt{\sum_{i=1}^D q_i^2}}, \quad \hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2} = \frac{\mathbf{d}}{\sqrt{\sum_{i=1}^D d_i^2}}
$$

---

### 2. Cosine Similarity Scoring
The relevance score measures the geometric cosine of the angle $\theta$ between the query and document vectors:

$$
\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^D q_i d_i}{\sqrt{\sum_{i=1}^D q_i^2} \sqrt{\sum_{i=1}^D d_i^2}}
$$

For unit-normalized vectors ($\|\hat{\mathbf{q}}\|_2 = \|\hat{\mathbf{d}}\|_2 = 1$), this reduces directly to the dot (inner) product:

$$
\text{Score}_{\text{dense}}(q, d) = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \sum_{i=1}^D \hat{q}_i \hat{d}_i \quad \in [-1, 1]
$$

Higher scores (closer to $1.0$) indicate strong semantic alignment.

---

### 3. Top-$k$ Nearest Neighbor Search (MIPS)
Given a query $q$ and a Milvus corpus $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$, retrieval solves a **Maximum Inner Product Search (MIPS)**:

$$
\text{Top-}k(q, \mathcal{D}) = \operatorname*{arg\,top\text{-}k}_{d \in \mathcal{D}} \; \langle \hat{\mathbf{q}}, \hat{\mathbf{d}} \rangle
$$

Milvus optimizes this nearest-neighbor search to return the top-$k$ ranked chunks with the highest cosine scores.
""")

st.markdown(r"""
Here, I will use the same evaluation metrics as BM25 (see BM25 page for more details).



""")

# 5. Benchmark Evaluation
st.subheader("Benchmark Evaluation")
if st.button("Calculate average metrics (k = 1, 3, 5, 10)"):
    metrics_df_local_path = Path("metrics_df.csv")
    metrics_df = None

    # 1. Try to load from Hugging Face first
    try:
        from huggingface_hub import hf_hub_download

        with st.spinner(f"Downloading metrics from Hugging Face ({HF_METRICS_REPO})..."):
            hf_metrics_path = hf_hub_download(
                repo_id=HF_METRICS_REPO,
                filename="metrics_df.csv",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
                or os.environ.get("SEC_RAG_HF_TOKEN")
                or None,
            )
            metrics_df = pd.read_csv(hf_metrics_path)
            st.info(f"Loaded metrics from Hugging Face ({HF_METRICS_REPO})")
            metrics_df.to_csv(metrics_df_local_path, index=False)
    except Exception as e:
        log.warning("Could not download metrics from Hugging Face (%s): %s", HF_METRICS_REPO, e)

    # 2. Check local file if Hugging Face failed / not found
    if metrics_df is None and metrics_df_local_path.is_file():
        st.info("Loaded metrics from local cache (metrics_df.csv)")
        metrics_df = pd.read_csv(metrics_df_local_path)

    # 3. Only if both fail / not found, recompute the metrics
    if metrics_df is None:
        st.info("Metrics not found on Hugging Face or locally. Computing metrics...")
        with st.spinner("Calculating Dense Vector retrieval metrics across FinanceBench questions..."):
            k_values = [1, 3, 5, 10]
            progress_bar = st.progress(0.0)
            retrieved_indices, _ = search_questions(
                mclient,
                collection_name,
                all_questions,
                k=max(k_values),
                progress_bar=progress_bar,
            )
            metrics_df = calculate_metrics(retrieved_indices, gold_indices, k_values=k_values)
            metrics_df.to_csv(metrics_df_local_path, index=False)

    st.session_state["embed_benchmark_summary"] = get_avg_metrics(metrics_df)

if "embed_benchmark_summary" in st.session_state:
    st.dataframe(
        st.session_state["embed_benchmark_summary"].style.format({
            "Recall@k": "{:.4f}",
            "MRR@k": "{:.4f}",
            "Hit Rate@k": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

# 6. Interactive Search Demo
st.markdown("---")
st.subheader("Interactive Dense Vector Search Demo")
st.markdown("Below is a simulation of corpus retrieval using dense embeddings and Milvus Lite vector search.")

template_questions = st.selectbox(
    "Select from a question in FinanceBench dataset",
    options=[""] + all_questions,
    key="selected_template_embed",
    on_change=on_template_change,
)

query = st.text_input(
    "Query",
    placeholder="Choose a template question to prefill or type your own question",
    key="query_text_embed",
)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query or select a template question.")
    else:
        with st.spinner("Embedding query and searching Milvus..."):
            query_vectors = embed_text(query)
            docs = retrieve_docs(mclient, collection_name, query_vectors, limit=10)

        if not docs:
            st.info("No matching documents found.")
        else:
            for i, doc in enumerate(docs, 1):
                doc_id = doc["doc_id"]
                doc_meta = id_to_doc_meta.get(doc_id, (doc["doc_name"], doc["page_num"]))
                with st.expander(
                    f"[Rank {i}] doc_id={doc_id} | chunk={doc['chunk_id']} | "
                    f"doc_name={doc_meta[0]} | doc_page={doc_meta[1]} | "
                    f"score={doc['score']:.4f}"
                ):
                    st.write(doc["text"])
