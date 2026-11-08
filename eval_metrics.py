import pandas as pd


def recall_k(retrieved: list, relevant: list, k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    retrieved_set = set(retrieved[:k])
    intersection_set = retrieved_set.intersection(relevant_set)
    return float(len(intersection_set)) / len(relevant_set)


def rr_k(retrieved: list, relevant: list, k: int) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_set = set(retrieved[:k])
    for re in relevant:
        if re in retrieved_set:
            return 1.0
    return 0.0


def calculate_metrics(
    retrieved_indices: list,
    gold_indices: list,
    k_values: list[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    results = []
    for k in k_values:
        for retrieved, relevant in zip(retrieved_indices, gold_indices):
            recall_score = recall_k(retrieved, relevant, k)
            rr_score = rr_k(retrieved, relevant, k)
            hit_score = hit_k(retrieved, relevant, k)
            results.append([recall_score, rr_score, hit_score, k])
    return pd.DataFrame(results, columns=["recall", "rr", "hit", "k"])


def get_avg_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        metrics_df.groupby("k")[["recall", "rr", "hit"]]
        .mean()
        .reset_index()
        .rename(columns={"k": "k", "recall": "Recall@k", "rr": "MRR@k", "hit": "Hit Rate@k"})
    )
    return summary_df

