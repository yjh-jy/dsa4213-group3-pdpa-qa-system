#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_retriever.py — Evaluate Hybrid retriever performance
Metrics: Recall@k, MRR, NDCG@k, Latency
Usage:
    python evaluate_retriever.py
"""

import json, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from hybrid_retriever import load_bm25_index, load_dense_index, HybridRetriever

# --- Paths ---
ROOT = Path(__file__).resolve().parents[0]      # repo root if script is at top-level
DATA = ROOT / "data"
QA_PATH = DATA / "qa" / "pdpa_qa_500.jsonl"
OUTDIR = DATA / "hybrid" / "pdpa_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10  # default top-k

# --- Evaluation metric functions ---
def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return 1.0 if any(r in retrieved_k for r in relevant) else 0.0

def reciprocal_rank(relevant, retrieved):
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        rel = 1.0 if cid in relevant else 0.0
        dcg += rel / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

# --- Load QA dataset ---
def load_qa_dataset(path):
    qa_items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                gold_chunks = [link["chunk_id"] for link in obj.get("corpus_links", [])]
                qa_items.append({
                    "id": obj["id"],
                    "question": obj["question_user"],
                    "gold_chunks": gold_chunks
                })
    return qa_items

# --- Evaluation function ---
def main():
    print("Loading BM25 and Dense indexes...")
    bm25, bm25_chunk_ids, bm25_sections = load_bm25_index()
    dense_emb, dense_chunk_ids, dense_sections = load_dense_index()
    retriever = HybridRetriever(bm25, bm25_chunk_ids, bm25_sections,
                                dense_emb, dense_chunk_ids, dense_sections)

    qa_items = load_qa_dataset(QA_PATH)
    print(f"Loaded {len(qa_items)} QA items from {QA_PATH}.")

    recall_scores, mrr_scores, ndcg_scores, latencies = [], [], [], []

    for qa in tqdm(qa_items, desc="Evaluating"):
        q = qa["question"]
        gold = qa["gold_chunks"]

        start = time.time()
        results = retriever.hybrid_retrieve(q, top_k=TOP_K)
        latency = time.time() - start
        latencies.append(latency)

        retrieved_chunks = [r[0] for r in results]

        recall_scores.append(recall_at_k(gold, retrieved_chunks, TOP_K))
        mrr_scores.append(reciprocal_rank(gold, retrieved_chunks))
        ndcg_scores.append(ndcg_at_k(gold, retrieved_chunks, TOP_K))

    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
    print(f"MRR: {np.mean(mrr_scores):.4f}")
    print(f"NDCG@{TOP_K}: {np.mean(ndcg_scores):.4f}")
    print(f"Avg Latency: {np.mean(latencies):.4f} sec/query")

    # Save report
    outpath = OUTDIR / "hybrid_eval_report.json"
    report = {
        "Recall@k": np.mean(recall_scores),
        "MRR": np.mean(mrr_scores),
        "NDCG@k": np.mean(ndcg_scores),
        "AvgLatency": np.mean(latencies)
    }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {outpath}")

if __name__ == "__main__":
    main()
