#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_cross_reranker_from_retriever_eval.py —
Rerank results from a previously evaluated retriever (e.g. hybrid_optimized)
and output results in the same structured JSON schema used for retriever evaluation.
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------
# PATHS & IMPORTS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cross_encoder_reranker import CrossEncoderReranker

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RETRIEVER_EVAL_PATH = ROOT / "src" / "retrievers" / "stratified_results" / "eval" / "evaluation_results.json"
BM25_SECTION_MAP = ROOT / "artefacts" / "bm25_index" / "sections.map.json"
CROSS_MODEL_DIR = Path(__file__).resolve().parents[3] / "artefacts" / "cross_encoder" / "model"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVER_KEY = "hybrid_optimized"   # select retriever block
RERANKER_NAME = "CrossEncoder_Reranker"
TOP_K_RERANKER = 10

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def precision_at_k(relevant, retrieved, k):
    retrieved_at_k = retrieved[:k]
    hits = len(set(retrieved_at_k) & relevant)
    return hits / k if k > 0 else 0.0

def hit_rate_at_k(relevant, retrieved, k):
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0

def ndcg_at_k(relevant, retrieved, k):
    def dcg(rels): return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
    actual = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    ideal = sorted(actual, reverse=True)
    return dcg(actual) / dcg(ideal) if dcg(ideal) > 0 else 0.0

def mean_reciprocal_rank_at_k(relevant, retrieved, k):
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

# ---------------------------------------------------------------------
# MAIN RERANKING LOGIC
# ---------------------------------------------------------------------
def evaluate_from_retriever_results(reranker, retriever_results, section_map):
    print(f"\n Evaluating reranker using retriever eval outputs...")
    results, total_times = [], []
    metric_store = defaultdict(list)

    for i, item in enumerate(retriever_results, 1):
        query = item.get("query")
        qa_id = item.get("qa_id")
        retrieved_chunks = item.get("retrieved_chunks", [])
        relevant_chunks = set(item.get("relevant_chunks", []))

        if not query or not retrieved_chunks:
            continue

        # Build candidate list with full text
        candidates = []
        for cid in retrieved_chunks:
            text = section_map.get(cid, {}).get("text", "")
            candidates.append({"chunk_id": cid, "text": text, "score": 0.0})
        if not candidates:
            continue

        # Rerank
        start = time.time()
        reranked = reranker.rerank(query, candidates, top_k=TOP_K_RERANKER)
        latency_ms = (time.time() - start) * 1000
        total_times.append(latency_ms)

        ranked_ids = [c["chunk_id"] for c in reranked]

        metrics = {}
        for k in [1, 3, 5, 10]:
            metrics[f"precision@{k}"] = precision_at_k(relevant_chunks, ranked_ids, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(relevant_chunks, ranked_ids, k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(relevant_chunks, ranked_ids, k)
            metrics[f"mrr@{k}"] = mean_reciprocal_rank_at_k(relevant_chunks, ranked_ids, k)
        metrics["latency_ms"] = latency_ms

        # store for aggregation
        for m, v in metrics.items():
            metric_store[m].append(v)

        results.append({
            "qa_id": qa_id,
            "query": query,
            "relevant_chunks": list(relevant_chunks),
            "reranked_chunks": ranked_ids,
            "metrics": metrics
        })

        if i % 20 == 0:
            print(f"  Processed {i}/{len(retriever_results)} queries...")

    # --- aggregate with mean, median, std, min, max, count ---
    summary = {}
    for m, vals in metric_store.items():
        if not vals:
            continue
        summary[m] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": len(vals)
        }

    print("\n=== Cross-Encoder Reranker Evaluation (from retriever eval) ===")
    for m in ["precision@1", "precision@3", "precision@5", "precision@10",
              "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
              "hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", 
              "mrr@1", "mrr@3", "mrr@5", "mrr@10"]:
        if m in summary:
            print(f"{m:<15s}: {summary[m]['mean']:.4f}")
    print(f"Latency (ms/query): {np.mean(total_times):.2f}")

    return {
        RERANKER_NAME: {
            "reranker_name": RERANKER_NAME,
            "evaluation_summary": summary,
            "individual_results": results
        }
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print(f" Loading retriever evaluation from {RETRIEVER_EVAL_PATH}")
    with open(RETRIEVER_EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    retriever_results = data[RETRIEVER_KEY]["individual_results"]

    print(f" Loading section mapping from {BM25_SECTION_MAP}")
    with open(BM25_SECTION_MAP, "r", encoding="utf-8") as f:
        section_map = json.load(f)

    reranker = CrossEncoderReranker(model_name_or_path=str(CROSS_MODEL_DIR))
    eval_output = evaluate_from_retriever_results(reranker, retriever_results, section_map)

    out_path = RESULTS_DIR / f"reranked_{RETRIEVER_KEY}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\n Structured reranked results saved to {out_path}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
