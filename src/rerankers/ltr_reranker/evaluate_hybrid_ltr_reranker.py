#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_hybrid_ltr_reranker.py — Evaluate a trained LTR LambdaRank reranker on hybrid retriever results
Computes Precision@k, NDCG@k, HitRate@k, MRR@k, and latency.
- Reads:
    artefacts/ltr_reranker/model/lgbm_model_ltr_reranker.txt
    src/retrievers/stratified_results/eval/evaluation_results.json
    data/ltr_processed/test_ltr_data.jsonl
- Writes:
    src/rerankers/ltr_reranker/results/hybrid_results_ltr_reranker.json
"""

import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb


# --- Paths ---
ROOT = Path(__file__).resolve().parents[3]      # repo root 
MODEL_PATH = ROOT / "artefacts" / "ltr_reranker" / "model" / "lgbm_model_ltr_reranker.txt"
RETRIEVER_EVAL_PATH = ROOT / "src" / "retrievers" / "stratified_results" / "eval" / "evaluation_results.json"
RETRIEVER_KEY = "hybrid_optimized"
PROCESSED_LTR_PATH = ROOT / "data" / "ltr_processed" / "test_ltr_data.jsonl"
RESULTS_DIR = ROOT / "src" / "rerankers" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 10  # number of top docs to evaluate

# --- Load data ---
def load_ltr_dataset(path: Path):
    df = pd.read_json(path, lines=True)
    print(f"Loaded {len(df)} samples from {path}")
    return df

# --- Evaluation metrics ---
def precision_at_k(labels, k):
    labels = np.asarray(labels)[:k]
    return np.mean(labels)

def hit_rate_at_k(labels, k):
    return 1.0 if np.any(np.asarray(labels)[:k]) else 0.0

def ndcg_at_k(labels, k):
    labels = np.asarray(labels)[:k]
    if not np.any(labels):
        return 0.0
    dcg = np.sum(labels / np.log2(np.arange(2, len(labels) + 2)))
    ideal = np.sum(sorted(labels, reverse=True) / np.log2(np.arange(2, len(labels) + 2)))
    return dcg / ideal if ideal > 0 else 0.0

def mrr_at_k(labels, k):
    labels = np.asarray(labels)[:k]
    for i, rel in enumerate(labels, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0

# --- Load features ---
def build_features_map(path: Path, model_feature_names):
    """
    Build dict {chunk_id: numeric_features_dict} using only the features model expects.
    """
    df = pd.read_json(path, lines=True)
    features_map = {}
    for _, row in df.iterrows():
        feat = {f: row[f] for f in model_feature_names if f in row}
        features_map[row["chunk_id"]] = feat
    print(f"Loaded features for {len(features_map)} chunks")
    return features_map

# --- Main evaluation ---
def evaluate_ltr_from_retriever(model, retriever_results, features_map):
    print("\nEvaluating LTR reranker on hybrid retriever output...")
    results = []
    total_times = []
    metric_store = defaultdict(list)

    model_feature_names = model.feature_name()

    for i, item in enumerate(retriever_results, 1):
        query = item["query"]
        qa_id = item["qa_id"]
        candidate_chunks = item["retrieved_chunks"]
        relevant_chunks = set(item["relevant_chunks"])

        # Build feature matrix with only model features
        feature_rows = []
        chunk_ids = []
        for cid in candidate_chunks:
            feat = features_map.get(cid)
            if feat:
                # ensure same order of columns as model
                feature_rows.append([feat[f] for f in model_feature_names])
                chunk_ids.append(cid)
        if not feature_rows:
            continue
        X = pd.DataFrame(feature_rows, columns=model_feature_names)

        # Predict
        start = time.time()
        preds = model.predict(X)
        latency_ms = (time.time() - start) * 1000
        total_times.append(latency_ms)

        # Rank
        ranked_idx = np.argsort(preds)[::-1]
        ranked_ids = [chunk_ids[idx] for idx in ranked_idx]

        # Metrics
        labels = [1 if cid in relevant_chunks else 0 for cid in ranked_ids]
        metrics = {}
        for k in [1, 3, 5, 10]:
            metrics[f"precision@{k}"] = precision_at_k(labels, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(labels, k)
            metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
        metrics["latency_ms"] = latency_ms

        for m, v in metrics.items():
            metric_store[m].append(v)

        results.append({
            "qa_id": qa_id,
            "query": query,
            "relevant_chunks": list(relevant_chunks),
            "reranked_chunks": ranked_ids,
            "metrics": metrics
        })

    # Aggregate summary
    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} 
               for m, v in metric_store.items()}

    print("\n=== LTR Reranker Evaluation Summary (Hybrid) ===")
    for m in sorted(summary.keys()):
        print(f"{m:<15s}: {summary[m]['mean']:.4f}")
    print(f"Latency (ms/query): {np.mean(total_times):.2f}")

    return {"summary": summary, "results": results}

if __name__ == "__main__":
    with open(RETRIEVER_EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    retriever_results = data[RETRIEVER_KEY]["individual_results"]

    model = lgb.Booster(model_file=str(MODEL_PATH))
    model_feature_names = model.feature_name()

    features_map = build_features_map(PROCESSED_LTR_PATH, model_feature_names)

    eval_output = evaluate_ltr_from_retriever(model, retriever_results, features_map)

    out_path = RESULTS_DIR / f"hybrid_results_ltr_reranker.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\nStructured reranked results saved to {out_path}")
