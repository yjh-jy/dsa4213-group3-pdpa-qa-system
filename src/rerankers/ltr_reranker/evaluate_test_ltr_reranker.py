#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_test_ltr_reranker.py — Evaluate a trained LightGBM LambdaRank reranker on test dataset
Computes Precision@k, NDCG@k, HitRate@k, MRR@k, and latency.
- Reads:
    ltr_reranker/model/lgbm_model_ltr_reranker.txt
    ltr_reranker/processed_data/test_ltr_data.jsonl
- Writes:
    ltr_reranker/model/results/test_results_ltr_reranker.json
"""

import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb


# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]      # repo root 
MODEL_DIR = ROOT / "ltr_reranker" / "model"
MODEL_PATH = MODEL_DIR / "lgbm_model_ltr_reranker.txt"
TEST_PATH = ROOT / "ltr_reranker" / "processed_data" / "test_ltr_data.jsonl"
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# --- Main evaluation ---
def evaluate_ltr_model(model, df):
    print(f"\nEvaluating LTR reranker...")
    total_times = []
    results = []

    # Use only numeric feature columns
    feature_cols = df.select_dtypes(include=["int", "float", "bool"]).columns.tolist()
    if "label" in feature_cols: feature_cols.remove("label")

    for qid, group in df.groupby("qid"):
        X = group[feature_cols]
        y = group["label"]

        start = time.time()
        preds = model.predict(X)
        latency_ms = (time.time() - start) * 1000
        total_times.append(latency_ms)

        ranked = group.assign(pred_score=preds).sort_values("pred_score", ascending=False)
        labels = (ranked["label"] == 2).astype(int).tolist()

        metrics = {}
        for k in [1, 3, 5, 10]:
            metrics[f"precision@{k}"] = precision_at_k(labels, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(labels, k)
            metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
        metrics["latency_ms"] = latency_ms

        results.append({"query_id": qid, "metrics": metrics})

    # Aggregate results
    agg = defaultdict(list)
    for r in results:
        for m, v in r["metrics"].items():
            agg[m].append(v)

    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in agg.items()}

    print("\n=== LTR Reranker Evaluation Summary (Test) ===")
    for m in [
        "precision@1", "precision@3", "precision@5", "precision@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
        "hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10",
        "mrr@1", "mrr@3", "mrr@5", "mrr@10",
    ]:
        if m in summary:
            print(f"{m:<15s}: {summary[m]['mean']:.4f}")
    print(f"Latency (ms/query): {np.mean(total_times):.2f}")

    return {"summary": summary, "results": results}

if __name__ == "__main__":
    df = load_ltr_dataset(TEST_PATH)
    model = lgb.Booster(model_file=str(MODEL_PATH))

    feature_names = model.feature_name()  
    missing_feats = [f for f in feature_names if f not in df.columns]
    extra_feats = [f for f in df.columns if f in feature_names]

    print(f"Using {len(feature_names)} features for prediction.")
    print(f"Missing in test data: {missing_feats}")
    print(f"Extra in test data: {[c for c in df.columns if c not in feature_names]}")

    df = df[feature_names + ["qid", "label"]] 

    eval_output = evaluate_ltr_model(model, df)

    out_path = RESULTS_DIR / "ltr_reranker_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {out_path}")
