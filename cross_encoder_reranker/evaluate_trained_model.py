#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_cross_reranker.py — evaluate a fine-tuned Cross-Encoder reranker
Computes Precision@k, NDCG@k, HitRate@k, and latency.
"""

import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from cross_encoder_reranker import CrossEncoderReranker

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "data" / "dense_training" / "stratified_splits" / "test_triples.jsonl"
CROSS_MODEL_DIR = Path(__file__).resolve().parent / "cross_encoder_model"
RESULTS_DIR = CROSS_MODEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_qa_dataset(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f" Loaded {len(data)} QA samples from {path}")
    return data

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

# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
def evaluate_cross_encoder(model, qa_data):
    print(f"\n Evaluating reranker model...")
    results, total_times = [], []

    for i, item in enumerate(qa_data, 1):
        query = item.get("query") or item.get("question_user")
        if not query or not item.get("pos_text"):
            continue

        candidates = [item["pos_text"]]
        rel_map = {"pos": 1}
        if "neg_text" in item:
            candidates.append(item["neg_text"])
            rel_map["neg"] = 0

        pairs = [(query, doc) for doc in candidates]

        start = time.time()
        scores = model.predict(pairs, batch_size=2, convert_to_numpy=True)
        latency_ms = (time.time() - start) * 1000
        total_times.append(latency_ms)

        sorted_idx = np.argsort(scores)[::-1]
        ranked_docs = [list(rel_map.keys())[idx] for idx in sorted_idx]

        relevant = {k for k, v in rel_map.items() if v == 1}
        metrics = {f"precision@{k}": precision_at_k(relevant, ranked_docs, k)
                   for k in [1, 3, 5, 10]}
        metrics.update({f"hit_rate@{k}": hit_rate_at_k(relevant, ranked_docs, k)
                        for k in [1, 3, 5, 10]})
        metrics.update({f"ndcg@{k}": ndcg_at_k(relevant, ranked_docs, k)
                        for k in [1, 3, 5, 10]})
        metrics["latency_ms"] = latency_ms

        results.append({"qa_id": item.get("qid") or f"sample_{i}",
                        "query": query, "metrics": metrics})

        if i % 50 == 0:
            print(f"  Processed {i}/{len(qa_data)} queries...")

    # Aggregate
    agg = defaultdict(list)
    for r in results:
        for m, v in r["metrics"].items():
            agg[m].append(v)

    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for m, v in agg.items()}
    print("\n=== Cross-Encoder Evaluation Summary ===")
    for m in ["precision@1", "precision@3", "precision@5", "precision@10",
              "ndcg@5", "ndcg@10", "hit_rate@5", "hit_rate@10"]:
        if m in summary:
            print(f"{m:<15s}: {summary[m]['mean']:.4f}")
    print(f"Latency (ms/query): {np.mean(total_times):.2f}")

    return {"summary": summary, "results": results}

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    qa_data = load_qa_dataset(QA_PATH)
    reranker = CrossEncoderReranker(model_name_or_path=str(CROSS_MODEL_DIR))
    model = reranker.model
    eval_output = evaluate_cross_encoder(model, qa_data)

    out_path = RESULTS_DIR / "cross_encoder_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\n Evaluation results saved to {out_path}")

if __name__ == "__main__":
    main()
