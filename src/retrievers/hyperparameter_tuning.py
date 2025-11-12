#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperparameter_tuning.py — Tuning for Hybrid and BM25 retrievers
- Reads validation split from data/dense_training/stratified_splits/val_triples.jsonl
- Uses existing fine-tuned dense retriever model for hybrid tuning
- Saves results to src/retrievers/stratified_results/tuning/hyperparameter_tuning.json
"""
import json
from pathlib import Path
import sys

def run_hybrid_tuning(val_triples_path: Path, dense_model_path: Path) -> dict:
    # Ensure local imports work
    sys.path.append(str(Path(__file__).resolve().parent / "hybrid_retrieval"))
    sys.path.append(str(Path(__file__).resolve().parent / "dense_retrieval"))
    from hybrid_retriever import HybridHyperparameterOptimizer, create_hybrid_retriever
    from dense_retriever import DenseRetriever

    print(f"Initializing dense retriever from: {dense_model_path}")
    dense_retriever = DenseRetriever(model_name=str(dense_model_path))
    optimizer = HybridHyperparameterOptimizer(val_triples_path)

    tuning_results = []
    # Linear alpha grid
    alpha_results = optimizer.optimize_alpha(
        alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        sample_size=100,
        dense_retriever=dense_retriever
    )
    tuning_results.extend(alpha_results)
    # RRF compare
    rrf_results = optimizer.compare_fusion_methods(
        sample_size=100,
        dense_retriever=dense_retriever
    )
    tuning_results.extend(rrf_results)

    best_config = max(tuning_results, key=lambda x: x.get('composite_score', 0))
    print(f"Hybrid best config: {best_config}")
    return {"best_config": best_config, "all_results": tuning_results}


def run_bm25_tuning(val_triples_path: Path) -> dict:
    sys.path.append(str(Path(__file__).resolve().parent / "bm25_retrieval"))
    from bm25_retriever import BM25HyperparameterOptimizer

    print(f"Running BM25 tuning on: {val_triples_path}")
    bm25_optimizer = BM25HyperparameterOptimizer(val_triples_path)
    k1_values = [0.5, 1.0, 1.2, 1.5, 2.0]
    b_values = [0.0, 0.3, 0.5, 0.75, 1.0]
    bm25_results = bm25_optimizer.grid_search(k1_values=k1_values, b_values=b_values, sample_size=100)
    best_bm25 = bm25_results[0] if bm25_results else {"k1": 0.5, "b": 1.0, "composite_score": 0.0}
    print(f"BM25 best params: {best_bm25}")
    return {"bm25_best_params": {"k1": best_bm25["k1"], "b": best_bm25["b"]}, "bm25_all_results": bm25_results}


def main():
    root = Path(__file__).resolve().parents[2]
    val_triples_path = root / "data" / "dense_training" / "stratified_splits" / "val_triples.jsonl"
    dense_model_path = root / "artefacts" / "dense_retriever" / "model"
    if not val_triples_path.exists():
        print(f"Validation triples not found: {val_triples_path}")
        return
    if not dense_model_path.exists():
        print(f"Dense model not found: {dense_model_path}")
        return

    hybrid_out = run_hybrid_tuning(val_triples_path, dense_model_path)
    bm25_out = run_bm25_tuning(val_triples_path)

    out_dir = Path(__file__).resolve().parent / "stratified_results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hyperparameter_tuning.json"
    payload = {
        **hybrid_out,
        **bm25_out,
        "tuning_timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Tuning results saved to: {out_path}")

if __name__ == "__main__":
    main()