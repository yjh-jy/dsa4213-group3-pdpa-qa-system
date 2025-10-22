#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_bm25_retrieval.py — Comprehensive evaluation of BM25 retrieval system
- Evaluates against PDPA QA dataset (pdpa_qa_500.jsonl)
- Computes Recall, Precision, MRR, NDCG, and latency metrics
- Generates detailed performance analysis
"""

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from bm25_retriever import BM25Retriever


# ----------------------------------------------------------------------
# BM25 Evaluation Class
# ----------------------------------------------------------------------
class BM25RetrievalEvaluator:
    """Evaluate BM25 retriever on a QA dataset."""

    def __init__(self, qa_dataset_path: Path, retriever: BM25Retriever):
        self.qa_dataset_path = qa_dataset_path
        self.retriever = retriever
        self.qa_data = self._load_qa_dataset()

    # ------------------------------------------------------------------
    def _load_qa_dataset(self) -> List[Dict]:
        """Load QA dataset from JSONL file."""
        if not self.qa_dataset_path.exists():
            raise FileNotFoundError(f"QA dataset not found at {self.qa_dataset_path}")

        qa_data = []
        with self.qa_dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
                        continue

        if not qa_data:
            raise ValueError("No valid QA pairs found in dataset")

        print(f"Loaded {len(qa_data)} QA pairs from {self.qa_dataset_path}")
        return qa_data

    # ------------------------------------------------------------------
    def _extract_relevant_chunks(self, qa_item: Dict) -> Set[str]:
        """Extract ground truth relevant chunk IDs from QA item."""
        relevant_chunks = set()
        for link in qa_item.get("corpus_links", []):
            if "chunk_id" in link:
                relevant_chunks.add(link["chunk_id"])
        return relevant_chunks

    # ------------------------------------------------------------------
    def compute_recall_at_k(self, retrieved_chunks: List[str], relevant_chunks: Set[str], k: int) -> float:
        """Compute Recall@k metric."""
        if not relevant_chunks:
            return 0.0
        retrieved_at_k = set(retrieved_chunks[:k])
        hits = len(retrieved_at_k.intersection(relevant_chunks))
        return hits / len(relevant_chunks)

    def compute_precision_at_k(self, retrieved_chunks: List[str], relevant_chunks: Set[str], k: int) -> float:
        """Compute Precision@k metric."""
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved_chunks[:k])
        hits = len(retrieved_at_k.intersection(relevant_chunks))
        return hits / min(k, len(retrieved_chunks))

    def compute_mrr(self, retrieved_chunks: List[str], relevant_chunks: Set[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        for i, chunk_id in enumerate(retrieved_chunks, 1):
            if chunk_id in relevant_chunks:
                return 1.0 / i
        return 0.0

    def compute_ndcg_at_k(self, retrieved_chunks: List[str], relevant_chunks: Set[str], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain (NDCG@k)."""
        def dcg(relevances: List[int]) -> float:
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

        actual_relevances = [1 if chunk in relevant_chunks else 0 for chunk in retrieved_chunks[:k]]
        actual_dcg = dcg(actual_relevances)
        ideal_relevances = [1] * min(len(relevant_chunks), k)
        ideal_dcg = dcg(ideal_relevances)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # ------------------------------------------------------------------
    def evaluate_single_query(self, qa_item: Dict, top_k: int = 10) -> Dict:
        """Evaluate retrieval performance for a single query."""
        query = qa_item["question_user"]
        relevant_chunks = self._extract_relevant_chunks(qa_item)

        # Run retrieval
        start_time = time.time()
        search_result = self.retriever.search(query, top_k=top_k)
        search_time = time.time() - start_time

        retrieved_chunks = [hit["chunk_id"] for hit in search_result["results"]]

        # Compute metrics
        metrics = {}
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                metrics[f"recall@{k}"] = self.compute_recall_at_k(retrieved_chunks, relevant_chunks, k)
                metrics[f"precision@{k}"] = self.compute_precision_at_k(retrieved_chunks, relevant_chunks, k)
                metrics[f"ndcg@{k}"] = self.compute_ndcg_at_k(retrieved_chunks, relevant_chunks, k)

        metrics["mrr"] = self.compute_mrr(retrieved_chunks, relevant_chunks)
        metrics["search_time_ms"] = search_time * 1000

        return {
            "qa_id": qa_item["id"],
            "query": query,
            "relevant_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "metrics": metrics,
            "search_result": search_result,
        }

    # ------------------------------------------------------------------
    def evaluate_dataset(self, top_k: int = 10, max_queries: int = None) -> Dict:
        """Evaluate retrieval performance on the entire dataset."""
        print(f"Starting BM25 evaluation with top_k={top_k}")
        queries_to_eval = self.qa_data[:max_queries] if max_queries else self.qa_data
        print(f"Evaluating {len(queries_to_eval)} queries...")

        all_results = []
        for i, qa_item in enumerate(queries_to_eval):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(queries_to_eval)} queries...")
            all_results.append(self.evaluate_single_query(qa_item, top_k))

        aggregated_metrics = self._aggregate_metrics(all_results)

        return {
            "evaluation_summary": aggregated_metrics,
            "individual_results": all_results,
            "dataset_info": {
                "total_queries": len(queries_to_eval),
                "top_k": top_k,
                "model": self.retriever.meta.get("version", "bm25_pdpa_v1"),
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    # ------------------------------------------------------------------
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all queries."""
        metrics_by_name = defaultdict(list)
        for result in results:
            for metric_name, value in result["metrics"].items():
                metrics_by_name[metric_name].append(value)

        aggregated = {}
        for metric_name, values in metrics_by_name.items():
            aggregated[metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return aggregated

    # ------------------------------------------------------------------
    def print_evaluation_summary(self, evaluation_result: Dict):
        """Print a formatted summary of evaluation results."""
        summary = evaluation_result["evaluation_summary"]
        dataset_info = evaluation_result["dataset_info"]

        print("\n" + "=" * 60)
        print("BM25 RETRIEVAL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Index: {dataset_info['model']}")
        print(f"Dataset: {dataset_info['total_queries']} queries")
        print(f"Top-K: {dataset_info['top_k']}")
        print(f"Evaluation Time: {dataset_info['evaluation_time']}")
        print()

        key_metrics = [
            "recall@1", "recall@5", "recall@10",
            "precision@1", "precision@5", "precision@10",
            "mrr", "ndcg@5", "ndcg@10", "search_time_ms"
        ]

        for metric in key_metrics:
            if metric in summary:
                stats = summary[metric]
                print(
                    f"{metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(median: {stats['median']:.4f}, range: {stats['min']:.4f}-{stats['max']:.4f})"
                )

        print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    def save_results(self, evaluation_result: Dict, output_path: Path):
        """Save evaluation results to JSON file."""
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
def main():
    """Main BM25 evaluation pipeline."""
    project_root = Path(__file__).resolve().parents[1]
    qa_dataset_path = project_root / "data" / "qa" / "pdpa_qa_500.jsonl"
    results_dir = Path(__file__).resolve().parents[0] / "results"
    results_dir.mkdir(exist_ok=True)

    print("Initializing BM25 retriever...")
    retriever = BM25Retriever()

    print("Loading QA dataset...")
    evaluator = BM25RetrievalEvaluator(qa_dataset_path, retriever)

    print("Running evaluation...")
    evaluation_result = evaluator.evaluate_dataset(top_k=10)

    evaluator.print_evaluation_summary(evaluation_result)

    date_str = time.strftime("%Y%m%d")
    output_path = results_dir / f"bm25_retrieval_eval_{date_str}.json"
    evaluator.save_results(evaluation_result, output_path)


if __name__ == "__main__":
    main()
