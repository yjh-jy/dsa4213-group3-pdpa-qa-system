#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_all_retrievers.py — Unified evaluation of all retrieval systems
- Evaluates BM25, Dense, and Hybrid retrievers against PDPA QA dataset
- Computes comprehensive metrics for comparison
- Generates single unified report
"""

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

# Add retriever paths to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "bm25_retrieval"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "dense_retrieval"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "hybrid_retrieval"))

# Import retrievers with error handling
try:
    from bm25_retriever import BM25Retriever
    BM25_AVAILABLE = True
except ImportError as e:
    print(f"BM25 Retriever not available: {e}")
    BM25_AVAILABLE = False

try:
    from dense_retriever import DenseRetriever
    DENSE_AVAILABLE = True
except ImportError as e:
    print(f"Dense Retriever not available: {e}")
    DENSE_AVAILABLE = False

try:
    from hybrid_retriever import create_hybrid_retriever
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"Hybrid Retriever not available: {e}")
    HYBRID_AVAILABLE = False

class UnifiedRetrievalEvaluator:
    def __init__(self, qa_dataset_path: Path):
        """Initialize evaluator with QA dataset."""
        self.qa_dataset_path = qa_dataset_path
        self.qa_data = self._load_qa_dataset()
        
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
    
    def _extract_relevant_chunks(self, qa_item: Dict) -> Set[str]:
        """Extract ground truth relevant chunk IDs from QA item."""
        relevant_chunks = set()
        for link in qa_item.get("corpus_links", []):
            if "chunk_id" in link:
                relevant_chunks.add(link["chunk_id"])
        return relevant_chunks
    
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
        
        # Actual DCG
        actual_relevances = [1 if chunk in relevant_chunks else 0 for chunk in retrieved_chunks[:k]]
        actual_dcg = dcg(actual_relevances)
        
        # Ideal DCG (perfect ranking)
        ideal_relevances = [1] * min(len(relevant_chunks), k) + [0] * max(0, k - len(relevant_chunks))
        ideal_dcg = dcg(ideal_relevances)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def evaluate_retriever(self, retriever, retriever_name: str, top_k: int = 10) -> Dict:
        """Evaluate a single retriever."""
        print(f"\nEvaluating {retriever_name}...")
        
        all_results = []
        for i, qa_item in enumerate(self.qa_data):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.qa_data)} queries...")
            
            query = qa_item["question_user"]
            relevant_chunks = self._extract_relevant_chunks(qa_item)
            
            # Perform retrieval
            start_time = time.time()
            if hasattr(retriever, 'search'):
                search_result = retriever.search(query, top_k=top_k)
                if isinstance(search_result, dict) and "results" in search_result:
                    retrieved_chunks = [hit["chunk_id"] for hit in search_result["results"]]
                else:
                    retrieved_chunks = [hit["chunk_id"] for hit in search_result]
            else:
                # Fallback for different retriever interfaces
                retrieved_chunks = retriever.retrieve(query, top_k=top_k)
            
            search_time = time.time() - start_time
            
            # Compute metrics
            metrics = {}
            for k in [1, 3, 5, 10]:
                if k <= top_k:
                    metrics[f"recall@{k}"] = self.compute_recall_at_k(retrieved_chunks, relevant_chunks, k)
                    metrics[f"precision@{k}"] = self.compute_precision_at_k(retrieved_chunks, relevant_chunks, k)
                    metrics[f"ndcg@{k}"] = self.compute_ndcg_at_k(retrieved_chunks, relevant_chunks, k)
            
            metrics["mrr"] = self.compute_mrr(retrieved_chunks, relevant_chunks)
            metrics["search_time_ms"] = search_time * 1000
            
            all_results.append({
                "qa_id": qa_item["id"],
                "query": query,
                "relevant_chunks": list(relevant_chunks),
                "retrieved_chunks": retrieved_chunks,
                "metrics": metrics
            })
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_results)
        
        return {
            "retriever_name": retriever_name,
            "evaluation_summary": aggregated_metrics,
            "individual_results": all_results,
            "dataset_info": {
                "total_queries": len(self.qa_data),
                "top_k": top_k,
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all queries."""
        metrics_by_name = defaultdict(list)
        
        # Collect all metric values
        for result in results:
            for metric_name, value in result["metrics"].items():
                metrics_by_name[metric_name].append(value)
        
        # Compute statistics
        aggregated = {}
        for metric_name, values in metrics_by_name.items():
            aggregated[metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return aggregated
    
    def print_comparison_summary(self, all_results: Dict):
        """Print a comparison summary of all retrievers."""
        print("\n" + "="*80)
        print("RETRIEVAL SYSTEMS COMPARISON SUMMARY")
        print("="*80)
        
        # Key metrics to compare
        key_metrics = ["recall@1", "recall@5", "recall@10", "precision@1", "precision@5", 
                      "mrr", "ndcg@5", "ndcg@10", "search_time_ms"]
        
        # Print header
        print(f"{'Metric':<15s}", end="")
        for retriever_name in all_results.keys():
            print(f"{retriever_name:>20s}", end="")
        print()
        print("-" * (15 + 20 * len(all_results)))
        
        # Print metrics comparison
        for metric in key_metrics:
            print(f"{metric:<15s}", end="")
            for retriever_name, results in all_results.items():
                if metric in results["evaluation_summary"]:
                    mean_val = results["evaluation_summary"][metric]["mean"]
                    print(f"{mean_val:>20.4f}", end="")
                else:
                    print(f"{'N/A':>20s}", end="")
            print()
        
        print("\n" + "="*80)
    
    def save_performance_summary(self, all_results: Dict):
        """Save only the performance summary report."""
        # Create simplified report with just mean values
        simplified_results = {}
        for retriever_name, results in all_results.items():
            simplified_results[retriever_name] = {}
            for metric_name, stats in results["evaluation_summary"].items():
                simplified_results[retriever_name][metric_name] = round(stats["mean"], 4)
        
        # Save only the simple summary file
        results_dir = Path(__file__).resolve().parents[0]
        summary_path = results_dir / f"performance_summary_{time.strftime('%Y%m%d')}.json"
        summary_report = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(self.qa_data),
            "performance_metrics": simplified_results
        }
        
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"Performance summary saved to: {summary_path}")

def main():
    """Main evaluation script."""
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    qa_dataset_path = project_root / "data" / "qa" / "pdpa_qa_500.jsonl"
    results_dir = Path(__file__).resolve().parents[0]
    
    # Initialize evaluator
    print("Initializing unified retrieval evaluator...")
    evaluator = UnifiedRetrievalEvaluator(qa_dataset_path)
    
    all_results = {}
    
    # Evaluate Dense Retriever
    if DENSE_AVAILABLE:
        try:
            print("Initializing Dense Retriever...")
            dense_retriever = DenseRetriever()
            dense_results = evaluator.evaluate_retriever(dense_retriever, "Dense", top_k=10)
            all_results["Dense"] = dense_results
        except Exception as e:
            print(f"Error evaluating Dense Retriever: {e}")
    
    # Evaluate BM25 Retriever (if available)
    if BM25_AVAILABLE:
        try:
            print("Initializing BM25 Retriever...")
            bm25_retriever = BM25Retriever()
            bm25_results = evaluator.evaluate_retriever(bm25_retriever, "BM25", top_k=10)
            all_results["BM25"] = bm25_results
        except Exception as e:
            print(f"Error evaluating BM25 Retriever: {e}")
    
    # Evaluate Hybrid Retriever (if available)
    if HYBRID_AVAILABLE:
        try:
            print("Initializing Hybrid Retriever...")
            hybrid_retriever = create_hybrid_retriever()
            hybrid_results = evaluator.evaluate_retriever(hybrid_retriever, "Hybrid", top_k=10)
            all_results["Hybrid"] = hybrid_results
        except Exception as e:
            print(f"Error evaluating Hybrid Retriever: {e}")
    
    # Print comparison summary
    if all_results:
        evaluator.print_comparison_summary(all_results)
        
        # Save performance summary only
        evaluator.save_performance_summary(all_results)
    else:
        print("No retrievers were successfully evaluated.")

if __name__ == "__main__":
    main()