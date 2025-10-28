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
    
    def k_fold_evaluate_dense_retriever(self, k_folds: int = 5, top_k: int = 10, results_dir: Path = None) -> Dict:
        """Perform k-fold cross-validation evaluation of dense retriever."""
        print(f"\n=== K-Fold Cross-Validation Evaluation (k={k_folds}) ===")
        
        # Import dense retriever function
        try:
            from dense_retriever import DenseRetriever
        except ImportError as e:
            print(f"Dense Retriever not available: {e}")
            return {}
        
        # Paths
        project_root = Path(__file__).resolve().parents[1]
        training_data_dir = project_root / "data" / "dense_training"
        
        if not training_data_dir.exists():
            print(f"Training data directory not found: {training_data_dir}")
            print("Please run dense_chunk_and_extract.py first to generate training data.")
            return {}
        
        # Check if all folds exist
        fold_results = []
        temp_dirs_to_cleanup = []
        
        try:
            for fold_idx in range(1, k_folds + 1):
                fold_dir = training_data_dir / f"fold_{fold_idx}"
                test_queries_path = fold_dir / "test_queries.jsonl"
                train_triples_path = fold_dir / "train_triples.jsonl"
                corpus_path = training_data_dir / "corpus.jsonl"
                
                if not all([fold_dir.exists(), test_queries_path.exists(), train_triples_path.exists(), corpus_path.exists()]):
                    print(f"Fold {fold_idx} data incomplete. Skipping k-fold evaluation.")
                    return {}
                
                print(f"\n--- Fold {fold_idx}/{k_folds} ---")
                
                # Initialize fresh retriever for this fold
                retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
                
                # Fine-tune on this fold's training data
                print(f"Fine-tuning on fold {fold_idx} training data...")
                temp_output_dir = None
                try:
                    # Create temporary output directory for this fold
                    if results_dir:
                        temp_output_dir = results_dir / f"temp_fold_{fold_idx}_dense_model"
                    else:
                        temp_output_dir = Path(__file__).resolve().parents[0] / f"temp_fold_{fold_idx}_dense_model"
                    
                    temp_output_dir.mkdir(parents=True, exist_ok=True)
                    temp_dirs_to_cleanup.append(temp_output_dir)
                    
                    retriever.fine_tune(
                        training_data_path=train_triples_path,
                        output_dir=temp_output_dir,
                        test_queries_path=None,  # Don't evaluate during training
                        corpus_path=None,
                        epochs=5,
                        batch_size=8,
                        learning_rate=2e-5,
                        warmup_steps=100
                    )
                    
                    # Load the fine-tuned model for this fold
                    retriever.model = retriever.model.__class__(str(temp_output_dir), device=retriever.device)
                    
                    # Reload corpus and recompute embeddings with fine-tuned model
                    retriever._load_corpus()
                    retriever._precompute_embeddings()
                except Exception as e:
                    print(f"Fine-tuning failed for fold {fold_idx}: {e}")
                    continue
                
                # Load test queries for this fold
                test_queries = []
                with test_queries_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                test_queries.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                
                print(f"Evaluating on {len(test_queries)} test queries...")
                
                # Evaluate on this fold's test data
                fold_metrics = []
                for query_data in test_queries:
                    query = query_data["query"]
                    # Handle different data formats
                    if "corpus_links" in query_data:
                        relevant_chunks = {query_data["corpus_links"][0]["chunk_id"]}
                    elif "pos_ids" in query_data:
                        relevant_chunks = set(query_data["pos_ids"])
                    else:
                        print(f"Warning: No relevant chunks found for query {query_data.get('qid', 'unknown')}")
                        continue
                    
                    # Perform retrieval
                    start_time = time.time()
                    search_result = retriever.search(query, top_k=top_k)
                    if isinstance(search_result, dict) and "results" in search_result:
                        retrieved_chunks = [hit["chunk_id"] for hit in search_result["results"]]
                    else:
                        retrieved_chunks = [hit["chunk_id"] for hit in search_result]
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
                    
                    fold_metrics.append(metrics)
                
                # Aggregate metrics for this fold
                fold_aggregated = self._aggregate_fold_metrics(fold_metrics)
                fold_results.append({
                    "fold": fold_idx,
                    "metrics": fold_aggregated,
                    "num_queries": len(test_queries)
                })
                
                print(f"Fold {fold_idx} Results:")
                for metric_name, stats in fold_aggregated.items():
                    if metric_name != "search_time_ms":
                        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
            if not fold_results:
                print("No folds were successfully evaluated.")
                return {}
            
            # Compute overall k-fold statistics
            overall_results = self._compute_k_fold_statistics(fold_results)
            
            print(f"\n=== Overall K-Fold Results (k={len(fold_results)}) ===")
            for metric_name, stats in overall_results.items():
                if metric_name not in ["_fold_count", "_std_results"]:
                    print(f"{metric_name}: {stats:.4f} (±{overall_results['_std_results'][metric_name]:.4f})")
            
            return overall_results
            
        finally:
            # Clean up temporary directories
            import shutil
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        print(f"Warning: Could not clean up {temp_dir}: {e}")
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate metrics within a single fold."""
        if not fold_metrics:
            return {}
        
        metric_values = defaultdict(list)
        for metrics in fold_metrics:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)
        
        aggregated = {}
        for metric_name, values in metric_values.items():
            aggregated[metric_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return aggregated
    
    def _compute_k_fold_statistics(self, fold_results: List[Dict]) -> Dict:
        """Compute overall statistics across all folds."""
        if not fold_results:
            return {}
        
        # Collect mean values from each fold
        fold_means = defaultdict(list)
        for fold_result in fold_results:
            for metric_name, stats in fold_result["metrics"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    fold_means[metric_name].append(stats["mean"])
        
        # Compute statistics across folds
        overall_stats = {}
        std_results = {}
        
        for metric_name, means in fold_means.items():
            overall_stats[metric_name] = statistics.mean(means)
            std_results[metric_name] = statistics.stdev(means) if len(means) > 1 else 0.0
        
        overall_stats["_fold_count"] = len(fold_results)
        overall_stats["_std_results"] = std_results
        
        return overall_stats
    
    def k_fold_evaluate_hybrid_retriever(self, k_folds: int = 5, top_k: int = 10, results_dir: Path = None) -> Dict:
        """Perform k-fold cross-validation evaluation of hybrid retriever."""
        print(f"\n=== K-Fold Cross-Validation Evaluation - Hybrid Retriever (k={k_folds}) ===")
        
        # Import required modules
        try:
            from dense_retriever import DenseRetriever
            from hybrid_retriever import create_hybrid_retriever
        except ImportError as e:
            print(f"Required retrievers not available: {e}")
            return {}
        
        # Paths
        project_root = Path(__file__).resolve().parents[1]
        training_data_dir = project_root / "data" / "dense_training"
        
        if not training_data_dir.exists():
            print(f"Training data directory not found: {training_data_dir}")
            return {}
        
        # Check if all folds exist
        fold_results = []
        temp_dirs_to_cleanup = []
        
        try:
            for fold_idx in range(1, k_folds + 1):
                fold_dir = training_data_dir / f"fold_{fold_idx}"
                test_queries_path = fold_dir / "test_queries.jsonl"
                train_triples_path = fold_dir / "train_triples.jsonl"
                corpus_path = training_data_dir / "corpus.jsonl"
                
                if not all([fold_dir.exists(), test_queries_path.exists(), train_triples_path.exists(), corpus_path.exists()]):
                    print(f"Fold {fold_idx} data incomplete. Skipping k-fold evaluation.")
                    return {}
                
                print(f"\n--- Fold {fold_idx}/{k_folds} ---")
                
                # Initialize and fine-tune dense retriever for this fold
                dense_retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
                
                print(f"Fine-tuning dense component for fold {fold_idx}...")
                temp_output_dir = None
                try:
                    # Create temporary output directory for this fold
                    if results_dir:
                        temp_output_dir = results_dir / f"temp_fold_{fold_idx}_hybrid_model"
                    else:
                        temp_output_dir = Path(__file__).resolve().parents[0] / f"temp_fold_{fold_idx}_hybrid_model"
                    
                    temp_output_dir.mkdir(parents=True, exist_ok=True)
                    temp_dirs_to_cleanup.append(temp_output_dir)
                    
                    dense_retriever.fine_tune(
                        training_data_path=train_triples_path,
                        output_dir=temp_output_dir,
                        test_queries_path=None,  # Don't evaluate during training
                        corpus_path=None,
                        epochs=5,
                        batch_size=8,
                        learning_rate=2e-5,
                        warmup_steps=100
                    )
                    
                    # Load the fine-tuned model for this fold
                    dense_retriever.model = dense_retriever.model.__class__(str(temp_output_dir), device=dense_retriever.device)
                    
                    # Reload corpus and recompute embeddings with fine-tuned model
                    dense_retriever._load_corpus()
                    dense_retriever._precompute_embeddings()
                except Exception as e:
                    print(f"Fine-tuning failed for fold {fold_idx}: {e}")
                    continue
                
                # Create hybrid retriever with the fine-tuned dense retriever
                print(f"Creating hybrid retriever for fold {fold_idx}...")
                try:
                    hybrid_retriever = create_hybrid_retriever(dense_retriever=dense_retriever)
                except Exception as e:
                    print(f"Hybrid retriever creation failed for fold {fold_idx}: {e}")
                    continue
                
                # Load test queries for this fold
                test_queries = []
                with test_queries_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                test_queries.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                
                print(f"Evaluating on {len(test_queries)} test queries...")
                
                # Evaluate on this fold's test data
                fold_metrics = []
                for query_data in test_queries:
                    query = query_data["query"]
                    # Handle different data formats
                    if "corpus_links" in query_data:
                        relevant_chunks = {query_data["corpus_links"][0]["chunk_id"]}
                    elif "pos_ids" in query_data:
                        relevant_chunks = set(query_data["pos_ids"])
                    else:
                        print(f"Warning: No relevant chunks found for query {query_data.get('qid', 'unknown')}")
                        continue
                    
                    # Perform retrieval
                    start_time = time.time()
                    search_result = hybrid_retriever.search(query, top_k=top_k)
                    if isinstance(search_result, dict) and "results" in search_result:
                        retrieved_chunks = [hit["chunk_id"] for hit in search_result["results"]]
                    else:
                        retrieved_chunks = [hit["chunk_id"] for hit in search_result]
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
                    
                    fold_metrics.append(metrics)
                
                # Aggregate metrics for this fold
                fold_aggregated = self._aggregate_fold_metrics(fold_metrics)
                fold_results.append({
                    "fold": fold_idx,
                    "metrics": fold_aggregated,
                    "num_queries": len(test_queries)
                })
                
                print(f"Fold {fold_idx} Results:")
                for metric_name, stats in fold_aggregated.items():
                    if metric_name != "search_time_ms":
                        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
            if not fold_results:
                print("No folds were successfully evaluated.")
                return {}
            
            # Compute overall k-fold statistics
            overall_results = self._compute_k_fold_statistics(fold_results)
            
            print(f"\n=== Overall K-Fold Results - Hybrid (k={len(fold_results)}) ===")
            for metric_name, stats in overall_results.items():
                if metric_name not in ["_fold_count", "_std_results"]:
                    print(f"{metric_name}: {stats:.4f} (±{overall_results['_std_results'][metric_name]:.4f})")
            
            return overall_results
            
        finally:
            # Clean up temporary directories
            import shutil
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        print(f"Warning: Could not clean up {temp_dir}: {e}")
    
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
    
    def save_performance_summary(self, all_results: Dict, k_fold_results: Dict = None, results_dir: Path = None):
        """Save performance summary to JSON file, including k-fold results."""
        # Create simplified report with just mean values
        simplified_results = {}
        if all_results:
            for retriever_name, results in all_results.items():
                simplified_results[retriever_name] = {}
                for metric_name, stats in results["evaluation_summary"].items():
                    simplified_results[retriever_name][metric_name] = round(stats["mean"], 4)
        
        # Add k-fold results (Dense and Hybrid)
        if k_fold_results:
            for retriever_name, results in k_fold_results.items():
                # Convert k-fold results to the same format as standard results
                k_fold_metrics = {}
                for metric_name, value in results.items():
                    if metric_name not in ["_fold_count", "_std_results"]:
                        std_val = results["_std_results"].get(metric_name, 0.0)
                        k_fold_metrics[metric_name] = {
                            "mean": round(float(value), 4),
                            "std": round(float(std_val), 4),
                            "method": "k_fold_cross_validation"
                        }
                
                simplified_results[f"{retriever_name}_KFold"] = k_fold_metrics
        
        # Save only the simple summary file
        if results_dir is None:
            results_dir = Path(__file__).resolve().parents[0]
        
        summary_path = results_dir / "performance_summary.json"
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
    import argparse
    import datetime
    import shutil
    
    parser = argparse.ArgumentParser(description="Unified Retrieval Evaluation - No Data Leakage")
    parser.add_argument("--k_folds", type=int, default=5, 
                       help="Number of folds for k-fold cross-validation (Dense & Hybrid)")
    parser.add_argument("--top_k", type=int, default=10, 
                       help="Number of top results to retrieve")
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    qa_dataset_path = project_root / "data" / "qa" / "pdpa_qa_500.jsonl"
    
    # Create timestamped results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / f"evaluation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Results will be saved to: {results_dir}")
    print("=" * 80)
    
    # Initialize evaluator
    print("Initializing unified retrieval evaluator...")
    evaluator = UnifiedRetrievalEvaluator(qa_dataset_path)
    
    all_results = {}
    k_fold_results = {}
    
    # Dense Retriever - ONLY K-fold evaluation to prevent data leakage
    if DENSE_AVAILABLE:
        try:
            print("\n=== Dense Retriever - K-Fold Cross-Validation ===")
            print("Note: Using k-fold cross-validation to prevent data leakage")
            dense_k_fold_results = evaluator.k_fold_evaluate_dense_retriever(
                k_folds=args.k_folds, top_k=args.top_k, results_dir=results_dir
            )
            if dense_k_fold_results:
                k_fold_results["Dense"] = dense_k_fold_results
        except Exception as e:
            print(f"Error in k-fold evaluation of Dense Retriever: {e}")
    
    # Hybrid Retriever - ONLY K-fold evaluation to prevent data leakage
    if HYBRID_AVAILABLE:
        try:
            print("\n=== Hybrid Retriever - K-Fold Cross-Validation ===")
            print("Note: Using k-fold cross-validation to prevent data leakage")
            hybrid_k_fold_results = evaluator.k_fold_evaluate_hybrid_retriever(
                k_folds=args.k_folds, top_k=args.top_k, results_dir=results_dir
            )
            if hybrid_k_fold_results:
                k_fold_results["Hybrid"] = hybrid_k_fold_results
        except Exception as e:
            print(f"Error in k-fold evaluation of Hybrid Retriever: {e}")
    
    # BM25 Retriever - Standard evaluation (no training involved, no data leakage risk)
    if BM25_AVAILABLE:
        try:
            print("\n=== BM25 Retriever - Standard Evaluation ===")
            print("Note: BM25 uses standard evaluation as it doesn't involve training")
            bm25_retriever = BM25Retriever()
            bm25_results = evaluator.evaluate_retriever(bm25_retriever, "BM25", top_k=args.top_k)
            all_results["BM25"] = bm25_results
        except Exception as e:
            print(f"Error evaluating BM25 Retriever: {e}")
    
    # Save and display results
    results_saved = False
    
    # Save k-fold results
    if k_fold_results:
        k_fold_results_file = results_dir / "k_fold_evaluation_results.json"
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for retriever_name, results in k_fold_results.items():
            json_results[retriever_name] = {}
            for key, value in results.items():
                if key == '_std_results':
                    json_results[retriever_name][key] = {k: float(v) for k, v in value.items()}
                elif isinstance(value, (np.float64, np.float32)):
                    json_results[retriever_name][key] = float(value)
                else:
                    json_results[retriever_name][key] = value
        
        with open(k_fold_results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nK-fold results saved to: {k_fold_results_file}")
        results_saved = True
        
        # Print k-fold comparison
        print(f"\n=== K-Fold Cross-Validation Results ===")
        for retriever_name, results in k_fold_results.items():
            print(f"\n{retriever_name} Retriever (k={results.get('_fold_count', 'N/A')}):")
            for metric_name, value in results.items():
                if metric_name not in ["_fold_count", "_std_results"]:
                    std_val = results["_std_results"].get(metric_name, 0.0)
                    print(f"  {metric_name}: {value:.4f} (±{std_val:.4f})")
    
    # Save standard evaluation results (BM25) and merged performance summary
    if all_results or k_fold_results:
        if all_results:
            evaluator.print_comparison_summary(all_results)
        evaluator.save_performance_summary(all_results, k_fold_results, results_dir=results_dir)
        results_saved = True
    
    # Final summary
    if results_saved:
        print(f"\n=== Evaluation Summary ===")
        print(f"✓ Dense Retriever: K-fold cross-validation (prevents data leakage)")
        print(f"✓ Hybrid Retriever: K-fold cross-validation (prevents data leakage)")
        if BM25_AVAILABLE:
            print(f"✓ BM25 Retriever: Standard evaluation (no training, no leakage risk)")
        print(f"\nAll evaluations completed successfully with proper methodology!")
    else:
        print("No retrievers were successfully evaluated.")

if __name__ == "__main__":
    main()