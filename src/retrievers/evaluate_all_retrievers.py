#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_all_retrievers.py — Retriever Evaluation Pipeline:
- Final evaluation (on unseen test set) with dense retriever, BM25, and hybrid retriever
- Outputs consolidated results to stratified_results/consolidated_results.json
"""

import json
import time
import statistics
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add retriever paths to sys.path
IMMEDIATE_PARENT = Path(__file__).resolve().parent
sys.path.append(str(IMMEDIATE_PARENT / "dense_retrieval"))
sys.path.append(str(IMMEDIATE_PARENT / "hybrid_retrieval"))
sys.path.append(str(IMMEDIATE_PARENT / "retrieval_evaluation"))

# Import modules
from dense_retriever import DenseRetriever
from hybrid_retriever import create_hybrid_retriever, HybridHyperparameterOptimizer
from evaluate_all_retrievers_kfold import UnifiedRetrievalEvaluator

class PDPAPipeline:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.artifacts_dir = self.root / "artefacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Create artifact subdirectories
        (self.artifacts_dir / "dense_retriever" / "model").mkdir(exist_ok=True)
        (self.artifacts_dir / "dense_retriever" / "faiss").mkdir(exist_ok=True)

        # Create stratified results subdirectories
        (IMMEDIATE_PARENT / "stratified_results" / "tuning").mkdir(exist_ok=True)
        (IMMEDIATE_PARENT / "stratified_results" / "eval").mkdir(exist_ok=True)
        
        # Data paths
        self.training_data_dir = self.root / "data" / "dense_training"
        
        # Results storage
        self.results = {
            "dense_retriever": {},
            "hybrid_default": {},
            "hybrid_optimized": {}
        }
        
        # Ensure BM25 index exists
        self._ensure_bm25_index()
        
        # Validate required data exists
        self._validate_data()
    
    def _ensure_bm25_index(self):
        """Ensure BM25 index exists for hybrid retriever."""
        bm25_index_dir = self.root / "artefacts" / "bm25_index"
        bm25_index_file = bm25_index_dir / "bm25_index.npz"
        
        if not bm25_index_file.exists():
            print("BM25 index not found. Building BM25 index...")
            sys.path.append(str(self.root / "bm25_retrieval"))
            
            try:
                from bm25_indexer import main as build_bm25_index
                build_bm25_index()
                print("BM25 index built successfully.")
            except ImportError:
                print("Warning: Could not import BM25 indexer. Please run bm25_indexer.py manually.")
            except Exception as e:
                print(f"Warning: BM25 index building failed: {e}")
        else:
            print("BM25 index found.")
    
    def _validate_data(self):
        """Validate that required datasets exist."""
        required_files = [
            self.training_data_dir / "stratified_splits" / "train_triples.jsonl",
            self.root / "data" / "corpus" / "corpus_subsection_v1.jsonl",
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print("Missing required data files:")
            for f in missing_files:
                print(f"  - {f}")
            print("\nPlease run the following scripts to generate missing data:")
            print("  - dense_chunk_and_extract.py (for training data)")
            print("  - create_stratified_splits.py (for stratified splits)")
            raise FileNotFoundError("Required data files are missing")
        
        print("All required data files found.")
    
    def _cleanup_artifacts(self):
        """Clean up all artifact directories to ensure fresh runs."""
        print("=" * 60)
        print("CLEANUP: Preparing fresh artifact directories")
        print("=" * 60)
        
        cleanup_dirs = [
            ("Dense Retriever Model", self.artifacts_dir / "dense_retriever" / "model"),
            ("FAISS Index", self.artifacts_dir / "dense_retriever" / "faiss"),
            ("Tuning Results", IMMEDIATE_PARENT / "stratified_results" / "tuning"),
            ("Evaluation Results", IMMEDIATE_PARENT / "stratified_results" / "eval")
        ]
        
        cleanup_files = [
            ("Final Results", IMMEDIATE_PARENT / "stratified_results" / "consolidated_results.json")
        ]
        
        # Clean up directories
        for name, dir_path in cleanup_dirs:
            if dir_path.exists():
                print(f"Cleaning up {name}: {dir_path}")
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {name} directory ready")
        
        # Clean up files
        for name, file_path in cleanup_files:
            if file_path.exists():
                print(f"Removing previous {name}: {file_path}")
                file_path.unlink()
                print(f"✓ {name} cleaned")
        
        print("✓ All artifacts cleaned - ready for fresh pipeline run")
        
    def train_dense_retriever(self) -> str:
        """Train dense retriever for 8 epochs using stratified training data."""
        print("=" * 60)
        print("STEP 1: Dense Retriever Training")
        print("=" * 60)
        
        # Use stratified splits for training
        train_triples_path = self.training_data_dir / "stratified_splits" / "train_triples.jsonl"
        val_queries_path = self.training_data_dir / "stratified_splits" / "val_triples.jsonl"
        corpus_path = self.root / "data" / "corpus" / "corpus_subsection_v1.jsonl"
        
        if not all([train_triples_path.exists(), corpus_path.exists()]):
            raise FileNotFoundError("Training data not found. Run dense_chunk_and_extract.py first.")
        
        # Initialize dense retriever with base model for training
        dense_retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
        print(f"Dense retriever using device: {dense_retriever.device}")
        
        # Verify GPU usage
        if str(dense_retriever.device) != str(device):
            print(f"Warning: Device mismatch - Main: {device}, Dense: {dense_retriever.device}")
        else:
            print(f"✓ GPU acceleration confirmed for dense retriever training")
            
        # Confirm model type
        if hasattr(dense_retriever, 'is_fine_tuned'):
            model_type = "fine-tuned" if dense_retriever.is_fine_tuned else "base"
            print(f"✓ Using {model_type} model for training")
        
        # Set output directory
        model_output_dir = self.artifacts_dir  / "dense_retriever" / "model"
        
        print(f"Training dense retriever...")
        print(f"Training data: {train_triples_path}")
        print(f"Output directory: {model_output_dir}")
        
        # Fine-tune with memory-optimized parameters to prevent GPU OOM
        dense_retriever.fine_tune(
            training_data_path=train_triples_path,
            output_dir=model_output_dir,
            test_queries_path=None,  # Disable evaluation for maximum speed
            corpus_path=None,        # Disable evaluation for maximum speed
            epochs=50,               # Reduced epochs for faster training
            batch_size=16,          # Memory-safe batch size for MPS
            gradient_accumulation_steps=8,  # Effective batch size = 16 * 8 = 128
            learning_rate=5e-5,     # Higher LR for faster convergence
            warmup_steps=50,        # Reduced warmup for faster start
            early_stopping=False,   # Disable early stopping for speed
            mixed_precision=True,   # FP16/BF16 for ~40-60% speed boost
            dataloader_num_workers=0,  # Disable multiprocessing for compatibility
            gradient_checkpointing=False,  # Disable since not supported by MPNet
            optimizer_type="adamw"  # Standard optimizer for compatibility
        )
        
        print(f"Dense retriever training completed. Model saved to: {model_output_dir}")
        return str(model_output_dir)
    
    def build_and_save_faiss_index(self, corpus_path):
        """Wrapper function for building and saving FAISS index"""
        # Build FAISS index
        print("Building FAISS index...")
        bundle, sections_map, index = build_index(corpus_path=corpus_path, model_path=self.artifacts_dir / "dense_retriever" / "model")
        
        # Save FAISS index to disk
        faiss_dir = self.artifacts_dir / "dense_retriever" / "faiss"
        save_faiss_bundle(faiss_dir, index, bundle["dense"]["chunk_ids"], bundle["meta"])
        
        print(f"FAISS index saved to: {faiss_dir}")
        
    
    def tune_hybrid_retriever(self, dense_model_path: str) -> Dict:
        """Tune hybrid retriever hyperparameters using validation set."""
        print("=" * 60)
        print("STEP 2: Hybrid Retriever Tuning")
        print("=" * 60)
        
        # Create dense retriever instance with fine-tuned model
        print(f"Initializing dense retriever for tuning with model: {dense_model_path}")
        dense_retriever = DenseRetriever(model_name=dense_model_path)
        
        # Confirm model type loaded
        if hasattr(dense_retriever, 'is_fine_tuned'):
            if dense_retriever.is_fine_tuned:
                print(f"✓ Fine-tuned dense retriever successfully loaded for tuning")
            else:
                print(f"⚠️ Warning: Expected fine-tuned model but loaded base model")
        else:
            print(f"✓ Dense retriever loaded for tuning")
        
        # Use validation triples for hyperparameter tuning (proper data split)
        val_triples_path = self.training_data_dir / "stratified_splits" / "val_triples.jsonl"
        print(f"Using validation triples for hyperparameter tuning: {val_triples_path}")
        tuning_dataset_path = val_triples_path
        
        # Initialize optimizer with validation dataset
        optimizer = HybridHyperparameterOptimizer(tuning_dataset_path)
        
        # Test different fusion methods and parameters
        print("Tuning hybrid retriever parameters...")
        
        tuning_results = []
        
        # Test linear fusion with different alpha values
        print("\nTesting linear fusion...")
        alpha_results = optimizer.optimize_alpha(
            alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            sample_size=100,
            dense_retriever=dense_retriever  # Pass the fine-tuned model
        )
        tuning_results.extend(alpha_results)
        
        # Test RRF fusion with different k values
        print("\nTesting RRF fusion...")
        rrf_results = optimizer.compare_fusion_methods(
            sample_size=100,
            dense_retriever=dense_retriever  # Pass the fine-tuned model
        )
        tuning_results.extend(rrf_results)
        
        # Find best configuration
        best_config = max(tuning_results, key=lambda x: x.get('composite_score', 0))
        
        print(f"\nBest configuration found:")
        print(f"  Method: {best_config.get('method_name', best_config.get('fusion_method', 'linear'))}")
        print(f"  Score: {best_config['composite_score']:.4f}")
        
        # Save tuning results
        tuning_file = IMMEDIATE_PARENT / "stratified_results" / "tuning" / "hyperparameter_tuning.json"
        with open(tuning_file, "w") as f:
            json.dump({
                "best_config": best_config,
                "all_results": tuning_results,
                "tuning_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"Tuning results saved to: {tuning_file}")
        
        return best_config
    
    def evaluate_retrievers(self, dense_model_path: str, best_hybrid_config: Dict) -> Dict:
        """Evaluate all retrievers on test set"""
        print("=" * 60)
        print("STEP 3: Final Evaluation")
        print("=" * 60)
        
        # Use test triples for final evaluation (proper data split)
        test_triples_path = self.training_data_dir / "stratified_splits" / "test_triples.jsonl"
        print(f"Using test triples for final evaluation: {test_triples_path}")
        evaluation_dataset_path = test_triples_path
        
        # Load tuning results (hybrid + bm25) from file
        tuning_file = Path(__file__).resolve().parent / "stratified_results" / "tuning" / "hyperparameter_tuning.json"
        bm25_best_params = {"k1": 0.5, "b": 1.0}
        if tuning_file.exists():
            with open(tuning_file, "r", encoding="utf-8") as f:
                tuning_json = json.load(f)
            # Hybrid best config
            if not best_hybrid_config:
                best_hybrid_config = tuning_json.get("best_config", {})
            # BM25 best params if present
            if "bm25_best_params" in tuning_json:
                bm25_best_params = tuning_json["bm25_best_params"]
        else:
            print(f"Warning: Tuning file not found at {tuning_file}. Using default BM25 params and provided hybrid config.")
        
        evaluator = UnifiedRetrievalEvaluator(evaluation_dataset_path)
        eval_results = {}
        
        # 1. Evaluate fine-tuned dense retriever
        print("Evaluating fine-tuned dense retriever...")
        dense_retriever = DenseRetriever(model_name=dense_model_path)
        print(f"Evaluation dense retriever using device: {dense_retriever.device}")
        if hasattr(dense_retriever, 'is_fine_tuned') and not dense_retriever.is_fine_tuned:
            print("Warning: Expected fine-tuned model but loaded base model for evaluation")
        dense_result = evaluator.evaluate_retriever(dense_retriever, "Dense_Finetuned", top_k=10)
        eval_results["dense_finetuned"] = dense_result
        
        # 2. Evaluate hybrid retriever with optimized parameters ONLY
        print("Evaluating hybrid retriever (optimized only)...")
        if best_hybrid_config.get("fusion_method") == "linear" and best_hybrid_config.get("alpha") is not None:
            hybrid_optimized = create_hybrid_retriever(
                alpha=best_hybrid_config["alpha"],
                fusion_method="linear",
                dense_retriever=dense_retriever
            )
        else:
            hybrid_optimized = create_hybrid_retriever(
                fusion_method="rrf",
                rrf_k=best_hybrid_config.get("rrf_k", 30),
                dense_retriever=dense_retriever
            )
        hybrid_optimized_result = evaluator.evaluate_retriever(hybrid_optimized, "Hybrid_Optimized", top_k=10)
        eval_results["hybrid_optimized"] = hybrid_optimized_result
        
        # 3. Evaluate BM25 retriever with optimized parameters
        print("Evaluating BM25 retriever (optimized params)...")
        sys.path.append(str("bm25_retrieval"))
        from bm25_retriever import BM25Retriever
        bm25_retriever = BM25Retriever(k1=bm25_best_params.get('k1'), b=bm25_best_params.get('b'))
        bm25_result = evaluator.evaluate_retriever(bm25_retriever, "BM25", top_k=10)
        eval_results["bm25"] = bm25_result
        
        # Save individual evaluation results
        eval_file = Path(__file__).resolve().parent / "stratified_results" / "eval" / "evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_file}")
        
        return eval_results, bm25_best_params

    def compute_metrics(self, eval_result: Dict) -> Dict:
        """Extract and compute metrics from evaluation result """
        summary = eval_result["evaluation_summary"]
        metrics = {}
        # Extract recall metrics
        for k in [1, 5, 10]:
            recall_key = f"recall@{k}"
            if recall_key in summary:
                metrics[recall_key] = summary[recall_key]["mean"]
        # Extract NDCG metrics
        for k in [1, 3, 5, 10]:
            ndcg_key = f"ndcg@{k}"
            if ndcg_key in summary:
                metrics[ndcg_key] = summary[ndcg_key]["mean"]
        # Extract MRR and MRR@k
        if "mrr" in summary:
            metrics["mrr"] = summary["mrr"]["mean"]
        for k in [1, 3, 5, 10]:
            mrrk = f"mrr@{k}"
            if mrrk in summary:
                metrics[mrrk] = summary[mrrk]["mean"]
        # Latency
        if "search_time_ms" in summary:
            latency_stats = summary["search_time_ms"]
            metrics["latency"] = {
                "p50_ms": latency_stats.get("median", 0),
                "p90_ms": latency_stats.get("median", 0) + 1.645 * latency_stats.get("std", 0),
                "avg_ms": latency_stats.get("mean", 0)
            }
        return metrics

    def run_pipeline(self):
        """Run the evaluation-only pipeline """
        start_time = time.time()
        print("Starting retriever evaluation pipeline...")
        print(f"Artifacts directory: {self.artifacts_dir}")
        try:
            # Load existing fine-tuned dense model
            dense_model_path = str(self.artifacts_dir / "dense_retriever" / "model")
            if not Path(dense_model_path).exists():
                print(f"Warning: Dense model directory not found at {dense_model_path}. Trying fallback path.")
                fallback_model = Path(__file__).resolve().parents[0] / "dense_retrieval" / "fine_tuned_model"
                dense_model_path = str(fallback_model)
            
            # Load tuning config from file
            tuning_file = Path(__file__).resolve().parent / "stratified_results" / "tuning" / "hyperparameter_tuning.json"
            best_hybrid_config = {}
            if tuning_file.exists():
                with open(tuning_file, "r", encoding="utf-8") as f:
                    tuning_json = json.load(f)
                best_hybrid_config = tuning_json.get("best_config", {})
            else:
                print(f"Warning: Tuning file not found at {tuning_file}. Using default hybrid parameters.")
                best_hybrid_config = {"fusion_method": "rrf", "rrf_k": 30}
            
            # Evaluate all retrievers (top_k=10, optimized hybrid, optimized BM25)
            eval_results, bm25_best_params = self.evaluate_retrievers(dense_model_path, best_hybrid_config)
            
            # Compile final results
            print("=" * 60)
            print("STEP 4: Compiling Results")
            print("=" * 60)
            self.results["dense_retriever"] = {
                "model_dir": str(self.artifacts_dir / "dense_retriever" / "model"),
                "faiss_index_dir": str(self.artifacts_dir / "dense_retriever" / "faiss"),
                "metrics": self.compute_metrics(eval_results["dense_finetuned"]) 
            }
            # Optimized Hybrid only
            optimized_params = {}
            if best_hybrid_config.get("fusion_method") == "linear" and best_hybrid_config.get("alpha") is not None:
                optimized_params = {"alpha": best_hybrid_config["alpha"], "fusion_method": "linear"}
            else:
                optimized_params = {"fusion_method": "rrf", "rrf_k": best_hybrid_config.get("rrf_k")}
            self.results["hybrid_optimized"] = {
                "best_params": optimized_params,
                "metrics": self.compute_metrics(eval_results["hybrid_optimized"]) 
            }
            self.results["bm25"] = {
                "best_params": bm25_best_params,
                "metrics": self.compute_metrics(eval_results["bm25"]) 
            }
            
            # Save final consolidated results
            results_file = Path(__file__).resolve().parent / "stratified_results" / "conslidated_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            total_time = time.time() - start_time
            print(f"\nEvaluation-only pipeline completed successfully in {total_time:.1f} seconds")
            print(f"Final results saved to: {results_file}")
            
            print("\n" + "=" * 60)
            print("FINAL RESULTS SUMMARY (top_k=10)")
            print("=" * 60)
            for retriever_name, result in [
                ("Dense Retriever (Fine-tuned)", self.results["dense_retriever"]),
                ("Hybrid Optimized", self.results["hybrid_optimized"]),
                ("BM25 (Optimized)", self.results["bm25"]) ]:
                print(f"\n{retriever_name}:")
                metrics = result["metrics"]
                for k in [1, 5, 10]:
                    rk = f"recall@{k}"
                    if rk in metrics:
                        print(f"  Recall@{k}: {metrics[rk]:.4f}")
                for k in [1,3,5,10]:
                    mk = f"mrr@{k}"
                    if mk in metrics:
                        print(f"  MRR@{k}: {metrics[mk]:.4f}")
                if "mrr" in metrics:
                    print(f"  MRR: {metrics['mrr']:.4f}")
                if "latency" in metrics:
                    print(f"  Avg Latency: {metrics['latency']['avg_ms']:.2f}ms")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise

def main():
    """Main entry point (evaluation-only)."""
    pipeline = PDPAPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()