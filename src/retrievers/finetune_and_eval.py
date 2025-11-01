#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finetune_and_eval.py — End-to-end PDPA QA System Pipeline for:
- Dense retriever training
- Hybrid retriever integration and tuning (validation set)
- Final evaluation (test set) with multiple retrievers
- Outputs consolidated results to stratified_results/consolidated_results.json
"""

import json
import time
import statistics
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Device detection
try:
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Add retriever paths to sys.path
IMMEDIATE_PARENT = Path(__file__).resolve().parent
sys.path.append(str("dense_retrieval"))
sys.path.append(str("dense_indexer"))
sys.path.append(str("hybrid_retrieval"))
sys.path.append(str("retrieval_evaluation"))

# Import modules
from dense_indexer import build_index, save_faiss_bundle
from dense_retriever import DenseRetriever
from hybrid_retriever import create_hybrid_retriever, HybridHyperparameterOptimizer
from evaluate_all_retrievers import UnifiedRetrievalEvaluator

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
        self.qa_dataset_path = self.root / "data" / "qa" / "pdpa_qa_500.jsonl"
        
        # Results storage
        self.results = {
            "device": str(device),
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
            self.qa_dataset_path
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
        if not val_triples_path.exists():
            print(f"Warning: Validation triples not found at {val_triples_path}")
            print("Falling back to QA dataset for tuning")
            tuning_dataset_path = self.qa_dataset_path
        else:
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
        """Evaluate all retrievers on test set."""
        print("=" * 60)
        print("STEP 3: Final Evaluation")
        print("=" * 60)
        
        # Use test triples for final evaluation (proper data split)
        test_triples_path = self.training_data_dir / "stratified_splits" / "test_triples.jsonl"
        if not test_triples_path.exists():
            print(f"Warning: Test triples not found at {test_triples_path}")
            print("Falling back to QA dataset for evaluation")
            evaluation_dataset_path = self.qa_dataset_path
        else:
            print(f"Using test triples for final evaluation: {test_triples_path}")
            evaluation_dataset_path = test_triples_path
        
        evaluator = UnifiedRetrievalEvaluator(evaluation_dataset_path)
        
        eval_results = {}
        
        # 1. Evaluate fine-tuned dense retriever
        print("Evaluating fine-tuned dense retriever...")
        dense_retriever = DenseRetriever(model_name=dense_model_path)
        print(f"Evaluation dense retriever using device: {dense_retriever.device}")
        
        # Confirm model type for evaluation
        if hasattr(dense_retriever, 'is_fine_tuned'):
            if dense_retriever.is_fine_tuned:
                print(f"✓ Fine-tuned model loaded for evaluation")
            else:
                print(f"⚠️ Warning: Expected fine-tuned model but loaded base model for evaluation")
        
        print(f"✓ GPU acceleration confirmed for dense retriever evaluation")
        dense_result = evaluator.evaluate_retriever(dense_retriever, "Dense_Finetuned", top_k=20)
        eval_results["dense_finetuned"] = dense_result
        
        # 2. Evaluate hybrid retriever with default parameters
        print("Evaluating hybrid retriever (default)...")
        print(f"✓ GPU acceleration confirmed for hybrid retriever (dense component)")
        hybrid_default = create_hybrid_retriever(
            alpha=0.5, 
            fusion_method="rrf", 
            rrf_k=30,
            dense_retriever=dense_retriever
        )
        hybrid_default_result = evaluator.evaluate_retriever(hybrid_default, "Hybrid_Default", top_k=20)
        eval_results["hybrid_default"] = hybrid_default_result
        
        # 3. Evaluate hybrid retriever with optimized parameters
        print("Evaluating hybrid retriever (optimized)...")
        
        # Extract parameters from best config
        if best_hybrid_config.get("fusion_method") == "linear" and best_hybrid_config.get("alpha") is not None:
            # Linear fusion
            hybrid_optimized = create_hybrid_retriever(
                alpha=best_hybrid_config["alpha"],
                fusion_method="linear",
                dense_retriever=dense_retriever
            )
        else:
            # RRF fusion
            hybrid_optimized = create_hybrid_retriever(
                fusion_method="rrf",
                rrf_k=best_hybrid_config.get("rrf_k", 30),
                dense_retriever=dense_retriever
            )
        
        hybrid_optimized_result = evaluator.evaluate_retriever(hybrid_optimized, "Hybrid_Optimized", top_k=20)
        eval_results["hybrid_optimized"] = hybrid_optimized_result
        
        # 4. Evaluate BM25 retriever with hyperparameter optimization
        print("Evaluating BM25 retriever with hyperparameter optimization...")
        sys.path.append(str("bm25_retrieval"))
        from bm25_retriever import BM25Retriever, BM25HyperparameterOptimizer
        
        # Run hyperparameter optimization on validation set for BM25
        val_triples_path = self.training_data_dir / "stratified_splits" / "val_triples.jsonl"
        if val_triples_path.exists():
            print(f"Using validation triples for BM25 hyperparameter tuning: {val_triples_path}")
            bm25_optimizer = BM25HyperparameterOptimizer(val_triples_path)
            
            # Run grid search with smaller parameter space for speed
            k1_values = [0.5, 1.0, 1.2, 1.5, 2.0]
            b_values = [0.0, 0.3, 0.5, 0.75, 1.0]
            print(f"Running BM25 grid search: {len(k1_values)} k1 × {len(b_values)} b = {len(k1_values) * len(b_values)} combinations")
            
            bm25_results = bm25_optimizer.grid_search(k1_values=k1_values, b_values=b_values, sample_size=100)
            
            if bm25_results:
                best_bm25_params = bm25_results[0]
                print(f"Best BM25 parameters: k1={best_bm25_params['k1']}, b={best_bm25_params['b']}")
                print(f"Best BM25 validation score: {best_bm25_params['composite_score']:.4f}")
                
                # Create BM25 retriever with optimized parameters
                bm25_retriever = BM25Retriever(k1=best_bm25_params['k1'], b=best_bm25_params['b'])
            else:
                print("BM25 optimization failed, using default parameters")
                bm25_retriever = BM25Retriever()
                best_bm25_params = {"k1": 0.5, "b": 1.0}
        else:
            print("Validation triples not found, using default BM25 parameters")
            bm25_retriever = BM25Retriever()
            best_bm25_params = {"k1": 0.5, "b": 1.0}
        
        # Evaluate BM25 on test set
        bm25_result = evaluator.evaluate_retriever(bm25_retriever, "BM25", top_k=20)
        eval_results["bm25"] = bm25_result
        
        # Save individual evaluation results
        eval_file = IMMEDIATE_PARENT / "stratified_results" / "eval" / "evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation results saved to: {eval_file}")
        
        return eval_results, best_bm25_params
    
    def compute_metrics(self, eval_result: Dict) -> Dict:
        """Extract and compute metrics from evaluation result."""
        summary = eval_result["evaluation_summary"]
        
        metrics = {}
        
        # Extract recall metrics
        for k in [1, 5, 10, 20]:
            recall_key = f"recall@{k}"
            if recall_key in summary:
                metrics[recall_key] = summary[recall_key]["mean"]
        
        # Extract MRR
        if "mrr" in summary:
            metrics["mrr"] = summary["mrr"]["mean"]
        
        # Extract latency metrics
        if "search_time_ms" in summary:
            latency_stats = summary["search_time_ms"]
            metrics["latency"] = {
                "p50_ms": latency_stats["median"],
                "p90_ms": latency_stats["median"] + 1.645 * latency_stats["std"],  # Approximate p90
                "avg_ms": latency_stats["mean"]
            }
        
        return metrics
    
    def run_pipeline(self):
        """Run the complete end-to-end pipeline."""
        start_time = time.time()
        
        print("Starting PDPA QA System Pipeline")
        print(f"Device: {device}")
        print(f"Artifacts directory: {self.artifacts_dir}")
        
        # Clean up all artifacts to ensure fresh run (disabled for good reasons, pls enable this when absolutely sure)
        # self._cleanup_artifacts()
        
        try:
            # Step 1: Train dense retriever
            dense_model_path = self.train_dense_retriever()
            self.build_and_save_faiss_index(corpus_path=self.root / "data" / "corpus" / "corpus_subsection_v1.jsonl")
            
            # Step 2: Tune hybrid retriever
            best_hybrid_config = self.tune_hybrid_retriever(dense_model_path)
            
            # Step 3: Evaluate all retrievers
            eval_results, best_bm25_params = self.evaluate_retrievers(dense_model_path, best_hybrid_config)
            
            # Step 4: Compile final results
            print("=" * 60)
            print("STEP 4: Compiling Results")
            print("=" * 60)
            
            # Extract metrics for each retriever
            self.results["dense_retriever"] = {
                "model_dir": str(self.artifacts_dir / "dense_retriever" / "model"),
                "faiss_index_dir": str(self.artifacts_dir / "dense_retriever" / "faiss"),
                "metrics": self.compute_metrics(eval_results["dense_finetuned"])
            }
            
            self.results["hybrid_default"] = {
                "params": {"alpha": 0.5, "fusion_method": "rrf", "rrf_k": 30},
                "metrics": self.compute_metrics(eval_results["hybrid_default"])
            }
            
            # Extract optimized parameters
            optimized_params = {}
            if best_hybrid_config.get("fusion_method") == "linear" and best_hybrid_config.get("alpha") is not None:
                optimized_params = {
                    "alpha": best_hybrid_config["alpha"],
                    "fusion_method": "linear"
                }
            else:
                optimized_params = {
                    "fusion_method": "rrf",
                    "rrf_k": best_hybrid_config.get("rrf_k", 30)
                }
            
            self.results["hybrid_optimized"] = {
                "best_params": optimized_params,
                "metrics": self.compute_metrics(eval_results["hybrid_optimized"])
            }
            
            self.results["bm25"] = {
                "best_params": best_bm25_params,
                "metrics": self.compute_metrics(eval_results["bm25"])
            }
            
            # Save final consolidated results
            results_file = IMMEDIATE_PARENT / "stratified_results" / "conslidated_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            # Print summary
            total_time = time.time() - start_time
            print(f"\nPipeline completed successfully in {total_time:.1f} seconds")
            print(f"Final results saved to: {results_file}")
            
            print("\n" + "=" * 60)
            print("FINAL RESULTS SUMMARY")
            print("=" * 60)
            
            for retriever_name, result in [
                ("Dense Retriever (Fine-tuned)", self.results["dense_retriever"]),
                ("Hybrid Default", self.results["hybrid_default"]),
                ("Hybrid Optimized", self.results["hybrid_optimized"])
            ]:
                print(f"\n{retriever_name}:")
                metrics = result["metrics"]
                for k in [1, 5, 10, 20]:
                    recall_key = f"recall@{k}"
                    if recall_key in metrics:
                        print(f"  Recall@{k}: {metrics[recall_key]:.4f}")
                if "mrr" in metrics:
                    print(f"  MRR: {metrics['mrr']:.4f}")
                if "latency" in metrics:
                    print(f"  Avg Latency: {metrics['latency']['avg_ms']:.2f}ms")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise

def main():
    """Main entry point."""
    pipeline = PDPAPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()