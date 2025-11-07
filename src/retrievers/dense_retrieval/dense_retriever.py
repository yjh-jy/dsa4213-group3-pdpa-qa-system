#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dense_retriever.py — Trainable Dense retrieval system for PDPA corpus
- Loads pre-built dense index from dense_indexer.py
- Provides search functionality with similarity scoring
- Supports fine-tuning with MPS acceleration
- Implements training pipeline for sentence transformers
"""

import json
import time
import faiss
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", faiss_path: str = "artefacts/dense_retriever/faiss"):
        """Initialize dense retriever with base model or fine-tuned model.
        
        Args:
            model_name: Model name or path (HuggingFace model name or local path to fine-tuned model)
        """
        self.model_name = model_name
        self.device = self._get_device()
        self.index = None
        self.corpus_embeddings = None
        
        model_path = Path(__file__).resolve().parents[3] / self.model_name

        # Determine if model_name is a local path or HuggingFace model name
        if model_path.exists() and model_path.is_dir():
            # Local fine-tuned model path
            self.model = SentenceTransformer(str(model_path), device=self.device)
            self.is_fine_tuned = True
            print(f"Loaded fine-tuned model from: {model_path} on {self.device}")

        elif model_name.startswith("sentence-transformers/") or "/" not in model_name:
            # HuggingFace model name
            print(f"Loading base model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.is_fine_tuned = False
        else:
            # Try to load as path first, fallback to HuggingFace
            try:
                print(f"Attempting to load model from path: {model_name}")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.is_fine_tuned = True
            except Exception as e:
                print(f"Failed to load from path, trying as HuggingFace model: {model_name}")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.is_fine_tuned = False
        
        # Load corpus data
        self._load_corpus()

        # Try loading the saved faiss index first
        try:
            print("Loading Saved FAISS index... ")
            self._load_faiss_bundle(faiss_path)
        except Exception as e:
            print(e)
            print("Loading failed for FAISS index. Switching to computing embedding on the spot...")
            # Fallback to computing corpus embeddings on the spot
            self._precompute_embeddings()

    def _load_faiss_bundle(self, in_dir: Path):
        index_path = Path(__file__).resolve().parents[3]/ in_dir / "index.faiss"
        chunk_ids_path = Path(__file__).resolve().parents[3] / in_dir / "chunk_ids.npy"
        meta_json_path = Path(__file__).resolve().parents[3] / in_dir / "meta.json"
        
        index = faiss.read_index(str(index_path))
        chunk_ids = np.load(str(chunk_ids_path), allow_pickle=True)
        with open(str(meta_json_path)) as f:
            meta = json.load(f)

        self.chunk_ids = chunk_ids
        self.meta = meta
        self.index = index
    
    def _load_corpus(self, corpus_path: Optional[Path] = None):
        """Load corpus data for dynamic embedding computation."""
        if corpus_path is None:
            corpus_path = Path(__file__).resolve().parents[3] / "data" / "corpus" / "corpus_subsection_v1.jsonl"
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")
        
        self.chunk_ids = []
        self.texts = []
        self.sections_map = {}
        
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk_id = chunk["chunk_id"]
                    text = chunk["text"]
                    
                    self.chunk_ids.append(chunk_id)
                    self.texts.append(text)
                    
                    # Store section metadata
                    self.sections_map[chunk_id] = {
                        "canonical_citation": chunk.get("canonical_citation", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "part": chunk.get("part", ""),
                        "section": chunk.get("section", ""),
                        "subsection": chunk.get("subsection", ""),
                        "text": text
                    }
        
        print(f"Loaded corpus: {len(self.chunk_ids)} chunks")
    
    def _precompute_embeddings(self):
        """Pre-compute corpus embeddings using the fine-tuned model for efficiency."""
        print("Pre-computing corpus embeddings...")
        self.corpus_embeddings = self.model.encode(
            self.texts, 
            normalize_embeddings=True, 
            show_progress_bar=True,
            batch_size=32
        )
        print(f"Pre-computed embeddings: {self.corpus_embeddings.shape}")
    
    def _get_device(self) -> str:
        """Get the best available device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    

    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into embedding vector."""
        return self.model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    def search(self, query: str, top_k: int = 10) -> Dict:
        """
        Search for relevant chunks using dense retrieval.
        
        Args:
            query: Natural language question
            top_k: Number of results to return
            
        Returns:
            Dict containing results list and search metadata
        """
        start_time = time.time()
        
        # Encode query using fine-tuned model
        query_embedding = self.encode_query(query)
        
        # Use pre-computed corpus embeddings
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            result = {
                "chunk_id": chunk_id,
                "score": float(similarities[idx]),
                "text": self.texts[idx],
                "rank": len(results) + 1
            }
            
            # Add section metadata if available
            if chunk_id in self.sections_map:
                result.update(self.sections_map[chunk_id])
            
            results.append(result)
        
        # Add timing info
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_time_ms": search_time * 1000,
            "model": self.model_name
        }
    
    def faiss_search(self, query: str, top_k: int) -> List[Tuple[str, float, int]]:
        start_time = time.time()

        # encode + normalize query
        qv = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        # (already normalized) → cosine == inner product
        D, I = self.index.search(qv, top_k)  # D: scores, I: row ids
        hits = []
        for rank, (row, score) in enumerate(zip(I[0], D[0]), start=1):
            cid = self.chunk_ids[row]
            hits.append((cid, float(score), rank))
        
        # Build results
        results = []
        for cid, score, rank in hits:
            result = {
                "chunk_id": cid,
                "score": score,
                "text": self.sections_map.get(cid)['text'],
                "rank": rank
            }
            
            # Add section metadata if available
            if cid in self.sections_map:
                result.update(self.sections_map[cid])
            
            results.append(result)
        
        # Add timing info
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_time_ms": search_time * 1000,
            "model": self.model_name
        }
    

    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of natural language questions
            top_k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        results = []
        for query in queries:
            result = self.search(query, top_k)
            results.append(result)
        return results
    
    def load_training_data(self, training_data_path: Path) -> List[InputExample]:
        """Load training triples and convert to InputExample format."""
        examples = []
        
        with training_data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triple = json.loads(line)
                    
                    # Create InputExample for contrastive learning
                    example = InputExample(
                        texts=[triple["query"], triple["pos_text"], triple["neg_text"]],
                        label=1.0  # Positive similarity for query-positive pair
                    )
                    examples.append(example)
        
        print(f"Loaded {len(examples)} training examples from {training_data_path}")
        return examples
    
    def create_evaluation_data(self, test_queries_path: Path, corpus_path: Path) -> InformationRetrievalEvaluator:
        """Create evaluation data for training monitoring."""
        # Load test queries
        queries = {}
        relevant_docs = {}
        
        with test_queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    query_data = json.loads(line)
                    qid = query_data["qid"]
                    queries[qid] = query_data["query"]
                    relevant_docs[qid] = set(query_data["pos_ids"])
        
        # Load corpus
        corpus = {}
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    corpus[chunk["chunk_id"]] = chunk["text"]
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="pdpa_eval"
        )
        
        return evaluator
    
    def fine_tune(self, 
                  training_data_path: Path,
                  output_dir: Path,
                  test_queries_path: Optional[Path] = None,
                  corpus_path: Optional[Path] = None,
                  epochs: int = 3,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 100,
                  early_stopping: bool = False,
                  gradient_accumulation_steps: int = 1,
                  mixed_precision: bool = True,
                  dataloader_num_workers: int = 0,
                  gradient_checkpointing: bool = False,
                  optimizer_type: str = "adamw") -> str:
        """
        Fine-tune the dense retriever on PDPA data with advanced optimizations.
        
        Args:
            training_data_path: Path to training triples JSONL file
            output_dir: Directory to save the fine-tuned model
            test_queries_path: Path to test queries for evaluation (optional)
            corpus_path: Path to corpus for evaluation (optional)
            epochs: Number of training epochs
            batch_size: Per-device training batch size
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            early_stopping: Enable early stopping based on evaluation metrics
            gradient_accumulation_steps: Steps to accumulate gradients (effective_batch = batch_size * steps)
            mixed_precision: Enable FP16/BF16 mixed precision training
            dataloader_num_workers: Number of workers for data loading
            gradient_checkpointing: Enable gradient checkpointing to save memory
            optimizer_type: Optimizer type ("adamw", "adamw_8bit", "adafactor")
            
        Returns:
            Path to the saved fine-tuned model
        """
        # Clear GPU memory before training
        if self.device != "cpu":
            import torch
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Starting fine-tuning on device: {self.device}")
        print(f"Training parameters:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   - Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Warmup steps: {warmup_steps}")
        print(f"   - Mixed precision: {mixed_precision}")
        
        # Load training data
        train_examples = self.load_training_data(training_data_path)
        
        # Create optimized DataLoader (disable multiprocessing for compatibility)
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size,
            num_workers=0,  # Disable multiprocessing to avoid pickling issues
            pin_memory=False  # Disable pin_memory for MPS compatibility
        )
        
        # Define loss function (Multiple Negatives Ranking Loss)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Create evaluator if test data is provided
        evaluator = None
        if test_queries_path and corpus_path and test_queries_path.exists() and corpus_path.exists():
            try:
                evaluator = self.create_evaluation_data(test_queries_path, corpus_path)
                print(f"Created evaluator with test data")
            except Exception as e:
                print(f"Could not create evaluator: {e}")
        
        # Set up output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure training
        num_train_steps = len(train_dataloader) * epochs
        
        # Train the model
        print(f"Training for {epochs} epochs ({num_train_steps} steps)...")
        
        # Configure advanced training parameters with all optimizations
        effective_batch_size = batch_size * gradient_accumulation_steps
        total_steps = (len(train_dataloader) // gradient_accumulation_steps) * epochs
        
        print(f"Training configuration:")
        print(f"  - Per-device batch size: {batch_size}")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size: {effective_batch_size}")
        print(f"  - Total training steps: {total_steps}")
        print(f"  - Mixed precision: {mixed_precision}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        print(f"  - Optimizer: {optimizer_type}")
        
        # Use optimized AdamW parameters for maximum speed
        optimizer_params = {
            'lr': learning_rate,
            'weight_decay': 0.01,
            'eps': 1e-6
        }
        
        # Configure streamlined training for maximum speed
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'epochs': epochs,
            'warmup_steps': warmup_steps,
            'output_path': str(output_dir),
            'optimizer_params': optimizer_params,
            'show_progress_bar': True,
            'use_amp': mixed_precision
        }
        
        print(f"✓ Speed-optimized training configuration:")
        print(f"  - Mixed precision: {mixed_precision}")
        print(f"  - Streamlined for maximum speed")
        print(f"  - Evaluation disabled during training")
        
        # Execute training with maximum speed (no evaluation overhead)
        print("Starting streamlined training for maximum speed...")
        self.model.fit(**training_args)
        
        print(f"Fine-tuning completed!")
        print(f"Model saved to: {output_dir}")
        
        # Clear GPU memory after training
        if self.device != "cpu":
            import torch
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print("✓ MPS memory cache cleared")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ CUDA memory cache cleared")
        
        return str(output_dir)
    
    def evaluate_model(self, test_queries_path: Path, corpus_path: Path) -> Dict:
        """Evaluate the current model on test data."""
        evaluator = self.create_evaluation_data(test_queries_path, corpus_path)
        
        print("Evaluating model...")
        results = evaluator(self.model)
        
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.4f}")
        
        return results
    
    def k_fold_cross_validation(self, 
                               training_data_dir: Path,
                               k_folds: int = 5,
                               epochs: int = 3,
                               batch_size: int = 16,
                               learning_rate: float = 2e-5,
                               warmup_steps: int = 100) -> Dict:
        """
        Perform k-fold cross-validation to evaluate model performance without data leakage.
        
        Args:
            training_data_dir: Directory containing fold_1, fold_2, ..., fold_k subdirectories
            k_folds: Number of folds to use (default: 5)
            epochs: Number of training epochs per fold
            batch_size: Training batch size
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            
        Returns:
            Dictionary containing average metrics across all folds
        """
        print(f"=== K-Fold Cross-Validation (k={k_folds}) ===")
        
        all_fold_results = []
        corpus_path = training_data_dir / "corpus.jsonl"
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        for fold in range(1, k_folds + 1):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            # Paths for current fold
            fold_dir = training_data_dir / f"fold_{fold}"
            train_triples_path = fold_dir / "train_triples.jsonl"
            test_queries_path = fold_dir / "test_queries.jsonl"
            
            if not train_triples_path.exists() or not test_queries_path.exists():
                print(f"Skipping fold {fold}: missing data files")
                continue
            
            # Create a fresh model for this fold (reset to base model)
            print(f"Initializing fresh model for fold {fold}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Create temporary output directory for this fold
            temp_output_dir = Path(__file__).resolve().parents[0] / f"temp_fold_{fold}_model"
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Fine-tune on training data for this fold
                print(f"Training on fold {fold} data...")
                self.fine_tune(
                    training_data_path=train_triples_path,
                    output_dir=temp_output_dir,
                    test_queries_path=None,  # Don't evaluate during training
                    corpus_path=None,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps
                )
                
                # Load the fine-tuned model for this fold
                self.model = SentenceTransformer(str(temp_output_dir), device=self.device)
                
                # Reload corpus and recompute embeddings with fine-tuned model
                self._load_corpus(corpus_path)
                self._precompute_embeddings()
                
                # Evaluate on test data for this fold
                print(f"Evaluating fold {fold}...")
                fold_results = self.evaluate_model(test_queries_path, corpus_path)
                all_fold_results.append(fold_results)
                
                print(f"Fold {fold} Results:")
                for metric, value in fold_results.items():
                    print(f"   {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                continue
            
            finally:
                # Clean up temporary model directory
                import shutil
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
        
        if not all_fold_results:
            raise RuntimeError("No folds completed successfully")
        
        # Calculate average metrics across all folds
        print(f"\n=== K-Fold Cross-Validation Results (k={k_folds}) ===")
        
        # Get all metric names from first fold
        metric_names = list(all_fold_results[0].keys())
        average_results = {}
        std_results = {}
        
        for metric in metric_names:
            values = [fold_results[metric] for fold_results in all_fold_results]
            average_results[metric] = np.mean(values)
            std_results[metric] = np.std(values)
            
            print(f"{metric}:")
            print(f"   Mean: {average_results[metric]:.4f}")
            print(f"   Std:  {std_results[metric]:.4f}")
            print(f"   Folds: {[f'{v:.4f}' for v in values]}")
        
        # Add summary statistics
        average_results['_fold_count'] = len(all_fold_results)
        average_results['_std_results'] = std_results
        
        return average_results

def train_dense_retriever():
    """Train a dense retriever on PDPA data."""
    print("=== Dense Retriever Training ===")
    
    # Paths
    root_dir = Path(__file__).resolve().parents[3]
    training_data_dir = root_dir / "data" / "dense_training"
    
    # Check if training data exists
    full_train_path = training_data_dir / "full_train_triples.jsonl"
    if not full_train_path.exists():
        print(f"Training data not found at {full_train_path}")
        print("Please run dense_chunk_and_extract.py first to generate training data.")
        return
    
    # Initialize retriever in training mode
    retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Set up training
    output_dir = Path(__file__).resolve().parents[0] / "fine_tuned_model"
    corpus_path = training_data_dir / "corpus.jsonl"
    
    # Use fold 1 for validation if available
    test_queries_path = training_data_dir / "fold_1" / "test_queries.jsonl"
    if not test_queries_path.exists():
        test_queries_path = None
        corpus_path = None
    
    # Fine-tune the model
    model_path = retriever.fine_tune(
        training_data_path=full_train_path,
        output_dir=output_dir,
        test_queries_path=test_queries_path,
        corpus_path=corpus_path,
        epochs=3,
        batch_size=8,  # Smaller batch size for MPS
        learning_rate=2e-5,
        warmup_steps=100
    )
    
    print(f"Training completed! Model saved to: {model_path}")
    
    # Evaluate if test data is available
    if test_queries_path and corpus_path:
        print("\nEvaluating fine-tuned model...")
        results = retriever.evaluate_model(test_queries_path, corpus_path)

def evaluate_trained_model():
    """Evaluate a trained dense retriever."""
    print("=== Dense Retriever Evaluation ===")
    
    # Paths
    root_dir = Path(__file__).resolve().parents[3]
    model_dir = Path(__file__).resolve().parents[0] / "fine_tuned_model"
    training_data_dir = root_dir / "data" / "dense_training"
    
    if not model_dir.exists():
        print(f"Trained model not found at {model_dir}")
        print("Please run training first.")
        return
    
    # Load trained model
    retriever = DenseRetriever(model_name=str(model_dir))
    
    # Evaluate on test data
    test_queries_path = training_data_dir / "fold_1" / "test_queries.jsonl"
    corpus_path = training_data_dir / "corpus.jsonl"
    
    if test_queries_path.exists() and corpus_path.exists():
        results = retriever.evaluate_model(test_queries_path, corpus_path)
    else:
        print("Test data not found. Please generate training data first.")

def k_fold_evaluate_dense_retriever():
    """Perform k-fold cross-validation evaluation of dense retriever."""
    print("=== K-Fold Cross-Validation Evaluation ===")
    
    # Paths
    root_dir = Path(__file__).resolve().parents[3]
    training_data_dir = root_dir / "data" / "dense_training"
    
    if not training_data_dir.exists():
        print(f"Training data directory not found: {training_data_dir}")
        print("Please run dense_chunk_and_extract.py first to generate training data.")
        return
    
    # Initialize retriever with base model (will be reset for each fold)
    retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Perform k-fold cross-validation
    try:
        results = retriever.k_fold_cross_validation(
            training_data_dir=training_data_dir,
            k_folds=5,
            epochs=5,
            batch_size=8,  # Smaller batch size for MPS
            learning_rate=2e-5,
            warmup_steps=100
        )
        
        print(f"\n=== Final K-Fold Results ===")
        print(f"Completed {results['_fold_count']} folds successfully")
        
        # Save results to file
        results_dir = Path(__file__).resolve().parents[0] / "results"
        results_dir.mkdir(exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"k_fold_evaluation_{timestamp}.json"
        
        # Convert numpy types to regular Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == '_std_results':
                json_results[key] = {k: float(v) for k, v in value.items()}
            elif isinstance(value, (np.float64, np.float32)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"K-fold evaluation failed: {e}")
        return None

def demo_retriever():
    """Demo usage of dense retriever."""
    try:
        retriever = DenseRetriever()
    except FileNotFoundError:
        print("No pre-built index found. Please run dense_indexer.py first or train a model.")
        return
    
    # Example queries
    test_queries = [
        "What is the official name of the PDPA?",
        "Does the PDPA apply to individuals acting in a personal capacity?",
        "What are the consent requirements for data collection?"
    ]
    
    print("=== Dense Retriever Demo ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = retriever.search(query, top_k=3)
        print(f"Search time: {result['search_time_ms']:.2f}ms")
        
        for i, hit in enumerate(result["results"], 1):
            print(f"{i}. [{hit['chunk_id']}] Score: {hit['score']:.3f}")
            print(f"   Citation: {hit.get('canonical_citation', 'N/A')}")
            print(f"   Text: {hit['text'][:100]}...")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dense Retriever Training and Evaluation")
    parser.add_argument("--mode", choices=["train", "evaluate", "k_fold", "demo"], 
                       default="demo", help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_dense_retriever()
    elif args.mode == "evaluate":
        evaluate_trained_model()
    elif args.mode == "k_fold":
        k_fold_evaluate_dense_retriever()
    elif args.mode == "demo":
        demo_retriever()

if __name__ == "__main__":
    main()