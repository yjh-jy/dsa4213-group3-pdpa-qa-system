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
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    def __init__(self, index_dir: Optional[Path] = None, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize dense retriever with pre-built index or for training.
        
        Args:
            index_dir: Directory containing pre-built index (optional for training mode)
            model_name: Model name for training or inference
        """
        if index_dir is None:
            index_dir = Path(__file__).resolve().parents[0] / "indexer_results" / "pdpa_v1"
        
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.chunk_ids = None
        self.texts = None
        self.sections_map = None
        self.meta = None
        self.device = self._get_device()
        
        # Try to load index, but don't fail if not available (training mode)
        try:
            self._load_index()
        except FileNotFoundError as e:
            print(f"Index not found: {e}. Running in training mode.")
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def _get_device(self) -> str:
        """Get the best available device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_index(self):
        """Load the pre-built dense index and metadata."""
        # Load embeddings and data
        embeddings_path = self.index_dir / "embeddings.npz"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Dense index not found at {embeddings_path}")
        
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.chunk_ids = data["chunk_ids"]
        self.texts = data["texts"]
        
        # Load metadata
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        # Load sections mapping
        sections_path = self.index_dir / "sections.map.json"
        if not sections_path.exists():
            raise FileNotFoundError(f"Sections mapping not found at {sections_path}")
        with sections_path.open("r", encoding="utf-8") as f:
            self.sections_map = json.load(f)
        
        # Initialize model
        model_name = self.meta.get("model", self.model_name)
        
        # Check if it's a local path relative to this directory
        if not model_name.startswith("sentence-transformers/") and not model_name.startswith("/"):
            # Try relative to the dense_retrieval directory (same level as indexer_results)
            local_model_path = self.index_dir.parent.parent / model_name
            if local_model_path.exists():
                model_name = str(local_model_path)
                print(f"Loading fine-tuned model from: {model_name}")
            else:
                print(f"Fine-tuned model not found at {local_model_path}, using default: {self.model_name}")
                model_name = self.model_name
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        print(f"Loaded dense index: {len(self.chunk_ids)} chunks, {self.meta['embedding_dim']}D embeddings")
        print(f"Using device: {self.device}")
    
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
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
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
            "model": self.meta["model"]
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
                  warmup_steps: int = 100) -> str:
        """
        Fine-tune the dense retriever on PDPA data.
        
        Args:
            training_data_path: Path to training triples JSONL file
            output_dir: Directory to save the fine-tuned model
            test_queries_path: Path to test queries for evaluation (optional)
            corpus_path: Path to corpus for evaluation (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            
        Returns:
            Path to the saved fine-tuned model
        """
        print(f"Starting fine-tuning on device: {self.device}")
        print(f"Training parameters:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Warmup steps: {warmup_steps}")
        
        # Load training data
        train_examples = self.load_training_data(training_data_path)
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
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
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
            warmup_steps=warmup_steps,
            output_path=str(output_dir),
            save_best_model=True,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )
        
        print(f"Fine-tuning completed!")
        print(f"Model saved to: {output_dir}")
        
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

def train_dense_retriever():
    """Train a dense retriever on PDPA data."""
    print("=== Dense Retriever Training ===")
    
    # Paths
    root_dir = Path(__file__).resolve().parents[1]
    training_data_dir = root_dir / "data" / "dense_training"
    
    # Check if training data exists
    full_train_path = training_data_dir / "full_train_triples.jsonl"
    if not full_train_path.exists():
        print(f"Training data not found at {full_train_path}")
        print("Please run dense_chunk_and_extract.py first to generate training data.")
        return
    
    # Initialize retriever in training mode
    retriever = DenseRetriever(index_dir=None, model_name="sentence-transformers/all-mpnet-base-v2")
    
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
    root_dir = Path(__file__).resolve().parents[1]
    model_dir = Path(__file__).resolve().parents[0] / "fine_tuned_model"
    training_data_dir = root_dir / "data" / "dense_training"
    
    if not model_dir.exists():
        print(f"Trained model not found at {model_dir}")
        print("Please run training first.")
        return
    
    # Load trained model
    retriever = DenseRetriever(index_dir=None, model_name=str(model_dir))
    
    # Evaluate on test data
    test_queries_path = training_data_dir / "fold_1" / "test_queries.jsonl"
    corpus_path = training_data_dir / "corpus.jsonl"
    
    if test_queries_path.exists() and corpus_path.exists():
        results = retriever.evaluate_model(test_queries_path, corpus_path)
    else:
        print("Test data not found. Please generate training data first.")

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
    """Main function with training and evaluation options."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_dense_retriever()
            return
        elif sys.argv[1] == "evaluate":
            evaluate_trained_model()
            return
    
    # Default demo
    demo_retriever()
    
    print("\n" + "="*60)
    print("USAGE:")
    print("  python3 dense_retriever.py          # Demo retrieval")
    print("  python3 dense_retriever.py train    # Train model on PDPA data")
    print("  python3 dense_retriever.py evaluate # Evaluate trained model")

if __name__ == "__main__":
    main()