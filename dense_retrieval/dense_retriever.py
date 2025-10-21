#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dense_retriever.py — Dense retrieval system for PDPA corpus
- Loads pre-built dense index from dense_indexer.py
- Provides search functionality with similarity scoring
- Supports batch queries for evaluation
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    def __init__(self, index_dir: Optional[Path] = None):
        """Initialize dense retriever with pre-built index."""
        if index_dir is None:
            index_dir = Path(__file__).resolve().parents[0] / "data" / "dense" / "pdpa_v1"
        
        self.index_dir = Path(index_dir)
        self.model = None
        self.embeddings = None
        self.chunk_ids = None
        self.texts = None
        self.sections_map = None
        self.meta = None
        
        self._load_index()
    
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
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        # Load sections mapping
        sections_path = self.index_dir / "sections.map.json"
        with sections_path.open("r", encoding="utf-8") as f:
            self.sections_map = json.load(f)
        
        # Initialize model
        self.model = SentenceTransformer(self.meta["model"])
        
        print(f"Loaded dense index: {len(self.chunk_ids)} chunks, {self.meta['embedding_dim']}D embeddings")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into embedding vector."""
        return self.model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant chunks using dense retrieval.
        
        Args:
            query: Natural language question
            top_k: Number of results to return
            
        Returns:
            List of results with chunk_id, score, text, and metadata
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

def main():
    """Demo usage of dense retriever."""
    retriever = DenseRetriever()
    
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

if __name__ == "__main__":
    main()