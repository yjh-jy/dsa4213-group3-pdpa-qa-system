#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bm25_wrapper.py — Class-based BM25 retriever for programmatic evaluation
    - Loads pre-built BM25 index from: data/bm25/pdpa_v1/
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

nltk.download('stopwords', quiet=True)
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep alphanumeric
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    toks = [STEMMER.stem(t) for t in toks]  
    return toks


class BM25Retriever:
    def __init__(self, index_dir=None, k1=0.5, b=1.0):
        """Initialize BM25 retriever using prebuilt index.
        
        Args:
            index_dir: Path to index directory
            k1: BM25 k1 parameter (term frequency saturation point)
            b: BM25 b parameter (length normalization)
        """
        if index_dir is None:
            index_dir = Path(__file__).resolve().parents[0] / "indexer_results" / "pdpa_v1"
        index_dir = Path(index_dir)

        npz = np.load(index_dir / "bm25_index.npz", allow_pickle=True)
        self.tokenized = list(npz["tokenized_corpus"])
        self.texts = list(npz["texts"])
        self.chunk_ids = list(npz["chunk_ids"])
        self.sections_map = json.loads((index_dir / "sections.map.json").read_text(encoding="utf-8"))
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        # Store hyperparameters
        self.k1 = k1
        self.b = b
        self.bm25 = BM25Okapi(self.tokenized, k1=k1, b=b)

    def search(self, query, top_k=10):
        """Perform BM25 retrieval for a single query."""
        import time
        start_time = time.time()

        query_toks = simple_tokenize(query)
        scores = self.bm25.get_scores(query_toks)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        hits = []
        for rank, i in enumerate(top_indices, 1):
            chunk_id = self.chunk_ids[i]
            meta = self.sections_map.get(chunk_id, {})
            hit = {
                "rank": rank,
                "chunk_id": chunk_id,
                "bm25_score": float(scores[i]),
                "text": self.texts[i],
                **meta,
            }
            hits.append(hit)

        search_time = (time.time() - start_time) * 1000
        return {
            "results": hits,
            "query": query,
            "retriever": "bm25",
            "search_time_ms": search_time,
            "index_snapshot": self.meta.get("version", "pdpa_v1"),
        }

class BM25HyperparameterOptimizer:
    """Grid search optimization for BM25 hyperparameters."""
    
    def __init__(self, qa_dataset_path, index_dir=None):
        """Initialize optimizer with QA dataset."""
        self.qa_dataset_path = Path(qa_dataset_path)
        self.index_dir = index_dir
        self.qa_data = self._load_qa_dataset()
        
    def _load_qa_dataset(self):
        """Load QA dataset from JSONL file."""
        qa_data = []
        with self.qa_dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_data.append(json.loads(line))
        return qa_data
    
    def _extract_relevant_chunks(self, qa_item):
        """Extract ground truth relevant chunk IDs from QA item or triples format."""
        relevant_chunks = set()
        
        # Handle QA dataset format (with corpus_links)
        if "corpus_links" in qa_item:
            for link in qa_item.get("corpus_links", []):
                if "chunk_id" in link:
                    relevant_chunks.add(link["chunk_id"])
        
        # Handle triples format (with pos_id directly)
        elif "pos_id" in qa_item:
            relevant_chunks.add(qa_item["pos_id"])
        
        return relevant_chunks
    
    def _evaluate_hyperparams(self, k1, b, sample_size=100):
        """Evaluate BM25 with specific hyperparameters on a sample of queries."""
        retriever = BM25Retriever(index_dir=self.index_dir, k1=k1, b=b)
        
        # Use a sample for faster evaluation
        sample_data = self.qa_data[:sample_size] if sample_size else self.qa_data
        
        total_recall_1 = 0
        total_recall_5 = 0
        total_recall_10 = 0
        total_mrr = 0
        total_ndcg_10 = 0
        
        for qa_item in sample_data:
            # Handle both QA dataset format and triples format
            if "question_user" in qa_item:
                query = qa_item["question_user"]  # QA dataset format
            elif "query" in qa_item:
                query = qa_item["query"]  # Triples format
            else:
                continue  # Skip invalid items
                
            relevant_chunks = self._extract_relevant_chunks(qa_item)
            
            if not relevant_chunks:
                continue
                
            # Get search results
            result = retriever.search(query, top_k=10)
            retrieved_chunks = [hit["chunk_id"] for hit in result["results"]]
            
            # Calculate metrics
            total_recall_1 += self._compute_recall_at_k(retrieved_chunks, relevant_chunks, 1)
            total_recall_5 += self._compute_recall_at_k(retrieved_chunks, relevant_chunks, 5)
            total_recall_10 += self._compute_recall_at_k(retrieved_chunks, relevant_chunks, 10)
            total_mrr += self._compute_mrr(retrieved_chunks, relevant_chunks)
            total_ndcg_10 += self._compute_ndcg_at_k(retrieved_chunks, relevant_chunks, 10)
        
        n_queries = len(sample_data)
        return {
            "k1": k1,
            "b": b,
            "recall@1": total_recall_1 / n_queries,
            "recall@5": total_recall_5 / n_queries,
            "recall@10": total_recall_10 / n_queries,
            "mrr": total_mrr / n_queries,
            "ndcg@10": total_ndcg_10 / n_queries,
            "composite_score": (total_recall_10 + total_mrr + total_ndcg_10) / (3 * n_queries)
        }
    
    def _compute_recall_at_k(self, retrieved_chunks, relevant_chunks, k):
        """Compute Recall@k metric."""
        if not relevant_chunks:
            return 0.0
        retrieved_at_k = set(retrieved_chunks[:k])
        hits = len(retrieved_at_k.intersection(relevant_chunks))
        return hits / len(relevant_chunks)
    
    def _compute_mrr(self, retrieved_chunks, relevant_chunks):
        """Compute Mean Reciprocal Rank (MRR)."""
        for i, chunk_id in enumerate(retrieved_chunks, 1):
            if chunk_id in relevant_chunks:
                return 1.0 / i
        return 0.0
    
    def _compute_ndcg_at_k(self, retrieved_chunks, relevant_chunks, k):
        """Compute Normalized Discounted Cumulative Gain (NDCG@k)."""
        def dcg(relevances):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        # Actual DCG
        actual_relevances = [1 if chunk in relevant_chunks else 0 for chunk in retrieved_chunks[:k]]
        actual_dcg = dcg(actual_relevances)
        
        # Ideal DCG (perfect ranking)
        ideal_relevances = [1] * min(len(relevant_chunks), k) + [0] * max(0, k - len(relevant_chunks))
        ideal_dcg = dcg(ideal_relevances)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def grid_search(self, k1_values=None, b_values=None, sample_size=100):
        """Perform grid search over BM25 hyperparameters.
        
        Args:
            k1_values: List of k1 values to try
            b_values: List of b values to try
            sample_size: Number of queries to use for evaluation (for speed)
        
        Returns:
            List of results sorted by composite score
        """
        if k1_values is None:
            k1_values = [0.5, 0.8, 1.0, 1.2, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
        
        if b_values is None:
            b_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
        
        print(f"Starting BM25 hyperparameter grid search...")
        print(f"k1 values: {k1_values}")
        print(f"b values: {b_values}")
        print(f"Total combinations: {len(k1_values) * len(b_values)}")
        print(f"Sample size: {sample_size} queries")
        print("-" * 80)
        
        results = []
        total_combinations = len(k1_values) * len(b_values)
        
        for i, k1 in enumerate(k1_values):
            for j, b in enumerate(b_values):
                combination_num = i * len(b_values) + j + 1
                print(f"[{combination_num:2d}/{total_combinations}] Testing k1={k1:.1f}, b={b:.1f}...", end=" ")
                
                try:
                    result = self._evaluate_hyperparams(k1, b, sample_size)
                    results.append(result)
                    print(f"Score: {result['composite_score']:.4f}")
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        # Sort by composite score (descending)
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print("\n" + "="*80)
        print("BM25 HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*80)
        print(f"{'Rank':<4} {'k1':<4} {'b':<4} {'R@1':<6} {'R@5':<6} {'R@10':<6} {'MRR':<6} {'NDCG@10':<8} {'Score':<6}")
        print("-" * 80)
        
        for rank, result in enumerate(results[:10], 1):  # Show top 10
            print(f"{rank:<4} {result['k1']:<4.1f} {result['b']:<4.1f} "
                  f"{result['recall@1']:<6.3f} {result['recall@5']:<6.3f} {result['recall@10']:<6.3f} "
                  f"{result['mrr']:<6.3f} {result['ndcg@10']:<8.3f} {result['composite_score']:<6.3f}")
        
        print("\n" + "="*80)
        
        if results:
            best = results[0]
            print(f"BEST HYPERPARAMETERS:")
            print(f"   k1 = {best['k1']}")
            print(f"   b = {best['b']}")
            print(f"   Composite Score = {best['composite_score']:.4f}")
            print(f"   Recall@10 = {best['recall@10']:.3f}")
            print(f"   MRR = {best['mrr']:.3f}")
            print(f"   NDCG@10 = {best['ndcg@10']:.3f}")
        
        return results

def analyze_current_performance():
    """Analyze current BM25 performance with default hyperparameters."""
    print("=== BM25 CURRENT PERFORMANCE ANALYSIS ===")
    
    # Load QA dataset
    qa_path = Path(__file__).resolve().parents[1] / "data" / "qa" / "pdpa_qa_500.jsonl"
    
    if not qa_path.exists():
        print(f"QA dataset not found at {qa_path}")
        return
    
    # Initialize optimizer
    optimizer = BM25HyperparameterOptimizer(qa_path)
    
    # Test current default parameters
    print("Current default parameters: k1=0.5, b=1.0 (optimized)")
    current_result = optimizer._evaluate_hyperparams(k1=0.5, b=1.0, sample_size=100)
    
    print(f"Performance on 100 sample queries:")
    print(f"  Recall@1:  {current_result['recall@1']:.3f}")
    print(f"  Recall@5:  {current_result['recall@5']:.3f}")
    print(f"  Recall@10: {current_result['recall@10']:.3f}")
    print(f"  MRR:       {current_result['mrr']:.3f}")
    print(f"  NDCG@10:   {current_result['ndcg@10']:.3f}")
    print(f"  Composite: {current_result['composite_score']:.3f}")
    
    return current_result

def optimize_bm25_hyperparameters():
    """Run hyperparameter optimization for BM25."""
    print("=== BM25 HYPERPARAMETER OPTIMIZATION ===")
    
    # Load QA dataset
    qa_path = Path(__file__).resolve().parents[1] / "data" / "qa" / "pdpa_qa_500.jsonl"
    
    if not qa_path.exists():
        print(f"QA dataset not found at {qa_path}")
        return
    
    # Initialize optimizer
    optimizer = BM25HyperparameterOptimizer(qa_path)
    
    # Run grid search
    results = optimizer.grid_search(sample_size=100)
    
    return results

def main():
    """Main function with options for demo, analysis, and optimization."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_current_performance()
            return
        elif sys.argv[1] == "optimize":
            optimize_bm25_hyperparameters()
            return
    
    # Default demo
    print("=== BM25 Retriever Demo ===")
    retriever = BM25Retriever()
    
    # Example queries
    test_queries = [
        "What is the official name of the PDPA?",
        "Does the PDPA apply to individuals acting in a personal capacity?",
        "What are the consent requirements for data collection?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = retriever.search(query, top_k=3)
        print(f"Search time: {result['search_time_ms']:.2f}ms")
        
        for i, hit in enumerate(result["results"], 1):
            print(f"{i}. [{hit['chunk_id']}] Score: {hit['bm25_score']:.3f}")
            print(f"   Citation: {hit.get('canonical_citation', 'N/A')}")
            print(f"   Text: {hit['text'][:100]}...")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("  python3 bm25_retriever.py          # Demo retrieval")
    print("  python3 bm25_retriever.py analyze  # Analyze current performance")
    print("  python3 bm25_retriever.py optimize # Run hyperparameter optimization")

if __name__ == "__main__":
    main()





