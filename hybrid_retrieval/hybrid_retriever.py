#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hybrid_retriever.py — Combine BM25 and Dense retrieval results using hybrid scoring
- Reads:
    data/bm25/pdpa_v1/{bm25_index.npz, sections.map.json}
    data/dense/pdpa_v1/{embeddings.npz, sections.map.json}
- Writes:
    data/hybrid/pdpa_v1/hybrid_topk.json 
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]      # repo root (go up one level from hybrid_retrieval/)
BM25_DIR = ROOT / "bm25_retrieval" / "indexer_results" / "pdpa_v1"
OUTDIR = Path(__file__).resolve().parents[0] / "indexer_results" / "pdpa_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Configurations ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim
ALPHA = 0.1  # optimized for linear fusion
RRF_K = 30  # optimized RRF constant
DEFAULT_FUSION = "rrf"  # best performing fusion method

# --- Load BM25 ---
def load_bm25_index() -> Tuple[BM25Okapi, List[str], Dict]:
    bm25_idx = np.load(BM25_DIR / "bm25_index.npz", allow_pickle=True)
    tokenized_corpus = bm25_idx["tokenized_corpus"]
    chunk_ids = bm25_idx["chunk_ids"]
    bm25 = BM25Okapi(tokenized_corpus.tolist())
    with open(BM25_DIR / "sections.map.json", "r", encoding="utf-8") as f:
        bm25_sections = json.load(f)
    return bm25, chunk_ids, bm25_sections

# --- Load dense retriever (using fine-tuned model) ---
def load_dense_retriever():
    """Load the fine-tuned dense retriever instead of static embeddings."""
    import sys
    sys.path.append(str(ROOT / "dense_retrieval"))
    from dense_retriever import DenseRetriever
    
    # Initialize dense retriever (will automatically use fine-tuned model if available)
    dense_retriever = DenseRetriever()
    return dense_retriever

# --- Hybrid retriever ---
class HybridRetriever:
    def __init__(self, bm25, bm25_chunk_ids, bm25_sections, dense_retriever, 
                 alpha=ALPHA, fusion_method="linear", rrf_k=RRF_K):
        """Initialize hybrid retriever with multiple fusion methods.
        
        Args:
            bm25: BM25 index
            bm25_chunk_ids: BM25 chunk IDs
            bm25_sections: BM25 sections mapping
            dense_retriever: Fine-tuned dense retriever instance
            fusion_method: "linear" for weighted combination, "rrf" for reciprocal rank fusion
            alpha: Weight for linear combination (BM25 weight, Dense weight = 1-alpha)
            rrf_k: Constant for RRF formula
        """
        self.bm25 = bm25
        self.bm25_chunk_ids = bm25_chunk_ids
        self.bm25_sections = bm25_sections

        self.dense_retriever = dense_retriever
        
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    # --- BM25 retrieval ---
    def retrieve_bm25(self, query: str, top_k=20) -> Dict[str, float]:
        """Retrieve using BM25 with proper tokenization."""
        # Use the same tokenization as in BM25 indexer
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        nltk.download('stopwords', quiet=True)
        stopwords_set = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
        # Tokenize query properly
        text = query.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        toks = [t for t in text.split() if t and t not in stopwords_set]
        tokenized_query = [stemmer.stem(t) for t in toks]
        
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {str(self.bm25_chunk_ids[i]): float(scores[i]) for i in top_idx}

    def retrieve_bm25_ranked(self, query: str, top_k=20) -> List[Tuple[str, int]]:
        """Retrieve BM25 results with ranks for RRF."""
        bm25_scores = self.retrieve_bm25(query, top_k)
        # Sort by score and return (chunk_id, rank) pairs
        sorted_results = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_id, rank + 1) for rank, (chunk_id, score) in enumerate(sorted_results)]

    # --- Dense retrieval ---
    def retrieve_dense(self, query: str, top_k=20) -> Dict[str, float]:
        """Retrieve using fine-tuned dense retriever."""
        result = self.dense_retriever.search(query, top_k=top_k)
        return {item["chunk_id"]: item["score"] for item in result["results"]}

    def retrieve_dense_ranked(self, query: str, top_k=20) -> List[Tuple[str, int]]:
        """Retrieve dense results with ranks for RRF."""
        dense_scores = self.retrieve_dense(query, top_k)
        # Sort by score and return (chunk_id, rank) pairs
        sorted_results = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_id, rank + 1) for rank, (chunk_id, score) in enumerate(sorted_results)]

    # --- Combine scores ---
    def hybrid_retrieve(self, query: str, top_k=10) -> List[Tuple[str, float, str]]:
        """Hybrid retrieval using either linear combination or RRF."""
        if self.fusion_method == "rrf":
            return self._rrf_fusion(query, top_k)
        else:
            return self._linear_fusion(query, top_k)
    
    def _linear_fusion(self, query: str, top_k=10) -> List[Tuple[str, float, str]]:
        """Linear weighted combination of BM25 and Dense scores."""
        bm25_scores = self.retrieve_bm25(query, top_k * 2)
        dense_scores = self.retrieve_dense(query, top_k * 2)

        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        combined = {}
        for cid in all_ids:
            s_bm25 = bm25_scores.get(cid, 0.0)
            s_dense = dense_scores.get(cid, 0.0)
            combined[cid] = self.alpha * s_bm25 + (1 - self.alpha) * s_dense

        top_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        formatted = []
        for cid, score in top_results:
            section_info = self.bm25_sections.get(cid)
            section_id = section_info.get("section_id", "Unknown") if section_info else "Unknown"
            formatted.append((cid, score, section_id))

        return formatted
    
    def _rrf_fusion(self, query: str, top_k=10) -> List[Tuple[str, float, str]]:
        """Reciprocal Rank Fusion of BM25 and Dense results."""
        # Get ranked results from both retrievers
        bm25_ranked = self.retrieve_bm25_ranked(query, top_k * 2)
        dense_ranked = self.retrieve_dense_ranked(query, top_k * 2)
        
        # Create rank dictionaries
        bm25_ranks = {chunk_id: rank for chunk_id, rank in bm25_ranked}
        dense_ranks = {chunk_id: rank for chunk_id, rank in dense_ranked}
        
        # Get all unique chunk IDs
        all_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_ids:
            rrf_score = 0.0
            
            # Add BM25 contribution
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id])
            
            # Add Dense contribution
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score and get top-k
        top_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        formatted = []
        for cid, score in top_results:
            section_info = self.bm25_sections.get(cid)
            section_id = section_info.get("section_id", "Unknown") if section_info else "Unknown"
            formatted.append((cid, score, section_id))

        return formatted
    
    def search(self, query: str, top_k=10):
        """Search method compatible with evaluation framework."""
        import time
        start_time = time.time()
        
        # Get hybrid results
        results = self.hybrid_retrieve(query, top_k)
        
        # Format for evaluation framework
        formatted_results = []
        for rank, (chunk_id, score, section_id) in enumerate(results, 1):
            section_info = self.bm25_sections.get(chunk_id, {})
            result = {
                "chunk_id": chunk_id,
                "score": float(score),
                "rank": rank,
                "text": section_info.get("text", ""),
            }
            # Add section metadata if available
            if section_info:
                result.update({
                    "canonical_citation": section_info.get("canonical_citation", ""),
                    "section_id": section_info.get("section_id", ""),
                    "doc_id": section_info.get("doc_id", ""),
                    "part": section_info.get("part", ""),
                    "section": section_info.get("section", ""),
                    "subsection": section_info.get("subsection", "")
                })
            formatted_results.append(result)
        
        search_time = time.time() - start_time
        
        return {
            "results": formatted_results,
            "query": query,
            "total_results": len(formatted_results),
            "search_time_ms": search_time * 1000,
            "model": "hybrid_bm25_dense"
        }

class HybridHyperparameterOptimizer:
    """Optimization for hybrid retriever hyperparameters."""
    
    def __init__(self, qa_dataset_path):
        """Initialize optimizer with QA dataset."""
        self.qa_dataset_path = Path(qa_dataset_path)
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
        """Extract ground truth relevant chunk IDs."""
        relevant_chunks = set()
        for link in qa_item.get("corpus_links", []):
            if "chunk_id" in link:
                relevant_chunks.add(link["chunk_id"])
        return relevant_chunks
    
    def _evaluate_hyperparams(self, alpha=None, fusion_method="linear", rrf_k=60, sample_size=100):
        """Evaluate hybrid retriever with specific hyperparameters."""
        if fusion_method == "linear" and alpha is None:
            alpha = 0.5
        
        retriever = create_hybrid_retriever(alpha=alpha, fusion_method=fusion_method, rrf_k=rrf_k)
        
        # Use a sample for faster evaluation
        sample_data = self.qa_data[:sample_size] if sample_size else self.qa_data
        
        total_recall_1 = 0
        total_recall_5 = 0
        total_recall_10 = 0
        total_mrr = 0
        total_ndcg_10 = 0
        
        for qa_item in sample_data:
            query = qa_item["question_user"]
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
            "alpha": alpha,
            "fusion_method": fusion_method,
            "rrf_k": rrf_k,
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
    
    def optimize_alpha(self, alpha_values=None, sample_size=100):
        """Optimize alpha parameter for linear fusion."""
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print(f"Optimizing alpha parameter for linear fusion...")
        print(f"Alpha values: {alpha_values}")
        print(f"Sample size: {sample_size} queries")
        print("-" * 60)
        
        results = []
        for i, alpha in enumerate(alpha_values):
            print(f"[{i+1:2d}/{len(alpha_values)}] Testing alpha={alpha:.1f}...", end=" ")
            
            try:
                result = self._evaluate_hyperparams(alpha=alpha, fusion_method="linear", sample_size=sample_size)
                results.append(result)
                print(f"Score: {result['composite_score']:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print("\n" + "="*60)
        print("ALPHA OPTIMIZATION RESULTS")
        print("="*60)
        print(f"{'Rank':<4} {'Alpha':<6} {'R@1':<6} {'R@5':<6} {'R@10':<6} {'MRR':<6} {'NDCG@10':<8} {'Score':<6}")
        print("-" * 60)
        
        for rank, result in enumerate(results[:5], 1):  # Show top 5
            print(f"{rank:<4} {result['alpha']:<6.1f} "
                  f"{result['recall@1']:<6.3f} {result['recall@5']:<6.3f} {result['recall@10']:<6.3f} "
                  f"{result['mrr']:<6.3f} {result['ndcg@10']:<8.3f} {result['composite_score']:<6.3f}")
        
        return results
    
    def compare_fusion_methods(self, sample_size=100):
        """Compare linear fusion vs RRF."""
        print(f"Comparing fusion methods...")
        print(f"Sample size: {sample_size} queries")
        print("-" * 60)
        
        results = []
        
        # Test linear fusion with best alpha (if known) or default
        print("Testing Linear Fusion (alpha=0.5)...", end=" ")
        linear_result = self._evaluate_hyperparams(alpha=0.5, fusion_method="linear", sample_size=sample_size)
        linear_result["method_name"] = "Linear (α=0.5)"
        results.append(linear_result)
        print(f"Score: {linear_result['composite_score']:.4f}")
        
        # Test RRF with different k values
        rrf_k_values = [30, 60, 100]
        for k in rrf_k_values:
            print(f"Testing RRF (k={k})...", end=" ")
            rrf_result = self._evaluate_hyperparams(fusion_method="rrf", rrf_k=k, sample_size=sample_size)
            rrf_result["method_name"] = f"RRF (k={k})"
            results.append(rrf_result)
            print(f"Score: {rrf_result['composite_score']:.4f}")
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print("\n" + "="*70)
        print("FUSION METHOD COMPARISON")
        print("="*70)
        print(f"{'Rank':<4} {'Method':<12} {'R@1':<6} {'R@5':<6} {'R@10':<6} {'MRR':<6} {'NDCG@10':<8} {'Score':<6}")
        print("-" * 70)
        
        for rank, result in enumerate(results, 1):
            print(f"{rank:<4} {result['method_name']:<12} "
                  f"{result['recall@1']:<6.3f} {result['recall@5']:<6.3f} {result['recall@10']:<6.3f} "
                  f"{result['mrr']:<6.3f} {result['ndcg@10']:<8.3f} {result['composite_score']:<6.3f}")
        
        return results

# Convenience function for easy initialization
def create_hybrid_retriever(alpha=ALPHA, fusion_method=DEFAULT_FUSION, rrf_k=RRF_K, dense_retriever=None):
    """Create a HybridRetriever instance with loaded indices.
    
    Args:
        alpha: Weight for linear combination
        fusion_method: "linear" or "rrf"
        rrf_k: RRF constant
        dense_retriever: Pre-trained DenseRetriever instance (optional)
    """
    try:
        bm25, bm25_chunk_ids, bm25_sections = load_bm25_index()
        
        # Use provided dense retriever or load default one
        if dense_retriever is None:
            dense_retriever = load_dense_retriever()
        
        return HybridRetriever(bm25, bm25_chunk_ids, bm25_sections, dense_retriever, 
                             alpha=alpha, fusion_method=fusion_method, rrf_k=rrf_k)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize hybrid retriever: {e}")

def analyze_current_performance():
    """Analyze current hybrid performance."""
    print("=== HYBRID RETRIEVER CURRENT PERFORMANCE ANALYSIS ===")
    
    # Load QA dataset
    qa_path = Path(__file__).resolve().parents[1] / "data" / "qa" / "pdpa_qa_500.jsonl"
    
    if not qa_path.exists():
        print(f"QA dataset not found at {qa_path}")
        return
    
    # Initialize optimizer
    optimizer = HybridHyperparameterOptimizer(qa_path)
    
    # Test current default parameters
    print("Current default: RRF fusion with k=30 (optimized)")
    current_result = optimizer._evaluate_hyperparams(fusion_method="rrf", rrf_k=30, sample_size=100)
    
    print(f"Performance on 100 sample queries:")
    print(f"  Recall@1:  {current_result['recall@1']:.3f}")
    print(f"  Recall@5:  {current_result['recall@5']:.3f}")
    print(f"  Recall@10: {current_result['recall@10']:.3f}")
    print(f"  MRR:       {current_result['mrr']:.3f}")
    print(f"  NDCG@10:   {current_result['ndcg@10']:.3f}")
    print(f"  Composite: {current_result['composite_score']:.3f}")
    
    return current_result

def optimize_hybrid_parameters():
    """Run comprehensive hybrid optimization."""
    print("=== HYBRID RETRIEVER OPTIMIZATION ===")
    
    # Load QA dataset
    qa_path = Path(__file__).resolve().parents[1] / "data" / "qa" / "pdpa_qa_500.jsonl"
    
    if not qa_path.exists():
        print(f"QA dataset not found at {qa_path}")
        return
    
    # Initialize optimizer
    optimizer = HybridHyperparameterOptimizer(qa_path)
    
    # First optimize alpha for linear fusion
    print("\n1. OPTIMIZING ALPHA FOR LINEAR FUSION")
    alpha_results = optimizer.optimize_alpha(sample_size=100)
    
    # Then compare fusion methods
    print("\n2. COMPARING FUSION METHODS")
    fusion_results = optimizer.compare_fusion_methods(sample_size=100)
    
    # Show best overall result
    if fusion_results:
        best = fusion_results[0]
        print(f"\nBEST HYBRID CONFIGURATION:")
        print(f"   Method: {best['method_name']}")
        if best['fusion_method'] == 'linear':
            print(f"   Alpha: {best['alpha']}")
        else:
            print(f"   RRF k: {best['rrf_k']}")
        print(f"   Composite Score: {best['composite_score']:.4f}")
        print(f"   Recall@10: {best['recall@10']:.3f}")
        print(f"   MRR: {best['mrr']:.3f}")
        print(f"   NDCG@10: {best['ndcg@10']:.3f}")
    
    return fusion_results

def main():
    """Main function with options for demo, analysis, and optimization."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_current_performance()
            return
        elif sys.argv[1] == "optimize":
            optimize_hybrid_parameters()
            return
    
    # Default demo
    print("=== Hybrid Retriever Demo ===")
    retriever = create_hybrid_retriever()
    
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
            print(f"{i}. [{hit['chunk_id']}] Score: {hit['score']:.3f}")
            print(f"   Citation: {hit.get('canonical_citation', 'N/A')}")
            print(f"   Text: {hit.get('text', '')[:100]}...")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("  python3 hybrid_retriever.py          # Demo retrieval")
    print("  python3 hybrid_retriever.py analyze  # Analyze current performance")
    print("  python3 hybrid_retriever.py optimize # Run hyperparameter optimization")

if __name__ == "__main__":
    main()
