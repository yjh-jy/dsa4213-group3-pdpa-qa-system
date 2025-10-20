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
ROOT = Path(__file__).resolve().parents[0]      # repo root if script is at top-level
DATA = ROOT / "data"
BM25_DIR = DATA / "bm25" / "pdpa_v1"
DENSE_DIR = DATA / "dense" / "pdpa_v1"
OUTDIR = DATA / "hybrid" / "pdpa_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Configurations ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
ALPHA = 0.5  # initial setting

# --- Load BM25 ---
def load_bm25_index() -> Tuple[BM25Okapi, List[str], Dict]:
    bm25_idx = np.load(BM25_DIR / "bm25_index.npz", allow_pickle=True)
    tokenized_corpus = bm25_idx["tokenized_corpus"]
    chunk_ids = bm25_idx["chunk_ids"]
    bm25 = BM25Okapi(tokenized_corpus.tolist())
    with open(BM25_DIR / "sections.map.json", "r", encoding="utf-8") as f:
        bm25_sections = json.load(f)
    return bm25, chunk_ids, bm25_sections

# --- Load dense embeddings ---
def load_dense_index() -> Tuple[np.ndarray, List[str], Dict]:
    dense_idx = np.load(DENSE_DIR / "embeddings.npz", allow_pickle=True)
    embeddings = dense_idx["embeddings"]
    chunk_ids = dense_idx["chunk_ids"]
    with open(DENSE_DIR / "sections.map.json", "r", encoding="utf-8") as f:
        dense_sections = json.load(f)
    return embeddings, chunk_ids, dense_sections

# --- Hybrid retriever ---
class HybridRetriever:
    def __init__(self, bm25, bm25_chunk_ids, bm25_sections, dense_emb, dense_chunk_ids, dense_sections, alpha=ALPHA):
        self.bm25 = bm25
        self.bm25_chunk_ids = bm25_chunk_ids
        self.bm25_sections = bm25_sections

        self.dense_emb = dense_emb
        self.dense_chunk_ids = dense_chunk_ids
        self.dense_sections = dense_sections

        self.alpha = alpha
        self.model = SentenceTransformer(MODEL_NAME)

    # --- BM25 retrieval ---
    def retrieve_bm25(self, query: str, top_k=20) -> Dict[str, float]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {str(self.bm25_chunk_ids[i]): float(scores[i]) for i in top_idx}

    # --- Dense retrieval ---
    def retrieve_dense(self, query: str, top_k=20) -> Dict[str, float]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.dense_emb, query_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {str(self.dense_chunk_ids[i]): float(scores[i]) for i in top_idx}

    # --- Combine scores ---
    def hybrid_retrieve(self, query: str, top_k=10) -> List[Tuple[str, float, str]]:
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
            section_info = self.dense_sections.get(cid) or self.bm25_sections.get(cid)
            section_id = section_info.get("section_id", "Unknown") if section_info else "Unknown"
            formatted.append((cid, score, section_id))

        outpath = OUTDIR / "hybrid_results.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(
                [{"chunk_id": cid, "score": score, "section_id": section_id} for cid, score, section_id in formatted],
                f, ensure_ascii=False, indent=2
            )
        print(f"\nHybrid top k results saved to: {outpath}")
        
        return formatted

def main():
    bm25, bm25_chunk_ids, bm25_sections = load_bm25_index()
    dense_emb, dense_chunk_ids, dense_sections = load_dense_index()

    retriever = HybridRetriever(bm25, bm25_chunk_ids, bm25_sections, dense_emb, dense_chunk_ids, dense_sections, alpha=ALPHA)

if __name__ == "__main__":
    main()
