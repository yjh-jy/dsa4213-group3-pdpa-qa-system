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
    def __init__(self, index_dir=None):
        """Initialize BM25 retriever using prebuilt index."""
        if index_dir is None:
            index_dir = Path(__file__).resolve().parents[1] / "data" / "bm25" / "pdpa_v1"
        index_dir = Path(index_dir)

        npz = np.load(index_dir / "bm25_index.npz", allow_pickle=True)
        self.tokenized = list(npz["tokenized_corpus"])
        self.texts = list(npz["texts"])
        self.chunk_ids = list(npz["chunk_ids"])
        self.sections_map = json.loads((index_dir / "sections.map.json").read_text(encoding="utf-8"))
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        self.bm25 = BM25Okapi(self.tokenized, k1=1.6, b=0.7)

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





