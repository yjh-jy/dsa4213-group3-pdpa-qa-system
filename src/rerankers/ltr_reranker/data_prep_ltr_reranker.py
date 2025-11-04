#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_prep_ltr_reranker.py — Combine BM25 and Dense retrieval results and metadata for training a LightGBM LambdaRank model
- Reads:
    artefacts/bm25_index/{bm25_index.npz, sections.map.json}
    data/corpus/corpus_subsection_v1.jsonl
    src/retrievers/dense_retrieval/dense_retriever.py
    data/dense_training/stratified_splits/{train_triples.jsonl, val_triples.jsonl, test_triples.jsonl}
- Writes:
    data/ltr_processed/{train_ltr_data.jsonl, val_ltr_data.jsonl, test_ltr_data.jsonl}
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

# --- Paths ---
ROOT = Path(__file__).resolve().parents[3]      # repo root 
BM25_DIR = ROOT / "artefacts" / "bm25_index" 
CORPUS_PATH = ROOT / "data" / "corpus" / "corpus_subsection_v1.jsonl"
DENSE_DIR = ROOT / "src" / "retrievers" 
TRAIN_FILE = ROOT / "data" / "dense_training" / "stratified_splits" / "train_triples.jsonl" 
VAL_FILE = ROOT / "data" / "dense_training" / "stratified_splits" / "val_triples.jsonl" 
TEST_FILE = ROOT / "data" / "dense_training" / "stratified_splits" / "test_triples.jsonl" 
OUTDIR = ROOT / "data" / "ltr_processed" 
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Configurations ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim

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
    import sys
    sys.path.append(str(DENSE_DIR / "dense_retrieval"))
    from dense_retriever import DenseRetriever
    return DenseRetriever()

# --- Load main corpus ---
def load_corpus():
    chunk_ids = []
    texts = []
    sections_map = {}
    
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]
                
                chunk_ids.append(chunk_id)
                texts.append(text)
                
                # Store section metadata
                sections_map[chunk_id] = {
                    "canonical_citation": chunk.get("canonical_citation", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "part": chunk.get("part", ""),
                    "section": chunk.get("section", ""),
                    "subsection": chunk.get("subsection", ""),
                    "text": text
                }
    
    print(f"Loaded corpus: {len(chunk_ids)} chunks")
    return sections_map

# --- Main retriever ---
class MainRetriever:
    def __init__(self, bm25, bm25_chunk_ids, bm25_sections, dense_retriever, corpus):
        self.bm25 = bm25
        self.bm25_chunk_ids = bm25_chunk_ids
        self.bm25_sections = bm25_sections
        self.dense_retriever = dense_retriever
        self.corpus = corpus

    def retrieve_bm25(self, query: str, top_k=20) -> Dict[str, float]:
        import nltk, re
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        nltk.download('stopwords', quiet=True)
        stopwords_set = set(stopwords.words('english'))
        stemmer = PorterStemmer()

        toks = [stemmer.stem(t) for t in re.sub(r"[^a-z0-9\s]", " ", query.lower()).split() if t not in stopwords_set]
        scores = self.bm25.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {str(self.bm25_chunk_ids[i]): float(scores[i]) for i in top_idx}

    def retrieve_dense(self, query: str, top_k=20) -> Dict[str, float]:
        result = self.dense_retriever.search(query, top_k=top_k)
        return {item["chunk_id"]: item["score"] for item in result["results"]}

    def extract_features(self, query: str, top_k=50) -> pd.DataFrame:
        bm25_scores = self.retrieve_bm25(query, top_k*2)
        dense_scores = self.retrieve_dense(query, top_k*2)
        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        records = []
        for cid in all_ids:
            section_info = self.corpus.get(cid, {})
            records.append({
                "query": query,
                "chunk_id": cid,
                "bm25_score": bm25_scores.get(cid, 0.0),
                "dense_score": dense_scores.get(cid, 0.0),
                "text": section_info.get("text", ""),
                "canonical_citation": section_info.get("canonical_citation", ""),
                "doc_id": section_info.get("doc_id", ""),
                "part": section_info.get("part", ""),
                "section": section_info.get("section", ""),
                "subsection": section_info.get("subsection", ""),
                "section_len": len(section_info.get("text", "").split())
            })
        return pd.DataFrame(records)

if __name__ == "__main__":
    bm25, bm25_chunk_ids, bm25_sections = load_bm25_index()
    dense_retriever = load_dense_retriever()

    corpus = load_corpus()
    retriever = MainRetriever(bm25, bm25_chunk_ids, bm25_sections, dense_retriever, corpus)

    # Iterate through train, val & test 
    data_files = [(TRAIN_FILE, "train_ltr_data.jsonl"), (VAL_FILE, "val_ltr_data.jsonl"), (TEST_FILE, "test_ltr_data.jsonl")]

    for data_path, out_filename in data_files:
        features_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                query_text = row["query"]
                pos_id = row.get("pos_id")
                neg_id = row.get("neg_id")
                qid = row.get("qid", "")

                # --- Get retrieval results ---
                df_features = retriever.extract_features(query_text, top_k=20)

                # --- Add relevance labels - 0 (negative), 1 (neutral), 2 (positive) ---
                def relevance_label(cid):
                    if cid == pos_id:
                        return 2   
                    if cid == neg_id:
                        return 0   
                    return 1      

                df_features["label"] = df_features["chunk_id"].apply(relevance_label)

                # Add positive/negative flags
                df_features["is_pos"] = (df_features["chunk_id"] == pos_id).astype(int)
                df_features["is_neg"] = (df_features["chunk_id"] == neg_id).astype(int)

                # Add other metadata 
                df_features["qid"] = qid
                df_features["pos_id"] = pos_id
                df_features["neg_id"] = neg_id
                df_features["pos_citation"] = row.get("pos_citation", "")
                df_features["neg_citation"] = row.get("neg_citation", "")

                # Add instance weights to penalise negatives more and reward positives more 
                df_features["weight"] = df_features["label"].map({2: 2.0, 1: 1.0, 0: 2.0})

                features_list.append(df_features)

        final_df = pd.concat(features_list, ignore_index=True)
        out_file = OUTDIR / out_filename
        final_df.to_json(out_file, orient="records", lines=True)
        print(f"Saved LTR features to {out_file}")