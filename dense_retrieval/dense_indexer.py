#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dense_indexer.py — Build dense embedding index for PDPA corpus
- Reads:  data/corpus/corpus_subsection_v1.jsonl
- Writes: data/dense/pdpa_v1/{embeddings.npz, meta.json, sections.map.json}
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings

# Paths
ROOT = Path(__file__).resolve().parents[1]  # repo root (go up one level from dense_retrieval/)
CORPUS = ROOT / "data" / "corpus" / "corpus_subsection_v1.jsonl"
OUTDIR = Path(__file__).resolve().parents[0] / "data" / "dense" / "pdpa_v1"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def make_section_id(doc_id: str, part: str, section: str, subsection) -> str:
    base = f"{doc_id} > Part {part} > Section {section}"
    return f"{base} > Subsection {subsection}" if subsection not in (None, "", "0") else base

def build_index(corpus_path: Path) -> Tuple[Dict, Dict]:
    texts, chunk_ids = [], []
    sections_map: Dict[str, Dict] = {}

    # Load corpus and collect texts
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunk_id = obj["chunk_id"]
            doc_id   = obj["doc_id"]
            part     = str(obj.get("part", ""))
            section  = str(obj.get("section", ""))
            subsection = obj.get("subsection")
            citation = obj.get("canonical_citation", "")
            text     = obj.get("text", "")

            texts.append(text)
            chunk_ids.append(chunk_id)

            sections_map[chunk_id] = {
                "section_id": make_section_id(doc_id, part, section, subsection),
                "canonical_citation": citation,
                "doc_id": doc_id,
                "part": part,
                "section": section,
                "subsection": subsection,
            }

    # Load model and encode texts
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Encoding {len(texts)} chunks...")
    # Normalize embeddings for cosine similarity w dot product
    embeddings = model.encode(
        texts, 
        batch_size=64, 
        show_progress_bar=True, 
        normalize_embeddings=True
    ).astype(np.float32)

    dense_blob = {
        "embeddings": embeddings,
        "texts": np.array(texts, dtype=object),
        "chunk_ids": np.array(chunk_ids, dtype=object),
        "embedding_dim": embeddings.shape[1],
    }
    meta = {
        "n_docs": len(texts),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "corpus_sha256": sha256_file(corpus_path),
        "version": "pdpa_v1",
    }
    return {"dense": dense_blob, "meta": meta}, sections_map

def main():
    """Build dense embedding index for PDPA corpus."""
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")
    
    # Ensure output directory exists
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Build index
    artifacts, sections_map = build_index(CORPUS)
    
    # Save artifacts
    np.savez_compressed(OUTDIR / "embeddings.npz", **artifacts["dense"])
    (OUTDIR / "meta.json").write_text(
        json.dumps(artifacts["meta"], ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    (OUTDIR / "sections.map.json").write_text(
        json.dumps(sections_map, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    
    # Print summary
    print(f"Dense index built: {artifacts['meta']['n_docs']} chunks")
    print(f"Embedding dimension: {artifacts['meta']['embedding_dim']}")
    print(f"Saved to: {OUTDIR}")

if __name__ == "__main__":
    main()