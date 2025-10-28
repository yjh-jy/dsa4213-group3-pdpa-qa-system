#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dense_indexer.py — Build dense embedding index for PDPA corpus with FAISS support
- Reads:  data/corpus/corpus_subsection_v1.jsonl
- Writes: dense_retrieval/indexer_results/pdpa_v1/{embeddings.npz, faiss_index.bin, meta.json, sections.map.json}
- Supports both base models and fine-tuned checkpoints
- Uses FAISS for efficient similarity search
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim embeddings (better performance)

# Paths
ROOT = Path(__file__).resolve().parents[1]  # repo root (go up one level from dense_retrieval/)
CORPUS = ROOT / "data" / "corpus" / "corpus_subsection_v1.jsonl"
OUTDIR = Path(__file__).resolve().parents[0] / "indexer_results" / "pdpa_v1"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def make_section_id(doc_id: str, part: str, section: str, subsection) -> str:
    base = f"{doc_id} > Part {part} > Section {section}"
    return f"{base} > Subsection {subsection}" if subsection not in (None, "", "0") else base

def build_index(corpus_path: Path) -> Tuple[Dict, Dict, faiss.Index]:
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
                "text": text  # Include text in sections_map for retrieval
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

    # Create FAISS index
    print(f"Building FAISS index...")
    embedding_dim = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings)
    
    print(f"FAISS index built with {faiss_index.ntotal} vectors")

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
        "faiss_index_type": "IndexFlatIP",
    }
    return {"dense": dense_blob, "meta": meta}, sections_map, faiss_index

def main():
    """Build dense index for PDPA corpus with FAISS support."""
    import sys
    
    # Check for custom model path (including fine-tuned checkpoints)
    global MODEL_NAME
    model_name = MODEL_NAME
    if len(sys.argv) > 1:
        custom_model_path = Path(sys.argv[1])
        if custom_model_path.exists():
            model_name = str(custom_model_path)
            print(f"Using custom trained model: {model_name}")
        else:
            # Check for fine-tuned model in standard location
            fine_tuned_path = Path(__file__).resolve().parents[0] / "fine_tuned_model"
            if fine_tuned_path.exists():
                model_name = str(fine_tuned_path)
                print(f"Using fine-tuned model: {model_name}")
            else:
                print(f"Custom model path not found: {custom_model_path}")
                print(f"Using default model: {MODEL_NAME}")
    else:
        # Check for fine-tuned model in standard location
        fine_tuned_path = Path(__file__).resolve().parents[0] / "fine_tuned_model"
        if fine_tuned_path.exists():
            model_name = str(fine_tuned_path)
            print(f"Using fine-tuned model: {model_name}")
    
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")
    
    # Ensure output directory exists
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Build index with specified model
    print(f"Building index with model: {model_name}")
    
    # Temporarily update the global MODEL_NAME for build_index
    original_model = MODEL_NAME
    MODEL_NAME = model_name
    
    try:
        artifacts, sections_map, faiss_index = build_index(CORPUS)
    finally:
        MODEL_NAME = original_model
    
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
    
    # Save FAISS index
    faiss_index_path = OUTDIR / "faiss_index.bin"
    faiss.write_index(faiss_index, str(faiss_index_path))
    print(f"FAISS index saved to: {faiss_index_path}")
    
    # Print summary
    print(f"Dense index built: {artifacts['meta']['n_docs']} chunks")
    print(f"Embedding dimension: {artifacts['meta']['embedding_dim']}")
    print(f"Model used: {artifacts['meta']['model']}")
    print(f"FAISS index type: {artifacts['meta']['faiss_index_type']}")
    print(f"Saved to: {OUTDIR}")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("  python3 dense_indexer.py                    # Use default/fine-tuned model")
    print("  python3 dense_indexer.py /path/to/model     # Use custom trained model")
    print("  python3 dense_indexer.py /path/to/checkpoint # Use specific checkpoint")

if __name__ == "__main__":
    main()