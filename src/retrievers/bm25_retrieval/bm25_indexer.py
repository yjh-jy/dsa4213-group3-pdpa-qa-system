#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bm25_indexer.py — Build BM25 index for PDPA corpus
- Reads:  data/corpus/corpus_subsection_v1.jsonl
- Writes: data/bm25/pdpa_v1/{bm25_index.npz, meta.json, sections.map.json}
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Paths
ROOT = Path(__file__).resolve().parents[3] 
CORPUS = ROOT / "data" / "corpus" / "corpus_subsection_v1.jsonl"
OUTDIR = ROOT / "artefacts" / "bm25_index"

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep alphanumeric
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    toks = [STEMMER.stem(t) for t in toks]  
    return toks

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
    tokenized, texts, chunk_ids, doclens = [], [], [], []
    sections_map: Dict[str, Dict] = {}

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

            toks = simple_tokenize(text)
            tokenized.append(toks)
            texts.append(text)
            chunk_ids.append(chunk_id)
            doclens.append(len(toks))

            sections_map[chunk_id] = {
                "section_id": make_section_id(doc_id, part, section, subsection),
                "canonical_citation": citation,
                "doc_id": doc_id,
                "part": part,
                "section": section,
                "subsection": subsection,
            }

    bm25_blob = {
        "tokenized_corpus": np.array(tokenized, dtype=object),
        "texts": np.array(texts, dtype=object),
        "chunk_ids": np.array(chunk_ids, dtype=object),
        "doclens": np.array(doclens, dtype=np.int32),
    }
    meta = {
        "n_docs": len(texts),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": "simple_tokenize(v1)",
        "corpus_sha256": sha256_file(corpus_path),
        "version": "pdpa_v1",
    }
    return {"bm25": bm25_blob, "meta": meta}, sections_map

def main():
    """Build BM25 index for PDPA corpus."""
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")
    
    # Ensure output directory exists
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Build index
    artifacts, sections_map = build_index(CORPUS)
    
    # Save artifacts
    np.savez_compressed(OUTDIR / "bm25_index.npz", **artifacts["bm25"])
    (OUTDIR / "meta.json").write_text(
        json.dumps(artifacts["meta"], ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    (OUTDIR / "sections.map.json").write_text(
        json.dumps(sections_map, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    
    # Print summary
    print(f"BM25 index built: {artifacts['meta']['n_docs']} chunks")
    print(f"Saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
