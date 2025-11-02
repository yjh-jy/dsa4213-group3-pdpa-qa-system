#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cross_encoder_reranker.py — inference wrapper for PDPA Cross-Encoder reranker
Assumes a fine-tuned model exists locally.
"""

from sentence_transformers import CrossEncoder
import torch, time
from typing import List, Dict
from pathlib import Path


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, device: str | None = None):
        """
        Args:
            model_name_or_path: local path to fine-tuned model
            device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        self.model_path = Path(__file__).resolve().parents[3] / str(model_name_or_path)

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = CrossEncoder(self.model_path, device=self.device)
        print(f"Loaded cross-encoder reranker from '{self.model_path}' on {self.device}")

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank candidate chunks by relevance to the query."""
        if not candidates:
            print("No candidates provided for reranking.")
            return []

        start = time.time()
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for i, c in enumerate(candidates):
            c["ce_score"] = float(scores[i])
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)[:top_k]

        for rank, item in enumerate(reranked, 1):
            item["rank"] = rank

        elapsed = (time.time() - start) * 1000
        print(f" Reranked {len(candidates)} candidates → top {top_k} in {elapsed:.1f} ms")
        return reranked


# ---------------------------------------------------------------------
# Demo usage (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    query = "Does the PDPA apply to individuals acting in a personal capacity?"
    candidates = [{"text": "The PDPA applies to organizations...", "chunk_id": "001"},
                  {"text": "This law concerns personal data breaches...", "chunk_id": "002"}]

    reranker = CrossEncoderReranker("cross_encoder_reranker/cross_encoder_model")
    results = reranker.rerank(query, candidates, top_k=2)

    print("\n=== Demo Results ===")
    for r in results:
        preview = r["text"][:120].replace("\n", " ")
        print(f"[{r['rank']}] Score = {r['ce_score']:.4f} → {preview}…")
