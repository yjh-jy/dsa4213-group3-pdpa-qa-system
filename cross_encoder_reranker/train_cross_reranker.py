#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_cross_reranker.py — fine-tune PDPA cross-encoder reranker and save in .safetensors format
Optimized for Apple Silicon (MPS) stability and automatic input truncation.
"""

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from safetensors.torch import save_file
import json, torch, gc
from pathlib import Path
from transformers import logging as hf_logging

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "dense_training" / "stratified_splits" / "train_triples.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "cross_encoder_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 8        
EPOCHS = 2

# silence tokenizer truncation warnings
hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
print(f" Loading training data from {DATA_FILE}")
train_samples = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        q = obj["query"]
        pos = obj["pos_text"]
        neg = obj["neg_text"]
        # create 2 pairwise examples
        train_samples.append(InputExample(texts=[q, pos], label=1.0))
        train_samples.append(InputExample(texts=[q, neg], label=0.0))
print(f" Created {len(train_samples)} pairwise samples")

# ---------------------------------------------------------------------
# TRAINING SETUP
# ---------------------------------------------------------------------
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

# enforce truncation automatically
model = CrossEncoder(BASE_MODEL, num_labels=1)

# ---------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------
print(f" Fine-tuning base model: {BASE_MODEL}")

# manual cleanup before training (important for MPS)
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

model.fit(
    train_dataloader=train_dataloader,
    epochs=EPOCHS,
    warmup_steps=int(0.1 * len(train_dataloader)),
    output_path=str(OUT_DIR),
    use_amp=True
)

# ---------------------------------------------------------------------
# MANUALLY SAVE THE MODEL
# ---------------------------------------------------------------------
print(" Manually saving trained model...")
model.model.save_pretrained(OUT_DIR)
model.tokenizer.save_pretrained(OUT_DIR)
print(f" Model weights and tokenizer saved to {OUT_DIR}")

# cleanup again after training
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

print(" Training complete!")
