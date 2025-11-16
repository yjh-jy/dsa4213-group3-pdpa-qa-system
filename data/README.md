# PDPABench and PDPA Corpus — Dataset Documentation

This subdirectory contains all dataset components used in our PDPA-focused citation-grounded RAG system and the accompanying paper. It includes the authoritative PDPA corpus, dense retriever training data, learning-to-rank data, and the full 500-sample PDPABench benchmark.

The goal of this dataset is to support **transparent**, **reproducible**, and **evidence-grounded** legal QA for Singapore’s Personal Data Protection Act (PDPA).

---

## 1. Directory Structure

```

data/
├── corpus/
│   ├── raw_pdpa.txt
│   ├── raw_pdpa.docx
│   └── corpus_subsection_v1.jsonl  # main corpus
│
├── dense_training/
│   ├── fold_1/ .. fold_5/          # depreciated old k-fold evaluation, kept for completeness but not used for any evaluations/ablation
│   ├── stratified_splits/
│   ├── corpus.jsonl
│   └── full_train_triples.jsonl
│
├── ltr_processed/
│   ├── train_ltr_data.jsonl
│   ├── val_ltr_data.jsonl
│   └── test_ltr_data.jsonl
│
└── qa/
    ├── prompts/
    ├── pdpa_golden30_seed.jsonl
    ├── manifest.json
    ├── schema_card.md
    ├── authoring_checklist_500.md
    └── pdpa_qa_500.jsonl

```

---

## 2. PDPA Corpus (`data/corpus/`)

### 2.1 Source
- Derived from the **Personal Data Protection Act 2012 (rev. 2020)** available on Singapore Statutes Online.
- Licensed under the **Open Data Licence**.

### 2.2 Chunking Strategy
The statute is parsed into:
- **Parts**
- **Sections**
- **Subsections**

Each chunk is assigned a **canonical citation**, e.g.:

```
PDPA s.4(1)
PDPA s.26(2)(b)
```

Stored in `corpus_subsection_v1.jsonl`, each entry has:
- `chunk_id`
- `canonical_section`
- `text`
- `part`, `section`, `subsection`, etc.

This ensures deterministic retrieval alignment.

---

## 3. Dense Retriever Training Data (`data/dense_training/`)

### 3.1 Composition
- `full_train_triples.jsonl` containing:
  - `query`
  - `pos_id` positive chunk id
  - `pos_citation` positive chunk's canonical citation
  - `neg_id` negative chunk id
  - `neg_citation`positive chunk's canonical citation

### 3.2 Motivation
Dense retriever training uses **contrastive learning** (`MultipleNegativesRankingLoss`) with **section-disjoint** splits, preventing models from memorising statute spans.

### 3.3 Stratification
Stored under `stratified_splits/`:
- Balanced across QA type,
- Disjoint across PDPA sections,
- Ensures fair generalisation testing.

---

## 4. Learning-to-Rank Data (`data/ltr_processed/`)

These datasets are used to train and evaluate LTR rerankers.

Each JSONL entry includes:
- `query`
- `chunk_id`
- `bm25_score`, `dense_score` model-generated scores (BM25, dense)
- `label` (0/1 relevance)
- other features

This enables training of LTR models such as LambdaMART or transformer rerankers.

---

## 5. PDPABench — 500-Sample PDPA QA Benchmark (`data/qa/`)

### 5.1 Contents
- `pdpa_qa_500.jsonl` — full benchmark  
- `pdpa_golden30_seed.jsonl` — manually validated seed set  
- `manifest.json` — metadata, versioning  
- `schema_card.md` — JSONL field descriptions  
- `authoring_checklist_500.md` — QA creation guidelines  
- `prompts/` — generation templates

### 5.2 Schema Summary
Each entry includes:

```

{
"id": "..."
"question_user": "...",
"gold_answer_short": "...",
"canonical_sections": ["PDPA s.xx(xx)"],
"qa_type": "...",
"difficulty": "...",
...
}

```
The complete schema with full descriptions can be found under `qa/schema_card.md`

### 5.3 QA Types
- **Pure-Definitive**  
- **Definitive-With-Condition**  
- **Scenario-Ambiguous**  
- **Pure-Abstain** (insufficient statutory grounding)

### 5.4 Splitting Protocol
- 80% train  
- 10% validation  
- 10% test  
- Strict **section-disjointness** ensures no leakage of statutory fragments.

---

## 6. Licensing

- PDPA text:  
  © Government of Singapore, reproduced from Singapore Statutes Online under the Open Data Licence.

- All annotations, QA pairs, schema, and benchmark content:  
  © Authors, released under the project’s root LICENSE.

---

## 7. Citation

If you use this dataset or benchmark:

```

@inproceedings{2025pdpa_rag,
title     = {Citation-Constrained, Abstention-Capable RAG for Singapore’s PDPA:
From Corpus Construction to Reliable Legal QA},
author    = {Ashley Toh Ke Wei and Choy Qi Hui and Sybella Tan and Yoong Jun Han},
institution = {National University of Singapore},
year      = {2025}
}

```

---

## 8. Contact

For questions regarding PDPABench or the PDPA corpus, please open an issue on the repository or contact the authors.