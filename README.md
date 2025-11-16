# Citation-Constrained, Abstention-Capable RAG for Singapore’s PDPA

**Authors:** Ashley Toh Ke Wei, Choy Qi Hui, Sybella Tan, Yoong Jun Han (Group 3).  
**Institution:** National University of Singapore  
**Release Date:** November 2025  

This repository contains the full implementation accompanying the report:  
**“Citation-Constrained, Abstention-Capable RAG for Singapore’s PDPA:  
From Corpus Construction to Reliable Legal QA”**.

Our work contributes the first **reproducible, statute-grounded** retrieval-augmented generation (RAG) system and evaluation benchmark dedicated to Singapore’s **Personal Data Protection Act (PDPA)**. The system produces **evidence-backed**, **canonically cited**, and **selectively abstaining** answers to natural-language questions grounded strictly in authoritative PDPA statute text.

## Table of Contents
- [Citation-Constrained, Abstention-Capable RAG for Singapore’s PDPA](#citation-constrained-abstention-capable-rag-for-singapores-pdpa)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Contributions](#key-contributions)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
  - [Setup (Model Artefacts)](#setup-model-artefacts)
  - [Usage](#usage)
    - [1. Start the PDPA RAG API Service](#1-start-the-pdpa-rag-api-service)
    - [2. Interactive API Documentation (FastAPI)](#2-interactive-api-documentation-fastapi)
    - [3. Query the API via cURL (via command line)](#3-query-the-api-via-curl-via-command-line)
      - [`/ask` — Full RAG pipeline](#ask--full-rag-pipeline)
      - [`/ask_no_rag` — SLM-only baseline](#ask_no_rag--slm-only-baseline)
      - [`/evaluate` — Batch evaluate against PDPABench-Test](#evaluate--batch-evaluate-against-pdpabench-test)
  - [PDPABench and PDPA Corpus (Data Overview)](#pdpabench-and-pdpa-corpus-data-overview)
    - [PDPA Corpus](#pdpa-corpus)
    - [Dense Retriever Training Data](#dense-retriever-training-data)
    - [Learning-to-Rank (LTR) Data](#learning-to-rank-ltr-data)
    - [PDPABench (500-sample Legal QA Benchmark)](#pdpabench-500-sample-legal-qa-benchmark)
  - [Results (PDPABench-Test)](#results-pdpabench-test)
  - [Reproducibility Notes](#reproducibility-notes)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)


## Overview

Existing legal NLP benchmarks (LegalBench, LexGLUE, MLEB) do not include Singapore-specific PDPA QA.  
PDPA-related NLP work focuses on privacy-policy compliance or enforcement decision corpora, but **no publicly documented benchmark or RAG system** exists for PDPA statute question answering.

This repository fills that gap by providing:

### Key Contributions
1. **Authoritative PDPA Corpus Pipeline**  
   - Programmatic ingestion of PDPA Parts 1–6 and 9–10 from Singapore Statutes Online.  
   - Subsection-level chunking with canonical citation labels (e.g., `PDPA s. 4B(1)`).

2. **Hybrid Retrieval Stack**  
   - BM25 + dense retrieval fused via Reciprocal Rank Fusion (RRF).  
   - Fine-tuned cross-encoder reranker optimised for short legal clauses.

3. **Citation-Constrained Generation**  
   - Qwen3-4B SLM with strict evidence-grounding rules.  
   - Every claim must cite retrieved statutory text.  
   - If evidence is insufficient, the model abstains.

4. **PDPABench (500 QAs)**  
   - Canonically cited, section-disjoint, multi-type PDPA QA benchmark.  
   - Metadata includes QA type, difficulty, canonical citations, and support spans.

5. **Reproducible Backend**  
   - Modular orchestrator for retrieval → reranking → generation.  
   - Structured logging for auditability.

An end-to-end architecture diagram is provided in `architectural_diagram.png`.


## Repository Structure

```
DSA4213-GROUP3-PDPA-QA-SYSTEM/
│
├── artefacts/
│   ├── bm25_index/           # Sparse index
│   ├── dense_retriever/      # Dense retriever (fine-tuned)
│   ├── cross_encoder/        # Cross-encoder reranker
│   └── ltr_reranker/         # LightGBM LTR model
│
├── data/
│   ├── corpus/               # PDPA chunks with canonical citations
│   ├── dense_training/       # Dense retriever training triples
│   ├── ltr_processed/        # Reranker training data
│   └── qa/                   # PDPABench (Train/Val/Test)
│
├── src/
│   ├── corpus-gen/           # Corpus construction and chunking
│   ├── generator/            # SLM generation + guardrails
│   ├── qa-gen/               # QA synthesis, cleaning, validation
│   ├── rag_service/          # Orchestrator and API entrypoint
│   ├── rerankers/            # Cross-encoder / LTR rerankers
│   └── retrievers/           # BM25, dense, hybrid retrievers
│
├── checkpoints/              # Optional HuggingFace model weight checkpoints
│── architectural_diagram.png
│── requirements.txt
└── LICENSE              
````

## Installation

Tested on **Python 3.10.6**, **macOS/Linux**, and Apple Silicon (M1/M1 Pro).
```bash
git clone https://github.com/yjh-jy/dsa4213-group3-pdpa-qa-system
cd dsa4213-group3-pdpa-qa-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

This installs all dependencies. No additional training, preprocessing, or corpus construction is required.

## Setup (Model Artefacts)

The dense retriever includes a large model file (`model.safetensors`) that exceeds GitHub’s file-size limit.  
To keep the repository manageable, it is stored in ~90 MB chunks.  
Reassemble the file once before running the API:
```bash
cd artefacts/dense_retriever/model

# reconstruct the full safetensors model
cat model.safetensors.part.* > model.safetensors
```
Once reconstructed, all retrieval and reranking components will load normally.


## Usage

### 1. Start the PDPA RAG API Service

The orchestrator exposes the complete PDPA RAG pipeline (retrieval → reranking → SLM generation → citation constraints → abstention logic).
```bash
cd src/rag_service
uvicorn orchestrator:app --reload --port 8000
```
The API will be available at:
```bash
http://localhost:8000/
```
---
### 2. Interactive API Documentation (FastAPI)

FastAPI automatically provides interactive testing interfaces:

**Swagger UI (interactive testing):**

    http://localhost:8000/docs

**ReDoc UI (schema-first documentation):**

    http://localhost:8000/redoc

Both allow you to run RAG queries directly in your browser without writing code.

---
### 3. Query the API via cURL (via command line)

#### `/ask` — Full RAG pipeline

Runs hybrid retrieval, cross-encoder reranking, citation-constrained generation, and abstention.

```bash
curl -s -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"qid":"q1","question":"What penalty applies for improper use of personal data resulting in harm but without proven gain?"}' \
    | jq .
```
#### `/ask_no_rag` — SLM-only baseline

Bypasses retrieval and reranking.
```bash
curl -s -X POST http://localhost:8000/ask_no_rag \
    -H "Content-Type: application/json" \
    -d '{"qid":"q1","question":"What penalty applies for improper use of personal data resulting in harm but without proven gain?"}' \
    | jq .
```
#### `/evaluate` — Batch evaluate against PDPABench-Test

Useful for PDPABench experiments.

```bash
# Evaluate with RAG
curl -s -X POST http://localhost:8000/evaluate \
            -H "Content-Type: application/json" \
            -d '{"run_name":"rag","with_rag":"True","test_path":"../../data/dense_training/stratified_splits/test_triples.jsonl"}' \
            | jq .
```

```bash
# Evaluate without RAG
curl -s -X POST http://localhost:8000/evaluate \
            -H "Content-Type: application/json" \
            -d '{"run_name":"no_rag","with_rag":"False","test_path":"../../data/dense_training/stratified_splits/test_triples.jsonl"}' \
            | jq .        
```
Each response returns structured JSON containing:
- the generated answer (or abstention),
- retrieved and reranked PDPA evidence,
- emitted canonical citations,
- reranker margin signals,
- latency and metadata.

Quantitative results for `/evaluate` will be generated under the `src/rag_service/eval_runs` folder, where each run's result will create a subfolder with timestamp and whether rag was used or not in its name. Each folder will consist of 2 files: a summary of the eval run, named `summary.json` and the full results for each test question, named `detailed_results.jsonl`.

Qualitative results are under `src/rag_service/eval_runs/qwen_3_4b/qualitative_eval/human`. More details regarding the setup of the qualitative evals are provided in the README_eval.md file under that subdirectory.


## PDPABench and PDPA Corpus (Data Overview)

This repository includes a fully curated and citation-grounded dataset supporting our PDPA legal QA system. The data is organised to allow authoritative retrieval, reproducible evaluation, and transparent benchmarking.

### PDPA Corpus
Located in `data/corpus/`, this corpus contains:
- **Subsection-level PDPA chunks** with canonical citation IDs (`PDPA s.xx(xx)`),
- **Raw PDPA text** (pre-processed from Singapore Statutes Online),
- **Corpus metadata** aligning each chunk to its Part, Section, and Subsection,
- **Citation-preserving JSONL format** for deterministic retrieval.

All statutory content is derived from the PDPA 2012 (rev. 2020) under the Open Data Licence.

### Dense Retriever Training Data
Under `data/dense_training/`, we provide:
- **Five-fold stratified splits** for contrastive retrieval training,
- **Full training triples** (`query`, `positive`, `negative`),
- **Section-disjoint splits** to prevent leakage of statutory provisions across train/validation/test.

### Learning-to-Rank (LTR) Data
Located in `data/ltr_processed/`:
- Processed relevance judgements for the cross-encoder reranker,
- Train/validation/test JSONL formatted sets.

These are generated by combining BM25, dense retrieval hits, and human-validated citation labels.

### PDPABench (500-sample Legal QA Benchmark)
Located in `data/qa/`, PDPABench contains:
- **500 canonical QA pairs** grounded strictly in PDPA statute,
- **Gold citation spans** per question,
- **Four QA types** (pure-definitive, definitive-with-conditions, scenario-ambigious, pure-abstain),
- **Schema card**, **authoring checklist**, and **dataset manifest**,
- **Golden30 seed set** used for evaluation sanity checks,
- **Split structure** (80/10/10) with **section-disjointness guarantees**.

This benchmark supports measurement of:
- citation correctness (Hit@3, Citation F1),
- textual quality (BERTScore, ROUGE-L),
- abstention reliability (confusion matrix),
- end-to-end RAG performance.

A detailed dataset specification is available in `data/README.md`.

## Results (PDPABench-Test)

| Category              | Metric            | Score      |
| --------------------- | ----------------- | ---------- |
| **Citation Fidelity** | Citation Hit Rate | **71.2%**  |
|                       | Citation Recall   | **0.673**  |
| **Textual Quality**   | BERTScore F1      | **0.894**  |
| **Abstention**        | Coverage          | **88.5%**  |
| **Latency**           | Avg per query     | **15.2 s** |

These results demonstrate substantial improvements over a non-RAG baseline (citation hit rate +69 pp).

All statistics are reproducible via the scripts in `/src/`. Please refer to the README.md under `/src` for more details on the respective components of the RAG system.

## Reproducibility Notes

* **Statutory Source:**
  Singapore Statutes Online — *Personal Data Protection Act 2012*
  (Parts 1–6, 9–10; revision as of 2020)

* **Benchmark:**
  PDPABench (500 QAs), stratified and section-disjoint (80/10/10)

* **Hardware:**
  Apple M1 / M1 Pro (16 GB RAM)

* **Model:**
  Qwen3-4B (non-reasoning mode), with strict citation constraints


## Citation

Please cite the accompanying paper as:

```
@inproceedings{2025pdpa_rag,
  title     = {Citation-Constrained, Abstention-Capable RAG for Singapore’s PDPA:
               From Corpus Construction to Reliable Legal QA},
  author    = {Ashley Toh Ke Wei and Choy Qi Hui and Sybella Tan and Yoong Jun Han},
  institution = {National University of Singapore},
  year      = {2025},
  url       = {https://github.com/yjh-jy/dsa4213-group3-pdpa-qa-system}
}
```

## License

Released under the **MIT License**.
PDPA statute text © Government of Singapore, reproduced under the Open Data Licence.

## Acknowledgements

We thank the NUS DSA4213 teaching staff for guidance.
We also acknowledge public initiatives by the **Singapore Academy of Law** and **IMDA** that motivated the exploration of transparent, statute-grounded legal-AI tools.
