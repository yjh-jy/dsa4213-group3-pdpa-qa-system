"""
Legal RAG Orchestrator (Generator, Prompting, Guardrails, API)

This file orchestrate the **rest of the stack**:
  ✓ Intent detection & dynamic prompt templates
  ✓ Prompt assembler with few-shots
  ✓ Self-check pass & abstention logic
  ✓ Strict citation/format validation
  ✓ FastAPI app with structured JSON responses

Usage
-----
1. Start up the API service:
    cd src/rag_service
    uvicorn orchestrator:app --reload --port 8000

2. Run using either the /ask, /ask_no_rag or /evaluate endpoint, some examples are provided below:

    For pure-definitive questions:
    curl -s -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"qid":"q1","question":"What penalty applies for improper use of personal data resulting in harm but without proven gain?"}' \
    | jq .

    For scenario-ambiguous questions:
    curl -s -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"qid":"q1","question":"If a company’s chatbot sends promotional messages using a third-party API, is the company responsible?"}' \
    | jq .

    For end-to-end evaluation (batch testing on unseen test set):
        evaluate with rag:
            curl -s -X POST http://localhost:8000/evaluate \
            -H "Content-Type: application/json" \
            -d '{"run_name":"rag","with_rag":"True","test_path":"../../data/dense_training/stratified_splits/test_triples.jsonl"}' \
            | jq .

        evaluate without rag:
            curl -s -X POST http://localhost:8000/evaluate \
            -H "Content-Type: application/json" \
            -d '{"run_name":"no_rag","with_rag":"False","test_path":"../../data/dense_training/stratified_splits/test_triples.jsonl"}' \
            | jq .
"""

from __future__ import annotations
from datetime import datetime
import os, re, time, json, random, hashlib, sys, csv
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from bert_score import score

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


script_dir = os.path.dirname(__file__)
target_dir = os.path.join(script_dir, '..', 'generator')
sys.path.insert(0, target_dir) # Add the directory to the beginning of the path
target_dir = os.path.join(script_dir, '..', 'retrievers', 'bm25_retrieval')
sys.path.insert(0, target_dir) 
target_dir = os.path.join(script_dir, '..', 'retrievers', 'dense_retrieval')
sys.path.insert(0, target_dir) 
target_dir = os.path.join(script_dir, '..', 'retrievers', 'hybrid_retrieval')
sys.path.insert(0, target_dir) 
target_dir = os.path.join(script_dir, '..',  'rerankers', 'cross_encoder_reranker')
sys.path.insert(0, target_dir) 

from config import *
from guardrails import *
from slm import *

random.seed(RANDOM_SEED)

# ---------------------- External hooks -----------------
try:
    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever
    from hybrid_retriever import HybridRetriever
    from hybrid_retriever import create_hybrid_retriever

    # Initalize the retriever models first
    # DENSE = DenseRetriever(model_name='artefacts/dense_retriever/model')
    # BM25 = BM25Retriever()
    HYBRID = create_hybrid_retriever(checkpoint_path='artefacts/dense_retriever/model')

    @dataclass
    class Retrieved:
        ids: List[str]
        scores: Dict[str, float]

    def bm25_topk(query: str, k: int, rerank: bool = True) -> List[str]:
        BM25 = BM25Retriever()
        all_results = BM25.search(query=query, top_k=k)
        if rerank:
            results = [res for res in all_results['results']]
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return results, ids, scores
        else:
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return Retrieved(ids=ids, scores=scores)

    def dense_topk(query: str, k: int, rerank: bool = True) -> List[str]:
        DENSE = DenseRetriever(model_name='artefacts/dense_retriever/model')
        all_results = DENSE.search(query=query, top_k=k)
        if rerank:
            results = [res for res in all_results['results']]
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return results, ids, scores
        else:
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return Retrieved(ids=ids, scores=scores)

    def hybrid_topk(query: str, k: int, rerank: bool = True):
        all_results = HYBRID.search(query=query, top_k=k)
        if rerank:
            results = [res for res in all_results['results']]
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return results, ids, scores
        else:
            ids = [res['chunk_id'] for res in all_results['results']]
            scores = [res['score'] for res in all_results['results']]
            return Retrieved(ids=ids, scores=scores)

except Exception as e:
    print(e)
    print("[orchestrator.py] Failed to load model")
    def bm25_topk(query: str, k: int) -> List[str]:
        raise NotImplementedError("bm25_topk not provided")
    def dense_topk(query: str, k: int) -> List[str]:
        raise NotImplementedError("dense_topk not provided")
    def hybrid_topk(query: str, k: int) -> List[str]:
        raise NotImplementedError("hybrid_topk not provided")
    

try:
    from cross_encoder_reranker import CrossEncoderReranker
    CE_RERANKER = CrossEncoderReranker(model_name_or_path='artefacts/cross_encoder/model')
    def cross_encoder_rerank(query: str, candidates: List[str], final_k: int) -> List[str]:
        all_results = CE_RERANKER.rerank(query=query, candidates=candidates, top_k=final_k)
        ids = [res['chunk_id'] for res in all_results]
        scores = [res['ce_score'] for res in all_results]
        return Retrieved(ids=ids, scores=scores)

except Exception:
    def cross_encoder_rerank(query: str, candidates: List[str], final_k: int) -> List[str]:
        NotImplementedError("cross_encoder reranker not provided")
    
# Useful dictionaries for metadata retrievals
try:
    STATUTE_PATH = os.getenv("STATUTE_JSONL", "../../data/corpus/corpus_subsection_v1.jsonl")
    CORPUS: List[Dict[str,Any]] = []
    ID2ROW: Dict[str,Dict[str,Any]] = {}
    FULL_QA_PATH = '../../data/qa/pdpa_qa_500.jsonl'
    QA: List[Dict[str,Any]] = []
    ID2QA: Dict[str, Dict[str, Any]] = {}
    CANONICAL2CID: Dict[str, str] = {}

    def _load_corpus_and_qas():
        global CORPUS, ID2ROW, ID2QA, CANONICAL2CID
        CORPUS = []
        with open(STATUTE_PATH, "rb") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                CORPUS.append(row)
        ID2ROW = {r["chunk_id"]: r for r in CORPUS}
        CANONICAL2CID = {r["canonical_citation"]: r['chunk_id'] for r in CORPUS}

        with open(FULL_QA_PATH, "rb") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                QA.append(row)

        ID2QA = {r['id']: r for r in QA}

    _load_corpus_and_qas()

except Exception as e:
    CORPUS: Dict[str, Dict[str, Any]] = {}
    ID2ROW: Dict[str, Dict[str, Any]] = {}
    print(e)


# --------------------- Evidence building ------------------
def enrich_evidence_context(selected_ids: List[str], rrf_scores = None, rerank_scores = None) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen = set()
    for did in selected_ids:
        row = ID2ROW.get(did)
        if not row or did in seen:
            continue
        seen.add(did)
        evidence.append({
            "id": row.get("canonical_citation"),
            "marginal_note": row.get("marginal_note", ""),
            "rrf_score": None if rrf_scores is None else float(rrf_scores[selected_ids.index(did)]),
            "rerank_score": None if rerank_scores is None else float(rerank_scores[selected_ids.index(did)]),
            "text": row.get("text", "")
        })
        # Simple cross-ref pull-through if text hints at cross refs (customize pattern as needed)
        if re.search(r"subject to s\.[0-9A-Za-z]+", row.get("text", "")):
            # You can parse and include that clause once
            pass

    return evidence

# --------------------------- API types --------------------------
class AskIn(BaseModel):
    qid: str
    question: str

class AskOut(BaseModel):
    qid: str
    question: str
    answer: Dict[str, Any]
    retrieval: Dict[str, Any]
    rerank: Dict[str, Any]
    diagnostics: Dict[str, Any]

# ----------------------------- API ------------------------------
app = FastAPI(title="Legal RAG Orchestrator", version="0.1")


# ----------------------------- Vanilia SLM endpoint ------------------------------
@app.post("/ask_no_rag", response_model=AskOut)
def ask_no_rag(payload: AskIn):
    t0 = time.time()
    
    sys_prompt = '''You are a careful Singapore data-privacy assistant. You will be given a question about Singapore's PDPA (Personal Data Protection Act) and answer to the best of your abilities.
    If you deem the context to be insufficient, ABSTAIN from answering: \"I’m not sure; this appears outside the provided statute.\" AND ask ONE clarifying question.\n"
    
    FORMAT:\nWrite in lawyer-plain English. Answer concisely in 2-5 sentences. Preserve numbers and legal terms. Include the specific canonical citations in your sentences like this: PDPA s.xx(xx). Do not use any other format to quote.'''
    
    prompt_hash = hashlib.md5(sys_prompt.encode()).hexdigest()[:8]

    # 1) generate
    text, reasoning_text = slm_generate(
        sys_prompt=sys_prompt, 
        question=payload.question, 
        max_new_tokens=MAX_NEW_TOKENS_QWEN3_THINKING if ENABLE_REASONING else MAX_NEW_TOKENS_QWEN3_NON_THINKING,
        use_reasoning=ENABLE_REASONING
        )

    # 2) validate citations
    cov = citation_coverage(text)
    cites = extract_citations(text)
    answer = {
        "text": text,
        "citations": cites,
        "abstained": False if cites else True, # Assume if not citations captured => it abstained on its own
        "reasoning_text": reasoning_text if ENABLE_REASONING else None
    }

    return AskOut(
        qid=payload.qid,
        question=payload.question,
        answer=answer,
        retrieval={
            "top_retrieved_passages": [],
            "max_retrieval_score": None,
            "max_threshold": None,
            "margin_threshold": None
        },
        rerank={
            "top_reranked_passages": [],
            "max_reranked_score": None,
            "max_threshold": None,
            "margin_threshold": None,
        },
        diagnostics={
            "prompt_hash": prompt_hash,
            "model": HF_GENERATOR_MODEL,
            "reasoning_enabled": ENABLE_REASONING,
            "total_latency_s": round(time.time() - t0, 2),
            "retrieval_latency_ms": None,
            "reranking_latency_ms": None,
            "citation_coverage": cov,
            "system_prompt_used": sys_prompt
        }
    )

# ----------------------------- RAG + SLM endpoint ------------------------------
@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    t0 = time.time()

    # 1) intent & retrieval (BM25/ Dense/ RRF)
    intent = detect_intent(payload.question)
    t1 = time.time()
    retrieved_context, retrieved_ids, retrieved_scores = hybrid_topk(payload.question, TOPK_RETRIEVER, rerank=True)
    retrieval_time_ms = round((time.time() - t1)*1000, 2)

    # 2) rerank and select
    t2 = time.time()
    rerank_out = cross_encoder_rerank(payload.question, retrieved_context, final_k=K_GEN)
    reranking_time_ms = round((time.time() - t2)*1000, 2)
    reranked_ids, reranked_scores = rerank_out.ids, rerank_out.scores

    # 3) build evidence & prompt
    evidence = enrich_evidence_context(selected_ids=reranked_ids, rrf_scores=retrieved_scores, rerank_scores=reranked_scores)
    sys_prompt = build_prompt(payload.question, evidence, intent=intent, fewshot_k=1)    
    prompt_hash = hashlib.md5(sys_prompt.encode()).hexdigest()[:8]

    # 4) generate
    text, reasoning_text = slm_generate(
        sys_prompt=sys_prompt, 
        question=payload.question, 
        max_new_tokens=MAX_NEW_TOKENS_QWEN3_THINKING if ENABLE_REASONING else MAX_NEW_TOKENS_QWEN3_NON_THINKING,
        use_reasoning=ENABLE_REASONING
        )

    # 5) validate citations
    cov = citation_coverage(text)
    cites = extract_citations(text)
    answer = {
        "text": text,
        "citations": cites,
        "abstained": False if cites else True, # Assume if not citations captured => it abstained on its own
        "reasoning_text": reasoning_text if ENABLE_REASONING else None
    }

    # 6) format retrieval and reranker block
    top_retrieved_passages = []
    for did in retrieved_ids:
        row = ID2ROW.get(did)
        top_retrieved_passages.append({
            "id": did,
            "sample_text": (row.get("text", "") or "")[:150],
            "retrieval_score": float(retrieved_scores[retrieved_ids.index(did)]),
        })
    reranked_passages = []
    for did in reranked_ids:
        row = ID2ROW.get(did)
        reranked_passages.append({
            "id": did,
            "sample_text": (row.get("text", "") or "")[:300],
            "reranked_score": float(reranked_scores[reranked_ids.index(did)]),
        })

    # 7) Abstention Gates (Hard Logic)
    retrieved_scores.sort(reverse=True)
    reranked_scores.sort(reverse=True)
    thr = {"rrf_top": TAU_RETR, "rrf_margin": TAU_RETR_MARGIN, "ce_top": TAU_RERANK_MARGIN, "ce_margin": TAU_RERANK_MARGIN}
    abstain, abstain_reason = decide_abstain(
        rrf_top=float(max(retrieved_scores, default=0.0)), 
        rrf_margin=retrieved_scores[0] - retrieved_scores[1],
        ce_top=float(max(reranked_scores, default=0.0)),
        ce_margin=reranked_scores[0] - reranked_scores[1],
        thresholds=thr
    )

    if abstain:
        answer = {
            "text": "I'm not sure; this appears outside the provided statute.",
            "original_text": text,
            "citations": [],
            "reasoning_text": reasoning_text if ENABLE_REASONING else None,
            "abstained": True,
            "abstain_reason": abstain_reason
        }
        return AskOut(
            qid=payload.qid,
            question=payload.question,
            answer=answer,
            retrieval={
                "top_retrieved_passages": top_retrieved_passages,
                "max_retrieval_score": float(max(retrieved_scores, default=0.0)),
                "retrieval_margin": retrieved_scores[0] - retrieved_scores[1],
                "max_threshold": TAU_RETR,
                "margin_threshold": TAU_RETR_MARGIN
            },
            rerank={
                "top_reranked_passages": reranked_passages,
                "max_reranked_score": float(max(reranked_scores, default=0.0)),
                "rerank_margin": reranked_scores[0] - reranked_scores[1],
                "max_threshold": TAU_RERANK,
                "margin_threshold": TAU_RERANK_MARGIN,
            },
            diagnostics={
                "intent": intent,
                "prompt_hash": prompt_hash,
                "model": HF_GENERATOR_MODEL,
                "reasoning_enabled": ENABLE_REASONING,
                "total_latency_s": round(time.time() - t0, 2),
                "retrieval_latency_ms": retrieval_time_ms,
                "reranking_latency_ms": reranking_time_ms,
                "citation_coverage": cov,
                "system_prompt_used": sys_prompt
            }
        )

    return AskOut(
        qid=payload.qid,
        question=payload.question,
        answer=answer,
        retrieval={
            "top_retrieved_passages": top_retrieved_passages,
            "max_retrieval_score": float(max(retrieved_scores, default=0.0)),
            "retrieval_margin": retrieved_scores[0] - retrieved_scores[1],
            "max_threshold": TAU_RETR,
            "margin_threshold": TAU_RETR_MARGIN
        },
        rerank={
            "top_reranked_passages": reranked_passages,
            "max_reranked_score": float(max(reranked_scores, default=0.0)),
            "rerank_margin": reranked_scores[0] - reranked_scores[1],
            "max_threshold": TAU_RERANK,
            "margin_threshold": TAU_RERANK_MARGIN,
        },
        diagnostics={
            "intent": intent,
            "prompt_hash": prompt_hash,
            "model": HF_GENERATOR_MODEL,
            "reasoning_enabled": ENABLE_REASONING,
            "total_latency_s": round(time.time() - t0, 2),
            "retrieval_latency_ms": retrieval_time_ms,
            "reranking_latency_ms": reranking_time_ms,
            "citation_coverage": cov,
            "system_prompt_used": sys_prompt
        }
    )

# ------------------------ Evaluate -------------------
class EvalIn(BaseModel):
    run_name: str
    with_rag: bool = True
    test_path: str  # JSONL with {qid, question, gold_ids:[...], intent?, is_abstain?}

# ---------- Evaluate endpoint (for batch testing) ----------
@app.post("/evaluate")
def evaluate(payload: EvalIn):
    score(['22'], ['22'], lang='en', verbose=True)  # warm up BERTScore

    R10, R3 = [], []
    total_latency_list = []
    Abst_tp = Abst_fp = Abst_fn = Abst_tn = 0

    Support_hits = Support_pred = Support_gold = 0
    Citation_hit_examples = 0  # fraction of examples with any overlap

    em_list: List[int] = []
    rouge_list: List[float] = []
    BERTScore_f1_list: List[float] = []
    BERTScore_recall_list: List[float] = []
    BERTScore_precision_list: List[float] = []

    ece_conf: List[float] = []
    ece_corr: List[int] = []

    # per-example logs for disk
    per_example_logs: List[Dict[str, Any]] = []

    # F1 variants
    f1_answerable_list: List[float] = []      # F1 over all answerable items (conditional)
    f1_answered_only_list: List[float] = []   # F1 over items the model actually answered (and are answerable)

    total_examples = 0
    seen_qid = set()
    system_prompt_used, model_name = None, None

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    outdir = f"eval_runs/{payload.run_name}_{ts}"
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = os.path.join(outdir, "detailed_results.jsonl")

    with open(payload.test_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            # try:
                if not line.strip():
                    continue
                ex = json.loads(line)
                qid = ex.get("qid")
                if qid in seen_qid:
                    continue  # stopgap measure to account for dupes
                query = ex.get("query")
                if not query:
                    continue  # skip malformed line

                t1 = time.time()
                if bool(payload.with_rag): 
                    print('#'*6+" Evaluating RAG VERSION "+'#'*6)
                    out = ask(AskIn(qid=qid, question=query))
                else:
                    print('#'*6+" Evaluating NON RAG VERSION "+'#'*6)
                    out = ask_no_rag(AskIn(qid=qid, question=query))

                out = out.model_dump()  # convert to json/dict
                latency_s = round(time.time() - t1, 3)
                total_latency_list.append(latency_s)

                total_examples += 1
                seen_qid.add(qid)

                if system_prompt_used is None:
                    system_prompt_used = out.get("diagnostics", {}).get("system_prompt_used", "")
                if model_name is None:
                    model_name = out.get("diagnostics", {}).get("model", "")

                # ----- Abstention metrics -----
                predicted_abstain = _get_pred_abstain(out)
                gold_abstain = ID2QA.get(qid)['abstain_allowed']

                if gold_abstain and predicted_abstain:
                    Abst_tp += 1
                elif gold_abstain and not predicted_abstain:
                    Abst_fn += 1
                elif not gold_abstain and predicted_abstain:
                    Abst_fp += 1
                else:
                    Abst_tn += 1

                # ----- Retrieval Recall@10 -----
                retrieved_ids = _get_used_ids(out, k=10, retrieve=True)
                gold_ids = set(
                    CANONICAL2CID.get(canonical)
                    for canonical in ID2QA.get(qid, {}).get('canonical_sections', [])
                )
                gold_ids.discard(None)
                r_at_10_hit = 1 if any(g in retrieved_ids for g in gold_ids) else 0
                R10.append(r_at_10_hit)

                # ----- Reranking Recall@3 -----
                reranked_ids = _get_used_ids(out, k=3, retrieve=False)
                r_at_3_hit = 1 if any(g in reranked_ids for g in gold_ids) else 0
                R3.append(r_at_3_hit)

                # ----- Citation metrics -----
                pred_cites = set(_get_pred_citations(out))
                pred_cites.discard(None)
                hits = len(pred_cites & gold_ids)
                Support_hits += hits
                Support_pred += len(pred_cites)
                Support_gold += len(gold_ids)
                if hits > 0:
                    Citation_hit_examples += 1

                # Per-example citation P/R/F1 (define defaults)
                if len(pred_cites) == 0 and len(gold_ids) == 0:
                    cit_P = cit_R = cit_F1 = 1.0  # degenerate perfect
                else:
                    cit_P = hits / max(1, len(pred_cites))
                    cit_R = hits / max(1, len(gold_ids))
                    cit_F1 = 0.0 if (cit_P == 0 and cit_R == 0) else (2 * cit_P * cit_R) / (cit_P + cit_R)

                # F1 variants (only for answerable items with any gold ids)
                if not gold_abstain and len(gold_ids) > 0:
                    f1_answerable_list.append(cit_F1)
                    if not predicted_abstain:
                        f1_answered_only_list.append(cit_F1)

                # ----- Generation Quality (EM / ROUGE-L / BERTScore) -----
                gold_answer_long = ID2QA.get(qid, {}).get('gold_answer_extended')
                pred_text = _get_pred_answer(out)

                em_i = None
                rouge_i = None
                b_p = b_r = b_f = None

                if isinstance(gold_answer_long, str) and gold_answer_long.strip():
                    # BERTScore returns tensors; take float mean
                    bp, br, bf = score([pred_text], [gold_answer_long], lang='en')
                    # convert to plain floats
                    b_p = float(bp.mean().item()) if hasattr(bp, "mean") else float(bp)
                    b_r = float(br.mean().item()) if hasattr(br, "mean") else float(br)
                    b_f = float(bf.mean().item()) if hasattr(bf, "mean") else float(bf)
                    BERTScore_precision_list.append(b_p)
                    BERTScore_recall_list.append(b_r)
                    BERTScore_f1_list.append(b_f)

                    em_i = exact_match(pred_text, gold_answer_long)
                    rouge_i = rouge_l_f(pred_text, gold_answer_long)
                    em_list.append(em_i)
                    rouge_list.append(rouge_i)

                # ----- Calibration (ECE over EM correctness if confidence available) -----
                conf = _get_pred_confidence(out)
                if conf is not None and isinstance(gold_answer_long, str) and gold_answer_long.strip():
                    ece_conf.append(conf)
                    ece_corr.append(exact_match(pred_text, gold_answer_long))

                # ----- write per-example JSONL row -----
                per_row = {
                    "qid": qid,
                    "query": query,
                    "gold": {
                        "abstain_allowed": bool(gold_abstain),
                        "canonical_sections": ID2QA.get(qid, {}).get("canonical_sections", []),
                        "gold_answer_extended": gold_answer_long,
                    },
                    "prediction": {
                        "abstained": bool(predicted_abstain),
                        "abstain_reason": out.get("answer", {}).get("abstain_reason", None),
                        "answer_text": pred_text, # what it actually outputted
                        "original_text": out.get("answer", {}).get("original_text", None) if bool(predicted_abstain) else None, # provide the original model output if it abstained for analytics
                        "reasoning_text": out.get("answer", {}).get("reasoning_text", None), # reasoning text for analytics for reasoning model
                        "citations": sorted(list(pred_cites)),
                    },
                    "retrieval": {
                        f"used_ids_k{TOPK_RETRIEVER}": retrieved_ids,
                        f"r_at_{TOPK_RETRIEVER}_hit": int(r_at_10_hit),
                        "top_passages": out.get("retrieval", {}).get("top_retrieved_passages", []),  # includes scores
                        "max_retrieval_score": out.get("retrieval", {}).get("max_retrieval_score"),
                        "retrieval_margin":  out.get("retrieval", {}).get("retrieval_margin"),
                        "max_threshold": out.get("retrieval", {}).get("max_threshold"),
                        "margin_threshold": out.get("retrieval", {}).get("margin_threshold"),
                    },
                    "reranking": {
                        f"used_ids_r{K_GEN}": reranked_ids,
                        f"r_at_{K_GEN}_hit": int(r_at_3_hit),
                        "top_passages": out.get("rerank", {}).get("top_reranked_passages", []),  # includes scores
                        "max_reranked_score": out.get("rerank", {}).get("max_reranked_score"),
                        "retrieval_margin":  out.get("rerank", {}).get("rerank_margin"),
                        "max_threshold": out.get("rerank", {}).get("max_threshold"),
                        "margin_threshold": out.get("rerank", {}).get("margin_threshold"),
                    },
                    "metrics": {
                        "citation_precision": float(cit_P),
                        "citation_recall": float(cit_R),
                        "citation_f1": float(cit_F1),
                        "em": em_i,
                        "rouge_l": rouge_i,
                        "bertscore_precision": b_p,
                        "bertscore_recall": b_r,
                        "bertscore_f1": b_f,
                        "confidence": conf,
                        "latency_s": latency_s,
                    },
                    "diagnostics": out.get("diagnostics", {}),
                }
                with open(jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(per_row, ensure_ascii=False) + "\n")
                
            # except Exception as e:
            #     print(e)
            #     print(f"Stopped at:\n{ex}\n")

    # ----- Aggregate -----
    def avg(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    hit_at_10_retrieval = float(sum(R10) / max(1, len(R10)))
    hit_at_3_reranking = float(sum(R3) / max(1, len(R3)))

    citation_precision = (Support_hits / max(1, Support_pred))
    citation_recall    = (Support_hits / max(1, Support_gold))
    citation_f1        = f1(Support_hits, Support_pred, Support_gold)
    citation_hit_rate  = float(Citation_hit_examples / max(1, total_examples))

    f1_answerable_cond = avg(f1_answerable_list)
    f1_answered_only   = avg(f1_answered_only_list)

    em = avg(em_list)
    rouge_l = avg(rouge_list)
    bertscore_f1 = avg(BERTScore_f1_list)
    bertscore_recall = avg(BERTScore_recall_list)
    bertscore_precision = avg(BERTScore_precision_list)

    ece = expected_calibration_error(ece_conf, ece_corr, n_bins=10)  # may be None

    abst_total = Abst_tp + Abst_fp + Abst_fn + Abst_tn
    abst_precision = Abst_tp / max(1, (Abst_tp + Abst_fp))
    abst_recall    = Abst_tp / max(1, (Abst_tp + Abst_fn))
    abst_f1        = (2 * abst_precision * abst_recall) / max(1e-12, (abst_precision + abst_recall))
    abst_acc       = (Abst_tp + Abst_tn) / max(1, abst_total)
    coverage       = (Abst_tn + Abst_fn) / max(1, abst_total)

    summarized_results = {
        "run": payload.run_name,
        "outdir": outdir,
        "files": {
            "per_question_jsonl": jsonl_path,
        },
        "n_examples": total_examples,
        "Retrieval": {
            f"Hit@{TOPK_RETRIEVER}": float(hit_at_10_retrieval)
        },
        "Reranking": {
            f"Hit@{K_GEN}": float(hit_at_3_reranking)
        },
        "Citations": {
            "Precision": float(citation_precision),
            "Recall": float(citation_recall),
            "F1": float(citation_f1),
            "Citation_HitRate": float(citation_hit_rate),
            "Support": {"hits": int(Support_hits), "pred": int(Support_pred), "gold": int(Support_gold)},
            "F1_AnswerableConditional": f1_answerable_cond,
            "F1_AnsweredOnly": f1_answered_only
        },
        "AnswerString": {
            "ExactMatch": em,
            "ROUGE-L": rouge_l,
            "BERTScore_Precision": bertscore_precision,
            "BERTScore_Recall": bertscore_recall,
            "BERTScore_F1": bertscore_f1,
        },
        "Abstention": {
            "TP": int(Abst_tp), "FP": int(Abst_fp), "FN": int(Abst_fn), "TN": int(Abst_tn),
            "Precision": float(abst_precision),
            "Recall": float(abst_recall),
            "F1": float(abst_f1),
            "Accuracy": float(abst_acc),
            "Coverage": float(coverage),
            "Thresholds": {
                "Max_retriever_score": TAU_RETR,
                "Max_retriever_margin": TAU_RETR_MARGIN,
                "Max_reranker_score": TAU_RERANK,
                "Max_reranker_margin": TAU_RERANK_MARGIN,
            }
        },
        "Calibration": {
            "ECE_10bin": ece
        },
        "Diagnostics": {
            "Model_name": f"{model_name}",
            "Reasoning_enabled": ENABLE_REASONING,
            "Model_sampling_parameters": {
                "Temperature": TEMPERATURE_QWEN3_THINKING if ENABLE_REASONING else TEMPERATURE_QWEN3_NON_THINKING,
                "Top_P": TOP_P_QWEN3_THINKING if ENABLE_REASONING else TOP_P_QWEN3_NON_THINKING,
                "Top_K": TOP_K_QWEN3_THINKING if ENABLE_REASONING else TOP_K_QWEN3_NON_THINKING
            },
            "Avg_latency_per_question_s": avg(total_latency_list),
            "Sys_prompt_hash": out.get("diagnostics", {}).get("prompt_hash", None),
            "Sys_prompt": system_prompt_used
        }
    }

    # Save summarized results
    try:
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summarized_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(e)
        print("Failed to save summary.json")

    return summarized_results

# ---------- Helpers ----------
_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r'[\W_]+', re.UNICODE)

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(' ', s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return ' '.join(tokens)

def exact_match(pred: str, gold: str) -> int:
    return int(_normalize_answer(pred) == _normalize_answer(gold))

def _lcs_len(a: List[str], b: List[str]) -> int:
    # classic LCS DP
    dp = [0]*(len(b)+1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j-1])
            prev = cur
    return dp[-1]

def rouge_l_f(pred: str, gold: str) -> float:
    # ROUGE-L F-score (token-level, beta=1)
    if not pred or not gold:
        return 0.0
    p_toks, g_toks = pred.split(), gold.split()
    lcs = _lcs_len(p_toks, g_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_toks)
    rec  = lcs / len(g_toks)
    return (2 * prec * rec) / (prec + rec + 1e-12)

def f1(n_hit, n_pred, n_gold) -> float:
    p = n_hit / max(1, n_pred)
    r = n_hit / max(1, n_gold)
    return (2*p*r) / max(1e-12, (p + r))

def expected_calibration_error(confidences: List[float], correct: List[int], n_bins: int = 10) -> Optional[float]:
    """
    confidences: list of predicted probabilities in [0,1]
    correct: list of 0/1 correctness (we use EM as correctness proxy)
    """
    if not confidences or not correct or len(confidences) != len(correct):
        return None
    # clamp and bin
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correct):
        c = min(1.0, max(0.0, float(c)))
        # rightmost edge goes into last bin
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, y))
    ece = 0.0
    n = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b)/n) * abs(acc - avg_conf)
    return float(ece)

def _get_used_ids(out: Dict[str, Any], k: int = 5, retrieve = None) -> List[str]:
    passages = out.get("retrieval", {}) if retrieve else out.get("rerank", {})
    seq = passages.get("top_retrieved_passages", []) if retrieve else passages.get("top_reranked_passages", [])
    return [p.get("id") for p in seq[:k]]

def _get_pred_answer(out: Dict[str, Any]) -> str:
    return out.get("answer", {}).get("text", "")

def _get_pred_citations(out: Dict[str, Any]) -> List[str]:
    raw_cites = out.get("answer", {}).get("citations", [])

    def _normalize_citation(c: str) -> str:
        if not c:
            return ""
        c = str(c)
        # Strip smart-excerpt labels like " [A]"
        c = re.sub(r"\s*\[[A-Z]\]\s*$", "", c)
        c = re.sub(r"(?i)^\s*pdpa\s*", "", c)
        c = re.sub(r"(?i)^\s*s\.\s*", "", c)
        c = re.sub(r"\s+", "", c)

        # Keep only up to the first parentheses group
        # Example: 15A(3)(b)(ii) → 15A(3)
        c = re.sub(r"^([0-9A-Za-z]+(?:\([^()]+\))?).*$", r"\1", c)

        return f"PDPA s.{c}"

    # normalize from canonical form to cid
    return set(CANONICAL2CID.get(_normalize_citation(c)) for c in raw_cites)

def _get_pred_abstain(out: Dict[str, Any]) -> bool:
    return bool(out.get("answer", {}).get("abstained", None))

def _get_pred_confidence(out: Dict[str, Any]) -> Optional[float]:
    # Try common keys; return None if unavailable
    ans = out.get("answer", {}) 
    for key in ("confidence", "prob", "score"):
        if key in ans and isinstance(ans[key], (int, float)):
            return float(ans[key])
    # sometimes top-level confidence
    for key in ("confidence", "prob", "score"):
        if key in out and isinstance(out[key], (int, float)):
            return float(out[key])
    return None